# vLLM Production Stack と LMCache KV オフロード

> vLLM の production-stack は参照用 Kubernetes デプロイメント — ルーター、エンジン、および可観測性が一体。LMCache は KV キャッシュを GPU メモリから抽出して複数のクエリとエンジン間で再利用する KV オフロード層（CPU DRAM、その後 disk/Ceph）。vLLM 0.11.0 KV オフロード コネクタ（2026 年 1 月）はこれを非同期で、Connector API（v0.9.0+）経由でプラグ可能にする。オフロード遅延はユーザー向けではない。LMCache は共有プレフィックスなしでも有価値 — GPU が KV スロットを使い果たすと、preempted リクエストは prefill を再計算せず CPU から復元できる。16x H100（80GB HBM）での発表ベンチマーク、4 個の a3-highgpu-4g にわたるとき：KV キャッシュが HBM を超えると、ネイティブ CPU オフロードと LMCache の両者がスループットを大幅に改善。KV フットプリントが低いと、すべての設定がベースラインと一致し、小さなオーバーヘッド。

**タイプ:** Learn
**言語:** Python (stdlib, toy KV-spill simulator)
**前提条件:** Phase 17 · 04 (vLLM Serving Internals)、Phase 17 · 06 (SGLang/RadixAttention)
**所要時間:** 約 60 分

## 学習目標

- vLLM production-stack レイヤーを図式化：ルーター、エンジン、KV オフロード、可観測性。
- KV オフロード コネクタ API（v0.9.0+）と 0.11.0 非同期パスがオフロード遅延を隠す方法を説明。
- LMCache CPU-DRAM が役立つ場合（KV > HBM）vs オーバーヘッド追加（KV HBM に十分小）を定量化。
- デプロイメント制約を指定して、ネイティブ vLLM CPU オフロード と LMCache コネクタの間を選択。

## 問題

vLLM サービングは GPU を 100% HBM で表示、並行処理が上がるたびに preemption イベント。リクエストが削除され、再キュー、1 分で同じ 2K トークン prompt を 4 回 prefill している。GPU コンピュートは redundant prefill に費やされ、goodput は生スループット以下。

GPU を追加するにはリニアコスト。HBM を追加することはできない。しかし CPU DRAM は安い — 1 ソケットは 512 GB+ で、HBM よりも遥かに悪いが「一時的にウォーム」KV キャッシュに対して OK な遅延。

LMCache は KV キャッシュを CPU DRAM に抽出して preempted リクエストが高速に回復でき、エンジン間の重複 prefix がキャッシュを共有でき各エンジンが re-prefill を避ける。

## コンセプト

### vLLM production-stack

`github.com/vllm-project/production-stack` は参照用 Kubernetes デプロイメント：

- **ルーター** — キャッシュ対応（Phase 17 · 11）。KV イベントを消費。
- **エンジン** — vLLM ワーカー。GPU ごと、または TP/PP グループごとに 1 つ。
- **KV キャッシュ オフロード** — LMCache デプロイメント、またはネイティブコネクタ。
- **可観測性** — Prometheus スクレープ、Grafana ダッシュボード、OTel トレース。
- **コントロール プレーン** — サービス発見、設定、ローリングアップデート。

Helm チャート + オペレーターとして配送。

### KV オフロード コネクタ API（v0.9.0+）

vLLM 0.9.0 はプラグ可能な KV キャッシュバックエンド用コネクタ API を導入。エンジンはブロックをコネクタにオフロード。コネクタはそれらを保存（RAM、disk、オブジェクトストレージ、LMCache）。リクエストがブロックを必要とするとき、コネクタはそれをロードバック。

vLLM 0.11.0（2026 年 1 月）は非同期オフロード パスを追加 — エンジンが共通の場合にブロックされない可能性があるようにオフロードはバックグラウンドで発生。エンドツーエンド遅延とスループットは依然としてワークロード形状、KV キャッシュヒット率、システム圧力に依存。vLLM 自身の注記は、カスタムカーネルオフロードが低ヒット率でスループットを低下させる可能性があり、非同期スケジューリングが推測復号化との既知相互作用問題を持つ可能性があることを指摘。

### ネイティブ CPU オフロード vs LMCache

**ネイティブ vLLM CPU オフロード**：エンジンローカル。KV ブロックをホスト RAM に保存。実装は高速、ゼロネットワークホップ。エンジン間を越えない。

**LMCache コネクタ**：クラスタスケール。ブロックを共有 LMCache サーバー（CPU DRAM + Ceph/S3 ティア）に保存。ブロックはどのエンジンからもアクセス可能。16x H100 ベンチマークが発表。

1 つのエンジンに HBM プレッシャーがあるときネイティブを選択。複数エンジンが prefix を共有するとき LMCache を選択（共通システム prompt での RAG、共有テンプレートのマルチテナント）。

### ベンチマーク振舞

16x H100（80 GB HBM）4 個の a3-highgpu-4g にわたるテスト：

- 低 KV フットプリント（短 prompt、低並行処理）：すべての設定がベースラインと一致、LMCache は約 3-5% オーバーヘッドを追加。
- 中程度フットプリント：LMCache はエンジン間の prefix 再利用で役立ち始める。
- KV が HBM を超えると：ネイティブ CPU オフロード と LMCache の両者がスループットを大幅に改善。LMCache はクロスエンジン共有のため大きな利益。

### LMCache が決定的な場合

- マルチテナントサービング、システム prompt がテナント間で共有。
- RAG、ドキュメント chunk がクエリ間で繰り返す。
- ファインチューン variant（LoRA）同じベースで base-model KV 再利用が redundant 作業をカット。
- Preemption-heavy ワークロード：CPU から復元が re-prefill より安価。

### 有効にしない場合

- 小さい HBM プレッシャー — 利益なしでオーバーヘッド支払う。
- 短い context（<1K トークン） — 転送時間 > re-prefill。
- シングルテナント シングル-prompt ワークロード — 再利用キャプチャなし。

### Disaggregated serving との統合

Phase 17 · 17 disaggregated serving + LMCache は複合：prefill プールから decode プールへの KV 転送は使用されなければ LMCache に着地。後続クエリは LMCache から pull。Phase 17 · 11 キャッシュ対応ルーターはローカル OR LMCache-shared キャッシュマッチエンジンにルーティング可能。

### 記憶すべき数字

- vLLM 0.9.0：コネクタ API 配送。
- vLLM 0.11.0（2026 年 1 月）：非同期オフロード パス。エンドツーエンド遅延への影響はワークロード、KV ヒット率、システムプレッシャーに依存（絶対保証ではない）。
- 16x H100 ベンチマーク：KV フットプリントが HBM を超えると LMCache が役立つ。
- 小さい HBM プレッシャー：利益なしで 3-5% オーバーヘッド。

## 使用方法

`code/main.py` は preemption-heavy ワークロード、LMCache ありなし、をシミュレート。避けられた re-prefill、スループット利益、break-even HBM 利用率を報告。

## 配送

このレッスンは `outputs/skill-vllm-stack-decider.md` を作成。ワークロード形状と vLLM デプロイメントを指定して、ネイティブ vs LMCache vs どちらでもない、を決定。

## 演習

1. `code/main.py` を実行。何の HBM 利用率で LMCache が報酬をもたらし始めるか？
2. テナントは 200 クエリ/時間で 6K トークン システム prompt を共有。テナントあたり期待される LMCache 節約を計算。
3. LMCache サーバーは単一障害点。HA ストラテジーを設計（レプリカ、ネイティブへのフォールバック）。
4. LMCache は Ceph spinning disk に保存。70B FP8（500 MB）の 4K トークン KV、読み取り時間 vs re-prefill は？
5. vLLM 0.11.0 非同期パスが「無料」かどうか議論。オーバーヘッドはどこに隠される？

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| Production-stack | "参照デプロイメント" | vLLM の Kubernetes Helm チャート + オペレーター |
| Connector API | "KV バックエンド インターフェース" | vLLM 0.9.0+ プラグ可能な KV ストア インターフェース |
| Native CPU offload | "エンジン ローカル spill" | 同じエンジンのホスト RAM に KV 保存 |
| LMCache | "クラスタ KV キャッシュ" | CPU DRAM + disk 上のクロスエンジン KV キャッシュ サーバー |
| 0.11.0 async | "ノンブロッキング オフロード" | エンジン ストリーム の背後に隠されたオフロード |
| Preemption | "スペース作成のため削除" | HBM フルときの KV キャッシュ シャッフル |
| Prefix reuse | "同じ システム prompt" | 複数クエリが始まりを共有。キャッシュ ヒット |
| Ceph tier | "disk ティア" | キャッシュ階層の DRAM 下の耐久ストレージ |

## 参考文献

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm チャート + オペレーター。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — コネクタ実装。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 非同期パス詳細。
