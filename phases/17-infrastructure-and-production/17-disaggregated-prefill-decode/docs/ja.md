# Disaggregated Prefill/Decode — NVIDIA Dynamo と llm-d

> Prefill は計算バウンド。Decode はメモリバウンド。両者を同じ GPU で実行するとどちらかのリソースが無駄になる。Disaggregation はそれぞれを別のプールに分割し、NIXL（RDMA/InfiniBand または TCP フォールバック）経由で KV キャッシュを転送する。NVIDIA Dynamo（GTC 2025 発表、1.0 GA）は vLLM/SGLang/TRT-LLM の上に位置するオーケストレーション層。Planner Profiler + SLA Planner は自動的に prefill:decode の比率を調整して SLO を達成する。NVIDIA は GB200 NVL72 + Dynamo の中程度の遅延レジーム における DeepSeek-R1 MoE で約 6 倍の改善を公開している（developer.nvidia.com, 2025-06）。Dynamo プロダクトページ（developer.nvidia.com, 非日付）では GB300 NVL72 + Dynamo で最大 50 倍の MoE スループット vs Hopper を謳っている。「30 倍」という数字はコミュニティの集約値であり、単一のプライマリソースは存在しないため、方向性のある主張として扱うべき。llm-d（Red Hat + AWS）は Kubernetes ネイティブ：prefill / decode / router は独立した Services で、per-role HPA を持つ。llm-d 0.5 は階層的 KV オフロード、キャッシュ対応 LoRA ルーティング、UCCL ネットワーキング、スケール・ツー・ゼロを追加する。エコノミクス：複数のカスタマー開示のロールアップにより、同じ SLA で collocated serving から disaggregated with Dynamo への切り替え時に $2M クラスの推論支出から年間 $600-800K（30-40% 削減）の節約が見込める。短い入力（<512 トークン、短い出力）は転送コストを正当化しない。

**タイプ:** Learn
**言語:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**前提条件:** Phase 17 · 04 (vLLM Serving Internals)、Phase 17 · 08 (Inference Metrics)
**所要時間:** 約 75 分

## 学習目標

- Prefill と Decode が異なる GPU 割り当てを最適とする理由を説明し、colocation による無駄を定量化する。
- Disaggregated アーキテクチャ図：prefill プール、decode プール、NIXL 経由の KV 転送、ルーター。
- Disaggregation が報酬をもたらさない条件を述べる（短い prompt、短い出力）。
- NVIDIA Dynamo（スタック上）と llm-d（Kubernetes ネイティブ）を区別し、各々を運用コンテキストに合わせる。

## 問題

Llama 3.3 70B を 8 個の H100 で運行している。混合ワークロード（長い prompt + 短い出力）では、prefill に計算を費やしているため、decode 中に GPU がアイドル状態になる。異なるワークロード（短い prompt + 長い出力）では反対になる。Collocated prefill + decode は両者の過度なプロビジョニングを意味する。

予算への影響：GPU 時間の 20-40% がリソース配分を誤って無駄になる。H100 compute を memory-bound decode 用に、または H100 HBM bandwidth を compute-bound prefill 用に購入している。どちらも高価な無駄。

Disaggregation は prefill と decode をそれぞれのボトルネックのために設計されたプール に分割する。KV キャッシュは高スループット相互接続経由で prefill プールから decode プールに転送される。

## コンセプト

### ボトルネックが異なる理由

**Prefill** — フルの入力 prompt に対してトランスフォーマーを一度に実行。行列乗算が支配的。計算バウンド。H100 FP8 は約 2000 TFLOPS の有効スループットを提供。バッチ効率は良好 — 1 回の forward は多くのトークンを処理する。

**Decode** — トークンを 1 つずつ生成。毎イテレーション全重みを読む。メモリバウンド。HBM3 は約 3 TB/s を提供。バッチ効率は高い並行性でのみ良好 — 重みの読み込みはバッチ全体に償却される。

Collocate すると、両者に最適化された GPU を購入する。H100 は両者に優れているがどちらにしても同じコストがかかる。スケールでは prefill プール を H100/compute-heavy、decode プール を H200/memory-heavy、または積極的な量子化で配置したい。

### アーキテクチャ

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL は NVIDIA のノード間トランスポート。利用可能な場合 RDMA/InfiniBand を使用、それ以外は TCP フォールバック。転送遅延は実在する — 通常 70B FP8 の 4K トークン prompt の KV キャッシュで 20-80 ms。これが短い prompt が disaggregation を正当化しない理由 — 転送税が節約を上回る。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 発表、1.0 GA）：
- vLLM、SGLang、TRT-LLM のオーケストレーターとして機能。
- Planner Profiler はワークロードを測定し、SLA Planner は自動的に prefill:decode 比率を設定。
- Rust コア、Python 拡張性。
- スループット改善：NVIDIA は GB200 NVL72 + Dynamo の中程度遅延レジームで DeepSeek-R1 MoE で 6 倍を報告（developer.nvidia.com, 2025-06）。コミュニティの「最大 30 倍」の報告は単一プライマリソースがなく方向性の標準とすべき。
- GB300 NVL72 + Dynamo：Dynamo プロダクトページ（developer.nvidia.com, 非日付）あたり Hopper vs 最大 50 倍 MoE スループット。

**llm-d**（Red Hat + AWS、Kubernetes ネイティブ）：
- Prefill / decode / router を独立した Kubernetes Services として機能。
- Per-role HPA with queue depth（prefill）/ KV utilization（decode）シグナル。
- `topologyConstraint packDomain: rack` は prefill+decode cliques を同じラックに詰めて高スループット KV 転送を実現。
- llm-d 0.5（2026）：階層的 KV オフロード、キャッシュ対応 LoRA ルーティング、UCCL ネットワーキング、スケール・ツー・ゼロ。

マネージド stack-above オーケストレーターが必要なら Dynamo を使う。Kubernetes ネイティブプリミティブを望み CNCF エコシステムにコミットしているなら llm-d を使う。

### エコノミクス

内部コンポジット（単一の発行されたケーススタディではない — 大きさのアンカー）：

- Collocated serving での $2M/年 推論支出。
- Disaggregated with Dynamo に切り替え。
- 同じリクエスト量、同じ P99 遅延 SLA。
- 報告された節約：年間 $600K–$800K（30–40% 削減）。
- 新しいハードウェアなし。

複数のカスタマー開示から合成した値であり、単一の引用可能なケーススタディではない。最も近い発行されたデータポイントは Baseten の 2x 高速 TTFT / Dynamo KV ルーティングで 61% 高スループット（baseten.co, 2025-10）と VAST + CoreWeave の 40-60% KV ヒット率での 60-130% more tokens/$（vastdata.com, 2025-12）の予測。節約は各プールをサイジングから来る。Prefill-heavy ワークロード（8K+ プレフィックスで RAG）はバランスのとれたものより恩恵を受ける。

### Disaggregate が報酬をもたらさない場合

- Prompts < 512 トークン、出力 < 200 トークン：転送税が利益を上回る。
- Small cluster（< 4 GPUs）：プール多様性が不十分。
- チームが per-role スケーリングで 2 つの GPU プールを運用できない：Dynamo は役立つが trivial ではない。
- RDMA ファブリックなし：TCP 転送税がより重い。

### ルーターは Phase 17 · 11 と統合

Disaggregated ルーターは KV-cache-aware（Phase 17 · 11）。リクエストはそのプレフィックスを保持する decode プールに到着 — マッチしなければ prefill → decode にフロー。ヒット率と disaggregation は複合 — キャッシュ対応ルーターは新しい prefill が必要かどうかを決定。

### MoE on Blackwell は実数が出ている場所

GB300 NVL72 + Dynamo は Hopper ベースラインで 50 倍 MoE スループットを示す。MoE エキスパートルーティングは prefill で compute-heavy だが decode で memory-heavy（エキスパートキャッシュ）なので、disaggregation は二重の勝利。2026 frontier モデルサービングは MoE 支配的（DeepSeek-V3、将来 GPT-5 バリアント）。

### 記憶すべき数字

ベンチマーク数字は漂流する — NVIDIA と推論スタックは四半期ごとに更新結果を投稿。引用前に再確認。

- GB200 NVL72 + Dynamo の DeepSeek-R1：中程度遅延レジームのベースラインと比較して約 6 倍スループット（developer.nvidia.com, 2025-06）。コミュニティの full Blackwell + Dynamo スタックで「最大 30 倍」の主張は単一プライマリソースのない方向性集約。
- GB300 NVL72 + Dynamo：Hopper と比較して最大 50 倍 MoE スループット（developer.nvidia.com, 非日付）。
- 節約アンカー（内部コンポジット、単一ケーススタディではない）：同じ SLA で $2M 年間支出で $600-800K/年。
- Disaggregation しきい値：prompts >512 トークン + 出力 >200 トークン。
- NIXL 経由 KV 転送：70B FP8 の 4K prompt KV で 20-80 ms。

## 使用方法

`code/main.py` は colocated vs disaggregated serving をシミュレート。スループット、リクエスト当たりコスト、prompt 長クロスオーバーを報告。

## 配送

このレッスンは `outputs/skill-disaggregation-decider.md` を作成。ワークロードとクラスタを指定して disaggregate すべきかを決定。

## 演習

1. `code/main.py` を実行。どの prompt 長で disaggregation が colocation を上回るか？
2. RAG サービスの prefill プール と decode プール を設計：P99 プレフィックス長 8K、出力 300。
3. Dynamo vs llm-d：純粋 Kubernetes ショップで Python ランタイム選好なし。どちらを選ぶか？
4. KV 転送コスト計算：70B FP8 での 4K prefill = 約 500 MB KV。RDMA 100 GB/s で転送 = 5 ms。TCP 10 GB/s = 50 ms。SLA には何が重要か？
5. MoE エキスパートルーティングは KV アクセスパターンを変更。異なるエキスパートをトークンあたり活性化する MoE では disaggregation はどのように動作するか？

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| Disaggregated serving | "prefill/decode を分割" | 各フェーズのための別の GPU プール |
| NIXL | "NVIDIA トランスポート" | Dynamo のノード間 KV 転送（RDMA/TCP）|
| NVIDIA Dynamo | "オーケストレーター" | vLLM/SGLang/TRT-LLM のスタック上コーディネーター |
| llm-d | "Kubernetes ネイティブ" | Red Hat + AWS K8s disaggregated スタック |
| Planner Profiler | "Dynamo 自動設定" | ワークロード測定、プール比率設定 |
| SLA Planner | "Dynamo ポリシー" | prefill:decode を自動レート照合して SLO を達成 |
| `packDomain: rack` | "llm-d トポロジ" | prefill+decode を高速 KV のため同じラック上に詰める |
| UCCL | "統一的共同" | llm-d 0.5 ネットワークレイヤーのスケール・ツー・ゼロ用 |
| MoE expert routing | "トークンあたりエキスパート" | DeepSeek-V3 パターン。Disaggregation が役立つ |

## 参考文献

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
