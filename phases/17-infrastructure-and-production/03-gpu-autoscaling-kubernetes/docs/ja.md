# Kubernetes での GPU オートスケーリング — Karpenter、KAI Scheduler、Gang Scheduling

> 3 つのレイヤー、1 つではありません。Karpenter はノードを動的にプロビジョニング（1 分以下、Cluster Autoscaler より 40% 高速）。KAI Scheduler は gang スケジューリング、トポロジ認識、階層キューを処理 — 7 個の 8 GPU 部分割り当てトラップを防止（7 つのノードが待機し、1 つの GPU が欠けるかかる 1 つで燃焼）。アプリケーション レベルのオートスケーラー（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）は推論に特定の信号でスケール — キューの深さ、KV キャッシュ利用率 — CPU/DCGM デューティサイクルではなく。古典的な HPA トラップは `DCGM_FI_DEV_GPU_UTIL` がデューティサイクル測定：100% は 10 リクエストまたは 100 リクエスト。vLLM は KV キャッシュメモリをプリアロケート、メモリはスケールダウンをトリガーしない。このレッスンは 3 つのレイヤーを構成し、デフォルト Karpenter `WhenEmptyOrUnderutilized` ポリシーを避ける方法を教えます（推論中に実行中 GPU ジョブを終了）。

**タイプ:** 学習
**言語:** Python（stdlib、簡単なキュー深度オートスケーラー シミュレータ）
**前提条件:** フェーズ 17 · 02（推論プラットフォーム経済学）、フェーズ 17 · 04（vLLM サービング インターナル）
**所要時間:** 約 75 分

## 学習目標

- 3 つのオートスケーリングレイヤー（ノードプロビジョニング、gang スケジューリング、アプリケーション レベル）を図解し、各レイヤーで使用されるツールに名前を付け。
- `DCGM_FI_DEV_GPU_UTIL` が vLLM の HPA 信号として間違っている理由を説明し、2 つの置き換え（キューの深さ、KV キャッシュ利用率）に名前を付け。
- Gang スケジューリングと、KAI Scheduler が防止する部分割り当て障害モード（8 個中 7 個の GPU アイドル）を説明。
- 実行中 GPU ジョブを終了する Karpenter 統合ポリシー（`WhenEmptyOrUnderutilized`）に名前を付け、2026 年の安全な代替案を述べ。

## 問題

チームは Kubernetes 上に LLM サービング サービスをシップ。HPA を `DCGM_FI_DEV_GPU_UTIL` で設定。サービスは業務時間中 100% で固定。HPA はスケール アップしない — 既に満杯と考えている。レプリカを手動で追加。TTFT は落ちる。HPA はまだスケール しない。信号は嘘をついている。

別途、Cluster Autoscaler をノード用に使用。1M トークン プロンプトが午前 2 時に到達。クラスタはノードプロビジョニングに 3 分かかり、リクエストはタイムアウト。

別にまた、8 GPU 間で 2 つのノードにまたがる 70B モデルをデプロイ。クラスタに 7 つの GPU が空き、1 つが 3 つのノード間に散在。Cluster Autoscaler は欠けている 1 GPU 用にノードをプロビジョニング。7 つのノードは 4 分待機し、Kubernetes が最後の GPU を起動する間、お金を燃焼。

3 つのレイヤー、3 つの異なる障害モード。2026 年の GPU 対応オートスケーリングは「HPA をオンにする」ではありません。ノードプロビジョニング、gang スケジューリング、アプリケーション信号スケーリングを構成することです。

## コンセプト

### レイヤー 1 — ノードプロビジョニング（Karpenter）

Karpenter は保留中のポッドを監視し、約 45-60 秒内にノードをプロビジョニング（Cluster Autoscaler は通常 GPU ノードで 90-120 秒かかる）。`NodePool` 制約に従って動的にインスタンスタイプを選択 — ポッドが 8 つの H100 を必要とし、クラスタに一致するノードがなければ、Karpenter は既存のグループをスケール する代わりに直接 1 つをプロビジョニング。

**統合トラップ**: Karpenter のデフォルト `consolidationPolicy: WhenEmptyOrUnderutilized` は GPU プールで危険。実行中 GPU ノードを終了し、ポッドをより安いサイズ調整済みインスタンスに移行します。推論ワークロードについてはこれが実行中リクエストを削除し、新しいノードで 70B モデルをリロードすることを意味。損失は容量の分単位 + リクエスト失敗。

GPU プール用の安全な設定：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Karpenter が本当に空のノードを 1 時間後に統合できるが、実行中ジョブを削除しない。

### レイヤー 2 — gang スケジューリング（KAI Scheduler）

KAI Scheduler（プロジェクト「Karp」その後リネーム）はデフォルト kube-scheduler が行わないことを処理：

**Gang スケジューリング** — すべてまたは何もない。8 つの GPU を必要とする分散推論ポッド、すべて 8 つが一緒に開始するか、どれも開始しない。これなしでは、部分割り当てトラップを得ます：7 個の 8 ポッドが開始、無期限に待機、お金を燃焼。

**トポロジ認識** — どの GPU が NVLink を共有するか、どれが同じラックに座るか、どれに InfiniBand があるか、知っている。ポッド配置に応じて。DeepSeek-V3 67B テンソル並列ワークロードは 1 つの NVLink ドメインに留まる必要。KAI Scheduler はそれを尊重。

**階層キュー** — 複数のチームが同じ GPU プール用に優先度とクォータで競争。チーム A の本番ピンチはチーム B の訓練ジョブによってのみプリエンプト（優先度ルール許可の場合）。

KAI は kube-scheduler の横に二次スケジューラとしてデプロイ。ワークロードに注釈を付けて使用。Ray と vLLM 本番スタックの両方が統合。

### レイヤー 3 — アプリケーション レベル信号

**HPA トラップ**: `DCGM_FI_DEV_GPU_UTIL` はデューティサイクルメトリック — GPU が各サンプリング間隔で作業をしていたかどうかを測定。100% 利用率は 10 同時リクエストまたは 100 を意味；GPU は どちらでも忙しい。デューティサイクルでスケーリングはブラインドスケーリング。

さらに悪いことに、vLLM と同様のエンジンは KV キャッシュメモリをプリアロケート（`--gpu-memory-utilization` まで）。メモリ使用率は 1 つのリクエストでさえ 90% 近くに留まる。メモリベース HPA はスケール ダウンしない。

**2026 年の置き換え信号**：

- キューの深さ（プリフィル待機中のリクエスト数）。
- KV キャッシュ利用率（アクティブなシーケンスに割り当てられたブロック の分数）。
- レプリカごと P99 TTFT（SLA 信号）。
- Goodput（すべての SLO を秒当たり満たすリクエスト）。

NVIDIA Dynamo Planner と llm-d Workload Variant Autoscaler はこれらの信号を消費し、レプリカをスケール。HPA 全体を置き換え（LLM サービング）。

### 何をいつ使用するか

| スケール判断 | ツール |
|------------|--------|
| ノード追加/削除 | Karpenter |
| 複数 GPU ジョブをスケジュール | KAI Scheduler |
| レプリカ追加/削除 | Dynamo Planner / llm-d WVA（または キューの深さで HPA をカスタム） |
| GPU タイプ選択 | Karpenter NodePool |
| 低優先度をプリエンプト | KAI Scheduler キュー |

### 分離プリフィル/デコードがすべてを複雑にする

分離プリフィル/デコードを実行（フェーズ 17 · 17）するなら、2 つのポッドクラス、異なるスケーリング トリガー：プリフィル ポッド はキューの深さでスケール、デコード ポッド は KV キャッシュ圧力でスケール。llm-d はこれらを別 `Services` として公開し、ロール当たり HPA。両方の前に単一 HPA を配置しようとしないでください。

### コールドスタートもここで重要

コールドスタート軽減（フェーズ 17 · 10）は、ノードプロビジョニング時間がユーザーに見えるようになるところ。Karpenter の 45-60 秒のウォームアップ + 20GB モデルロード + エンジン初期化は、ゼロからのリクエストが 2-5 分かかることを意味。SLO 重大パス用にウォームプール（`min_workers=1`）を保つか、アプリケーション レイヤー（Modal スタイル チェックポイント）でコールドスタート軽減を使用。

### 覚えるべき数値

- Karpenter ノードプロビジョニング：約 45-60s vs Cluster Autoscaler 約 90-120s（GPU ノード）。
- KAI Scheduler は部分割り当て廃棄を防止 — 8 個中 7 個のトラップ。
- `DCGM_FI_DEV_GPU_UTIL` を HPA 信号として：壊れている。キューの深さまたは KV 利用率を使用。
- Karpenter `WhenEmptyOrUnderutilized`：実行中 GPU ジョブを終了。推論用に `WhenEmpty + consolidateAfter: 1h` を使用。

## 使用

`code/main.py` はバースト GPU ワークロード上の 3 層オートスケーラーをシミュレート。素朴な HPA（デューティサイクル）、キューの深さ HPA、KAI-gang-scheduled スケーリングを比較。未満たリクエスト、アイドル GPU 分、複合スコアを報告。

## 配布

このレッスンは `outputs/skill-gpu-autoscaler-plan.md` を生成。クラスタトポロジ、ワークロード形状、SLO を考えると、3 層オートスケーリング計画を設計。

## 演習

1. `code/main.py` を実行。バースト ワークロードの下で、素朴なデューティサイクル HPA がキューの深さ HPA がキャッチする何個のリクエストを削除するか？差の出所は？
2. Llama 3.3 70B FP8 on H100 SXM5 を実行する 1 つのクラスタ用に Karpenter NodePool を設計。`capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter` を指定し、非 GPU ワークロード をこれらのノードから遠ざかるテイント。
3. チームが「GPU が利用可能だがポッド がスケジュール されない」と報告。診断 — これは Karpenter、kube-scheduler、KAI Scheduler？どのメトリクスが確認？
4. 分離プリフィル ポッドをスケール する信号と、異なるデコード ポッド信号を選択。両方を正当化。
5. `WhenEmptyOrUnderutilized` 統合トラップのコストを計算（24x7 本番サービス、P99 TTFT > 10s で平均 60 リクエスト削除イベント/日）。

## 重要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-------------|----------|
| Karpenter | 「ノードプロビジョナー」 | Kubernetes ノードオートスケーラー。サブミニュートプロビジョニング |
| Cluster Autoscaler | 「古いスケーラー」 | Kubernetes ノードオートスケーラー前身。遅い、グループベース |
| KAI Scheduler | 「GPU スケジューラー」 | Gang + トポロジ + キュー用の二次スケジューラー |
| Gang スケジューリング | 「すべてまたは何もない」 | N ポッドをアトミックにスケジュール またはすべてを延期 |
| トポロジ認識 | 「ラック対応」 | NVLink/IB/ラック配置に基づいてポッド を配置 |
| `DCGM_FI_DEV_GPU_UTIL` | 「GPU 利用率」 | デューティサイクルメトリック。LLM のスケーリング信号ではない |
| キューの深さ | 「待機中のリクエスト」 | プリフィル境界スケーリング用の正しい HPA 信号 |
| KV キャッシュ利用率 | 「メモリ圧力」 | デコード境界スケーリング用の正しい HPA 信号 |
| 統合 | 「Karpenter 統合」 | より安いインスタンスタイプへのノード終了 |
| `WhenEmpty + 1h` | 「安全な統合」 | ポリシーが実行中 GPU ジョブを削除しない |

## 参考文献

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 設計ドキュメント、設定例。
- [Karpenter 中断制御](https://karpenter.sh/docs/concepts/disruption/) — 統合ポリシー セマンティックス、GPU 安全デフォルト。
- [NVIDIA — Kubernetes 上の分離 LLM 推論](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner スケーリング信号。
- [Ray ドキュメント — RayClusters 用 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 統合パターン。
- [AWS EKS コンピューティング オートスケーリング ベストプラクティス](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — マネージド Kubernetes 特定ガイダンス。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 設計。
