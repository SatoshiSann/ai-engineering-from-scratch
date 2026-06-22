# LLM API ロードテスト — なぜ k6 と Locust は嘘つきか

> 伝統的ロードテスター はストリーミングレスポンス、変数出力長、トークンレベルメトリクス、GPU 飽和のためにデザインされなかった。2 つトラップが大半のチーム を噛む。GIL トラップ：Locust の token-level 測定は高い並行処理の下で Python GIL 下でのトークン化を実行。トークン化バックログ は reported inter-token レイテンシ を inflate — クライアントがボトルネック、サーバーではない。Prompt-uniformity トラップ：ループテストの同一 prompt は token 分布上の単一ポイントをテスト。実トラフィックは変数長、diverse prefix matches。LLMPerf は `--mean-input-tokens` + `--stddev-input-tokens` でこれを修正。2026 年ツール マッピング：LLM-specialized（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）token-level 正確性のため。**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** — ストリーミング対応、Kubernetes-native TestRun/PrivateLoadZone CRD 経由 distributed。CI/CD ゲートに最適。Vegeta は Go constant-rate 飽和。Locust 2.43.3 は LLM-Locust 拡張でのみストリーミング。ロード パターン：steady-state、ramp、spike（autoscaling テスト）、soak（メモリリーク）。

**タイプ:** Build
**言語:** Python (stdlib, toy realistic-prompt generator + latency collector)
**前提条件:** Phase 17 · 08 (Inference Metrics)、Phase 17 · 03 (GPU Autoscaling)
**所要時間:** 約 75 分

## 学習目標

- 2 つのアンチパターン（GIL トラップ、prompt-uniformity トラップ）を説明、generic ロードテスター が LLM API 向けに嘘つきになる。
- 目的に応じてツールを選択：LLMPerf（benchmark run）、k6 + ストリーミング拡張（CI gate）、guidellm（large-scale synthetic）、GenAI-Perf（NVIDIA reference）。
- 4 つのロードパターン（steady、ramp、spike、soak）を設計し、各々がキャッチする障害モード を名付ける。
- Fixed length ではなく mean + stddev input トークン使用、realistic prompt 分布を build。

## 問題

k6-tested LLM エンドポイント を 500 concurrent ユーザー。Hold した。出荷。200 実ユーザー でのプロダクション、サービス fell over — P99 TTFT exploded、GPU pinned。

2 つのことが起こった。First、k6 sent 500 identical prompt — request-coalescing と prefix キャッシング それを 500 concurrent decode 処理したように見させた、実際には 1 つ処理していた。Second、k6 はストリーミング response のインター-トークン レイテンシを eye 体験の方法でトラック しない。1 つ HTTP 接続、500 トークン が varying interval で到着しない。

LLM ロードテスト はそれ自身ディシプリン。

## コンセプト

### GIL トラップ（Locust）

Locust は Python を使用、高い並行処理の下で GIL 下でのトークン化クライアントサイド実行。トークナイザーは request generation の背後にキュー。Reported inter-token レイテンシはクライアント-サイド tokenization バックログ を include。サーバーが遅い と思う。テスト harness。

修正：LLM-Locust 拡張 tokenization を separate processes に move、または compiled-language harness（k6、LLMPerf tokenizers.rs 使用）を使用。

### Prompt-uniformity トラップ

すべての known ロードテスター はシングル prompt を設定することを許可。10,000 iteration ループテストで exact same prompt 毎回送る。Server は same prefix 毎回見ける — prefix キャッシュ ヒットは 100% に接近、スループット excellent に見える。

修正：prompt 分布からサンプル。LLMPerf は `--mean-input-tokens 500 --stddev-input-tokens 150` — diverse 長、diverse content。

### 4 つのロード パターン

1. **Steady-state** — constant RPS 30-60 分。キャッチ：baseline 性能 回帰。
2. **Ramp** — linearly RPS increase 0 から target 15 分で。キャッチ：capacity breakpoint、warm-up anomalies。
3. **Spike** — sudden 3-10x RPS 2 分 back。キャッチ：autoscaling レイテンシ、queue saturation、cold-start impact。
4. **Soak** — steady-state 4-8 時間。キャッチ：メモリ leaks、connection-pool drift、可観測性 overflow。

### 2026 年ツール マッピング

**LLMPerf**（Anyscale） — Python しか Rust-backed トークン化。Mean/stddev prompt。ストリーミング-aware。パフォーマンス run のためのベストデフォルト。

**NVIDIA GenAI-Perf** — NVIDIA reference。Triton クライアント使用。包括的メトリクス カバレッジ。注記：ITL TTFT exclude。LLMPerf include。2 つツール同じサーバーで異なる TPOT 作成。

**LLM-Locust**（TrueFoundry） — Locust 拡張 GIL トラップ修正。Familiar Locust DSL + ストリーミング メトリクス。

**guidellm** — large-scale synthetic ベンチマーク。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 itself（Go、compiled、GIL なし） ストリーミング-aware メトリクス追加。
- k6 Operator は Kubernetes-native distributed testing 用 TestRun / PrivateLoadZone CRD 使用。
- CI/CD gate と SLA テストに最適。

**Vegeta** — Go、k6 より simpler。Constant-rate HTTP 飽和。LLM-aware ではないが gateway / レート制限テストに良好。

**Locust 2.43.3 stock** — LLM GIL トラップがある。LLM-Locust 拡張でのみ。

### SLA ゲート CI 内

PR で k6 実行：

- 30-50 iteration 各々 baseline RPS で。
- ゲート：P50/P95 TTFT、5xx < 5%、TPOT under しきい値。
- Breach で build break。

### Realistic prompt 分布

実トラフィック samples から build（ある場合）または published 分布から（例えば、chat の ShareGPT prompt、code の HumanEval）。LLMPerf に mean + stddev を feed。Loop-with-one-prompt absolutely 回避。

### 記憶すべき数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：ストリーミング-aware メトリクス。
- 典型的 LLMPerf run：100-1000 リクエスト concurrency X で。
- 典型的 CI ゲート：30-50 iteration per PR。
- 4 つパターン：steady、ramp、spike、soak。

## 使用方法

`code/main.py` はリアリスティック prompt 分布でのロード テスト、effective TPOT を測定、uniform-prompt トラップを示す。

## 配送

このレッスンは `outputs/skill-load-test-plan.md` を作成。ワークロードと SLA を指定して、ツール と 4 つロード パターンを設計 pick。

## 演習

1. `code/main.py` 実行。Uniform vs realistic 分布 比較 — gap どこ？
2. CI ゲート の k6 スクリプト write：TTFT P95 < 800 ms 100 concurrent で、runtime 5 分。
3. Soak テスト メモリ growing 50 MB/hour を示す。3 つの原因を name し、それら間を pick するインストルメント。
4. Spike テスト 10 RPS から 100 RPS。期待復帰時間が Karpenter + vLLM production-stack ある場合（Phase 17 · 03 + 18）？
5. GenAI-Perf は TPOT=6ms report。LLMPerf は TPOT=11ms 同じサーバーで report。説明。

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| LLMPerf | "LLM harness" | Anyscale ベンチマーク ツール、ストリーミング-aware |
| GenAI-Perf | "NVIDIA ツール" | NVIDIA reference harness |
| LLM-Locust | "LLM 向け Locust" | GIL トラップ修正 Locust 拡張 |
| guidellm | "synthetic ベンチマーク" | Large-scale synthetic ツール |
| k6 Operator | "K8s k6" | CRD-based distributed k6 |
| GIL trap | "Python クライアント オーバーヘッド" | トークン化バックログ reported レイテンシ inflate |
| Prompt-uniformity trap | "single-prompt 嘘" | Same prompt ループ キャッシュ hit、スループット inflate |
| Steady-state | "constant ロード" | Flat RPS N 分 |
| Ramp | "linear up" | 0 target へ duration で |
| Spike | "burst テスト" | Sudden multiplier revert |
| Soak | "long テスト" | Leak 検出時間 |

## 参考文献

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
