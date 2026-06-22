# 推論メトリクス — TTFT、TPOT、ITL、Goodput、P99

> 4 つのメトリクスが推論デプロイメントが機能するかどうかを決定。TTFT はプリフィル + キュー + ネットワーク。TPOT（同等に ITL）はメモリ境界デコード コスト/トークン。エンドツーエンドレイテンシ は TTFT + TPOT × 出力長。スループット はフリート間で集計されるトークン/秒。しかし製品を重要にするのは goodput — すべての SLO を同時に満たしたリクエストの分数。高スループットで低 goodput はトークンを処理中でユーザーが時間内に到達しない。2026 年の TensorRT-LLM の参考数字：平均 TTFT 162 ms、平均 TPOT 7.33 ms、平均 E2E 1,093 ms。常に P50、P90、P99 を報告 — 平均のみ決してしない。測定トラップを監視：GenAI-Perf は ITL 計算から TTFT を除外、LLMPerf はこれを含む。2 つのツールが同じ実行で TPOT に同意しない。

**タイプ:** 学習
**言語:** Python（stdlib、簡単なパーセンタイル計算機と goodput レポーター）
**前提条件:** フェーズ 17 · 04（vLLM サービング インターナル）
**所要時間:** 約 60 分

## 学習目標

- TTFT、TPOT、ITL、E2E、スループット、goodput を正確に定義し、各々が測定するコンポーネントに名前を付け。
- 平均が LLM サービング で間違った統計である理由と P50/P90/P99 を読む方法を説明。
- マルチ制約 SLO（例 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s）を構築、それに対して goodput を計算。
- 同じ実行で TPOT に同意しない 2 つのベンチマークツールに名前を付け、理由を説明。

## 問題

「スループットは 15,000 トークン/秒。」で、どうしたということ？ 40% のリクエストが 2 秒エンドツーエンド を通り越したら、ユーザーはセッションを放棄。スループット のみはそれが製品が機能するかどうかを述べない。

推論は複数のレイテンシ軸を持ち、各々は異なる障害。プリフィル は計算境界でプロンプト長でスケール。デコード はメモリ境界でバッチサイズでスケール。キューイング遅延は運用問題。ネットワーク は物理距離問題。各々のための異なるメトリクスが必要、パーセンタイル が必要、「ユーザーが彼らが期待するものを得たか」を言う単一複合が必要 — それは goodput。

## コンセプト

### TTFT — 初トークン へ時間

`TTFT = queue_time + network_request + prefill_time`

プロンプトが長い場合プリフィル が優位。Llama-3.3-70B FP8 on H100 で、32k プロンプトは約 800 ms の純粋プリフィル 。キュー時間はロード下スケジューラー動作。ネットワーク要求は TLS 含むワイヤー時間。TTFT はユーザーが何かがストリーム バック前に見るレイテンシ。

### TPOT / ITL — トークン間レイテンシ

多くの名前を 1 つの量に。`TPOT`（トークン当たり時間）、`ITL`（トークン間レイテンシ）、`デコード レイテンシ/トークン` — すべて同じ。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

同じ Llama-3.3-70B H100 スタック チャンク化プリフィル で、TPOT 平均 約 7 ms。チャンク化プリフィル なし、隣接シーケンスで長いプリフィル 中、TPOT は 50 ms にスパイク可能。P99 を監視、平均ではなく。

### E2E レイテンシ

`E2E = TTFT + TPOT * output_tokens + network_response`

長い出力（>500 トークン）用、E2E は TPOT 支配。短い出力で長いプロンプト用、E2E は TTFT 支配。出力長条件 E2E を報告。

### スループット

`throughput = total_output_tokens / elapsed_time`

集計メトリック。フリート効率を述べる。個々リクエスト健康を述べない。

### Goodput — 実際にあなたが気にするメトリック

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO はマルチ制約。リクエストが「良い」のはすべての制約が保持した場合のみ。Goodput はシェア。高スループット を 60% goodput で は失敗。99% goodput で低スループット はターゲット。

2026 年、goodput は MLPerf Inference v6.0 提出と AI プラットフォーム プロバイダーでの内部 SLA 追跡で使用されるメトリック。

### 平均が間違った統計である理由

LLM レイテンシ分布は右歪み。1 つの長いプリフィル 隣接のデコード バッチ は 500 トークン TPOT 約 7 ms と 20 トークン TPOT 約 60 ms を配布できる。平均 TPOT は 9 ms。P99 TPOT は 65 ms。ユーザー は P99 を定期的にヒット — 彼らが出て行く理由。

常に（P50、P90、P99）トリプルを報告。ユーザー体験、P99 はあなたが最適化する 1 つ。

### 参考数字 — Llama-3.1-8B-Instruct on TensorRT-LLM、2026

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：チャンク化プリフィル 設定に応じ 10-25 ms で変動。

これらは公開されている NVIDIA 参考点。モデルサイズ（70B は 3-5x を示すだろう）、ハードウェア（H100 vs B200 約 3x）、負荷で変わります。

### 測定トラップ

最も使用されている 2 つの 2026 ベンチマークツール は同じ実行で TPOT に同意しない：

- **NVIDIA GenAI-Perf**：ITL 計算から TTFT を除外。ITL トークン 2 から開始。
- **LLMPerf**：TTFT を含む。ITL トークン 1 から開始。

100 出力トークン 700 ms 全デコード での 500 ms TTFT のリクエスト、GenAI-Perf は `ITL = 700/99 = 7.07 ms` を報告、LLMPerf は `ITL = 1200/100 = 12.00 ms` を報告。ツール選択数字を変える。

常にツールを述べる。常に定義を公開。

### SLO を構築

2026 年の 70B チャット モデルのための妥当なコンシューマーファーシング SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 s <300 トークン出力用。
- Goodput ターゲット >= 99%。

エンタープライズ SLO は TTFT をタイト（200-400 ms）にし、E2E をルーズ。要点はそれらを書き込み、3 つすべてを測定、複合として goodput を追跡。

### 計測する方法

- 実トラフィックまたは現実的合成を実行（LLMPerf で `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- ベンチマーク実行ために 2x ピーク同時度をターゲット。
- 30-50 反復を実行、結合サンプル のパーセンタイル を取る。
- ツール名、ツールバージョン、モデル、ハードウェア、同時度、プロンプト分布で公開。

## 使用

`code/main.py` はおもちゃ goodput 計算機。合成レイテンシ分布を生成、SLO を適用、goodput を計算。同じトレース上で GenAI-Perf vs LLMPerf TPOT 差も表示。

## 配布

このレッスンは `outputs/skill-slo-goodput-gate.md` を生成。ワークロードと SLO を考えると、スループット ではなく goodput でデプロイをゲート するCI/CD 準備ベンチマーク レシピを生成。

## 演習

1. `code/main.py` を実行。1% テイル スパイク付き分布を生成。P99 TPOT を 30 ms から 15 ms に タイト したら goodput はどう変わる？
2. ベンダーが「Llama 3.3 70B H100 で 15,000 tok/s」を引用。信頼する前に 3 つの質問に名前を付け。
3. チャンク化プリフィル が P99 TPOT を保護するが平均 TPOT ではない理由？
4. 音声アシスタント用にコンシューマー SLO を構築（最初トークンは読めず、聞こえる）。最もユーザー見える メトリクス は？
5. LLMPerf README と GenAI-Perf ドキュメント を読む。ツール が同意しない 3 つのメトリック を特定。

## 重要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-------------|----------|
| TTFT | 「初トークン へ時間」 | キュー + ネットワーク + プリフィル。長いプロンプトで プリフィル が優位 |
| TPOT | 「トークン当たり時間」 | 最初後のメモリ境界デコード コスト/トークン |
| ITL | 「トークン間レイテンシ」 | ほとんどのツールで TPOT と同じ（すべてではない — GenAI-Perf 参照） |
| E2E | 「エンド・ツー・エンド」 | TTFT + TPOT * output_len。応答側ネットワークの上。
| スループット | 「tok/s」 | フリート効率。レイテンシ パーセンタイル なしでは無用 |
| Goodput | 「SLO レート」 | すべての SLO 制約を同時に満たす リクエストの分数 |
| P99 | 「テイル」 | 1-100 最悪ケース レイテンシ。ユーザー体験 メトリック |
| SLO マルチ制約 | 「ジョイント」 | 3 つのレイテンシ 境界すべてのAND。リクエストはいずれか 1 つが違反したら失敗 |
| GenAI-Perf vs LLMPerf | 「ツール トラップ」 | ツール が ITL に TTFT を含むかどうかに同意しない |

## 参考文献

- [NVIDIA NIM — LLM ベンチマーク メトリック](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT の正規定義。
- [Anyscale — LLM サービング ベンチマーク メトリック](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 別定義と計測レシピ。
- [BentoML — LLM 推論 メトリック](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 実デプロイ上での応用計測。
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray ベースオープン ソース ベンチマーク。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA のベンチマーク ツール。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 業界許容 goodput ベースベンチマーク。
