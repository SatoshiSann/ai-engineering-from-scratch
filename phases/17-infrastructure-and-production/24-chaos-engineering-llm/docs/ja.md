# LLM Production 用 Chaos エンジニアリング

> 2026 年 Chaos エンジニアリング LLM 向けはそれ自身ディシプリン。Production で実験実行する前の前提条件：defined SLI/SLO、trace+metric+log 可観測性、自動ロールバック、runbooks、on-call。アーキテクチャは 4 つプレーン を持つ：control（experiment scheduler）、target（services、infra、data stores）、safety（guards + abort + トラフィック filters）、可観測性（metrics + traces + logs）、feedback（SLO adjustments へ）。ガードレール は mandatory：burn-rate アラート は daily error-budget burn > 2x expected の場合実験を pause。Suppression windows + trace-ID correlation は alert ノイズを dedupe。Cadence：weekly small canary + SLO review。Monthly game day + postmortem。Quarterly cross-team 復元力監査 + dependency mapping。LLM-specific 実験：メモリ overload、ネットワーク障害、プロバイダー outage、malformed prompt、KV キャッシュ eviction storm。ツーリング：Harness Chaos Engineering（LLM-derived 推奨、blast-radius ダウンスケーリング、MCP ツール統合）。LitmusChaos（CNCF）。Chaos Mesh（CNCF Kubernetes-native）。

**タイプ:** Learn
**言語:** Python (stdlib, toy chaos experiment runner)
**前提条件:** Phase 17 · 23 (SRE for AI)、Phase 17 · 13 (Observability)
**所要時間:** 約 60 分

## 学習目標

- 5 つの chaos エンジニアリング前提条件を name（SLI/SLO、可観測性、ロールバック、runbooks、on-call）、いかなるスキップが練習を break かを説明。
- 4 つプレーン を図式化（control、target、safety、可観測性）と SLO へのフィードバック loop。
- 5 つの LLM-specific 実験を列挙（メモリ overload、ネットワーク fail、プロバイダー outage、malformed prompt、KV eviction storm）。
- ツール pick — Harness、LitmusChaos、Chaos Mesh — stack を指定。

## 問題

Chaos テスト traditional スタック で確立。LLM スタック 新しい障害モード を add。4K-token prompt が poison character で tokenizer を 12 秒 stall。Upstream プロバイダーが 429。Gateway retry。Service OOM on retry-amplified 並行。KV キャッシュ eviction storm が burst ロード下 prefill cascade を cause、compute saturate。

すべてが unit テストに現れない。Chaos エンジニアリング はユーザーが見つける前に discover。

## コンセプト

### 前提条件

Production 内 chaos を実行しないで without：

1. **SLI/SLO** — defined service-level indicators と objectives。
2. **可観測性** — traces、metrics、logs、dashboards に wire。
3. **自動ロールバック** — Phase 17 · 20 policy-flag ロールバック。
4. **Runbooks** — structured、Phase 17 · 23。
5. **On-call** — 対応する誰か。

Anying miss は chaos が real インシデント になる。

### 4 つプレーン + feedback

**Control プレーン** — experiment scheduler（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target プレーン** — services、pods、nodes、load balancers、data stores。

**Safety プレーン** — kill switch、suppression windows、blast-radius limits、error-budget gates。

**可観測性 プレーン** — normal metrics + trace-ID correlation を chaos-induced を natural 障害から distinguish。

**Feedback ループ** — findings は SLO adjustment、runbook updates、code fixes へ back feed。

### ガードレール は mandatory

- **Burn-rate アラート**：experiment pause if daily error-budget burn expected を超える 2x。
- **Suppression windows**：experiment 中 blast radius で non-experiment アラート silence。
- **Trace-ID correlation**：すべての experiment-induced エラーは tag carry so on-call dedupe できる。

### 5 つの LLM-specific 実験

1. **メモリ overload** — long-context リクエスト高 並行 で KV キャッシュ preemption storm を force。観察：service gracefully shed または crash？

2. **ネットワーク 障害** — inference ゲートウェイ と プロバイダー 間の接続を cut。観察：fallback SLA 内 kick？（Phase 17 · 19）

3. **プロバイダー outage simulation** — OpenAI から 100% 429。観察：Anthropic へのルーティング failover？（Phase 17 · 16、19）

4. **Malformed prompt** — tokenizer-stalling payload を inject（例えば deeply nested unicode、huge UTF-8 codepoint）。観察：single リクエスト worker lock up？

5. **KV eviction storm** — vLLM block budget を saturate で eviction force。観察：LMCache recovery または service degrade？

### Cadence

- **Weekly** — small canary experiment staging、maybe 5% prod。
- **Monthly** — scheduled game day specific scenario 上。cross-team attendance。postmortem。
- **Quarterly** — cross-team 復元力監査。dependency map update。

### ツーリング

- **Harness Chaos Engineering** — commercial。AI-derived experiment 推奨。Blast-radius ダウンスケーリング。MCP ツール統合。
- **LitmusChaos** — CNCF graduated。Kubernetes workflow-based。
- **Chaos Mesh** — CNCF sandbox。Kubernetes-native CRD style。
- **Gremlin** — commercial。Broad サポート。
- **AWS FIS** / **Azure Chaos Studio** — managed クラウド offerings。

### 小さく開始

最初実験：steady トラフィック の下 decode レプリカの pod-kill。観察 rerouting と復帰。これが work そして safe に見えたら network chaos に graduate。

最初 LLM-specific 実験：1 つプロバイダー 429 inject 5 分 のために。観察 fallback。最も チーム their fallback がトリガー を fully-test できたことに discover。

### 記憶すべき数字

- 4 つプレーン：control、target、safety、可観測性。
- Burn-rate pause：2x expected daily budget burn。
- Cadence：weekly canary、monthly game day、quarterly audit。
- 5 つ LLM 実験：memory、network、provider、malformed prompt、KV storm。

## 使用方法

`code/main.py` は 3 つの chaos 実験を safety プレーン gates でシミュレート。どの実験が burn-rate abort を trip するかを報告。

## 配送

このレッスンは `outputs/skill-chaos-plan.md` を作成。Stack と成熟度を指定して、最初 3 つ実験と ツーリング を pick。

## 演習

1. `code/main.py` 実行。どの実験が burn-rate ゲート を trip、なぜ？
2. vLLM-based RAG サービスのために最初 5 つ chaos 実験を設計。Include success criteria。
3. Burn-rate アラート experiment を pause。Root cause を determine — chaos または natural？
4. Chaos が staging 内実行すべきか または production のみかを argue。Production が right answer のとき。
5. Generic network-chaos が reproduce できない 3 つ LLM-specific 障害モード を name。

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| SLI / SLO | "サービス target" | Indicator + objective。必須前提条件 |
| Blast radius | "scope" | Experiment の影響を受ける services / ユーザー set |
| Burn-rate alert | "バジェット ゲート" | Fire when error-budget burn rate > 2x expected |
| Game day | "monthly drill" | Scheduled cross-team chaos 演習 |
| LitmusChaos | "CNCF workflow" | Graduated CNCF Kubernetes chaos ツール |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes-native chaos |
| Harness CE | "commercial AI-assisted" | AI 推奨 付き Harness chaos |
| Malformed prompt | "tokenizer bomb" | Tokenization stall を入力 |
| KV eviction storm | "preemption cascade" | Mass eviction prefill re- trigger cascade |

## 参考文献

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
