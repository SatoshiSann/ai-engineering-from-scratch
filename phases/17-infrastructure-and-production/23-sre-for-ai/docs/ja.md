# AI 向け SRE — マルチエージェント インシデント対応、Runbook、予測検出

> AI SRE は LLM を infrastructure データ（logs、runbooks、service topology）に grounded via RAG を使用して調査、ドキュメンテーション、調整フェーズ を自動化。2026 年アーキテクチャ パターン はマルチエージェント オーケストレーション — specialized エージェント（logs、metrics、runbooks）supervisor により coordinated。AI は仮説 と クエリを proposing、ユーザーが judgment コール を approve。Datadog Bits AI と Azure SRE Agent はこれを managed プロダクト として ship。Runbook は進化：NeuBird Hawkeye は adversarial 評価（2 つモデル same インシデント analyze。agreement = confidence、disagreement = uncertainty）を使用。Operational memory チームの変化をまたぎ persist。自動修復は慎重に留まる：AI suggest、ユーザー approve。完全自動 action は narrow（pod restart、specific deploy ロールバック）tight ガードレール付き — 「set it and forget it」を売る誰もが overselling。浮上 frontier：pre-incident 予測。MIT 研究は historical logs + GPU temps + API エラーパターン に訓練された LLM は outage の 10-15 分早く 89% を predict。投影：2026 年末までに 95% enterprise LLM は自動フェイルオーバー を have。

**タイプ:** Learn
**言語:** Python (stdlib, toy multi-agent incident triage simulator)
**前提条件:** Phase 17 · 13 (Observability)、Phase 17 · 24 (Chaos Engineering)
**所要時間:** 約 60 分

## 学習目標

- マルチエージェント AI SRE アーキテクチャを図式化：supervisor + specialized エージェント（logs、metrics、runbooks）+ ユーザー approval ゲート。
- なぜ自動修復が narrow（pod restart、deploy revert）rather than broad（service 再構築）かを説明。
- Adversarial 評価パターン name（NeuBird Hawkeye）：2 つモデル agree = confidence、disagree = escalate。
- MIT 89% early-detection 結果 cite、operational constraint：predictions が actuation なし は dashboards。

## 問題

On-call エンジニアが 3 a.m. でページング。「Checkout 高 error rate。」Datadog、Loki、3 つ runbooks、deploy ログをチェック。30 分後 root cause が vLLM OOM from KV キャッシュ spike であることに realize。Pod restart。Error clear。

2026 年 investigation の最初 20 分は automatable。Logs を service でグループ、recent deploy に correlate、runbooks にマッチ — すべて RAG + tool-use。Supervised エージェント は first-pass triage できる hypothesis を present Datadog 開ける前に。

完全自動修復は異なる問題。Pod restart：安全。GPU pool scale：ポリシーが許可なら安全。Service 再構築：absolutely not。ディシプリンは narrow line を draw。

## コンセプト

### マルチエージェント アーキテクチャ

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor はインシデント を sub-queries に break。Specialized エージェント は tool access（log search、PromQL、doc retrieval）を持つ。Supervisor は synthesize、hypothesis + evidence を ユーザーに present。ユーザーが approve または redirect。

### 自動修復スコープ

**安全（narrow）**：pod restart、specific deploy revert、pre-approved bounds 内で scale プール、pre-approved feature フラグ enable。

**安全ではない（broad）**：service topology change、resource limits modify、new code deploy、IAM change、databases alter。

誰もが「set it and forget it」を売ることが overselling。安全セット は AI SRE が mature として成長、しかしボーダーは real。

### Adversarial 評価（NeuBird Hawkeye）

2 つモデルが independently same インシデント を analyze。Root cause に agree なら confidence high。Disagree なら human に escalate、両方 hypotheses visible。Simple パターン、effective フィルター against hallucinated root 原因。

### Operational メモリ

チーム turnover は silent kill traditional SRE の — tribal knowledge leaves。AI SRE は runbooks + post-mortems を vector DB に stores。エージェント はすべての new インシデント で retrieve。New エンジニアが join のとき、AI は full history を持つ。

### Pre-incident 予測

MIT 2025 研究：historical logs、GPU temperatures、API エラーパターンで訓練された LLM は test セット 上 outage の 10-15 分前に 89% を predicted。

現実チェック：predictions が actuation なし は dashboards。Operational question は「predict のとき何 do？」 Pre-emptive drain？Pager？Auto-scale？Answer は policy-specific。

### 2026 年プロダクト

- **Datadog Bits AI** — Datadog 内 managed SRE copilot。
- **Azure SRE Agent** — Azure-native。
- **NeuBird Hawkeye** — adversarial eval + operational memory。
- **PagerDuty AIOps** — triage + deduplication。
- **Incident.io Autopilot** — incident commander + coordination。

### Runbooks as code

Runbooks は Confluence ページから versioned markdown structured sections へ evolve（symptom、hypothesis、verify、act）。Structured runbooks は better RAG retrieval を feed。AI-SRE rollout をいかなる start することは unstructured runbooks を structured に turn することから。

### 記憶すべき数字

- MIT early-detection：89% outages、10-15 分 lead time。
- マルチエージェント triage：supervisor + (logs、metrics、runbooks) + human。
- Safe auto-remediation セット：pod restart、deploy revert、bounds 内スケール。
- Adversarial eval：2 つモデル independent。agreement = confidence。

## 使用方法

`code/main.py` は マルチエージェント triage をシミュレート：log エージェントが error find、metric エージェント CPU spike find、runbook エージェント known issue match。Supervisor hypotheses rank。

## 配送

このレッスンは `outputs/skill-ai-sre-plan.md` を作成。Current on-call、インシデント volume、team maturity を指定して AI SRE rollout を設計。

## 演習

1. `code/main.py` 実行。Log と metric エージェント disagree なら？Supervisor は resolve？
2. 3 つ「安全」自動修復アクション を定義する。各々を justify。
3. Structured runbook テンプレートを write：sections、required fields、verification コマンド。
4. Predictive 検出が 12 分 lead で fire。ポリシー何 — pager、pre-drain、または両方？
5. 3-person チームが 2026 年 AI SRE を adopt すべきか or wait かを argue。成熟度、volume、risk 考慮。

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| AI SRE | "on-call 向け エージェント" | LLM-backed インシデント 調査 + 調整 |
| Supervisor agent | "オーケストレーター" | Top-level エージェント インシデント を sub-queries に break |
| Specialized agent | "domain エージェント" | Sub-agent with tool access（logs、metrics、runbooks）|
| Auto-remediation | "AI fixes it" | Narrow pre-approved action。NOT broad 再構築 |
| Operational memory | "vector runbooks" | Vector DB で post-mortems + runbooks |
| Adversarial eval | "two-model チェック" | Independent 分析。agreement = confidence |
| NeuBird Hawkeye | "adversarial one" | Adversarial-eval + memory パターン付きプロダクト |
| Bits AI | "Datadog の SRE agent" | Datadog-managed AI SRE |
| Pre-incident prediction | "early 検出" | Outage 予測 上の 10-15 分 lead time |

## 参考文献

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
