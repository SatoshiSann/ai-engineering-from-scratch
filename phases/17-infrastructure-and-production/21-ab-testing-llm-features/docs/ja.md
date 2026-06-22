# A/B テスト LLM フィーチャー — GrowthBook、Statsig、および Vibes 問題

> 伝統的 A/B テスト は non-deterministic LLM のためにビルドされなかった。重大な区別：evals は「モデルがジョブをできるか？」に答える。A/B テスト は「ユーザーが気にするか？」に答える。両方が必須。vibes チェックで出荷は時代遅れ。2026 年で何をテストするか：prompt engineering（表現）、モデル選択（GPT-4 vs GPT-3.5 vs OSS。精度 vs コスト vs レイテンシ）、生成パラメーター（temperature、top-p）。実のケース：chatbot reward-model variant は +70% 会話長、+30% retention 配信。Nextdoor AI subject-line 実験は reward-function refinement 後 +1% CTR 配信。Khan Academy Khanmigo は latency-vs-math-accuracy 軸上で反復。プラットフォーム分割：**Statsig**（2025 年 9 月 OpenAI が $1.1B で買収）— sequential テスト、CUPED、all-in-one。**GrowthBook** — オープンソース、warehouse-native、Bayesian + Frequentist + Sequential エンジン、CUPED、SRM チェック、Benjamini-Hochberg + Bonferroni 補正。Warehouse-SQL 選好と「OpenAI により買収」が組織に重要かにより選択。

**タイプ:** Learn
**言語:** Python (stdlib, toy sequential test simulator)
**前提条件:** Phase 17 · 13 (Observability)、Phase 17 · 20 (Progressive Deployment)
**所要時間:** 約 60 分

## 学習目標

- Evals（「モデルがジョブをできるか」）と A/B テスト（「ユーザーが気にするか」）を区別。
- 3 つのテスト可能軸（prompt、モデル、パラメーター）を列挙し各々のメトリクスを選択。
- CUPED、sequential テスト、Benjamini-Hochberg multiple-comparison 補正を説明。
- Warehouse-SQL posture と corporate acquisition 姿勢に基づき Statsig または GrowthBook を選択。

## 問題

システム prompt を hand-tuned。より良く感じる。出荷。変換は noise により変化。メトリク ブレーム。または新しいモデルを出荷して conversion は移動しなかった — モデル 低下または変更は検出にはあまりにも小さい？分からない、A/B なしで出荷したため。

Eval は モデルが labeled セットで タスクできるかに答える。ユーザーが出力を好むかに答えない。制御された online 実験のみがそれに答え、実験が十分なパワーを持ち、non-determinism をコントロール、multiple comparisons を補正する場合のみ。

## コンセプト

### Evals vs A/B テスト

**Evals** — offline、labeled セット、judge（rubric または LLM-as-judge または human）。答え：「出力は correct / helpful / safe 固定分布？」

**A/B テスト** — online、live ユーザー、randomized。答え：「新 variant は ユーザーレベル メトリクスを移動するか、重要？」

両方が必須。Eval は回帰を exposure 前にキャッチ。A/B は product 影響を after に確認。

### テストする内容

1. **Prompt エンジニアリング** — 表現、system-prompt 構造、例。メトリクス：タスク成功、ユーザー retention、コスト/リクエスト。
2. **モデル選択** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。メトリクス：精度（タスク）+ コスト/リクエスト + レイテンシ P99。マルチ目的。
3. **生成パラメーター** — temperature、top-p、max_tokens。メトリクス：タスク固有（出力多様 vs 決定性）。

### CUPED — 分散削減

Controlled-experiments Using Pre-Experiment Data。Pre-period 分散を post-period 比較前に regression out。典型的分散削減：30-70%。有効サンプルサイズが無料でアップ。

実装：Statsig と GrowthBook 両方実装。

### Sequential テスト

Classical A/B は固定サンプル サイズ 仮定。Sequential テスト（「peek-and-decide」）は repeated looks の下で false-positive レートをコントロール。Always-valid sequential 手順（mSPRT、Howard の confidence sequences）は clear winner で early stop を許可。

### Multiple-comparison 補正

20 A/B テスト を 95% confidence で実行は 1 false positive を偶然作成。Bonferroni 補正は per-test α tighten。Benjamini-Hochberg は false-discovery レートをコントロール。GrowthBook は両方実装。

### SRM — sample ratio mismatch

割り当て hash は ユーザーを variant に randomize。50/50 split が 47/53 配信なら何か broken — SRM チェック flag。両プラットフォーム実装。

### Statsig vs GrowthBook

**Statsig**：
- OpenAI により $1.1B で買収（2025 年 9 月）。Hosted、SaaS。
- Sequential テスト、CUPED、held-out populations。
- All-in-one：feature フラグ + experimentation + 可観測性。
- ベストフィット：チーム既に bundled プロダクト欲しい、OpenAI ownership 気にしない。

**GrowthBook**：
- オープンソース（MIT）。Warehouse-native（Snowflake/BigQuery/Redshift から直接読む）。
- 複数エンジン：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH 補正。
- Self-host または managed クラウド。
- ベストフィット：warehouse-SQL ショップ、data チーム メトリクス層をコントロール、OSS 欲しい。

### Non-determinism が power を複雑化

同じ prompt は varying 出力を作成。伝統的 power 計算は IID observations 仮定。LLM non-determinism では、有効サンプル サイズは nominal より低い。必要サンプルサイズに ~1.3-1.5x safety margin を乗じる。

### 実のケース outcome

- Chatbot reward モデル variant：+70% 会話長、+30% retention。
- Nextdoor subject lines：+1% CTR reward-function refinement 後。
- Khan Academy Khanmigo：iterative latency-vs-math-accuracy トレード。

### アンチパターン：vibes で出荷

各 senior エンジニアが「より良く感じる」のに出荷された feature を名付けられ、A/B なし。ほとんどはチームが months に気付かない product メトリクスに regress。A/B は forcing 関数。

### 記憶すべき数字

- Statsig OpenAI により買収：$1.1B、2025 年 9 月。
- GrowthBook：オープンソース MIT。Bayesian + Frequentist + Sequential。
- CUPED 分散削減：30-70%。
- LLM 非決定性 → +30-50% サンプル-サイズ バッファー。

## 使用方法

`code/main.py` は fixed と sequential 境界での sequential A/B テストをシミュレート。sequential が early stop をどのように許可するかを示す。

## 配送

このレッスンは `outputs/skill-ab-plan.md` を作成。フィーチャー変更、ワークロード、ベースラインを指定してプラットフォーム、ゲート、サンプルサイズを選択。

## 演習

1. `code/main.py` 実行。期待 5% lift に baseline 3% conversion、80% power のサンプルサイズ？
2. Healthcare-regulated on-prem カスタマーに Statsig または GrowthBook 選択。
3. GPT-4 vs GPT-3.5 を cost-per-resolved-ticket でテストする A/B 設計。Primary メトリクス、guardrail メトリク、secondary 何？
4. カナリア passes しか A/B は -1.2% conversion を示す。これを出荷するか？エスケレーション基準を書く。
5. Pre-period が post の 60% 分散を持つ CUPED を適用。有効-サンプル-サイズ ブースト計算。

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| Eval | "offline テスト" | モデル 能力の labeled-set 評価 |
| A/B test | "実験" | ユーザー上での live randomized 比較 |
| CUPED | "分散削減" | 分散を削減するための pre-period regression |
| Sequential test | "peek-ok テスト" | Always-valid 手順 early stop を許可 |
| Multiple comparison | "ファミリー エラー" | 多くテスト実行 false positives inflate |
| Bonferroni | "tight 補正" | テスト数で α 除算 |
| Benjamini-Hochberg | "BH FDR" | False-discovery-rate コントロール、less conservative |
| SRM | "悪い split" | Sample ratio mismatch。割り当て バグ |
| Statsig | "OpenAI owned" | Commercial all-in-one、2025 買収 |
| GrowthBook | "OSS one" | MIT warehouse-native プラットフォーム |
| mSPRT | "sequential probability ratio test" | Classical sequential 手順 |

## 参考文献

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
