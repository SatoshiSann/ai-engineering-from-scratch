# LLM 評価 — RAGAS、DeepEval、G-Eval

> 完全一致と F1 は意味的等価性を見逃す。人手レビューはスケールしない。LLM-as-judge が本番の答えだ — その数値を信頼できるだけのキャリブレーションを伴って。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 13（質問応答）、Phase 5 · 14（情報検索）
**所要時間:** 約75分

## 問題

あなたの RAG システムはこう答える: 「June 29th, 2007.」
正解の参照は: 「June 29, 2007.」
完全一致（Exact Match）は0点。F1 は約75%。人間なら100点をつける。

これに10,000件のテストケースを掛ける。さらに検索器、チャンキング、プロンプト、モデルへのあらゆる変更を掛ける。意味を理解し、スケールでも安価に動き、回帰について嘘をつかず、正しい失敗モードを浮かび上がらせる評価器が必要だ。

2026年には、この問題を制する3つのフレームワークがある。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。4つの RAG 指標（忠実性、回答関連性、コンテキスト適合率、コンテキスト再現率）を、NLI + LLM-judge バックエンドで提供する。研究に裏付けられ、軽量。
- **DeepEval。** LLM 版の Pytest。G-Eval、タスク完了、幻覚、バイアスの指標。CI/CD ネイティブ。
- **G-Eval。** 1つの手法（かつ DeepEval の指標）: 思考連鎖、カスタム基準、0〜1のスコアを伴う LLM-as-judge。

3つすべてが LLM-as-judge に依存する。このレッスンでは、その手法と、それを取り巻く信頼レイヤーに対する直感を養う。

## コンセプト

![4つの評価次元、LLM-as-judge アーキテクチャ](../assets/llm-evaluation.svg)

**LLM-as-judge。** 静的な指標を、ルーブリックに基づいて出力をスコア付けする LLM で置き換える。`(query, context, answer)` が与えられたら、判定 LLM にプロンプトする: 「忠実性を0〜1で採点せよ。」スコアを返す。

なぜ機能するか: LLM は人間の判断をごくわずかなコストで近似する。GPT-4o-mini はケースあたり約$0.003で、1000サンプルの回帰評価を$5未満で実行できる。

なぜ静かに失敗するか:

1. **判定バイアス。** 判定者は長い回答、自分と同じモデルファミリーの回答、プロンプトのスタイルに合う回答を好む。
2. **JSON 解析失敗。** 不正な JSON → NaN スコア → 集計から黙って除外される。RAGAS ユーザーはこの苦痛を知っている。try/except + 明示的な失敗モードでガードすること。
3. **モデルバージョン間のドリフト。** 判定者をアップグレードすると、すべての指標が変わる。判定モデル + バージョンを凍結すること。

**RAG の4指標。**

| 指標 | 問い | バックエンド |
|--------|----------|---------|
| 忠実性（Faithfulness） | 回答中の各主張は検索されたコンテキストから来ているか？ | NLI ベースの含意 |
| 回答関連性（Answer relevance） | 回答は質問に答えているか？ | 回答から仮想的な質問を生成し、実際の質問と比較 |
| コンテキスト適合率（Context precision） | 検索されたチャンクのうち、関連していた割合は？ | LLM-judge |
| コンテキスト再現率（Context recall） | 検索は必要なものすべてを返したか？ | 正解回答に対する LLM-judge |

**G-Eval。** カスタム基準を定義する: 「回答は正しい出典を引用したか？」フレームワークが自動的に思考連鎖の評価ステップに展開し、0〜1で採点する。RAGAS がカバーしないドメイン固有の品質次元に適する。

**キャリブレーション。** 人間のラベルとの相関を取るまで、生の判定スコアを決して信頼しないこと。100件の手作業ラベル例を実行する。判定者対人間をプロットする。スピアマンの ρ を計算する。ρ < 0.7 なら、判定ルーブリックには改善が必要だ。

## 作ってみよう

### ステップ1: NLI による忠実性（RAGAS スタイル）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

回答を原子的な主張に分解する。各主張を検索されたコンテキストに対して NLI でチェックする。忠実性 = 裏付けられた割合。

### ステップ2: 回答関連性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

回答が、尋ねられたものとは異なる質問を示唆するなら、関連性は下がる。

### ステップ3: G-Eval カスタム指標

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

評価ステップがルーブリックだ。明示的なステップは、暗黙の「0〜1で採点せよ」プロンプトよりも安定する。

### ステップ4: CI ゲート

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

pytest ファイルとして出荷する。すべての PR で実行する。回帰時にマージをブロックする。

### ステップ5: ゼロから作るおもちゃ評価

`code/main.py` を参照。忠実性（回答の主張とコンテキストのオーバーラップ）と関連性（回答トークンと質問トークンのオーバーラップ）の stdlib のみの近似。本番用ではない。形を示す。

## 落とし穴

- **キャリブレーションなし。** 人間ラベルとの相関が0.3の判定者はノイズだ。出荷前にキャリブレーションの実行を必須とすること。
- **自己評価。** 生成と判定に同じ LLM を使うとスコアが10〜20%水増しされる。判定には別のモデルファミリーを使うこと。
- **ペアワイズ判定での位置バイアス。** 判定者は最初に提示された選択肢を好む。常に順序をランダム化し、両方を実行すること。
- **生の集計は失敗を隠す。** 平均スコア0.85は、しばしば5%の壊滅的失敗を隠す。常に下位分位を調べること。
- **ゴールデンデータセットの腐敗。** バージョン管理されず時間とともにドリフトする評価セットは、縦断的比較を壊す。変更のたびにデータセットにタグを付けること。
- **LLM コスト。** スケールでは判定呼び出しがコストを支配する。キャリブレーション閾値を満たす最も安価なモデルを使う。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使ってみよう

2026年のスタック:

| ユースケース | フレームワーク |
|---------|-----------|
| RAG 品質モニタリング | RAGAS（4指標） |
| CI/CD 回帰ゲート | DeepEval + pytest |
| カスタムドメイン基準 | DeepEval 内の G-Eval |
| オンラインのライブトラフィックモニタリング | 参照不要モードの RAGAS |
| ヒューマンインザループのスポットチェック | 注釈 UI 付きの LangSmith または Phoenix |
| レッドチーミング／安全性評価 | Promptfoo + DeepEval |

典型的なスタック: モニタリングに RAGAS、CI に DeepEval、新規次元に G-Eval。3つすべてを実行する。それらは有用な形で意見が食い違う。

## 出荷しよう

`outputs/skill-eval-architect.md` として保存する。

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 演習

1. **易。** 既知の幻覚を含む10件の RAG 例に RAGAS を使う。忠実性指標がそれぞれを捉えることを検証する。
2. **中。** 50件の QA 回答を正確性について0〜1で手作業ラベル付けする。G-Eval で採点する。判定者と人間の間のスピアマン ρ を測定する。
3. **難。** DeepEval で pytest の CI ゲートを構築する。意図的に検索器を回帰させる。ゲートが失敗することを検証する。最下位10%に対する閾値チェックで下位分位アラートを追加する。

## 重要用語

| 用語 | 一般に言われること | 実際の意味 |
|------|-----------------|-----------------------|
| LLM-as-judge | LLM での採点 | ルーブリックに基づき、判定モデルに出力を0〜1で採点させる。 |
| RAGAS | RAG 指標ライブラリ | 4つの参照不要 RAG 指標を持つオープンソース評価フレームワーク。 |
| 忠実性（Faithfulness） | 回答は接地されているか？ | 検索されたコンテキストに含意される回答の主張の割合。 |
| コンテキスト適合率（Context precision） | 検索されたチャンクは関連していたか？ | 実際に重要だった top-K チャンクの割合。 |
| コンテキスト再現率（Context recall） | 検索はすべてを見つけたか？ | 検索されたチャンクが裏付ける正解回答の主張の割合。 |
| G-Eval | カスタム LLM 判定 | ルーブリック + 思考連鎖の評価ステップ + 0〜1のスコア。 |
| キャリブレーション（Calibration） | 信頼するが検証する | 判定スコアと人間スコアの間のスピアマン相関。 |

## 参考文献

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS の論文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval の論文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — オープンな本番スタック。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — バイアス、キャリブレーション、限界。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — RAGAS、DeepEval、Phoenix を統合する統一フレームワーク。
