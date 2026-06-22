# 質問応答システム

> 3つのシステムは現代 QA を形作った。抽出的はスパンを見つけた。検索拡張されたドキュメントでそれらを接地。世代的は答えを生成。全ての現代的AI助手は3つの混合。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 11（機械翻訳）、Phase 5 · 10（注意メカニズム）
**所要時間:** 約75分

## 問題

ユーザーは「最初のiPhoneはいつ発売されましたか？」を入力し、「2007年6月29日」を期待。「Appleの歴史は長く多様」ではない。「2007」分離で座っていない理由の文。直接、接地、正しい答え。

3つのアーキテクチャは最後の十年 QA 上で支配。

- **抽出的 QA。** 質問と答えがスパンに含まれるはずの通路が与えられたら、スパン開始と終了インデックスを見つけ。SQuAD は標準的なベンチマーク。
- **オープン領域 QA。** 通路は与えられない。最初に関連する通路を取得、次にまたは答えを生成。これは今日全ての RAG パイプラインの基礎。
- **世代的 / クローズドブック QA。** 大きいモデル参数記憶から答え。検索はない。推論で最速、事実で最も信頼性なし。

2026 のトレンドはハイブリッド：最高のいくつかの通路を取得、最高に返答 RAG をプロンプト。これはレッスン 14 が深さで検索半分をカバーする。このレッスンは QA 半分をビルド。

## コンセプト

![QA アーキテクチャ：抽出的、検索拡張、世代的](../assets/qa.svg)

**抽出的。** 質問と通路をトランスフォーマー（BERT族）で一緒にエンコード。答えスパンの開始と終了トークンインデックスを予測する2つのヘッド訓練。損失は有効な位置の上でのクロスエントロピー。出力は通路のスパン。決に幻想（構造），決に通路を答えることができない質問を処理（構造）。

**検索拡張（RAG）。** 2ステージ。最初に、取得器はコーパスから最上位 `k` 通路を見つける。次に、読む（抽出または世代的）はそれらの通路を使って答えを生成。取得器読む分割は各独立で訓練と評価ができる。最新の RAG はしばしば間の reranker を追加。

**世代的。** デコーダのみ LLM（GPT、Claude、Llama）学習された重みから答え。取得ステップなし。一般的知識で優れた、希少または最新事実で壊滅的。幻想率は事前訓練データの事実頻度に反比例。

## ビルド

### ステップ 1：事前訓練モデル持つ抽出的 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` は SQuAD 2.0 で訓練、これは答えられない質問を含む。デフォルトで、`question-answering` パイプラインモデルの null スコアが勝つ場合でも最高得点スパン戻す — それは自動的に空の答えを戻さない。明示的な「答えなし」動作を取得、パス `handle_impossible_answer=True` をパイプライン呼び出しに。パイプラインは次に返す空の答え のみ null スコアがすべてのスパンスコアを超える場合。常に `score` フィールドをいずれかの方法でチェック。

### ステップ 2：検索拡張パイプライン（スケッチ）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

2ステージパイプライン。密な取得（Sentence-BERT）は意味類似度によって関連通路を見つけ。抽出的読む（RoBERTa-SQuAD）は組み合わされた最上位通路から答えスパンを取得。小さなコーパスで動作。百万文書コーパスのため、FAISS またはベクトルデータベースを使用。

### ステップ 3：RAG で生成的

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

プロンプトパターンは重要。明示的にモデルにコンテキストでグラウンド化し、コンテキスト不足の場合「分かりません」を戻す、素朴なプロンプティングに対して幻想率 40-60% を削減。より凝ったパターンは引用、信頼スコア、構造化抽出を追加。

### ステップ 4：現実の反映する評価

SQuAD は **正確マッチ（EM）** と **トークンレベルF1** を使用。EM は正規化後の厳密なマッチ（小文字、ストリップ句読点、削除冠詞）— 予測がマッチまたはスコア 0。F1 は予測と参照間の意見重複で計算され部分的クレジットを与える。ともに paraphrases under-credit：「2007年6月29日」対「2007年6月29日」は通常 0 EM を取得（序数は正規化を壊す）ですが依然実質的な F1 重複トークンから獲得。

本番 QA 用：

- **答え精度**（LLM-判定または人間-判定、メトリックは意味的等価をキャプチャしないため）。
- **引用精度。** 引用された通路は実際に答えをサポートするか？自動的に生成された引用と取得通路間の文字列マッチで容易にチェック。
- **拒否キャリブレーション。** 取得された通路に答えがない場合、システムは正しく「分かりません」と言うか？偽の信頼率を測定。
- **取得リコール。** 読む評価する前に、取得が最上位 `k` に右通路を取得するかを測定。読むは欠落している通路を修正できない。

### RAGAS：2026 本番評価フレームワーク

`RAGAS` は RAG システムのための目的ビルドされ、2026 で本番デフォルトを出荷。金の参照が必要なしに4つのディメンション：

- **忠実性。** 答えの各クレームは取得されたコンテキストから来るか？NLI ベース含意により測定。あなたの主要な幻想メトリック。
- **答え関連性。** 答えは質問に扱うか？答えから仮説質問を生成して実質問に比較することにより測定。
- **文脈精度。** 取得チャンクのうち何が実際に関連するか？低精度=プロンプトのノイズ。
- **文脈リコール。** 取得セットは全て必要な情報を含むか？低リコール=読むは成功できない。

参考フリースコアリングはあなたが策略された金の答えなしで実演本番トラフィック評価できる。開いた質問のための LLM-as-judge 階層 正確マッチメトリックが無用なとこ。

`pip install ragas`。あなたの取得 + 読むにプラグイン。各質問ごと4つのスカラー取得。回帰で警告。

## 使う

2026 スタック。

| ケース使用 | 推奨 |
|---------|-------------|
| 与えられた通路、答えスパンを見つけ | `deepset/roberta-base-squad2` |
| 固定コーパス、クローズドブック受け入れられない | RAG：密な取得 + LLM 読む |
| 文書保存上でリアルタイム | RAG ハイブリッド（BM25 + 密）取得 + reranker（レッスン14）|
| 会話型 QA（フォローアップ質問） | LLM 会話履歴 + 各ターンの RAG |
| 高度に計算的、規制領域 | 権威あるコーパスの上で抽出的；決に生成的単独 |

抽出的 QA は 2026 で流行遅れ理由 RAG with LLM はより多くの場合を処理する。それは依然する文脈の引用が必要なとこ出荷：法的研究、規制準拠、監査ツール。

## 出荷

`outputs/skill-qa-architect.md` として保存：

```markdown
---
name: qa-architect
description: QA アーキテクチャ、取得戦略、評価プラン選択。
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

要件が与えられたら（コーパスサイズ、質問タイプ、真実性制約、遅延予算）、出力：

1. アーキテクチャ。抽出的、抽出読む RAG、世代的読む RAG、またはクローズドブック LLM。1文の理由。
2. 取得。なし、BM25、密（エンコーダを名付け）、またはハイブリッド。
3. 読む。SQuAD チューンモデル、LLM 名によって、または「ドメイン微調整DistilBERT。」
4. 評価。EM + F1 抽出的ベンチマークのため；答え精度 + 引用精度 + 拒否キャリブレーション本番用。測定メトリック名とそれをどう測定。

規制または準拠敏感質問のためクローズドブック LLM 答え拒否。取得リコールベースラインなし任意 QA システム拒否（読む評価するあなたは知ることができない取得サーフ右通路）。マルチホップ推論が必要な質問をマルチホップ取得のような特化として HotpotQA-訓練システムとしてフラグ。
```

## 演習

1. **簡単。** 上で SQuAD 抽出パイプラインを 10 ウィキペディア通路に設定。10の質問を手作り。答えが正しい場合の頻度を測定。通路と質問が清潔である場合に 7-9 正しい見たい。
2. **中程度。** 拒否クラシファイアを追加。最上位取得スコア閾値（0.3コサインは言って）下のとき、読むを呼ぶ代わりに「分かりません」を戻す。保有セットで閾値をチューン。
3. **難しい。** あなたが選択した 10,000 ドキュメントコーパス上でRAGパイプラインをビルド。ハイブリッド取得（BM25 + 密）実装 RRF 融合で（レッスン14見よ）。ハイブリッドステップなしで答え精度を測定。どの質問タイプが最多恩恵を受けるかドキュメント。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 抽出的 QA | 答えスパンを見つけ | 与えられた通路内の答えの開始と終了インデックスを予測。|
| オープン領域 QA | コーパスの上で QA | 与えられた通路なし；取得の後答え必要。|
| RAG | 取得の後生成 | 検索拡張生成。取得 + 読むパイプライン。|
| SQuAD | 標準的なベンチマーク | スタンフォード質問応答データセット。EM + F1 メトリック。|
| 幻想 | 構成答え | 読む出力は取得文脈でサポートされない。|
| 拒否キャリブレーション | 黙るべき時を知ること | システム正しく「分かりません」と言う答えることできないとき。|

## 参考文献

- [Rajpurkar et al. (2016)。SQuAD：テキスト理解の100,000+質問](https://arxiv.org/abs/1606.05250) — ベンチマーク論文。
- [Karpukhin et al. (2020)。オープン領域QAのための密な通路取得](https://arxiv.org/abs/2004.04906) — DPR、QA のための標準的な密な取得。
- [Lewis et al. (2020)。知識集約型NLPタスクのため取得拡張生成](https://arxiv.org/abs/2005.11401) — RAG を名付けた論文。
- [Gao et al. (2023)。大きいモデルのための取得拡張生成：サーベイ](https://arxiv.org/abs/2312.10997) — 包括的 RAG サーベイ。
