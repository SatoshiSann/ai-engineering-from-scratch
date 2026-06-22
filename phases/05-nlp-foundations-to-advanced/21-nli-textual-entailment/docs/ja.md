# 自然言語推論 — テキスト含意

> 「tがhを含意する」は、tを読む典型的な人間がhが真であると結論づけることを意味する。NLIは含意/矛盾/ニュートラルの予測タスク。表面上は退屈だが、本番環境では負荷がかかっている。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 5 · 05 (感情分析)、Phase 5 · 13 (質問応答)
**所要時間:** 約60分

## 問題

要約機を構築した。要約を生成した。要約にハルシネーションが含まれていないことをどうやって知るか?

チャットボットを構築した。「はい」と答えた。答えが取得されたパッセージでサポートされていることをどうやって知るか?

10,000のニュース記事をトピック別に分類する必要がある。訓練ラベルがない。モデルを再利用できるか?

3つの問題すべてが自然言語推論に帰着する。NLIは、前提`t`と仮説`h`を与えられたとき、`h`が`t`に含意されているのか、矛盾しているのか、それともニュートラル(無関係)なのかを分類する。

- **ハルシネーション検出:** `t` = ソースドキュメント、`h` = 要約主張。含意でない = ハルシネーション。
- **根拠のあるQA:** `t` = 取得されたパッセージ、`h` = 生成された答え。含意でない = 捏造。
- **ゼロショット分類:** `t` = ドキュメント、`h` = 言語化ラベル(「これはスポーツについてです」)。含意 = 予測ラベル。

1つのタスク、3つの本番用途。これは、すべてのRAG評価フレームワークがボンネットの下にNLIモデルを出荷する理由だ。

## コンセプト

![NLI: 3方向分類、前提対仮説](../assets/nli.svg)

**3つのラベル。**

- **含意。** `t` → `h`。「猫はマットの上にいる」は「猫がいる」を含意する。
- **矛盾。** `t` → ¬`h`。「猫はマットの上にいる」は「猫はいない」に矛盾する。
- **ニュートラル。** どちらの方向でも推論なし。「猫はマットの上にいる」は「猫は空腹だ」に対してニュートラルだ。

**論理的含意ではない。** NLIは*自然な*言語推論 — 典型的な人間読者が推論するもの、厳密なロジックではない。「ジョンは彼の犬を散歩させた」はNLIで「ジョンは犬を持っている」を含意するが、厳密な一階述語論理は所有をaxiomatizeしない限りそれを認めるだけだ。

**データセット。**

- **SNLI** (2015)。570kの人間注釈付きペア、画像キャプションを前提として。狭いドメイン。
- **MultiNLI** (2017)。10ジャンル全体で433kペア。2026年の標準訓練コーパス。
- **ANLI** (2019)。敵対的NLI。人間は既存モデルを壊すために設計された例を書いた。より難しい。
- **DocNLI、ConTRoL** (2020–21)。ドキュメント長の前提。マルチホップと長距離推論をテストする。

**アーキテクチャ。** トランスフォーマーエンコーダー(BERT、RoBERTa、DeBERTa)は`[CLS] 前提 [SEP] 仮説 [SEP]`を読む。`[CLS]`表現は3方向ソフトマックスにフィードする。MNLIで訓練し、保持ベンチマークで評価し、分布内ペアで90%以上の精度を取得する。

**NLI経由のゼロショット。** ドキュメントと候補ラベルを与えられて、各ラベルを仮説に変える(「このテキストはスポーツについてです」)。各仮説の含意確率を計算する。最大値を選ぶ。これはHugging Faceの`zero-shot-classification`パイプラインの背後にあるメカニズムだ。

## ビルドしてみよう

### ステップ1: 事前訓練されたNLIモデルを実行

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # すべてのラベルを返す; 廃止されたreturn_all_scores=Trueを置き換える

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

本番用NLIでは、`facebook/bart-large-mnli`と`microsoft/deberta-v3-large-mnli`がオープンデフォルトだ。DeBERTa-v3はリーダーボードのトップだ。

### ステップ2: ゼロショット分類

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

テンプレートはデフォルトで「この例は{label}についてです。」だ。`hypothesis_template`でカスタマイズする。訓練データは不要だ。ファインチューニングはない。すぐに機能する。

### ステップ3: RAG用忠実性チェック

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

これはRAGAS忠実性のコアだ。生成された答えを原子的主張に分割する。各主張をチェックして、取得されたコンテキストに対して。含意する割合を報告する。

### ステップ4: 手巻きNLI分類器(概念的)

`code/main.py`のstdlibのみのおもちゃを参照: 前提と仮説は字句オーバーラップ+否定検出で比較される。トランスフォーマーモデルと競争力がない — しかし、それはタスクの形を示す: 2つのテキストを入力、3方向ラベルを出力、損失=`{entail, contradict, neutral}`の交叉エントロピー。

## ピットフォール

- **仮説のみのショートカット。** モデルは「not」、「nobody」、「never」が矛盾と相関するため、SNLIで仮説だけから約60%でラベルを予測できる。ラベル漏洩を検出するための強いベースライン。
- **字句オーバーラップヒューリスティック。** 部分列ヒューリスティック(「すべての部分列は含意される」)はSNLIに合格するがHANS/ANLIで失敗する。敵対的ベンチマークを使用する。
- **ドキュメント長の劣化。** 単一文NLIモデルはドキュメント長の前提で20以上のF1を低下させる。長いコンテキスト用にDocNLI訓練されたモデルを使用する。
- **ゼロショットテンプレート感度。** 「この例は{label}についてです」対「{label}」対「トピックは{label}です」は精度を10ポイント以上揺らがせることができる。テンプレートをチューニングする。
- **ドメイン不一致。** MNLIは一般的な英語で訓練される。法律、医療、科学的テキストには領域固有のNLIモデルが必要だ(例えば、SciNLI、MedNLI)。

## 使用方法

2026年スタック:

| ユースケース | モデル |
|---------|-------|
| 汎用NLI | `microsoft/deberta-v3-large-mnli` |
| 高速/エッジ | `cross-encoder/nli-deberta-v3-base` |
| ゼロショット分類(軽量) | `facebook/bart-large-mnli` |
| ドキュメントレベルNLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多言語 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAGのハルシネーション検出 | RAGAS / DeepEvalの内部NLIレイヤー |

2026年のメタパターン: NLIはテキスト理解の粘着性テープだ。「AはBをサポートしているか?」または「AはBに矛盾しているか?」が必要なときはいつでも — 別のLLM呼び出しに達する前にNLIに到達する。

## 出荷しよう

`outputs/skill-nli-picker.md`として保存:

```markdown
---
name: nli-picker
description: NLIモデル、ラベルテンプレート、分類/忠実性/ゼロショットタスク用の評価セットアップを選択する。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

ユースケース(忠実性チェック、ゼロショット分類、ドキュメントレベル推論)を指定して、以下を出力:

1. モデル。名前が付けられたNLIチェックポイント。ドメイン、長さ、言語に関連付けられた理由。
2. テンプレート(ゼロショットの場合)。言語化パターン。例。
3. しきい値。決定ルールのための含意カットオフ。キャリブレーションに基づく理由。
4. 評価。保持されたラベル付きセットの精度、仮説のみベースライン、敵対的サブセット。

仮説のみベースラインなしで100例のラベル付きサニティチェックのないゼロショット分類の出荷を拒否しなさい。ドキュメント長の前提で単一文NLIモデルの使用を拒否しなさい。NLIがハルシネーションを解決すると主張することにフラグを立てなさい — それはそれを減らす; それは排除しない。
```

## 演習

1. **簡単。** 3つすべてのクラスをカバーする20の手作り(前提、仮説、ラベル)トリプルで`facebook/bart-large-mnli`を実行する。精度を測定する。敵対的「部分列ヒューリスティック」トラップを追加して(「私はケーキを食べなかった」対「私はケーキを食べた」)、それが壊れるかどうかを確認する。
2. **中程度。** 100のAG Newsヘッドラインでゼロショットテンプレート`"This text is about {label}"`を`"The topic is {label}"`および`"{label}"`と比較する。精度スイングを報告する。
3. **難しい。** RAG忠実性チェッカーを構築: 原子的主張分解+主張ごとのNLI。50のRAG生成答え with ゴールドコンテキストで評価する。手ラベルに対する偽陽性率と偽陰性率を測定する。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| NLI | 自然言語推論 | 前提-仮説関係の3方向分類。 |
| RTE | テキスト含意の認識 | NLIの古い名前; 同じタスク。 |
| 含意 | 「tはhを暗示する」 | 典型的な読者はtを与えられてhが真であると結論づけるだろう。 |
| 矛盾 | 「tはhを除外する」 | 典型的な読者はtを与えられてhが偽であると結論づけるだろう。 |
| ニュートラル | 「決定不能」 | tからhへのいずれかの方向の推論がない。 |
| ゼロショット分類 | 分類器としてのNLI | ラベルを仮説として言語化し、最大含意を選択する。 |
| 忠実性 | 答えはサポートされているか? | (取得されたコンテキスト、生成された答え)の上のNLI。 |

## 参考文献

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLIベンチマーク。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI as classifier。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026年のNLIワークホース。
