# 名前付きエンティティ認識

> 名前を引き出す。曖昧な境界、ネストされたエンティティ、ドメインジャーゴンで扱うまで簡単に聞こえます。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 02（BoW + TF-IDF）、Phase 5 · 03（単語埋め込み）
**所要時間:** 約75分

## 問題

「Apple sued Google over its iPhone search deal in the US.」5つのエンティティ：Apple (ORG)、Google (ORG)、iPhone (PRODUCT)、search deal (maybe)、US (GPE)。良いNERシステムはそれらを正しいタイプで全て抽出。悪いもは iPhone を見落とし、Apple の果物と Apple の会社を混同し、「US」を PERSON としてラベル。

NERは全て構造化抽出パイプラインの下の働き馬。履歴書解析、準拠ログスキャン、医療記録匿名化、検索クエリ理解、チャットボット応答のためのグラウンディング、法的契約抽出。あなたはそれをほぼ決して見ない；あなたは常にそれに依存。

このレッスンは古典パス（規則基づ、HMM、CRF）を最新のもの（BiLSTM-CRF、次にトランスフォーマー）に歩く。各ステップは前のもの特定の制限を解決。パターンはレッスン。

## コンセプト

**BIOタギング**（または BILOU）エンティティ抽出をシーケンス標識問題に変える。各トークンに `B-TYPE`（エンティティ開始）、`I-TYPE`（エンティティ内）、または `O`（任意エンティティ外）でラベル。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

マルチトークンエンティティチェーン：`New B-GPE`、`York I-GPE`、`City I-GPE`。BIOを理解するモデルは任意スパン抽出できます。

アーキテクチャ進行：

- **規則基づ。** Regex + gazetteer 検索。既知エンティティで高精度、新しいもあたり0カバレッジ。
- **HMM。** 隠れマルコフモデル。トークン与えられたタグの放射確率、タグタグ遷移確率。Viterbi デコード。ラベル付き データで訓練。
- **CRF。** 条件付きランダムフィールド。HMM のような が判別的、任意機能混合できる（単語形、資本化、隣接単語）。依然として古典的な本番の働き馬 2026 低リソース展開のため。
- **BiLSTM-CRF。** ニューラル機能手作り代わり。LSTMは文を両方向読む、CRF層の上に一貫したタグシーケンスを実行。
- **トランスフォーマー基づ。** トークン分類の頭でBERTをファインチューン。最高精度。最高計算。

## ビルド

### ステップ1：BIOタギングヘルパー

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### ステップ2：手作りの機能

古典的な（ニューラル）NER のため、機能がゲーム。有用なもの：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` は `xXxxxx` を戻す。`word_shape("USA-2024")` は `XXX-dddd` を戻す。資本化パターンは適切な名詞のためハイシグナル。

### ステップ3：シンプルな規則基づ+辞書ベースライン

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

本番 gazetteers は Wikipedia と DBpedia からスクレイプされた数百万エントリを持つ。カバレッジは良い。曖昧（`Apple` 会社対果物）は悪い。それは統計モデルを勝ったなぜ。

### ステップ4：CRFステップ（スケッチ、完全実装ではない）

完全CRFからゼロが50行で照明的でない確率理論基礎なし。代わりに `sklearn-crfsuite` を使用：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` と `c2` は L1 と L2 正則化。`all_possible_transitions=True` はモデルを非合法シーケンス学習を許可（例えば、`I-ORG` は `O` の後）が、BIOでない一貫性を強制する方法は、制約を書く。

### ステップ5：BiLSTM-CRFは何を追加するか

機能は学習された。入力：トークン埋め込み（GloVe または fastText）。LSTM は左から右と右から左を読む。連結隠れた状態は CRF 出力層を通す。CRF はタグシーケンス一貫性を依然強制；LSTMは手作り機能を学習されたもので置き換え。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF層のため、`torchcrf.CRF` を使用（pip install pytorch-crf）。手作り CRF を超える利益は測定可能ですが、数十の訓練文を持つ場合より小さい。

## 使う

spaCy は本番グレード NER を外すボックスで出荷。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` は `PRODUCT` ではなく `ORG` としてラベル—spaCy の小さいモデルは弱い製品エンティティカバレッジ。大きいモデル（`en_core_web_lg`）がより良く。トランスフォーマーモデル（`en_core_web_trf`）より良い。

Hugging Face BERT ベース NER のため：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` は連続する B-X, I-X トークンをスパンにマージ。なしで、トークンレベルラベルを取得し自分でマージする必要。

### LLM ベース NER（2026オプション）

ゼロショットと数ショット LLM NER は多くドメインで今微調整モデルに競争でき、ラベル付きデータが稀のときドラマティックに良い。

- **ゼロショットプロンプティング。** LLM に エンティティタイプリストと例スキーマをあげ。JSON 出力を求め。外に出るボックスで動き；精度は新しいドメイン上で中。
- **ZeroTuneBio スタイルプロンプティング。** タスクを候補抽出 → 意味説明 → 判断 → 再チェックに分解。マルチステージプロンプト（1ショット）は生物医学 NER 上で実質的に精度を上げる。同じパターンは法律、財政、科学ドメイン動作。
- **RAG と動的プロンプティング。** 全推論呼び出しのための小さい注釈された種子セットから最も同様な標識例を取得；飛び飛びに数ショットプロンプト構築。2026 ベンチマークでは、これは GPT-4 生物医学 NER F1 を 11-12% 静的プロンプティング上で上げます。
- **あたりエンティティタイプ分解。** 長いドキュメントのため、単一の呼び出し一度にすべてのエンティティタイプを抽出させる長さが成長するにつれリコールを失う。あたり1抽出パスエンティティタイプを実行。高い推論コスト、実質的に高い精度。これは臨床ノートと法的契約のための標準パターン。

本番推奨で 2026：訓練データ収集前に LLM ゼロショット基準を始める。多くの場合 F1 は十分な、あなたは微調整する必要なし。

### 古典 NER が依然勝つ場所

LLM 利用可能でも、古典 NER は下のときに勝つ：

- 遅延予算は 50ms 下。
- あなたはそれぞれ数千の標識サンプルを持ち 98%+ F1 必要。
- ドメインが訓練 CRF または BiLSTM が良く転送の安定本体を持つ。
- 正則制約は デバイス上、生成以外のモデルを必要。

### それがどこで分かれるか

- **ドメインシフト。** CoNLL 訓練 NER 法的契約で gazeteer より実行が悪い。あなたのドメインでファインチューン。
- **ネストされたエンティティ。** 「Bank of America Tower」は同時に ORG と FACILITY。標準 BIO は重複スパン表現できない。ネストされた NER が必要（マルチパス または スパン基づモデル）。
- **長エンティティ。** 「United States Federal Deposit Insurance Corporation.」トークンレベルモデル時々これを分割。`aggregation_strategy` またはポスト処理を使用。
- **スパースタイプ。** 医学 NER ラベル DRUG_BRAND、ADVERSE_EVENT、DOSE。一般目的モデルはアイデアなし。Scispacy と BioBERT はそこで開始ポイント。

## 出荷

`outputs/skill-ner-picker.md` として保存：

```markdown
---
name: ner-picker
description: 与えられた抽出タスクのため正しい NER アプローチをピック。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

タスク記述が与えられたら（ドメイン、ラベルセット、言語、遅延、データ量）、出力：

1. アプローチ。規則基づ + gazetteer、CRF、BiLSTM-CRF、またはトランスフォーマーファインチューン。
2. 開始モデル。それを名付け（spaCy モデル ID、Hugging Face チェックポイント ID、または「カスタム、ゼロから訓練」）。
3. ラベル戦略。BIO、BILOU、またはスパン基づ。1文で正当化。
4. 評価。`seqeval` を使用。常に報告エンティティレベル F1（トークンレベルではなく）。

ラベル付きサンプル 500 下でトランスフォーマーをファインチューニングすることを推奨することを拒否、あなたがトレーニング済み領域モデルを既に持つ場合を除き。ネストされたエンティティをスパン基づまたはマルチパスモデルとしてフラグ。「本番スケール」ユーザーが提及し CoNLL-2003 から変更されないラベルの gazetteer 監査を必要。
```

## 演習

1. **簡単。** `bio_to_spans`（`spans_to_bio` の逆）を実装し、10文で往復一貫性を確認。
2. **中程度。** CoNLL-2003 英語 NER データセットで上記のscikit-crfsuite CRF を訓練。`seqeval` を使用あたりエンティティ F1 を報告。典型的な結果：~84 F1。
3. **難しい。** `distilbert-base-cased` をドメイン特定 NER データセット（医学、法律、または財政）でファインチューン。spaCy 小さいモデルに対して比較。データリークチェック文書し何があなたを驚かせたか書く。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| NER | 名前を抽出 | タイプ（PERSON、ORG、GPE、DATE、...）でトークンスパンにラベル。|
| BIO | タギングスキーム | `B-X` は始める、`I-X` は続ける、`O` は外。|
| BILOU | より良い BIO | `L-X`（最後）を追加、`U-X`（ユニット）より清潔境界のため。|
| CRF | 構造化分類 | ラベル間の遷移モデル、排出のみ。有効なシーケンスを強制。|
| ネストされた NER | 重複エンティティ | 1つのスパンは異なるエンティティで別のスパンのサブスパン。BIO は表現できない。|
| エンティティレベル F1 | 適切な NER メトリック | 予想スパンは真スパンと正確にマッチする必要。トークンレベル F1 は精度を誇大。|

## 参考文献

- [Lample et al. (2016)。名前付きエンティティ認識のための神経アーキテクチャ](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF論文。標準的。
- [Devlin et al. (2018)。BERT：深い双方向トランスフォーマー事前訓練](https://arxiv.org/abs/1810.04805) — トークン分類パターンを導入された標準になった。
- [spaCy 言語機能 — 名前付きエンティティ](https://spacy.io/usage/linguistic_features#named-entities) — 各属性に関する`Doc.ents` と `Span` に関する実装参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正しいメトリックライブラリ。常に使用。
