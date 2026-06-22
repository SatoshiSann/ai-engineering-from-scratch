# テキスト処理 — トークン化、ステミング、見出し語化

> 言語は連続的だ。モデルは離散的だ。前処理はその橋渡しである。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 2 · 14 (Naive Bayes)
**所要時間:** 約45分

## 問題

モデルは "The cats were running." を読めない。整数を読むのである。

あらゆる NLP システムは、同じ3つの問いで始まる。単語はどこで始まるのか。単語の語根は何か。「run」「running」「ran」を、それが役立つときには同じものとして、役立たないときには異なるものとして、どう扱うのか。

トークン化を誤れば、モデルはゴミから学習する。トークナイザーが `don't` を1トークンとして扱うのに `do n't` を2トークンとして扱えば、訓練分布が分裂する。ステマーが `organization` と `organ` を同じステムに潰せば、トピックモデリングは死ぬ。見出し語化器が品詞の文脈を必要とするのにそれを渡さなければ、動詞が名詞として扱われる。

このレッスンは3つの前処理ステップをゼロから構築し、それから NLTK と spaCy が同じ作業をどう行うかを示して、トレードオフを見られるようにする。

## コンセプト

3つの操作。それぞれに役割と失敗モードがある。

**トークン化 (Tokenization)** は文字列をトークンに分割する。「トークン」という言葉が意図的に曖昧なのは、適切な粒度がタスクに依存するからだ。古典的 NLP には単語レベル。トランスフォーマーにはサブワード。空白のない言語には文字。

**ステミング (Stemming)** はルールで接尾辞を切り落とす。高速、攻撃的、愚直。`running -> run`。`organization -> organ`。この2つ目が失敗モードである。

**見出し語化 (Lemmatization)** は文法知識を使って単語を辞書形に還元する。遅く、正確で、参照テーブルか形態素解析器が必要だ。`ran -> run`(「ran」が「run」の過去形だと知る必要がある)。`better -> good`(比較級の形を知る必要がある)。

経験則。速度が重要でノイズを許容できるときはステミングする(検索インデックス作成、おおまかな分類)。意味が重要なときは見出し語化する(質問応答、意味検索、ユーザーが読むものすべて)。

## ビルドする

### ステップ1: 正規表現による単語トークナイザー

最もシンプルで実用的なトークナイザーは、句読点をそれ自身のトークンとして保ちつつ、非英数字文字で分割する。完璧でも最終形でもないが、1行で動く。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

優先順位の順に3つのパターン。内側にアポストロフィを任意で含む単語(`don't`、`it's`)。純粋な数字。空白でも英数字でもない単一文字をスタンドアロンのトークンとして(句読点)。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注目すべき失敗モード。`3pm` は `['3', 'pm']` に分割される。なぜなら、文字の並びと数字の並びを交互に扱っているからだ。ほとんどのタスクには十分。URL、メール、ハッシュタグはすべて壊れる。本番では、一般的なパターンの前にこれらのパターンを追加する。

### ステップ2: Porter ステマー(ステップ1aのみ)

完全な Porter アルゴリズムには5つのルールフェーズがある。ステップ1aだけでも最も頻出する英語の接尾辞をカバーし、そのパターンを教えてくれる。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

ルールを上から下に読む。`ponies -> poni` であって `pony` でないのは、`ies -> i` ルールのせいだ。本物の Porter にはこれを修正するステップ1bがある。ルールは競合する。先のルールが勝つ。順序は、どの単一ルールよりも重要である。

### ステップ3: 参照ベースの見出し語化器

本来の見出し語化には形態論が必要だ。扱いやすい教育用バージョンは、小さな見出し語テーブルとフォールバックを使う。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最後のケースが重要な教育ポイントだ。`watched` はテーブルになく、フォールバックは `ing` しか処理しない。本物の見出し語化は `ed`、不規則動詞、比較級形容詞、音変化を伴う複数形(`children -> child`)をカバーする。だからこそ本番システムは WordNet、spaCy の形態素解析器、または完全な形態素解析器を使うのである。

### ステップ4: パイプでつなぐ

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

欠けている部分は POS タガーだ。Phase 5 · 07(POS タギング)で1つ構築する。今のところ、すべてを `NOUN` にデフォルトし、その限界を認めておく。

## 使ってみる

NLTK と spaCy は本番バージョンを提供する。それぞれ数行だ。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` は短縮形、Unicode、正規表現が見逃すエッジケースを処理する。`PorterStemmer` は5つのフェーズすべてを実行する。`WordNetLemmatizer` は、NLTK の Penn Treebank 方式から WordNet の略語セットに翻訳された POS タグを必要とする。上記の翻訳の配線は、ほとんどのチュートリアルがスキップする部分である。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy はパイプライン全体を `nlp(text)` の背後に隠す。トークン化、POS タギング、見出し語化がすべて実行される。大規模では NLTK より高速だ。すぐに使える状態でより正確だ。トレードオフは、個々のコンポーネントを簡単に差し替えられないことである。

### どちらをいつ選ぶか

| 状況 | 選ぶもの |
|-----------|------|
| 教育、研究、コンポーネントの差し替え | NLTK |
| 本番、多言語、速度が重要 | spaCy |
| トランスフォーマーのパイプライン(どのみちモデルのトークナイザーでトークン化する) | `tokenizers` / `transformers` を使い、古典的な前処理をスキップする |

### 誰も警告してくれない2つの失敗モード

ほとんどのチュートリアルはアルゴリズムを教えてそこで止まる。2つのことが本物の前処理パイプラインに噛みつくが、それらはほとんど扱われない。

**再現性ドリフト。** NLTK と spaCy は、バージョン間でトークン化と見出し語化器の挙動を変える。spaCy 2.x で `['do', "n't"]` を生成したものが、3.x では `["don't"]` を生成するかもしれない。あなたのモデルは1つの分布で訓練された。今や推論は別の分布で実行されている。精度が静かに劣化し、誰もその理由を知らない。`requirements.txt` でライブラリのバージョンを固定する。20個のサンプル文の期待されるトークン化を凍結する前処理の回帰テストを書く。アップグレードのたびに実行する。

**訓練/推論のミスマッチ。** 攻撃的な前処理(小文字化、ストップワード除去、ステミング)で訓練し、生のユーザー入力でデプロイし、性能が暴落するのを見る。これは最も一般的な本番 NLP の失敗である。訓練中に前処理するなら、推論中に同一の関数を実行しなければならない。前処理をモデルパッケージの中の関数として出荷する。サービングチームが書き直すノートブックのセルとしてではなく。

## 出荷する

3冊の教科書を読まずにエンジニアが前処理戦略を選ぶのを助ける、再利用可能なプロンプト。

`outputs/prompt-preprocessing-advisor.md` として保存する。

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 演習

1. **易。** `tokenize` を拡張して URL を単一のトークンとして保つようにする。テスト: `tokenize("Visit https://example.com today.")` は1つの URL トークンを生成するべきだ。
2. **中。** Porter ステップ1bを実装する。単語が母音を含み `ed` または `ing` で終わる場合、それを除去する。二重子音ルールを処理する(`hopping -> hop`、`hopp` ではなく)。
3. **難。** WordNet を参照テーブルとして使うが、WordNet にエントリがないときはあなたの Porter ステマーにフォールバックする見出し語化器を構築する。タグ付きコーパスで、素の WordNet と素の Porter に対する精度を測定する。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| トークン (Token) | 単語 | モデルが消費する任意の単位。単語、サブワード、文字、バイトのいずれにもなり得る。 |
| ステム (Stem) | 単語の語根 | ルールベースの接尾辞除去の結果。必ずしも実在の単語ではない。 |
| 見出し語 (Lemma) | 辞書形 | 辞書で引く形。正しく計算するには文法的文脈が必要。 |
| POS タグ (POS tag) | 品詞 | NOUN、VERB、ADJ のようなカテゴリ。正確に見出し語化するために必要。 |
| 形態論 (Morphology) | 語形のルール | 時制、数、格に基づいて単語がどう形を変えるか。見出し語化はこれに依存する。 |

## 参考文献

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 元の論文、5ページ、いまだに最も明快な説明。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 本物のパイプラインがどう配線されているか。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — まだ思いつかなかったトークン化のエッジケース。
