# 品詞タグ付けと構文解析

> 文法は一時期流行遅れだった。その後、あらゆる LLM パイプラインが構造化抽出を検証する必要に迫られ、文法は復活した。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 01 (Text Processing)、Phase 2 · 14 (Naive Bayes)
**所要時間:** 約45分

## 問題

レッスン 01 では、レンマ化には品詞タグが必要だと述べた。`running` が動詞であると知らなければ、レンマ化器はそれを `run` に還元できない。`better` が形容詞であると知らなければ、`good` に還元できない。

その約束は丸ごと一つのサブフィールドを隠していた。品詞タグ付けは文法的カテゴリを割り当てる。構文解析は文の木構造を復元する。どの語がどの語を修飾するのか、どの動詞がどの引数を支配するのか。古典的 NLP は両方を 20 年かけて洗練させた。その後、ディープラーニングがこれらを事前学習済みトランスフォーマー上のトークン分類タスクへと畳み込み、研究コミュニティは先へ進んだ。

応用コミュニティは違う。あらゆる構造化抽出パイプラインは、今なお内部で品詞と依存木を使っている。LLM が生成した JSON は文法的制約に照らして検証される。質問応答システムは依存構文解析を使ってクエリを分解する。機械翻訳の品質評価器は構文木のアラインメントをチェックする。

知っておく価値がある。このレッスンでは、タグセット、ベースライン、そして自前実装をやめて spaCy を呼び出すべき地点を紹介する。

## コンセプト

**品詞タグ付け (POS tagging)** は各トークンに文法的カテゴリのラベルを付ける。**Penn Treebank (PTB)** タグセットは英語のデフォルトだ。36 個のタグがあり、カジュアルな読者には細かすぎると感じる区別がある。`NN` 単数名詞、`NNS` 複数名詞、`NNP` 固有名詞単数、`VBD` 動詞過去形、`VBZ` 動詞三人称単数現在形、などだ。**Universal Dependencies (UD)** タグセットはより粗く (17 タグ)、言語非依存だ。これは言語横断作業のデフォルトになった。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**構文解析 (syntactic parsing)** は木を生成する。主要な 2 つのスタイルがある。

- **句構造解析 (constituency parsing)。** 名詞句、動詞句、前置詞句が互いに入れ子になる。出力は非終端カテゴリ (NP、VP、PP) の木で、語が葉となる。
- **依存構文解析 (dependency parsing)。** 各語は依存先となる唯一の主辞語を持ち、文法関係でラベル付けされる。出力は木で、各辺が (主辞、従属辞、関係) の三つ組となる。

依存構文解析は 2010 年代に勝利した。言語、特に自由語順言語をまたいできれいに一般化できるからだ。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## ビルド

### ステップ 1: 最頻タグベースライン

機能する最も単純な品詞タガー。各語について、訓練データで最も頻繁に付いていたタグを予測する。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Brown コーパスでは、このベースラインは約 85% の精度に達する。良くはないが、まともなモデルなら下回ってはならない下限だ。

### ステップ 2: バイグラム HMM タガー

系列の同時確率をモデル化する。

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

2 つのテーブル。遷移確率 (前のタグが与えられたときのタグ) と出力確率 (タグが与えられたときの語)。両方を Laplace スムージングつきの頻度から推定する。Viterbi (タグ格子上の動的計画法) でデコードする。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 上のバイグラム HMM は約 93% の精度に達する。85% から 93% への跳躍はほとんどが遷移確率による。モデルは `DET NOUN` が一般的で `NOUN DET` がまれであることを学習する。

### ステップ 3: 現代のタガーがこれを上回る理由

遷移確率と出力確率は局所的だ。それらは "I bought a saw" では `saw` が名詞だが "I saw the movie" では動詞であることを捉えられない。任意の特徴量 (接尾辞、語形、前後の語、語そのもの) を持つ CRF は約 97% に達する。BiLSTM-CRF やトランスフォーマーは約 98% 以上に達する。

このタスクの上限はアノテーター間の不一致によって決まる。Penn Treebank では人間のアノテーターは約 97% の確率で一致する。98% を超えるモデルはおそらくテストセットに過学習している。

### ステップ 4: 依存構文解析のスケッチ

依存構文解析を一から完全に実装するのは範囲外だ。標準的な教科書での扱いは Jurafsky と Martin にある。知っておくべき 2 つの古典的な系統がある。

- **遷移ベース (transition-based)** パーサー (arc-eager、arc-standard) はシフト・リデュースパーサーのように動作する。トークンを読み、スタックにシフトし、弧を作成するリデュースアクションを適用する。貪欲デコードは高速だ。古典的な実装は MaltParser。現代的なニューラル版は Chen と Manning の遷移ベースパーサー。
- **グラフベース (graph-based)** パーサー (Eisner のアルゴリズム、Dozat-Manning biaffine) は、考えられるすべての主辞・従属辞辺をスコア付けし、最大全域木を選ぶ。遅いがより正確だ。

ほとんどの応用作業では spaCy を呼び出す。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

`dep` 列を下から上へ読むと、文の文法構造が浮かび上がる。

## 使う

あらゆる本番 NLP ライブラリは、標準パイプラインの一部として品詞・依存構文解析器を同梱している。

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`)。高速、正確、トークン化 + NER + レンマ化と統合済み。`token.tag_` (Penn)、`token.pos_` (UD)、`token.dep_` (依存関係)。
- **Stanford NLP (stanza)**。Stanford の CoreNLP 後継。60 以上の言語で最先端。
- **trankit**。トランスフォーマーベース、良好な UD 精度。
- **NLTK**。`pos_tag`。使えるが遅く、古い。教育用には十分。

### 2026 年でもこれが重要な場面

- **レンマ化。** レッスン 01 は正しくレンマ化するために品詞を必要とする。常に。
- **LLM 出力からの構造化抽出。** 生成された文が文法的制約 (例: 主語と動詞の一致、必須修飾語) を尊重しているか検証する。
- **アスペクトベース感情分析。** 依存構文解析は、どの形容詞がどの名詞を修飾するかを教えてくれる。
- **クエリ理解。** "movies directed by Wes Anderson starring Bill Murray" は構文解析を通じて構造化された制約へと分解される。
- **言語横断転移。** UD タグと依存関係は言語非依存であり、新しい言語のゼロショット構造化分析を可能にする。
- **低計算量パイプライン。** トランスフォーマーを出荷できない場合、品詞 + 依存構文解析 + 地名辞典 (gazetteer) で驚くほど遠くまで行ける。

## 出荷

`outputs/skill-grammar-pipeline.md` として保存する。

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 演習

1. **易しい。** 小さなタグ付きコーパス (例: NLTK の Brown サブセット) で最頻タグベースラインを使い、ホールドアウト文での精度を測定せよ。約 85% という結果を検証せよ。
2. **中級。** 上記のバイグラム HMM を訓練し、タグごとの適合率/再現率を報告せよ。HMM が最も混同するのはどのタグか。
3. **難しい。** spaCy の依存構文解析を使って、1000 文のサンプルから主語・動詞・目的語の三つ組を抽出せよ。50 個の手動ラベル付き三つ組で評価せよ。抽出が失敗する箇所 (多くは受動態、等位接続、省略された主語) を文書化せよ。

## 重要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| 品詞タグ | 語のタイプ | 文法的カテゴリ。PTB は 36 個、UD は 17 個。 |
| Penn Treebank | 標準タグセット | 英語特化。動詞の時制と名詞の数を細かく区別。 |
| Universal Dependencies | 多言語タグセット | PTB より粗く、言語中立、言語横断作業のデフォルト。 |
| 依存構文解析 | 文の木 | 各語は一つの主辞を持ち、各辺は文法関係を持つ。 |
| Viterbi | 動的計画法 | 出力確率と遷移確率が与えられたとき、最も確率の高いタグ系列を見つける。 |

## 参考文献

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 品詞と構文解析の標準的な教科書での扱い。
- [Universal Dependencies project](https://universaldependencies.org/) — あらゆる多言語パーサーが使う言語横断タグセットとツリーバンクのコレクション。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` で公開されるすべての属性の実用的なリファレンス。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — ニューラルパーサーを主流にした論文。
