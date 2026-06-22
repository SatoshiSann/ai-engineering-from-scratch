# 単語の袋（Bag of Words）、TF-IDF、テキスト表現

> まず数えて、後で考える。TF-IDFは2026年でも、よく定義されたタスクではエンベディングを上回る。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 01（テキスト処理）、Phase 2 · 02（ゼロからの線形回帰）
**所要時間:** 約75分

## 問題

モデルは数字が必要だ。お前には文字列がある。

すべてのNLPパイプラインは同じ質問に答える必要がある。長さが可変のトークンのストリームを、分類器が消費できる固定サイズのベクトルにどのように変えるのか。その分野が最初に着目した答えは、最も単純に機能するものだった。単語を数える。ベクトルを作る。

そのベクトルは、どのエンベディングモデルよりも多くの本番NLPを運んでいる。スパムフィルター、トピック分類器、ログ異常検出、検索ランキング（BM25以前）、感情分析の初期段階、学術的NLPベンチマークの最初の10年間。2026年の実務家でもまだ、単語の存在が重要なタスクでは最初に手を伸ばす。それは高速で、解釈可能で、しばしば400M パラメータのエンベディングモデルと区別がつかない。

このレッスンでは、単語の袋を最初からビルドし、次にTF-IDFをビルドする。次にscikit-learnが3行で同じことをしているのを示す。次に、エンベディングに手を伸ばすようにする失敗モードを名付ける。

## コンセプト

**Bag of Words (BoW)** は順序を捨てる。各ドキュメントについて、語彙内の各単語が何回出現するかを数える。ベクトルの長さは語彙のサイズである。位置 `i` は単語 `i` のカウントである。

**TF-IDF** はBoWを再び重み付けする。すべてのドキュメントに表示される単語は情報量がないので、スケールダウンする。コーパス全体でまれですが、単一のドキュメントで頻繁な単語は信号なので、スケールアップする。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

ここで `TF` はドキュメント内の用語頻度、 `df` はドキュメント頻度（単語を含むドキュメント数）、 `N` は総ドキュメント数である。`log` は遍在する単語の重みを制限する。

重要な特性：どちらもスパースベクトルを解釈可能な軸で生成する。訓練されたクラシファイアの重みを見て、どの単語がドキュメントを各クラスに向かって押すかを読むことができる。768次元のBERT埋め込みではこれはできない。

## ビルド

### ステップ1：語彙を構築する

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

入力：トークン化されたドキュメントのリスト（任意のワードレベルのトークナイザーで十分；このレッスンの `code/main.py` は単純な小文字バリアントを使用）。出力：`{word: index}` 辞書。安定した挿入順は、単語インデックス 0 が最初のドキュメントで最初に見られた単語であることを意味する。慣例は異なる；scikit-learnはアルファベット順にソートする。

### ステップ2：単語の袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行はドキュメント。列は語彙インデックス。エントリ `[i][j]` は「単語 `j` がドキュメント `i` に何回出現するか」である。ドキュメント 1 は `cat` を2回持つ、そうだったから。ドキュメント 0 は `ran` をゼロ回持つ、そうでなかったから。

### ステップ3：用語頻度とドキュメント頻度

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

名前が値する2つのスムージングトリック。`(n+1)/(d+1)` は `log(x/0)` を回避する。末尾の `+1` は、すべてのドキュメントに含まれる単語でもまだ IDF 1 を持つ（0ではなく）ことを保証し、scikit-learnのデフォルトと一致する。他の実装は生の `log(N/df)` を使用する。どちらでも機能するが、スムージングされたバージョンはより親切である。

### ステップ4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

3つのドキュメント、5つの語彙単語（`the`、`cat`、`sat`、`dog`、`ran`）。`the` は3つすべてに出現するので、その IDF は低い。`dog` は1つに出現するので、その IDF は高い。ベクトルはスパース（ほとんどのエントリは小さい）で判別的な単語が飛び出す。

### ステップ5：L2-正規化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

正規化がなければ、より長いドキュメントはより大きなベクトルを取得し、類似度スコアを支配する。L2正規化は、すべてのドキュメントを単位超球面に置く。行間のコサイン類似度は、単なるドット積である。

## 使う

scikit-learnは本番バージョンを出荷する。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` は1つの呼び出しでトークン化、語彙、およびBoWを実行する。`TfidfVectorizer` はIDF重み付けとL2正規化を追加する。どちらもスパース行列を返す。100kドキュメントの場合、密なバージョンはメモリに適合しない；分類器が密を要求するまでスパースのままにする。

すべてを変更するノブ：

| 引数 | 効果 |
|-----|--------|
| `ngram_range=(1, 2)` | バイグラムを含める。通常は分類を高める。|
| `min_df=2` | 2つ未満のドキュメント内の単語をドロップ。ノイズの多いデータの語彙をトリム。|
| `max_df=0.95` | 95%以上のドキュメント内の単語をドロップ。ハードコードされたリストなしにストップワード削除に近づく。|
| `stop_words="english"` | scikit-learnのビルトインストップワードリスト。タスク依存—感情分析は否定をドロップすべきではない。|
| `sublinear_tf=True` | 生の `tf` の代わりに `1 + log(tf)` を使用。用語が1つのドキュメントで何度も繰り返される場合に役立つ。|

### TF-IDFがまだ勝つ場合（2026年現在）

- スパム検出、トピックラベリング、ログ異常フラグ付け。単語の存在が重要；意味的微妙さはしない。
- 低データレジーム（数百の標識付き例）。TF-IDFプラスロジスティック回帰に事前学習コストはない。
- 遅延が重要な場所。TF-IDFプラスリニアモデルはマイクロ秒で応答する。トランスフォーマーを通じてドキュメントを埋め込むには10-100ミリ秒かかる。
- 予測を説明する必要があるシステム。分類器の係数を検査。最上位の肯定的な単語が理由である。

### TF-IDFが失敗する場合

意味的な盲目失敗。これら2つのドキュメントを考える：

- 「The movie was not good at all.」
- 「The movie was excellent.」

1つはネガティブレビュー。1つはポジティブ。それらのTF-IDFオーバーラップは正確に `{the, movie, was}` である。単語バッグ分類器は、単語 `not` が `good` の近くにあるとラベルをフリップすることを暗記する必要がある。十分なデータでこれを学ぶことができますが、構文を理解するモデルほど優雅ではありません。

もう1つの失敗：推論時のボキャブラリー外単語。IMDbレビューで訓練されたBoWモデルは、そのトークンが訓練データに表示されなかった場合、`Zoomer-approved` で何をすべきかはっきりしない。サブワード埋め込み（レッスン04）はこれを処理する。TF-IDFはできない。

### ハイブリッド：TF-IDF重み付きエンベディング

2026年のプラグマティックなデフォルト（中程度のデータ分類）：TF-IDF重みを単語エンベディングに対する注意として使用する。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

埋め込みから意味容量を取得し、TF-IDFから希な単語強調を取得する。分類器はプール済みベクトル上で訓練する。これは約50k標識付き例以下の感情、トピック、インテント分類で単独での実行を上回る。

## 出荷

`outputs/prompt-vectorization-picker.md` として保存：

```markdown
---
name: vectorization-picker
description: テキスト分類タスクを与えられたら、BoW、TF-IDF、エンベディング、またはハイブリッドを推奨する。
phase: 5
lesson: 02
---

テキストベクトル化戦略を推奨する。タスク説明が与えられたら、出力：

1. 表現（BoW、TF-IDF、トランスフォーマーエンベディング、またはハイブリッド）。1文で理由を説明。
2. 具体的なベクトルライザー構成。ライブラリに名前を付ける。引数を引用（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 出荷前にテストする1つの失敗モード。

ユーザーが500未満の標識付き例を持っている場合、TF-IDFベースラインの意味的失敗の証拠を示さない限り、エンベディング推奨を拒否。感情分析のためのストップワード削除を拒否（否定は信号を運ぶ）。クラス不均衡をベクトルライザー変更以上が必要であるとフラグ。

例入力：「30kの顧客サポートチケットを12カテゴリに分類。ほとんどのチケットは2-3文。英語のみ。監査ログのための説明可能性が必要。」

例出力：

- 表現：TF-IDF。30kの例は小さくない；説明可能性要件は密なエンベディングを除外する。
- 構成：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。ストップワードを保持、カテゴリキーワードは時々ストップワード（「not working」対「working」）。
- テストする失敗：`min_df=3` が希なカテゴリキーワードをドロップしないことを確認。`get_feature_names_out` をクラスでフィルター実行し、確認。
```

## 演習

1. **簡単。** L2正規化されたTF-IDF出力で `cosine_similarity(doc_vec_a, doc_vec_b)` を実装。同一ドキュメントが1.0でスコアリングされ、互いに素なボキャブラリー ドキュメントが0.0でスコアリングされることを確認。
2. **中程度。** `bag_of_words` に `n-gram` サポートを追加。パラメータ `n` は `n`-グラム上でカウント生成。`n=2` で `["the", "cat", "sat"]` が `["the cat", "cat sat"]` のバイグラムカウント生成することをテスト。
3. **難しい。** GloVe 100dベクトル（一度ダウンロード、キャッシュ）を使用して上記のTF-IDF重み付けエンベディングハイブリッドをビルド。20 Newsgroups データセット上で、平純TF-IDFと平均プール埋め込みに対する分類精度を比較。どこで勝つかを報告。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| BoW | 単語頻度ベクトル | 1つのドキュメントの語彙単語のカウント。順序を捨てる。|
| TF | 用語頻度 | ドキュメント内の単語のカウント、オプションでドキュメント長で正規化。|
| DF | ドキュメント頻度 | 単語を少なくとも1回含むドキュメントのカウント。|
| IDF | 逆ドキュメント頻度 | `log(N / df)` スムーズ。至る所に出現する単語の重みを減らす。|
| スパースベクトル | ほとんどゼロ | ボキャブラリーは通常10k-100k単語；ほとんどは任意のドキュメントから欠落。|
| コサイン類似度 | ベクトル角 | L2正規化ベクトルのドット積。1は同一、0は直交。|

## 参考文献

- [scikit-learn — テキストからの特徴抽出](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 標準的なAPI参考、すべてのノブに関する注記。
- [Salton, G., & Buckley, C. (1988). 自動テキスト検索における用語重み付けアプローチ](https://www.sciencedirect.com/science/article/pii/0306457388900210) — TF-IDFを10年のデフォルトにした論文。
- [「TF-IDFがなぜまだエンベディングを上回るのか」— Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 古い方法が勝つときと理由について2026年の見方。
