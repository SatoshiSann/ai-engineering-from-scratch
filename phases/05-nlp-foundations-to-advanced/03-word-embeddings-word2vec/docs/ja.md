# 単語埋め込み — Word2Vecをゼロから

> 単語はそれが付き合う企業による。その考えに浅いネットを訓練すれば、幾何学が出てくる。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 02（BoW + TF-IDF）、Phase 3 · 03（バックプロパゲーションをゼロから）
**所要時間:** 約75分

## 問題

TF-IDFは `dog` と `puppy` が異なる単語であることを知っている。それらがほぼ同じ意味であることを知らない。`dog` で訓練されたクラシファイアは `puppy` についてのレビューに一般化できない。シノニムをリストアップしてこれを紙やすりでこすることができますが、まれな用語、ドメインジャーゴン、および予想しなかった全言語で失敗します。

`dog` と `puppy` が空間で互いに近く着地する表現が必要。`king - man + woman` が `queen` の近くに着地する。`dog` で訓練されたモデルが自由に `puppy` に信号を転送する。

Word2Vecはその空間をくれた。2層ニューラルネットワーク、1兆トークン訓練実行、2013年に発表。アーキテクチャはほぼ恥ずかしいほど単純。結果はNLPの10年間の形を変えた。

## コンセプト

**分布的仮説** (Firth, 1957)：「単語を企業によって知った。」2つの単語が同様のコンテキストに表示されたら、似たようなことを意味します。

Word2Vecは2つのフレーバーで来て、どちらもその考えを利用する。

- **Skip-gram。** 中心単語が与えられたら、周囲の単語を予測。`cat -> (the, sat, on)` ウィンドウサイズ2で。
- **CBOW（連続単語の袋）。** 周囲の単語が与えられたら、中心を予測。`(the, sat, on) -> cat`。

Skip-gramはトレーニングが遅いですが、希な単語をより良く処理。それはデフォルトになった。

ネットワークは非線形のない1つの隠れた層を持つ。入力は語彙上の1ホット ベクトル。出力は語彙上のソフトマックス。訓練後、出力層を捨てる。隠れた層の重みはエンベディング。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

トリック：100k単語のソフトマックスは法外に高い。Word2Vecは**ネガティブ サンプリング**を使用してそれをバイナリ分類タスクに変える。「このコンテキスト単語がこの中心単語の近くに出現したか、はい or いいえ」予測。語彙全体のソフトマックスを計算する代わりに、訓練ペアごとに複数のネガティブ（共起しない）単語をサンプル。

## ビルド

### ステップ1：コーパスから訓練ペアへ

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

ウィンドウ内のすべての (中心、コンテキスト) ペアは肯定的な訓練例。

### ステップ2：埋め込みテーブル

2つの行列。`W` は中心単語埋め込みテーブル（保持するもの）。`W'` はコンテキスト単語テーブル（しばしば破棄、時々 `W` と平均化）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小さなランダムイニット。語彙サイズ10kと暗い100は現実的；教えるために、50語彙x 16暗いは幾何学を見るのに十分。

### ステップ3：ネガティブサンプリングオブジェクティブ

各肯定ペア `(center, context)` について、語彙からランダムに単語 `k` をネガティブとしてサンプル。肯定的なもではドット積 `W[center] · W'[context]` を高く、ネガティブなもでは低くするようにモデルを訓練。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法の公式：肯定ペアの論理損失（sigmoid 近い 1）加えてネガティブペアの論理損失（sigmoid 近い 0）。勾配は両方のテーブルに流れる。完全な導出は原論文にあります；鉛筆と紙でそれを一度歩みましょう、それを固くしたい場合。

### ステップ4：小さなコーパスで訓練

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

大規模なコーパスで十分な時代の後、共有コンテキストを持つ単語は同様の中心埋め込みを持つ。小さなコーパスで、効果がかすかに見えます。数十億トークンで、それは劇的に見えます。

### ステップ5：類推トリック

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

事前訓練300dグーグルニュースベクトル上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。モデルが王族が何であるか知っているからではない。ベクトル `(king - man)` が「王立」のような何かをキャプチャし、それを `woman` に加える女性王族地域の近くに着地しているから。

## 使う

ゼロからWord2Vecを書くことは教え。本番NLPは `gensim` を使用。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

実際の仕事のために、あなたはほぼ決してWord2Vecを自分で訓練していない。事前訓練されたベクトルをダウンロード。

- **GloVe** — スタンフォードの共起行列因数分解アプローチ。50d、100d、200d、300dチェックポイント。良い一般的カバレッジ。レッスン04はGloVeについて特に説明。
- **fastText** — フェイスブックのWord2Vec拡張は文字n-グラムを埋め込む。語彙外単語をサブワード作成で処理。レッスン04。
- **グーグルニュースの事前訓練Word2Vec** — 300d、300万単語語彙、2013年発表。毎日まだダウンロード。

### Word2Vecがまだ2026年で勝つ場合

- 軽いドメイン特有検索。1時間でラップトップ医学要約で訓練、一般的なモデルキャプチャなしで特化したベクトル取得。
- 類推スタイル機能エンジニアリング。`gender_vector = mean(man - woman pairs)`。他の単語からそれを引く性別中立軸を取得。まだ公平研究で使用。
- 解釈可能性。100dはPCAまたはt-SNEで実際にプロット十分小さくてクラスタが形成されます。
- 推論がGPUなしでデバイス上で実行する必要がある場所。Word2Vec検索は単一行フェッチ。

### Word2Vecが失敗する場所

ポリシーミー壁。`bank` は1つのベクトル。`river bank` と `financial bank` はそれを共有。`table` （スプレッドシート対家具）はそれを共有。下流クラシファイアはベクトルから意味を区別できない。

文脈埋め込み（ELMo、BERT、すべてのトランスフォーマー以来）は周囲のコンテキストに基づいて単語の各出現に異なるベクトルを生成することでこれを解決。これはWord2VecからBERTへのジャンプ：静的から文脈へ。フェーズ7はトランスフォーマー半分をカバー。

語彙外問題はもう1つの失敗。Word2Vecは訓練データに含まれていなかった `Zoomer-approved` を決してみた。フォールバックなし。fastTextはサブワード作成でこれを修正（レッスン04）。

## 出荷

`outputs/skill-embedding-probe.md` として保存：

```markdown
---
name: embedding-probe
description: word2vecモデルを検査。アナロジー実行、近くを見つけ、品質を診断。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

訓練された単語エンベディングをプローブして、それらが働いていることを確認。`gensim.models.KeyedVectors` オブジェクトと語彙が与えられたら、実行：

1. 3つの標準的なアナロジーテスト。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。トップ1結果とそのコサインを報告。
2. ユーザーが供給するドメイン特有の単語での5つの最近傍テスト。コサイン付きトップ5近くをプリント。
3. 1つ対称性チェック。`similarity(a, b) == similarity(b, a)` float精度内。
4. 1つ退化チェック。任意の埋め込みが0.01以下または100以上のノルムを持つ場合、モデルに訓練バグがあります。フラグ。

アナロジー精度単独でモデルが良いと宣言することを拒否。アナロジーベンチマークはゲーム可能で下流タスクに転送しない。本来+下流評価一緒に推奨。
```

## 演習

1. **簡単。** 小さなコーパス上（猫と犬について20文）訓練ループを実行。200時代後、`nearest(vocab, W, W[vocab["cat"]])` は `dog` をトップ3に戻すことを確認。そうでなければ時代を増やすか語彙。
2. **中程度。** 頻繁な単語のサブサンプリングを追加。`10^-5` より高い頻度を持つ単語はそれらの頻度に比例する確率で訓練ペアからドロップ。希な単語類似性への効果測定。
3. **難しい。** 20 Newsgroups コーパスで訓練モデル。2つの偏差軸を計算：`he - she` と `doctor - nurse`。職業単語を両軸に投影。最大偏差ギャップを持つ職業を報告。これは公平研究者が使用するプロービングタイプ。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 単語埋め込み | ベクトルとしての単語 | 文脈から学習した密で低次元（通常100-300）表現。|
| Skip-gram | Word2Vecトリック | 中心単語から文脈単語を予測。CBOWより遅い、希な単語に良い。|
| ネガティブサンプリング | 訓練ショートカット | 完全語彙ソフトマックスをk個のランダム単語に対するバイナリ分類に置き換え。|
| 静的埋め込み | 単語ごと1ベクトル | コンテキストに関わらず同じベクトル。ポリセミで失敗。|
| 文脈埋め込み | 文脈感応ベクトル | 周囲の単語に基づいて各出現に異なるベクトル。トランスフォーマーが生成するもの。|
| OOV | 語彙外 | 訓練でみられない単語。Word2Vecはこれらのベクトルを生成できない。|

## 参考文献

- [Mikolov et al. (2013). 単語とフレーズの分散表現とそれらの合成性](https://arxiv.org/abs/1310.4546) — ネガティブサンプリング論文。短くて読める。
- [Rong, X. (2014). word2vec パラメータ学習説明](https://arxiv.org/abs/1411.2738) — 勾配の最も明確な導出、元論文の数学が密に感じる場合。
- [gensim Word2Vecチュートリアル](https://radimrehurek.com/gensim/models/word2vec.html) — 実際に機能する本番訓練設定。
