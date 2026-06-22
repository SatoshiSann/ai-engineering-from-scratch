# GloVe、FastText、サブワード埋め込み

> Word2Vecは単語ごと1埋め込みを訓練。GloVeは共起行列を因数分解。FastTextはピースを埋め込み。BPEはトランスフォーマーにブリッジ。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 03（Word2Vecをゼロから）
**所要時間:** 約45分

## 問題

Word2Vecは2つのオープン質問を残した。

最初に、共起行列を直接因数分解するスキップグラム更新オンラインをする代わりに研究の平行線（LSA、HAL）がありました。Word2Vecの反復的アプローチは本質的に良いのか、違いは2つの方法がカウント処理する方法のアーティファクトか？**GloVe** それに答えた：行列因数分解と考えられた損失マッチまたはWord2Vecを上回り、訓練に少なくかかった。

2番目に、どちらの方法も見たことのない単語のストーリーを持たなかった。`Zoomer-approved`、`dogecoin`、先週造られた任意の適切な名詞、希な根の全インフレクト形。**FastText** 文字n-グラムを埋め込むことでこれを修正：単語はモルフェム含むそれの部分の合計、語彙外単語でさえも理にかなったベクトルを取得。

3番目に、トランスフォーマーが到着したら、質問は再び変わった。単語レベルのボキャブラリーは約100万エントリでキャップ；実言語はそれより開いている。**Byte-pair encoding (BPE)** およびその親戚はすべてをカバーする頻繁なサブワードユニットのボキャブラリー学習によってこれを解決。最新LLMのため各現代的トークナイザーはサブワードトークナイザー。

このレッスンは3つすべてを歩き、次にそれぞれいつ手を伸ばすかを説明。

## コンセプト

**GloVe (Global Vectors)。** 単語単語共起行列 `X` ビルド `X[i][j]` は単語 `j` が単語 `i` のコンテキストで何度出現するか。訓練ベクトルそのような `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。頻繁なペア支配ので損失重み付け。完成。

**FastText。** 単語は文字n-グラムプラス単語自体の合計。`where` は `<wh, whe, her, ere, re>, <where>` になる。単語ベクトルはそれらのコンポーネントベクトルの合計。Word2Vecのよう訓練。利益：見たことない単語（`whereupon`）既知n-グラムから構成。

**BPE (Byte-Pair Encoding)。** 個別バイト（または文字）のボキャブラリーで開始。コーパス内の全てのアジャセント ペアをカウント。最も頻繁なペアを新しいトークンにマージ。`k` 回数繰り返す。結果：`k + 256` トークンのボキャブラリーどこで頻繁なシーケンス（`ing`、`tion`、`the`）単一トークンと希な単語は親しみある部分に壊れる。全文をトークン化。

## ビルド

### GloVe：共起行列因数分解

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

2つ動く部分名前が値する。重み付け機能 `f(x) = (x/x_max)^alpha` 非常に頻繁なペア（`(the, and)` のような）スケールダウン損失を支配しないように。最終的な埋め込みは `W` (中心) と `W_tilde` (コンテキスト) テーブルの合計。両方を合計は単独で1つだけを使用しておまけに傾向、発表トリック。

### FastText：サブワード認識埋め込み

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

各単語はn-グラム（通常3から6文字）で表現。単語埋め込みはそのn-グラム埋め込みの合計。Skip-gramトレーニングのため、これをWord2Vecが単一ベクトルを使用したところにプラグイン。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

見たことない単語のため、そのn-グラムの一部がまだ既知である限り、ベクトルをまだ取得。`whereupon` は `<wh`、`her`、`ere`、および `<where` を `where` と共有、近いところに着地。

### BPE：学習されたサブワードボキャブラリー

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第1回数最も一般的なアジャセント ペアをマージ。十分な回数後、頻繁なサブストリング（`low`、`est`、`tion`）は単一トークンになり希な単語はきれいに壊れる。

実際のGPT / BERT / T5 トークナイザーは30k-100kマージを学ぶ。結果：任意テキストは既知IDの制限された長さシーケンスにトークン化、OOVなし。

## 使う

実装では、これらのいずれかを自分で訓練することはめったにない。事前訓練されたチェックポイントロード。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

トランスフォーマー時代でBPEスタイルサブワードトークン化のため：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` プレフィックスは単語境界をマーク（GPT-2慣習）。全ての現代的トークナイザーはBPE変種、WordPiece（BERT）、またはSentencePiece（T5、LLaMA）。

### どれを選ぶかそしていつ

| 状況 | 選ぶ |
|-----------|------|
| 事前訓練一般目的単語ベクトル、OOV許容なし必要 | GloVe 300d |
| 事前訓練一般目的単語ベクトル、タイプミス / neologisms / 形態論的リッチ言語処理する必要 | FastText |
| トランスフォーマーに入るもの（訓練または推論） | モデルが出荷したどのトークナイザー。決して交換しない。|
| ゼロからあなたの言語モデルを訓練 | あなたのコーパスの上でBPEまたはSentencePieceトークナイザーを訓練して最初|
| 本番テキスト分類リニアモデル持つ | まだTF-IDF。レッスン02。|

## 出荷

`outputs/skill-embeddings-picker.md` として保存：

```markdown
---
name: tokenizer-picker
description: 新言語モデルまたはテキストパイプラインのトークン化アプローチをピック。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

タスクとデータセット説明が与えられたら、出力：

1. トークン化戦略（単語レベル、BPE、WordPiece、SentencePiece、バイトレベル）。1文の理由。
2. ボキャブラリーサイズ目標（例えば、英語のみLM場合32k、多言語場合64k-100k）。
3. ライブラリ呼び出し正確な訓練コマンド。ライブラリに名前を付け。引数を引用。
4. 1つの再現可能性陥阱。トークナイザーモデルミスマッチは単一の最も一般的な静かな本番バグ；どのペアが一緒に使用される必要があるかを指摘。

事前訓練されたLLMをファインチューニングしている場合は、カスタムトークナイザー訓練推奨を拒否。本番推論を目指しているモデルについて単語レベルトークン化推奨を拒否。非英語 / マルチスクリプトコーパスをバイトフォールバック持つSentencePieceとして必要としてフラグ。
```

## 演習

1. **簡単。** `char_ngrams("playing")` と `char_ngrams("played")` を実行。2つのn-グラムセットのJaccard オーバーラップを計算。`pla`、`lay`、`play` などの実質的な共有ピース見る、FastTextが形態論的変種にわたってなぜ良く転送されるか理由。
2. **中程度。** `learn_bpe` をボキャブラリー成長を追跡するために拡張。マージ数の関数としてtokens-per-corpus-character をプロット。最初の急速圧縮見る、約 ~2-3文字/トークンの近くで漸近。
3. **難しい。** シェークスピア全集で1k-マージBPEを訓練。一般的な単語対希な適切な名詞のトークン化を比較。マージ前後の単語ごと平均トークン測定。何があなたを驚かせたかを書く。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 共起行列 | 単語単語頻度テーブル | `X[i][j]` = 単語 `i` コンテキストで単語 `j` 表示頻度。|
| サブワード | 単語のピース | 文字n-グラム（FastText）または学習されたトークン（BPE/WordPiece/SentencePiece）。|
| BPE | Byte-pair encoding | 最も頻繁なアジャセント ペアのイテレーティブマージ、ボキャブラリーはターゲットサイズに達するまで。|
| OOV | 語彙外 | モデルが決してみたことない単語。Word2Vec/GloVeが失敗。FastTextとBPEはそれを処理。|
| Byte-level BPE | 生バイト上BPE | GPT-2のスキーム。ボキャブラリーは256バイトで開始、OOVは決にない。|

## 参考文献

- [Pennington、Socher、Manning (2014)。GloVe：単語表現のための全域ベクトル](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe論文、7ページ、まだ損失の最良導出。
- [Bojanowski et al. (2017)。サブワード情報で単語ベクトルを豊にする](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich、Haddow、Birch (2016)。希な単語でのニューラル機械翻訳サブワード単位持つ](https://arxiv.org/abs/1508.07909) — BPEを最新NLPに導入した論文。
- [ハギング顔トークナイザーサマリー](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece、SentencePieceが実装でどのように違うか。
