# トランスフォーマー以前のテキスト生成 — N-gram 言語モデル

> ある単語が意外なものであれば、そのモデルは悪い。パープレキシティは意外性を数値にする。スムージングはそれを有限に保つ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 01 (テキスト処理)、Phase 2 · 14 (ナイーブベイズ)
**所要時間:** 約45分

## 問題

トランスフォーマー以前、RNN 以前、単語埋め込み以前、言語モデルは直前の `n-1` 個の単語にどれだけの頻度で続いたかを数えることで次の単語を予測していた。「the cat」→「sat」を47回、「the cat」→「jumped」を12回、「the cat」→「refrigerator」を0回と数える。正規化して確率分布を得る。

それが n-gram 言語モデルである。1980年から2015年まで、あらゆる音声認識器、あらゆるスペルチェッカー、そしてあらゆるフレーズベースの機械翻訳システムを動かしていた。安価なオンデバイス言語モデリングが必要なときには、今なお動いている。

興味深い問題は、未見の n-gram にどう対処するかである。生のカウントベースのモデルは、見たことのないものすべてに確率ゼロを割り当てる。これは破滅的である。なぜなら文は長く、ほとんどすべての長い文には少なくとも1つの未見の系列が含まれるからだ。50年にわたるスムージングの研究がこれを解決した。Kneser-Ney スムージングがその成果であり、現代の深層学習はその経験主義的な伝統を受け継いでいる。

## コンセプト

![N-gram モデル: カウント、スムージング、生成](../assets/ngram.svg)

**N-gram の確率:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。`n` を固定する (典型的にはトライグラムなら3、4-gram なら4)。カウントから計算する:

```text
P(w | context) = count(context, w) / count(context)
```

**ゼロカウント問題。** 訓練データで見られなかった n-gram は確率ゼロになる。Brown コーパスに関する2007年の研究では、4-gram モデルですらホールドアウトの 4-gram の30%が訓練データで未見であった。スムージングなしでは、いかなる実際のテキストでも評価できない。

**スムージングの手法、洗練度の順に:**

1. **Laplace (add-one)。** すべてのカウントに1を加える。シンプルだが、稀なイベントには最悪。
2. **Good-Turing。** 頻度の頻度に基づいて、高頻度イベントから未見のイベントへ確率質量を再配分する。
3. **補間 (Interpolation)。** n-gram、(n-1)-gram などの推定値を調整可能な重みで組み合わせる。
4. **バックオフ (Backoff)。** n-gram のカウントがゼロなら、(n-1)-gram にフォールバックする。Katz バックオフはこれを正規化する。
5. **絶対割引 (Absolute discounting)。** すべてのカウントから固定の割引 `D` を引き、未見のものへ再配分する。
6. **Kneser-Ney。** 絶対割引に加えて、低次モデルに巧妙な選択を加える。生の頻度ではなく*継続確率* (ある単語が何個のコンテキストに現れるか) を使う。

Kneser-Ney の洞察は深い。「San Francisco」はよくあるバイグラムである。ユニグラム「Francisco」はほとんど「San」の後に現れる。ナイーブな絶対割引は「Francisco」に高いユニグラム確率を与える (カウントが高いため)。Kneser-Ney は「Francisco」が1つのコンテキストにしか現れないことに気づき、それに応じて継続確率を下げる。その結果、「Francisco」で終わる新規のバイグラムには適切に低い確率が与えられる。

**評価: パープレキシティ。** ホールドアウトのテストセットにおける単語あたりの平均負の対数尤度の指数である。低いほど良い。パープレキシティ100は、100個の単語の中から一様に選ぶときと同程度にモデルが混乱していることを意味する。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 作ってみよう

### ステップ1: トライグラムのカウント

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

入力はトークン化された文のリスト。出力は n-gram のカウントとコンテキストのカウント。`<s>` と `</s>` は文境界である。

### ステップ2: Laplace スムージング

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

すべてのカウントに1を加える。スムージングはされるが、未見のイベントに質量を過剰に割り当て、稀ではあるが既知のイベントも損なう。

### ステップ3: Kneser-Ney (バイグラム、補間)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

3つの可動部分がある。`continuation_prob` は「この単語はいくつの異なるコンテキストに現れるか」を捉える (Kneser-Ney の革新)。`lambda_prev` は割引によって解放された質量で、バックオフの重み付けに使われる。最終的な確率は、割引された主項に加えて重み付けされた継続項である。

### ステップ4: サンプリングによるテキスト生成

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

確率に比例したサンプリング。シードごとに常に異なる出力を与える。ビームサーチのような出力を得るには、各ステップで argmax を選び (貪欲法)、小さなランダム性のつまみ (温度) を加える。

### ステップ5: パープレキシティ

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

低いほど良い。Brown コーパスでは、よく調整された 4-gram KN モデルはパープレキシティ約140に達する。トランスフォーマー LM は同じテストセットで15〜30に達する。その差は約10倍である。この差こそが、この分野が先へ進んだ理由である。

## 使ってみよう

- **古典的 NLP の教育。** スムージング、MLE、パープレキシティに最も明快に触れられる。
- **KenLM。** 本番向けの n-gram ライブラリ。低レイテンシが重要な音声・MT システムでリスコアラーとして使われる。
- **オンデバイスのオートコンプリート。** キーボードのトライグラムモデル。今なお。
- **ベースライン。** ニューラル LM が良いと宣言する前に、必ず n-gram LM のパープレキシティを計算すること。もしあなたのトランスフォーマーが KN を大差で上回らないなら、何かが間違っている。

## 仕上げよう

`outputs/prompt-lm-baseline.md` として保存する:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 演習

1. **易しい。** 1,000文の Shakespeare コーパスでトライグラム LM を訓練する。20文を生成する。それらは局所的にはもっともらしいが、全体としては支離滅裂である。これが定番のデモである。
2. **普通。** ホールドアウトの Shakespeare 分割に対して、あなたの KN モデルのパープレキシティを実装する。Laplace と比較する。KN がパープレキシティを30〜50%下げるのが見られるはずである。
3. **難しい。** トライグラムのスペル修正器を作る。スペルミスのある単語とそのコンテキストが与えられたとき、修正候補を生成し、LM のもとでのコンテキスト確率でランク付けする。Birkbeck スペリングコーパス (公開) で評価する。

## 重要な用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| N-gram | 単語の系列 | `n` 個の連続するトークンの系列。 |
| スムージング | ゼロの回避 | 未見のイベントに非ゼロの確率を与えるよう確率質量を再配分すること。 |
| パープレキシティ | LM の品質指標 | ホールドアウトデータにおける `exp(-平均対数確率)`。低いほど良い。 |
| バックオフ | より短いコンテキストへのフォールバック | トライグラムのカウントがゼロなら、バイグラムを使う。Katz バックオフがこれを形式化する。 |
| Kneser-Ney | n-gram に最良のスムージング | 絶対割引 + 低次モデルのための継続確率。 |
| 継続確率 | KN 特有 | 生のカウントではなく、`w` が現れるコンテキストの数で重み付けされた `P(w)`。 |

## 参考文献

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM とスムージングの定番の扱い。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — Kneser-Ney を最良の n-gram スムーザーとして確立した論文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 元の KN 論文。
- [KenLM](https://kheafield.com/code/kenlm/) — 高速な本番向け n-gram LM。レイテンシに敏感なアプリケーションで2026年も使われている。
