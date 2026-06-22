# ゼロからの自己注意

> アテンションは検索テーブルである。あらゆる単語は「誰が私にとって重要か」と問い、答えを学ぶ。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 3 （深層学習コア）, Phase 5 レッスン 10 （Sequence-to-Sequence）
**所要時間:** 約90分

## 学習目標

- NumPyのみを使用してゼロからスケーリングされたドット積自己注意を実装し、クエリ/キー/値投影とsoftmax加重和を含む
- 複数の頭がマルチ頭アテンション層を分割し、並列アテンションを計算し、結果を連結する
- アテンション行列がトークン関係をどのようにキャプチャするかを追跡し、sqrt(d_k)でスケーリングがsoftmax飽和を防ぐ理由を説明
- 因果マスキングを応用して、双方向アテンションを自動回帰（デコーダスタイル）アテンションに変換

## 問題

RNNはシーケンスを1つずつトークン処理。トークン50に到達するまでに、トークン1からの情報はすでに50の圧縮ステップを通じて搾られた。長範囲依存は固定サイズ隠れ状態にしぼられ — LSTPゲーティングが完全に解決しないボトルネック。

2014年 Bahdanau アテンション論文は修正を示した: デコーダにあらゆるエンコーダ位置を見て戻し、現在のステップでどれが重要かを決めるようにさせる。しかしそれはまだRNNにボルトされていた。2017年「Attention Is All You Need」論文は鋭い問題を尋ねた: アテンションが *唯一* のメカニズムであったなら？再帰性なし。コンボリューション。ただアテンション。

自己注意はシーケンス内のあらゆる位置が、単一並列ステップであらゆる他位置に注目できるようにする。これがトランスフォーマーを速く、スケーラブル、支配的にする。

## コンセプト

### データベース検索アナロジー

アテンションをソフトデータベース検索と考えよう:

```
従来データベース:
  クエリ: 「フランスの首都」  -->  完全一致  -->  「パリ」

アテンション:
  クエリ: 「フランスの首都」  -->  すべてのキーとの類似度  -->  すべての値のサブレンド加重
```

あらゆるトークンは3つのベクトルを生成する:
- **クエリ (Q)**: 「何を探しているのか？」
- **キー (K)**: 「何を含んでいるのか？」
- **値 (V)**: 「選択された場合、何の情報を提供するのか？」

クエリとすべてのキーの間のドット積はアテンションスコアを生成。高スコアは「このキーは私のクエリに一致する」。これらのスコアは値を重み付ける。出力は値の加重合計。

### Q, K, V 計算

各トークンエンベディングは3つの学習された重み行列を通じて投影される:

```
入力エンベディング（nトークンのシーケンス、各d次元）:

  X = [x1, x2, x3, ..., xn]       形: (n, d)

3つの重み行列:

  Wq  形: (d, dk)
  Wk  形: (d, dk)
  Wv  形: (d, dv)

投影:

  Q = X @ Wq    形: (n, dk)      各トークンのクエリ
  K = X @ Wk    形: (n, dk)      各トークンのキー
  V = X @ Wv    形: (n, dv)      各トークンの値
```

視覚的に、1つのトークン:

```
             Wq
  x_i ------[*]------> q_i    「何を探しているのか？」
       |
       |     Wk
       +----[*]------> k_i    「何を含んでいるのか？」
       |
       |     Wv
       +----[*]------> v_i    「何を提供するのか？」
```

### アテンション行列

すべてのトークンに対してQ, K, Vを得たら、アテンションスコアは行列を形成:

```
スコア = Q @ K^T    形: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1 はあらゆるキーに注目
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

各行: 1つのトークンの全シーケンスに注目
```

### なぜスケール？

ドット積はdk次元で増大。dk = 64、ドット積は勾配が消滅するsoftmax領域に押出来る数十の範囲にいられる。修正: sqrt(dk)で割る。

```
スケーリングされたスコア = (Q @ K^T) / sqrt(dk)
```

これは、softmaxが有用な勾配を生成する範囲内に値を保つ。

### Softmax スコアを重み付けに変える

Softmaxは生スコアを、各行で確率分布に変える:

```
q1 の生スコア:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
アテンション重み:   [0.52, 0.09, 0.07, 0.14, 0.08]   （合計 ~1.0）
```

あらゆるトークンが、あらゆる他トークンに注目する量を言う重みのセット。

### 値の加重合計

各トークンの最終出力はすべての値ベクトルの加重合計:

```
output_i = sum( attention_weight[i][j] * v_j  すべての j )

トークン 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完全パイプライン

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

1行の公式:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## ビルド

### ステップ1: ゼロからの Softmax

Softmaxは生ロジットを確率に変える。数値安定性のためマックスを減算。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### ステップ2: スケーリングされたドット積アテンション

コア関数。Q, K, V行列を取り、アテンション出力と重み行列を返す。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### ステップ3: 学習投影を持つ自己注意クラス

Wq, Wk, Wv重み行列を持つ完全な自己注意モジュール、Xavier類似スケーリングで初期化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### ステップ4: 文上で実行

文のための偽エンベディングを作成し、アテンション重みを監視。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### ステップ5: ASCIIヒートマップでアテンション可視化

ビジュアル向けアテンション重みを文字にマップ。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 使用

PyTorchの `nn.MultiheadAttention` は正確に私たちが構築したもの、さらに マルチ頭分割と出力投影を実行:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

主な違い: マルチ頭アテンションは複数のアテンション関数を並列実行し、各dk = d_model / n_heads サイズの自身のQ, K, V投影を持つ、その後結果を連結。これはモデルが異なる関係タイプに同時に注目することを許可。

## 展開

このレッスンが生成:
- `outputs/prompt-attention-explainer.md` - データベース検索アナロジーを通じてアテンションを説明するプロンプト

## 演習

1. `scaled_dot_product_attention` を、softmax前に特定位置を負無限大に設定する任意マスク行列を受け入れるように修正（これが因果/デコーダマスキング方法）
2. ゼロからのマルチ頭アテンション実装: Q, K, V を `n_heads` チャンク分割、各で注目実行、連結、最終重み行列 Wo で投影
3. 同じ長さで異なる2つの文を取得し、同じ SelfAttention インスタンスを通じて供給し、それらのアテンションパターン比較。何が変わる？何が同じままですか？

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|----------------|----------------------|
| クエリ (Q) | 「質問ベクトル」 | このトークンが探している情報を表すいくつかの入力の投影 |
| キー (K) | 「ラベルベクトル」 | このトークンが含む情報を表すいくつかの投影、クエリに対して一致 |
| 値 (V) | 「内容ベクトル」 | アテンションスコアに基づいて集約される実際の情報を運ぶいくつかの投影 |
| スケーリングされたドット積アテンション | 「アテンション公式」 | softmax(QK^T / sqrt(dk)) @ V - スケーリング高次元でsoftmax飽和を防ぐ |
| 自己注意 | 「トークンが自分自身と他を見る」 | Q, K, V がすべて同じシーケンスから来るアテンション、あらゆる位置をあらゆる他位置に注目させる |
| アテンション重み | 「どの程度フォーカス」 | スケーリングされたドット積からのsoftmaxで生成される、位置上の確率分布 |
| マルチ頭アテンション | 「並列アテンション」 | 異なる投影を持つ複数のアテンション関数実行、その後結果を連結してのより豊かな表現 |

## 参考文献

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - 元のトランスフォーマー論文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 完全アーキテクチャの最良ビジュアルウォークスルー
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 行ごとのPyTorch実装説明付き
