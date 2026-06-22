# 確率と分布

> 確率は、AI が不確実性を表現するための言語である。

**タイプ:** 学習
**言語:** Python
**前提条件:** フェーズ1、レッスン01〜04
**所要時間:** 約75分

## 学習目標

- ベルヌーイ分布・カテゴリカル分布・ポアソン分布・一様分布・正規分布の PMF と PDF をゼロから実装する
- 期待値と分散を計算し、中心極限定理を使ってガウス分布があらゆる場所に現れる理由を説明する
- 数値安定化のトリック（最大ロジットを引く）を用いた softmax と log-softmax 関数を構築する
- ロジットからクロスエントロピー損失を計算し、負の対数尤度と結びつける

## 問題設定

分類器が `[0.03, 0.91, 0.06]` を出力する。言語モデルが50,000件の候補から次の単語を選ぶ。拡散モデルが学習済み分布からサンプリングして画像を生成する。これらはすべて確率が実際に動いている例だ。

モデルが行う予測はすべて確率分布である。損失関数はすべて、予測分布が真の分布からどれだけ離れているかを測る。訓練ステップはすべて、ある分布を別の分布に近づけるようパラメータを調整する。確率なしには、ML の論文を1本も読めず、モデルを1つもデバッグできず、訓練損失がなぜ NaN になるのかも理解できない。

## 概念

### 事象、標本空間、確率

標本空間 S はすべての起こりうる結果の集合である。事象は標本空間の部分集合である。確率は事象を 0 から 1 の数値に写す。

```
コイン投げ:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

サイコロ1回の出目:
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

確率の公理は3つあり、これがすべての基礎となる：
1. P(A) >= 0（任意の事象 A について）
2. P(S) = 1（何かは必ず起こる）
3. P(A or B) = P(A) + P(B)（A と B が同時に起こりえない場合）

その他すべて（ベイズの定理、期待値、分布）はこの3つの規則から導かれる。

### 条件付き確率と独立性

P(A|B) は B が起こったときに A が起こる確率である。

```
P(A|B) = P(A and B) / P(B)

例: トランプのカード
  P(キング | 絵札) = P(キング かつ 絵札) / P(絵札)
                   = (4/52) / (12/52)
                   = 4/12 = 1/3
```

一方の結果がもう一方について何も教えてくれないとき、2つの事象は独立である：

```
独立の定義:   P(A|B) = P(A)
同値条件:     P(A and B) = P(A) * P(B)
```

コイン投げは独立である。カードを引いて戻さない場合は独立でない。

### 確率質量関数と確率密度関数

離散確率変数は確率質量関数（PMF）を持つ。各結果には直接読み取れる具体的な確率がある。

```
PMF: P(X = k)

公正なサイコロ:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  すべての確率の合計 = 1
```

連続確率変数は確率密度関数（PDF）を持つ。ある1点における密度は確率ではない。確率は区間上で密度を積分することによって得られる。

```
PDF: f(x)

P(a <= X <= b) = a から b まで f(x) を積分した値

f(x) は 1 より大きくなりうる（密度であり確率ではない）
-inf から +inf まで f(x) dx を積分すると 1 になる
```

この区別は ML では重要である。分類の出力は PMF（離散的な選択）である。VAE の潜在空間は PDF（連続）を使う。

### 代表的な分布

**ベルヌーイ分布:** 1回の試行で2つの結果。二値分類をモデル化する。

```
P(X = 1) = p
P(X = 0) = 1 - p
平均 = p,  分散 = p(1-p)
```

**カテゴリカル分布:** 1回の試行で k 個の結果。多クラス分類（softmax の出力）をモデル化する。

```
P(X = i) = p_i,  ただし sum of p_i = 1
例: P(猫) = 0.7,  P(犬) = 0.2,  P(鳥) = 0.1
```

**一様分布:** すべての結果が等確率。ランダムな初期化に使われる。

```
離散: P(X = k) = 1/n for k in {1, ..., n}
連続: f(x) = 1/(b-a) for x in [a, b]
```

**正規分布（ガウス分布）:** ベルカーブ。平均（mu）と分散（sigma^2）でパラメータ化される。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

標準正規分布: mu = 0, sigma = 1
  データの68%が1 sigma 以内
  95%が2 sigma 以内
  99.7%が3 sigma 以内
```

**ポアソン分布:** 固定区間内に起こる稀な事象の件数。事象の発生率をモデル化する。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
平均 = lambda,  分散 = lambda
```

### 期待値と分散

期待値は結果の加重平均である。

```
離散:   E[X] = sum of x_i * P(X = x_i)
連続:   E[X] = integral of x * f(x) dx
```

分散は平均の周りのばらつきを測る。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
標準偏差 = sqrt(Var(X))
```

ML では、期待値は損失関数（データ分布上の平均損失）として現れる。分散はモデルの安定性を示す。勾配の分散が高いことは、訓練がノイジーであることを意味する。

### 同時分布と周辺分布

同時分布 P(X, Y) は2つの確率変数を一緒に記述する。

同時 PMF の例（X = 天気、Y = 傘）：

| | Y=0（傘なし） | Y=1（傘あり） | 周辺 P(X) |
|---|---|---|---|
| X=0（晴れ） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **周辺 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

周辺分布はもう一方の変数を足し合わせることで得られる：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上の表の行合計と列合計が周辺分布である。

### 正規分布があらゆる場所に現れる理由

中心極限定理：多くの独立な確率変数の和（または平均）は、元の分布によらず正規分布に収束する。

```
サイコロを1個振る: 一様分布（フラット）
2個の平均:         三角形（山型）
30個の平均:        ほぼ完璧なベルカーブ

これはどんな出発分布でも成り立つ。
```

これが理由で：
- 測定誤差は近似的に正規分布に従う（多くの小さな独立した誤差源の和）
- ニューラルネットワークの重みの初期化に正規分布が使われる
- SGD の勾配ノイズは近似的に正規分布に従う（多くのサンプル勾配の和）
- 正規分布は与えられた平均と分散に対して最大エントロピー分布である

### 対数確率

生の確率は数値的な問題を引き起こす。小さな確率を多数掛け合わせるとすぐにアンダーフローして0になる。

```
P(文) = P(単語1) * P(単語2) * ... * P(単語n)
      = 0.01 * 0.003 * 0.02 * ...
      -> 0.0（約30項でアンダーフロー）
```

対数確率はこれを解決する。積算が加算になる。

```
log P(文) = log P(単語1) + log P(単語2) + ... + log P(単語n)
           = -4.6 + -5.8 + -3.9 + ...
           -> 有限の数（アンダーフローなし）
```

規則：
- log(a * b) = log(a) + log(b)
- 対数確率は常に <= 0（0 < P <= 1 なので）
- より負 = より起こりにくい
- クロスエントロピー損失は正解クラスの負の対数確率である

### 確率分布としての Softmax

ニューラルネットワークは生のスコア（ロジット）を出力する。Softmax はそれを有効な確率分布に変換する。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

性質:
  - すべての出力は (0, 1) の範囲にある
  - すべての出力の合計は 1
  - 入力の相対的な順序を保持する
  - exp() はロジット間の差を増幅する
```

softmax のトリック：オーバーフローを防ぐために、指数化する前に最大ロジットを引く。

```
z = [100, 101, 102]
exp(102) = オーバーフロー

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

同じ結果、オーバーフローなし。
```

Log-softmax は数値的安定性のために softmax と対数を組み合わせる。PyTorch はクロスエントロピー損失の内部でこれを使用する。

### サンプリング

サンプリングとは分布からランダムな値を引き出すことである。ML では：
- Dropout はどのニューロンをゼロにするかをランダムにサンプリングする
- データ拡張はランダムな変換をサンプリングする
- 言語モデルは予測分布から次のトークンをサンプリングする
- 拡散モデルはノイズをサンプリングして段階的にノイズ除去する

任意の分布からのサンプリングには、逆変換サンプリング、棄却サンプリング、または再パラメータ化トリック（VAE で使用）などの技法が必要である。

## 実装

### ステップ1：確率の基礎

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### ステップ2：PMF と PDF をゼロから実装

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### ステップ3：期待値と分散

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### ステップ4：分布からのサンプリング

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### ステップ5：Softmax と対数確率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### ステップ6：中心極限定理のデモンストレーション

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### ステップ7：可視化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

すべての可視化を含む完全な実装は `code/probability.py` にある。

## 実用的な使い方

NumPy と SciPy を使えば、上記のすべてが1行で書ける：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

ゼロから実装したことで、ライブラリの関数が何をしているか理解できた。

## 演習

1. 指数分布の逆変換サンプリングを実装する。10,000個の値をサンプリングし、ヒストグラムを真の PDF と比較して検証する。

2. 2つのイカサマサイコロの同時分布表を作成する。周辺分布を計算し、サイコロが独立かどうかを確認する。

3. ロジット `[2.0, 0.5, -1.0, 3.0, 0.1]` を出力する5クラス分類器で、正解クラスがインデックス3のときのクロスエントロピー損失を計算する。次に、PyTorch の `nn.CrossEntropyLoss` で答えを検証する。

4. 対数確率のリストを受け取り、最も確率の高い系列・合計対数確率・等価な生の確率を返す関数を書く。各単語の確率が0.01の50単語の文でテストする。

## 重要用語

| 用語 | よく使われる表現 | 実際の意味 |
|------|----------------|----------------------|
| 標本空間 | 「すべての可能性」 | 実験で起こりうるすべての結果の集合 S |
| PMF | 「確率関数」 | 各離散的な結果に正確な確率を与える関数。合計は1 |
| PDF | 「確率曲線」 | 連続変数の密度関数。区間上で積分すると確率が得られる |
| 条件付き確率 | 「何かが起きたときの確率」 | P(A\|B) = P(A and B) / P(B)。ベイズ的思考とベイズの定理の基盤 |
| 独立性 | 「互いに影響しない」 | P(A and B) = P(A) * P(B)。一方の事象を知っても他方について何もわからない |
| 期待値 | 「平均」 | すべての結果の確率加重和。損失関数は期待値である |
| 分散 | 「ばらつきの大きさ」 | 平均からの偏差の二乗の期待値。分散が大きい = ノイジーで不安定な推定 |
| 正規分布 | 「ベルカーブ」 | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。CLT によりあらゆる場所に現れる |
| 中心極限定理 | 「平均は正規分布になる」 | 多くの独立なサンプルの平均は、元の分布によらず正規分布に収束する |
| 同時分布 | 「2変数を一緒に」 | P(X, Y) は X と Y のすべての結果の組み合わせの確率を記述する |
| 周辺分布 | 「他の変数を足し出す」 | P(X) = sum_y P(X, Y)。同時分布から1変数の分布を復元する |
| 対数確率 | 「確率の対数」 | log P(x)。積を和に変換し、長い系列での数値アンダーフローを防ぐ |
| Softmax | 「スコアを確率に変換」 | softmax(z_i) = exp(z_i) / sum(exp(z_j))。実数値のロジットを有効な確率分布に写す |
| クロスエントロピー | 「損失関数」 | -sum(p_true * log(p_predicted))。2つの分布がどれだけ異なるかを測る。低いほど良い |
| ロジット | 「モデルの生出力」 | softmax 前の正規化されていないスコア。ロジスティック関数に由来する名前 |
| サンプリング | 「ランダムな値を引き出す」 | 確率分布に従って値を生成すること。モデルが出力を生成する方法 |

## さらに学ぶには

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 平均が正規分布になる理由の視覚的証明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - ここで扱った内容以上をカバーする簡潔なリファレンス
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 数値的安定性がなぜ重要か、またどう達成するか
