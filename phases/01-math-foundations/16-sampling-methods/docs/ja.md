# サンプリング手法

> サンプリングとは、AI が可能性の空間を探索する方法である。

**種別:** 実装
**言語:** Python
**前提条件:** フェーズ1、レッスン06-07（確率論、ベイズの定理）
**所要時間:** 約120分

## 学習目標

- 一様乱数のみを使って、逆CDF法・棄却サンプリング・重点サンプリングをゼロから実装する
- 言語モデルのトークン生成のために、温度サンプリング・top-k・top-p（ニュークリアス）サンプリングを構築する
- 再パラメータ化トリックを説明し、それがVAEにおけるサンプリングを通した逆伝播を可能にする理由を理解する
- Metropolis-Hastings MCMCを実行して、非正規化されたターゲット分布からサンプリングする

## 問題

言語モデルがプロンプトの処理を終え、50,000個のロジットのベクトルを生成する。語彙の各トークンに1つずつ対応している。そこからどうやって1つを選ぶのか？

常に最も確率が高いトークンを選ぶと、すべての応答が同一になる。決定論的で、つまらない。一様ランダムに選ぶと、出力はでたらめになる。答えはこれら両極端の間のどこかにあり、その「どこか」はサンプリングによって制御される。

サンプリングはテキスト生成に限った話ではない。強化学習は軌跡をサンプリングすることで方策勾配を推定する。VAEは学習済み分布からサンプリングし、その確率性を通じて逆伝播することで潜在表現を学習する。拡散モデルはノイズをサンプリングして反復的にノイズ除去することで画像を生成する。モンテカルロ法は閉じた形の解を持たない積分を推定する。MCMCアルゴリズムは列挙が不可能な高次元の事後分布を探索する。

あらゆる生成AIシステムはサンプリングシステムである。サンプリング戦略が出力の品質・多様性・制御可能性を決定する。このレッスンでは、一様乱数から始まり、現代のLLMや生成モデルを支える手法に至るまで、主要なサンプリング手法をすべてゼロから構築する。

## 概念

### なぜサンプリングが重要か

サンプリングはAIと機械学習において4つの基本的な役割を担っている。

**生成。** 言語モデル・拡散モデル・GANはいずれもサンプリングによって出力を生成する。サンプリングアルゴリズムが創造性・一貫性・多様性を直接制御する。温度・top-k・ニュークリアスサンプリングは、エンジニアが日常的に調整するパラメータだ。

**学習。** 確率的勾配降下法はミニバッチをサンプリングする。Dropoutは非活性化するニューロンをサンプリングする。データ拡張はランダムな変換をサンプリングする。重点サンプリングは強化学習（PPO、TRPO）における勾配分散を削減するためにサンプルを再重み付けする。

**推定。** MLにおける多くの量は閉じた形の解を持たない。データ分布上の期待損失・エネルギーベースモデルの分配関数・ベイズ推論のエビデンスがそれに当たる。モンテカルロ推定はこれらすべてをサンプルの平均によって近似する。

**探索。** MCMCアルゴリズムはベイズ推論において事後分布を探索する。進化的戦略はパラメータ摂動をサンプリングする。トンプソンサンプリングはバンディット問題において探索と活用のバランスを取る。

核心となる課題: 直接サンプリングできるのは単純な分布（一様分布・正規分布）のみだ。それ以外のすべてに対しては、単純なサンプルをターゲット分布のサンプルに変換する手法が必要になる。

### 一様乱数サンプリング

すべてのサンプリング手法はここから始まる。一様乱数発生器は [0, 1) の値を生成し、同じ長さの任意の部分区間は等しい確率を持つ。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

n個の離散集合から一様にサンプリングするには、Uを生成してfloor(n * U)を返す。連続範囲 [a, b] からサンプリングするには、a + (b - a) * U を計算する。

重要な洞察: 1つの一様乱数には、任意の分布から1つのサンプルを生成するのに必要なちょうどよい量の乱数性が含まれている。コツは適切な変換を見つけることだ。

### 逆CDF法（逆変換サンプリング）

累積分布関数（CDF）は値を確率に写像する。

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆CDFは確率を値に逆写像する。U ~ Uniform(0, 1) であれば、X = F_inverse(U) はターゲット分布に従う。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布の例:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

これは F_inverse を閉じた形で書き下せる場合に完璧に機能する。正規分布には閉じた形の逆CDFが存在しないため、別の手法（Box-Muller法や数値近似）を使う。

**離散版:** 離散分布については、CDFを累積和として構築し、Uを生成して、累積和がUを超える最初のインデックスを見つける。これがレッスン06の `sample_categorical` の動作原理だ。

### 棄却サンプリング

CDFを反転できないが、ターゲットPDFを定数倍まで評価できる場合、棄却サンプリングが機能する。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

境界Mが小さいほど採択率が高くなる。低次元（1〜3次元）では棄却サンプリングは良好に機能する。高次元では、提案分布の体積の大部分が棄却されるため、採択率が指数的に低下する。これが棄却サンプリングにおける次元の呪いだ。

**例: 切断正規分布からのサンプリング。** 切断された範囲上で一様な提案分布を使う。エンベロープMはその範囲における正規分布PDFの最大値だ。

**例: 半円からのサンプリング。** 境界矩形内で一様に提案し、点が半円内に収まれば採択する。これがモンテカルロ法で円周率を計算する方法だ: 採択率は面積比 pi/4 に等しい。

### 重点サンプリング

ターゲット分布 p(x) からのサンプルが不要な場合もある。p(x) のもとでの期待値を推定したいが、別の分布 q(x) からのサンプルしか持っていない場合がある。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

これは強化学習において重要だ。PPO（近端方策最適化）では、古い方策 pi_old のもとで軌跡を収集するが、新しい方策 pi_new を最適化したい。重点重みは pi_new(a|s) / pi_old(a|s) である。PPOはこれらの重みをクリッピングして、新しい方策が古い方策から大きく乖離するのを防ぐ。

重点サンプリング推定量の分散は、q が p にどれだけ類似しているかに依存する。q が p と大きく異なると、一部のサンプルが巨大な重みを持って推定値を支配する。自己正規化重点サンプリングは、重みの合計で割ることでこの問題を軽減する。

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### モンテカルロ推定

モンテカルロ推定は、ランダムサンプルの平均を取ることで積分を近似する。大数の法則が収束を保証する。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

誤差率は次元に依存しない。これがグリッドベースの積分が不可能な高次元においてモンテカルロ法が主流となっている理由だ。

**円周率の推定:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**期待値の推定:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### マルコフ連鎖モンテカルロ（MCMC）：Metropolis-Hastings

MCMCは、定常分布がターゲット分布 p(x) であるマルコフ連鎖を構築する。十分なステップの後、チェーンからのサンプルは（近似的に）p(x) からのサンプルとなる。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

対称な提案分布（q(x'|x) = q(x|x')）の場合、比は p(x')/p(x) に簡略化される。これが元のMetropolisアルゴリズムだ。

**なぜ機能するか。** 採択ルールは詳細釣り合い条件を保証する: x にいて x' に移動する確率は、x' にいて x に移動する確率に等しい。詳細釣り合いは、p(x) がチェーンの定常分布であることを意味する。

**実践的な考慮事項:**
- バーンイン: チェーンが平衡状態に達する前の初期サンプルを破棄する
- 間引き: 自己相関を減らすためにk番目ごとのサンプルを保持する
- 提案スケール: 小さすぎるとチェーンの移動が遅くなり（採択率高・探索遅）、大きすぎると多くの提案が棄却される（採択率低・行き詰まり）
- 高次元でのガウス提案における最適採択率は約0.234

### ギブスサンプリング

ギブスサンプリングは多変量分布に対するMCMCの特殊ケースだ。すべての次元で一度に移動を提案するのではなく、条件分布から1変数ずつ更新する。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

ギブスサンプリングでは、各条件分布 p(x_i | x_{-i}) からサンプリングできる必要がある。多くのモデルではこれは簡単だ。
- ベイジアンネットワーク: 条件分布はグラフ構造から導出される
- ガウス混合モデル: 条件分布はガウス分布
- イジングモデル: 各スピンの条件分布は近傍にのみ依存する

採択率は常に1（すべての提案が採択される）だ。なぜなら、正確な条件分布からのサンプリングは自動的に詳細釣り合い条件を満たすからだ。

**制限。** 変数間の相関が高い場合、1変数ずつ更新しては分布の対角方向に大きく移動できないため、ギブスサンプリングの混合が遅くなる。

### 温度サンプリング（LLMで使用）

言語モデルは語彙の各トークンに対してロジット z_1, ..., z_V を出力する。ソフトマックスがこれを確率に変換する。温度はソフトマックス前にロジットを再スケールする。

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**なぜ機能するか。** ロジットをT < 1で割ると、ロジット間の差が増幅される。z_1 = 2、z_2 = 1 の場合、T = 0.5 で割ると z_1/T = 4、z_2/T = 2 となり、差が大きくなる。ソフトマックス後、最も高いロジットのトークンがより大きなシェアを獲得する。

**実践での目安:**
- T = 0.0: 貪欲デコーディング、事実確認型Q&Aに最適
- T = 0.3-0.7: やや創造的、コード生成に適切
- T = 0.7-1.0: バランス型、一般的な会話に適切
- T = 1.0-1.5: 創作文章、ブレインストーミング
- T > 1.5: ランダム性増加、実用上ほとんど役に立たない

温度はどのトークンが可能かを変えない。各トークンに割り当てられる確率質量を変える。

### Top-kサンプリング

Top-kサンプリングは、候補セットを最も高い確率を持つk個のトークンに限定し、そのセットを再正規化してサンプリングする。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-kは語彙分布のロングテールに存在する非常に低確率なトークン（タイポ・ナンセンス）を選択することを防ぐ。問題点: kはコンテキストに関わらず固定されている。モデルが確信を持っている場合（1つのトークンが95%の確率）でも、k = 40 は39個の代替を許してしまう。モデルが不確かな場合（確率が1000トークンに分散）は、k = 40 では妥当な選択肢を切り捨ててしまう。

### Top-p（ニュークリアス）サンプリング

Top-pサンプリングは候補セットのサイズを動的に調整する。固定数のトークンを保持するのではなく、累積確率がpを超える最小のトークンセットを保持する。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

モデルが確信を持っている場合、ニュークリアスサンプリングは少数のトークン（2〜3個程度）を保持する。モデルが不確かな場合は多数（200個程度）を保持する。この適応的な振る舞いが、ニュークリアスサンプリングが一般的にtop-kよりも良いテキストを生成する理由だ。

**一般的な組み合わせ:**
- 温度0.7 + top-p 0.9: 汎用的な設定として良好
- 温度0.0（貪欲）: 決定論的タスクに最適
- 温度1.0 + top-k 50: Fan et al.（2018）元論文の設定

Top-kとtop-pは組み合わせることができる。まずtop-kを適用し、残りのセットにtop-pを適用する。

### 再パラメータ化トリック（VAEで使用）

変分オートエンコーダ（VAE）は、入力を潜在空間の分布にエンコードし、その分布からサンプリングして、サンプルをデコードすることで学習する。問題点: サンプリング操作を通じて逆伝播することができない。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

再パラメータ化トリックは乱数性をパラメータから分離する。

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

これは N(mu, sigma^2) が mu + sigma * N(0, 1) と同じ分布を持つために機能する。重要な洞察: 乱数性をパラメータを持たないソース（epsilon）に移し、サンプルをパラメータの微分可能な変換として表現する。

**VAE学習ループにおける処理:**
1. エンコーダが各入力に対して mu と log(sigma^2) を出力する
2. epsilon ~ N(0, 1) をサンプリングする
3. z = mu + sigma * epsilon を計算する
4. z をデコードして入力を再構成する
5. ステップ4, 3, 2, 1を通じて逆伝播する（ステップ3が微分可能であるため可能）

再パラメータ化トリックがなければ、VAEは標準的な逆伝播で学習できない。この1つの洞察がVAEを実用的なものにした。

### Gumbel-Softmax（微分可能なカテゴリカルサンプリング）

再パラメータ化トリックは連続分布（ガウス分布）に対して機能する。離散カテゴリカル分布には別のアプローチが必要だ。Gumbel-Softmaxはカテゴリカルサンプリングへの微分可能な近似を提供する。

**Gumbel-Maxトリック（微分不可能）:**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（微分可能な近似）:**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmaxは離散サンプルの連続緩和を生成する。出力はハードな one-hot ではなく確率ベクトル（ソフト one-hot）だ。勾配はソフトマックスを通じて流れる。学習時の順伝播では「ストレートスルー」推定量を使える: 順伝播にはハードなargmaxを使い、逆伝播にはソフトなGumbel-Softmax勾配を使う。

**応用:**
- VAEにおける離散潜在変数
- ニューラルアーキテクチャ探索（離散演算の選択）
- ハードアテンション機構
- 離散行動を持つ強化学習

### 層別サンプリング

標準的なモンテカルロサンプリングは、偶然にサンプル空間にギャップを残すことがある。層別サンプリングは空間を層に分割して各層からサンプリングすることで、均一なカバレッジを強制する。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

層別サンプリングは常に標準的なモンテカルロと比べて分散が低いかまたは等しい。

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**応用:**
- 数値積分（準モンテカルロ）
- 学習データの分割（各フォールドのクラスバランスの確保）
- 層別化された重点サンプリング（両手法の組み合わせ）
- NeRF（Neural Radiance Fields）はカメラ光線に沿った層別サンプリングを使用する

### 拡散モデルとの関連

拡散モデルはサンプリングプロセスを通じて画像を生成する。順プロセスはT ステップにわたって画像にガウスノイズを加え、純粋なノイズになるまで続ける。逆プロセスはノイズ除去を学習し、元の画像を一歩ずつ復元する。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

このレッスンの手法との関連:
- 各ノイズ除去ステップは再パラメータ化トリックを使用する（ノイズをサンプリングし、決定論的変換を適用）
- ノイズスケジュール {alpha_t} は温度アニーリングの一形態を制御する
- 学習ではモンテカルロ推定を使ってELBO（証拠下界）を近似する
- 拡散モデルにおける祖先サンプリングはマルコフ連鎖（各ステップは現在の状態にのみ依存）

画像生成プロセス全体は反復的なサンプリングだ: ノイズから始め、各ステップで学習済みノイズ除去モデルを条件とした、わずかにノイズの少ないバージョンをサンプリングする。

## 実装

### ステップ1: 一様サンプリングと逆CDF法

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10,000個の指数分布サンプルを生成し、平均が1/lambdaであることを確認する。

### ステップ2: 棄却サンプリング

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

棄却サンプリングを使って切断正規分布からサンプリングする。サンプルのヒストグラムで形状を確認する。

### ステップ3: 重点サンプリング

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

一様提案分布を使って正規分布のもとで E[X^2] を推定する。既知の答え（mu^2 + sigma^2）と比較する。

### ステップ4: モンテカルロ法による円周率の推定

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### ステップ5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

双峰分布（2つのガウス分布の混合）からサンプリングする。チェーンの軌跡を可視化する。

### ステップ6: ギブスサンプリング

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### ステップ7: 温度サンプリング

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

トークンロジットのセットに対して、温度が出力分布をどのように変化させるかを示す。

### ステップ8: Top-kとtop-pサンプリング

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### ステップ9: 再パラメータ化トリック

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

再パラメータ化されたサンプルには勾配が流れるが、直接サンプリングには流れないことを示す。

### ステップ10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

温度を下げると出力が one-hot ベクトルに近づく様子を示す。

すべての可視化を含む完全な実装は `code/sampling.py` にある。

## 活用する

NumPy と SciPy を使った本番バージョン:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大規模なMCMCには専用ライブラリを使う:
- PyMC: NUTS（適応的HMC）を備えた完全なベイズモデリング
- emcee: アンサンブルMCMCサンプラー
- NumPyro/JAX: GPUアクセラレーションされたMCMC

ゼロから構築した。これでライブラリ呼び出しが何をしているか理解できる。

## 演習

1. コーシー分布に対して逆CDFサンプリングを実装する。CDFはF(x) = 0.5 + arctan(x)/piである。10,000個のサンプルを生成し、ヒストグラムを真のPDFと比較する。重い裾（中心から遠い極端な値）に注目する。

2. 棄却サンプリングを使ってUniform(0, 1)提案分布からBeta(2, 5)分布のサンプルを生成する。採択されたサンプルを真のBeta PDFと比較してプロットする。理論的な採択率はいくらか？

3. 1,000・10,000・100,000個のサンプルを使ったモンテカルロ法で0からpiまでのsin(x)の積分を推定する。各レベルでの誤差を比較する。誤差がO(1/sqrt(N))でスケールすることを確認する。

4. Metropolis-Hastingsを実装して、exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) に比例する2次元分布 p(x, y) からサンプリングする。サンプルとチェーンの軌跡をプロットする。提案分布の標準偏差を変えて実験する。

5. 完全なテキスト生成デモを構築する: ロジットを持つ10語の語彙を与え、（a）貪欲、（b）温度=0.7、（c）top-k=3、（d）top-p=0.9 を使って20トークンのシーケンスを生成する。5回実行して出力の多様性を比較する。

## 主要用語

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| サンプリング | 「ランダムな値を引き出すこと」 | 確率分布に従って値を生成すること。すべての生成AIの背後にある仕組み |
| 一様分布 | 「すべて等確率」 | [a, b] 内のすべての値が等しい確率密度 1/(b-a) を持つ。すべてのサンプリング手法の出発点 |
| 逆CDF | 「確率変換」 | F_inverse(U) は一様サンプルを既知のCDFを持つ任意の分布のサンプルに変換する。正確で効率的 |
| 棄却サンプリング | 「提案して採択/棄却」 | 単純な提案分布から生成し、ターゲット/提案比に比例する確率で採択する。正確だがサンプルを無駄にする |
| 重点サンプリング | 「サンプルの再重み付け」 | q(x) からのサンプルを使い、各サンプルをp(x)/q(x)で重み付けすることでp(x)のもとでの期待値を推定する。RLのPPOの核心 |
| モンテカルロ | 「ランダムサンプルの平均化」 | 積分をサンプル平均で近似する。次元に依存せずO(1/sqrt(N))の誤差 |
| MCMC | 「収束するランダムウォーク」 | 定常分布がターゲットであるマルコフ連鎖を構築する。Metropolis-Hastingsが基礎的アルゴリズム |
| Metropolis-Hastings | 「上りは採択、下りも時々採択」 | 移動を提案し、密度比に基づいて採択する。詳細釣り合い条件がターゲット分布への収束を保証 |
| ギブスサンプリング | 「1変数ずつ」 | 他の変数を固定して各変数をその条件分布から更新する。採択率100% |
| 温度 | 「確信度のつまみ」 | ソフトマックス前にロジットをTで割る。T<1は分布を鋭くし（より確信的）、T>1は平坦にする（より多様） |
| Top-kサンプリング | 「上位k個を保持」 | 上位k個の確率を持つトークン以外をゼロにし、再正規化してサンプリングする。候補セットサイズ固定 |
| ニュークリアスサンプリング（top-p） | 「確率が高いものを保持」 | 累積確率がpを超える最小のトークンセットを保持する。適応的な候補セットサイズ |
| 再パラメータ化トリック | 「乱数性を外に移す」 | epsilon ~ N(0,1) を使い z = mu + sigma * epsilon と書く。サンプリングを微分可能にする。VAE学習に必須 |
| Gumbel-Softmax | 「ソフトなカテゴリカルサンプリング」 | Gumbel ノイズと温度付きソフトマックスを使ったカテゴリカルサンプリングへの微分可能な近似 |
| 層別サンプリング | 「強制的なカバレッジ」 | サンプル空間を層に分割し各層からサンプリングする。常にナイーブなモンテカルロより低い分散 |
| バーンイン | 「ウォームアップ期間」 | チェーンが定常分布に達する前の初期MCMCサンプルを破棄する |
| 詳細釣り合い | 「可逆性条件」 | p(x) * T(x->y) = p(y) * T(y->x)。p がマルコフ連鎖の定常分布であるための十分条件 |
| 拡散サンプリング | 「反復的なノイズ除去」 | ノイズから始め、学習済みノイズ除去ステップを適用してデータを生成する。各ステップは条件付きサンプリング操作 |

## 参考文献

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMCの基礎に関する詳細なチュートリアル
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmaxの原著論文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - ニュークリアス（top-p）サンプリングの論文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 再パラメータ化トリックを導入したVAEの論文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPMがサンプリングと画像生成を結びつける
