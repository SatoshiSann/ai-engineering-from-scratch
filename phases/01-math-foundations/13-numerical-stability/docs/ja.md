# 数値安定性

> 浮動小数点数は漏れのある抽象化だ。学習中に必ず問題を起こす。しかも気づかないうちに。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 1、レッスン 01-04
**所要時間:** 約120分

## 学習目標

- max減算トリックを使って数値的に安定なsoftmaxとlog-sum-expを実装する
- 浮動小数点演算におけるオーバーフロー、アンダーフロー、桁落ちを特定する
- 中心差分法を使って解析的勾配を数値勾配と照合して検証する
- なぜbfloat16が学習においてfloat16より好まれるか、またlossスケーリングが勾配アンダーフローをどう防ぐかを説明する

## 問題

モデルが3時間学習する。するとlossがNaNになる。print文を追加する。ステップ9,000ではロジットは問題ない。ステップ9,001では`inf`になっている。ステップ9,002までにすべての勾配が`nan`となり、学習は終わる。

あるいは、モデルが最後まで学習するが、精度が論文の主張より2%低い。すべてを確認する。アーキテクチャは一致している。ハイパーパラメータも一致している。データも一致している。問題は、論文がfloat32を使い、あなたが正しいスケーリングなしにfloat16を使ったことだ。32ビット分の丸め誤差の蓄積が、静かに精度を食い尽くした。

あるいは、クロスエントロピーlossをゼロから実装する。小さいロジットでは動作する。ロジットが100を超えると`inf`を返す。`exp(100)`がfloat32で表現できる最大値を超えるため、softmaxがオーバーフローした。すべてのMLフレームワークはこれを2行のトリックで処理している。そのトリックの存在を知らなかっただけだ。

数値安定性は理論的な懸念ではない。学習が成功するか静かに失敗するかの違いだ。デバッグするあらゆる深刻なMLバグは、最終的に浮動小数点数の問題に行き着く。

## 概念

### IEEE 754: コンピュータが実数を格納する方法

コンピュータはIEEE 754標準に従い、実数を浮動小数点値として格納する。浮動小数点数は3つの部分からなる：符号ビット、指数部、仮数部（有効数字）。

```
Float32のレイアウト（合計32ビット）:
[1 符号] [8 指数] [23 仮数]

値 = (-1)^符号 * 2^(指数 - 127) * 1.仮数
```

仮数部は精度（有効桁数）を決定する。指数部は範囲（数値の大きさや小ささ）を決定する。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32は約7桁の10進数精度を提供する。つまり1.0000001と1.0000002は区別できるが、1.00000001と1.00000002は区別できない。7桁を超えると、すべて丸め誤差のノイズになる。

float16は約3桁しかない。表現できる最大値は65,504だ。MLではロジット、勾配、活性化値が常にこれを超えるため、不安なほど小さい。

bfloat16はGoogleがfloat16の範囲問題に対処するために作ったものだ。float32と同じ8ビットの指数部（同じ範囲、最大3.4e38）を持つが、仮数部は7ビットのみ（float16より精度が低い）。ニューラルネットワークの学習では精度より範囲の方が重要なため、bfloat16が通常有利だ。

### なぜ 0.1 + 0.2 != 0.3 なのか

0.1という数は2進浮動小数点で正確に表現できない。2進数では循環小数になる：

```
0.1 を2進数で表すと = 0.0001100110011001100110011... (無限に繰り返す)
```

float32はこれを23ビットの仮数部に丸める。格納された値は約0.100000001490116だ。同様に、0.2は約0.200000002980232として格納される。それらの和は0.300000004470348であり、0.3ではない。

```
Python での例:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

これがMLで重要な理由：

1. `if loss < threshold`のようなloss比較が誤った結果を返すことがある
2. 多くの小さな値の累積（何千ものステップにわたる勾配更新）が真の和からずれる
3. floatを`==`で比較するとチェックサムや再現性テストが失敗する

対処法：floatを`==`で比較しない。`abs(a - b) < epsilon`または`math.isclose()`を使う。

### 桁落ち（Catastrophic Cancellation）

ほぼ等しい2つの浮動小数点数を引き算すると、有効桁がキャンセルされ、丸め誤差が先頭桁に繰り上がる。

```
a = 1.0000001    (float32では 1.00000011920929 として格納)
b = 1.0000000    (float32では 1.00000000000000 として格納)

真の差:    0.0000001
計算結果:  0.00000011920929

相対誤差: 19.2%
```

1回の減算で相対誤差が19%になる。MLでは以下の場合にこれが起きる：

- 平均が大きいデータの分散を計算する：E[x]が大きいときの`E[x^2] - E[x]^2`
- ほぼ等しい対数確率を引き算する
- epsilon が小さすぎる有限差分勾配を計算する

対処法：大きくてほぼ等しい数の引き算を避けるよう式を変形する。分散にはWelfordアルゴリズムを使うか、先にデータを中心化する。対数確率には、全体を通じて対数空間で計算する。

### オーバーフローとアンダーフロー

オーバーフローは結果が大きすぎて表現できないときに起きる。アンダーフローは小さすぎるとき（表現できる最小の正の数よりゼロに近いとき）に起きる。

```
float32の境界:
  最大値:           3.4028235e+38
  最小正規化数:     1.175e-38
  最小非正規化数:   1.401e-45
  オーバーフロー:   3.4e38 を超えるものはすべて inf になる
  アンダーフロー:   1.4e-45 未満のものはすべて 0.0 になる
```

MLでオーバーフローの主な原因は`exp()`関数だ：

```
exp(88.7)  = 3.40e+38   (float32にギリギリ収まる)
exp(89.0)  = inf         (オーバーフロー)
exp(-87.3) = 1.18e-38   (アンダーフローのギリギリ上)
exp(-104)  = 0.0         (ゼロにアンダーフロー)
```

`log()`関数は逆方向で問題が起きる：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (問題なし)
log(1e-46) = -inf        (入力がゼロにアンダーフローし、log(0) = -inf)
```

MLでは、`exp()`はsoftmax、sigmoid、確率計算に現れる。`log()`はクロスエントロピー、対数尤度、KLダイバージェンスに現れる。適切なトリックなしに`log(exp(x))`を組み合わせると地雷原になる。

### Log-Sum-Expトリック

`log(sum(exp(x_i)))`を直接計算するのは数値的に危険だ。`x_i`のどれかが大きいと`exp(x_i)`がオーバーフローする。すべての`x_i`が非常に負の値なら、すべての`exp(x_i)`がゼロにアンダーフローし、`log(0)`が`-inf`になる。

トリック：指数をとる前に最大値を引く。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

なぜこれが機能するか：`max(x)`を引いた後、最大の指数は`exp(0) = 1`になる。オーバーフローは起こりえない。和の中に少なくとも1つの項が1になるため、和は少なくとも1であり、`log(1) = 0`となる。`-inf`へのアンダーフローは起こりえない。

証明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (c を足して引く)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (exp(c) を外に出す)
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

`c = max(x)`とおけばオーバーフローが解消される。

このトリックはMLのあらゆる場所に現れる：
- Softmax正規化
- クロスエントロピーlossの計算
- 系列モデルにおける対数確率の和
- 混合ガウスモデル
- 変分推論

### なぜSoftmaxにmax減算トリックが必要か

Softmaxはロジットを確率に変換する：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

トリックなしでは、[100, 101, 102]のロジットはオーバーフローを引き起こす：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

これらはfloat32 (最大 ~3.4e38) をオーバーフローするか？ いや、2.69e43 < 3.4e38? 実際には:
exp(88.7) がすでにfloat32の限界に達している。
exp(100) = float32ではinf になる。
```

トリックを使って max(x) = 102 を引くと：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

確率は同一だ。計算は安全だ。これは最適化ではない。正確さのための必須要件だ。

### NaNとInf：検出と防止

`nan`（Not a Number）と`inf`（無限大）は計算を通じてウイルスのように伝播する。勾配更新に1つの`nan`があれば、重みが`nan`になり、その後のすべての出力が`nan`になる。学習はたった1ステップで終わる。

`inf`が現れる原因：
- 大きな正の数の`exp()`
- ゼロ除算：`1.0 / 0.0`
- 累積での`float32`オーバーフロー

`nan`が現れる原因：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 負の数の`sqrt()`
- 負の数の`log()`
- 既存の`nan`を含む算術演算

検出方法：

```python
import math

math.isnan(x)       # x が nan なら True
math.isinf(x)       # x が +inf か -inf なら True
math.isfinite(x)    # x が nan でも inf でもなければ True
```

防止策：

1. `exp()`への入力をクランプする：`exp(clamp(x, -80, 80))`
2. 分母にepsilonを加える：`x / (y + 1e-8)`
3. `log()`の中にepsilonを加える：`log(x + 1e-8)`
4. 安定な実装を使う（log-sum-exp、安定なsoftmax）
5. 重みの爆発を防ぐ勾配クリッピング
6. デバッグ中はフォワードパスのたびに`nan`/`inf`をチェックする

### 数値勾配チェック

（誤差逆伝播法から得られた）解析的勾配にはバグが含まれることがある。数値的勾配チェックは、有限差分を使って勾配を計算することでこれを検証する。

中心差分の式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

これはO(h^2)の精度を持ち、O(h)しかない前進差分`(f(x+h) - f(x)) / h`よりはるかに優れている。

hの選び方：大きすぎると近似が不正確になる。小さすぎると桁落ちが解を台無しにする。`h = 1e-5`から`1e-7`が一般的だ。

チェック方法：解析的勾配と数値勾配の相対誤差を計算する。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

目安：
- relative_error < 1e-7：完璧、勾配は正しい
- relative_error < 1e-5：許容範囲、おそらく正しい
- relative_error > 1e-3：何かが間違っている
- relative_error > 1：勾配は完全に間違っている

新しいレイヤーやloss関数を実装するときは必ず勾配をチェックすること。PyTorchには`torch.autograd.gradcheck()`が用意されている。

### 混合精度学習

現代のGPUにはfloat16の行列乗算をfloat32より2〜8倍高速に計算する専用ハードウェア（Tensor Cores）がある。混合精度学習はこれを活用する：

```
1. float32のマスターコピーの重みを維持する
2. フォワードパスをfloat16で行う（高速）
3. lossをfloat32で計算する（オーバーフローを防ぐ）
4. バックワードパスをfloat16で行う（高速）
5. 勾配をfloat32にスケーリングする
6. float32のマスター重みを更新する
```

純粋なfloat16学習の問題点：勾配は非常に小さいことが多い（1e-8以下）。float16は約6e-8未満のものをすべてゼロにアンダーフローさせる。すべての勾配更新がゼロになるため、モデルが学習を停止する。

対処法はlossスケーリングだ：

```
1. lossに大きなスケールファクターを掛ける（例：1024）
2. バックワードパスが (loss * 1024) の勾配を計算する
3. すべての勾配が1024倍大きくなる（float16のアンダーフロー以上に押し上げられる）
4. 重みを更新する前に勾配を1024で割る
5. 正味の効果：同じ更新量、アンダーフローなし
```

動的lossスケーリングはスケールファクターを自動的に調整する。大きな値（65536）から始め、勾配が`inf`にオーバーフローしたら半分にする。Nステップオーバーフローなしで経過したら倍にする。

### bfloat16 vs float16: 学習でbfloat16が勝る理由

```
float16:   [1 符号] [5 指数]  [10 仮数]
bfloat16:  [1 符号] [8 指数]  [7 仮数]
```

float16はより高い精度（仮数10ビット対7ビット）を持つが、範囲が限られている（最大約65,504）。bfloat16は精度が低いが、float32と同じ範囲（最大約3.4e38）を持つ。

ニューラルネットワークの学習では：

- 学習スパイク中に活性化値やロジットが定期的に65,504を超える。float16はオーバーフローするが、bfloat16は処理できる。
- lossスケーリングはfloat16では必須だが、bfloat16ではその範囲が勾配の大きさのスペクトルをカバーするため、通常不要だ。
- bfloat16はfloat32を単純に切り捨てたもの：仮数部の下位16ビットを削除する。変換は些細であり、指数部においてロスレスだ。

float16は値が限定されていて精度がより重要な推論に適している。bfloat16は範囲がより重要な学習に適している。これがTPUと最新のNVIDIA GPU（A100、H100）がネイティブなbfloat16サポートを持つ理由だ。

### 勾配クリッピング

爆発する勾配は、多くのレイヤーを通じて勾配が指数関数的に増大するときに発生する（RNN、深いネットワーク、Transformerで一般的）。1つの大きな勾配が1ステップですべての重みを壊す可能性がある。

2種類のクリッピング：

**値によるクリッピング：** 各勾配要素を独立してクランプする。

```
grad = clamp(grad, -max_val, max_val)
```

シンプルだが、勾配ベクトルの方向が変わる可能性がある。

**ノルムによるクリッピング：** 勾配ベクトル全体のノルムが閾値を超えないようスケーリングする。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

勾配の方向を保持する。これが`torch.nn.utils.clip_grad_norm_()`の動作だ。標準的な選択肢だ。

典型的な値：Transformerには`max_norm=1.0`、RLには`max_norm=0.5`、シンプルなネットワークには`max_norm=5.0`。

勾配クリッピングはハックではない。安全機構だ。これなしでは、1つの外れ値バッチが何週間もの学習を台無しにするほど大きな勾配を生成する可能性がある。

### 数値安定器としての正規化レイヤー

バッチ正規化、レイヤー正規化、RMS正規化は通常、学習の収束を助ける正則化手法として紹介される。しかしこれらは数値安定化器でもある。

正規化なしでは、活性化値がレイヤーを通じて指数関数的に増大または減少しうる：

```
Layer 1: [0, 1] の値
Layer 5: [0, 100] の値
Layer 10: [0, 10,000] の値
Layer 50: [0, inf] の値
```

正規化はすべてのレイヤーで活性化値を再中心化・再スケーリングする：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常1e-5）は、すべての活性化値が同一のときにゼロ除算を防ぐ。学習可能なパラメータ`gamma`と`beta`により、ネットワークは必要なスケールを回復できる。

これによりネットワーク全体で値が数値的に安全な範囲に保たれ、フォワードパスのオーバーフローとバックワードパスの勾配爆発の両方を防ぐ。

### よくあるMLの数値バグ

**バグ：数エポック後にlossがNaNになる。**
原因：ロジットが大きくなりすぎ、softmaxがオーバーフローした。または学習率が高すぎて重みが発散した。
対処法：安定なsoftmax（max減算）を使う、学習率を下げる、勾配クリッピングを追加する。

**バグ：lossが log(クラス数) で止まる。**
原因：モデルの出力がほぼ一様な確率になっている。勾配が消失しているか、モデルがまったく学習していないことが多い。
対処法：データラベルが正しいか確認する、loss関数を検証する、dead ReLUを確認する。

**バグ：検証精度が期待より1〜3%低い。**
原因：適切なlossスケーリングなしの混合精度。勾配アンダーフローが静かに小さな更新をゼロにしている。
対処法：動的lossスケーリングを有効にするか、bfloat16に切り替える。

**バグ：一部のレイヤーの勾配ノルムが0.0だ。**
原因：dead ReLUニューロン（すべての入力が負）、またはfloat16アンダーフロー。
対処法：LeakyReLUまたはGELUを使う、勾配スケーリングを使う、重みの初期化を確認する。

**バグ：1つのGPUでは動作するが別のGPUでは異なる結果が出る。**
原因：非決定論的な浮動小数点の累積順序。GPUの並列リダクションは異なるハードウェアで異なる順序で和を計算し、浮動小数点の加算は結合的でない。
対処法：小さな差異（1e-6）を許容するか、`torch.use_deterministic_algorithms(True)`を設定して速度低下を受け入れる。

**バグ：loss計算で`exp()`が`inf`を返す。**
原因：max減算トリックなしで生のロジットを`exp()`に渡している。
対処法：内部でlog-sum-expを実装している`torch.nn.functional.log_softmax()`を使う。

**バグ：float32からfloat16に切り替えたら学習が発散する。**
原因：float16は6e-8未満の勾配の大きさや65,504を超える活性化値を表現できない。
対処法：lossスケーリングを伴う混合精度（AMP）を使うか、代わりにbfloat16を使う。

## 実装

### ステップ1：浮動小数点精度の限界を示す

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### ステップ2：ナイーブなsoftmaxと安定なsoftmaxを実装する

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) は [nan, nan, nan] を返す
```

### ステップ3：安定なlog-sum-expを実装する

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) は inf を返す
```

### ステップ4：安定なクロスエントロピーを実装する

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### ステップ5：勾配チェック

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 応用

### 混合精度シミュレーション

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 勾配クリッピング

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf検出

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

すべてのエッジケースを示した完全な実装については`code/numerical.py`を参照のこと。

## 成果物

このレッスンで作成するもの：
- `code/numerical.py`：安定なsoftmax、log-sum-exp、クロスエントロピー、勾配チェック、混合精度シミュレーション
- `outputs/prompt-numerical-debugger.md`：学習中のNaN/Infや数値問題を診断するためのプロンプト

これらの安定な実装は、Phase 3で学習ループを構築するときと、Phase 4でアテンションメカニズムを実装するときに再び登場する。

## 演習

1. **桁落ち（Catastrophic Cancellation）。** float32でナイーブな式`E[x^2] - E[x]^2`を使って[1000000.0, 1000001.0, 1000002.0]の分散を計算せよ。次にWelfordのオンラインアルゴリズムを使って計算せよ。真の分散（0.6667）に対する誤差を比較せよ。

2. **精度の探索。** Pythonにおいて`1.0 + x == 1.0`となる最小の正のfloat32値`x`を見つけよ。これが機械イプシロンだ。`numpy.finfo(numpy.float32).eps`と一致することを確認せよ。

3. **log-sum-expのエッジケース。** `logsumexp_stable`関数を以下でテストせよ：(a) すべての値が等しい、(b) 1つの値が残りよりはるかに大きい、(c) すべての値が非常に負（-1000）。ナイーブなバージョンが失敗するところで正しい結果が得られることを確認せよ。

4. **ニューラルネットワーク層の勾配チェック。** 単一の線形層`y = Wx + b`とその解析的バックワードパスを実装せよ。3x2の重み行列に対して`numerical_gradient`を使って正確さを検証せよ。

5. **lossスケーリング実験。** float16での学習をシミュレートせよ：[1e-9, 1e-3]の範囲でランダムな勾配を作成し、float16に変換して、ゼロになる割合を測定せよ。次にlossスケーリングを適用し（1024を掛ける）、float16に変換し、スケールを戻して、ゼロの割合を再度測定せよ。

## 主要用語

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| IEEE 754 | 「floatの標準」 | 2進浮動小数点形式、丸め規則、特殊値（inf、nan）を定義する国際標準。すべての現代的なCPUとGPUが実装している。 |
| 機械イプシロン | 「精度の限界」 | ある浮動小数点形式において1.0 + e != 1.0となる最小の値e。float32では約1.19e-7。 |
| 桁落ち | 「引き算による精度損失」 | ほぼ等しい浮動小数点数を引き算するとき、有効桁がキャンセルされ丸め誤差が結果を支配する。 |
| オーバーフロー | 「数が大きすぎる」 | 結果が表現できる最大値を超えてinfになる。exp(89)はfloat32をオーバーフローする。 |
| アンダーフロー | 「数が小さすぎる」 | 結果が表現できる最小の正の数よりゼロに近く、0.0になる。exp(-104)はfloat32をアンダーフローする。 |
| log-sum-expトリック | 「先に最大値を引く」 | exp(max(x))を因数分解することでオーバーフローとアンダーフローを防ぎながらlog(sum(exp(x)))を計算する。softmax、クロスエントロピー、対数確率計算で使われる。 |
| 安定なsoftmax | 「爆発しないsoftmax」 | 指数をとる前にmax(logits)を引く。数値的に同一の結果で、オーバーフローは起こりえない。 |
| 勾配チェック | 「バックプロパゲーションの検証」 | 実装バグを発見するために、誤差逆伝播法による解析的勾配を有限差分による数値勾配と比較する。 |
| 混合精度 | 「フォワードはfloat16、バックワードはfloat32」 | 速度が重要な演算には低精度の浮動小数点数を使い、数値的に敏感な演算には高精度を使う。典型的な高速化は2〜3倍。 |
| lossスケーリング | 「勾配アンダーフローを防ぐ」 | 勾配がfloat16の表現可能な範囲内に収まるよう、バックプロパゲーション前にlossに大きな定数を掛け、重みの更新前に同じ定数で割る。 |
| bfloat16 | 「ブレイン浮動小数点」 | 8ビットの指数部（float32と同じ範囲）と7ビットの仮数部（float16より精度が低い）を持つGoogleの16ビット形式。学習に適している。 |
| 勾配クリッピング | 「勾配ノルムを制限する」 | 勾配ベクトルのノルムが閾値を超えないようスケーリングする。爆発する勾配が重みを壊すのを防ぐ。 |
| NaN | 「Not a Number」 | 未定義の演算（0/0、inf-inf、sqrt(-1)）から生じる特殊なfloat値。以降のすべての演算に伝播する。 |
| Inf | 「無限大」 | オーバーフローやゼロ除算から生じる特殊なfloat値。NaNを生成するために組み合わさることがある（inf - inf、inf * 0）。 |
| 数値勾配 | 「力ずくの微分」 | f(x+h)とf(x-h)を評価して2hで割ることで微分を近似する。遅いが検証には信頼できる。 |

## 参考資料

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 決定版リファレンス、密度は高いが完全
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16学習のlossスケーリングを導入したNVIDIAの論文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorchでの混合精度の実践ガイド
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- GoogleがTPUにこの形式を選んだ理由
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 浮動小数点の和の丸め誤差を減らすアルゴリズム
