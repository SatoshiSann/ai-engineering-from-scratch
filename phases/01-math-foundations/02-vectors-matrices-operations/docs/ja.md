# ベクトル、行列、演算

> すべてのニューラルネットワークは、行列の掛け算に少し手を加えたものに過ぎない。

**タイプ:** 実装
**言語:** Python, Julia
**前提条件:** フェーズ1、レッスン01（線形代数の直感的理解）
**所要時間:** 約60分

## 学習目標

- 要素ごとの演算、行列積、転置、行列式、逆行列を持つ Matrix クラスを実装する
- 要素ごとの乗算と行列積の違いを区別し、それぞれの適用場面を説明できる
- 自作の Matrix クラスだけを使って、単一の全結合ニューラルネットワーク層（`relu(W @ x + b)`）を実装する
- ブロードキャストのルールと、ニューラルネットワークフレームワークにおけるバイアス加算の仕組みを説明できる

## 問題

ニューラルネットワークを構築したいとする。コードを読むと、次のような記述がある：

```
output = activation(weights @ input + bias)
```

この `@` は行列積だ。`weights` は行列で、`input` はベクトルだ。これらの演算が何をするのか知らなければ、この1行は魔法のように見える。知っていれば、3つの演算でレイヤーの順伝播全体を表していることがわかる。

モデルが処理する画像はすべてピクセル値の行列だ。すべての単語埋め込みはベクトルだ。すべてのニューラルネットワークのすべての層は行列変換だ。変数を理解せずにコードが書けないのと同様に、行列演算に習熟せずに AI システムを構築することはできない。

このレッスンでは、その習熟度をゼロから積み上げる。

## 概念

### ベクトル：数値の順序付きリスト

ベクトルは方向と大きさを持つ数値のリストだ。AI では、ベクトルはデータ点、特徴量、またはパラメータを表す。

```
v = [3, 4]        -- 2次元ベクトル
w = [1, 0, -2]    -- 3次元ベクトル
```

2次元ベクトル `[3, 4]` は平面上の座標 (3, 4) を指す。その長さ（大きさ）は 5 だ（3-4-5 の直角三角形）。

### 行列：数値のグリッド

行列は2次元のグリッドだ。行と列がある。m x n 行列は m 行 n 列を持つ。

```
A = | 1  2  3 |     -- 2x3 行列（2行、3列）
    | 4  5  6 |
```

ニューラルネットワークでは、重み行列が入力ベクトルを出力ベクトルに変換する。784個の入力と128個の出力を持つ層は、128x784の重み行列を使う。

### 形状が重要な理由

行列積には厳密なルールがある：`(m x n) @ (n x p) = (m x p)`。内側の次元が一致しなければならない。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  重み         入力         出力

内側の次元：784 = 784  -- 有効
```

PyTorch で形状の不一致エラーが出たら、これが原因だ。

### 演算の対応表

| 演算 | 内容 | ニューラルネットワークでの用途 |
|-----------|-------------|-------------------|
| 加算 | 要素ごとに結合 | 出力へのバイアス加算 |
| スカラー乗算 | すべての要素をスケーリング | 学習率 * 勾配 |
| 行列積 | ベクトルの変換 | 層の順伝播 |
| 転置 | 行と列を入れ替え | 誤差逆伝播 |
| 行列式 | 1つの数値による要約 | 可逆性の確認 |
| 逆行列 | 変換の取り消し | 線形方程式の求解 |
| 単位行列 | 何もしない行列 | 初期化、残差接続 |

### 要素ごとの乗算と行列積

この違いは初学者がよくつまずく点だ。

要素ごとの乗算：対応する位置同士を掛け算する。両方の行列は同じ形状でなければならない。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

行列積：行と列のドット積。内側の次元が一致しなければならない。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

演算が異なれば、結果もルールも異なる。

### ブロードキャスト

バイアスベクトルを出力行列に加算するとき、形状が一致しない。ブロードキャストは小さい配列を大きい配列に合わせて引き伸ばす。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

ブロードキャストはベクトルを行方向に引き伸ばす：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

最新のフレームワークはすべてこれを自動で行う。仕組みを理解しておくと、形状がおかしそうなのにコードが動く場合の混乱を防げる。

## 実装

### ステップ1：Vector クラス

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### ステップ2：コア演算を持つ Matrix クラス

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### ステップ3：動作確認

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### ステップ4：ニューラルネットワークとの接続

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

これが単一の全結合層だ：`output = relu(W @ x + b)`。すべてのニューラルネットワークのすべての全結合層は、まさにこれを行っている。

## 実際に使う

NumPy を使えば、上記のすべてをより少ない行数で、桁違いに高速に実行できる。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python の `@` 演算子は `__matmul__` を呼び出す。NumPy はそれを C と Fortran で書かれた最適化された BLAS ルーチンで実装している。同じ数学で、100倍高速だ。

NumPy でのブロードキャスト：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy は1次元のバイアスを自動的に両方の行にブロードキャストする。これがすべてのニューラルネットワークフレームワークにおけるバイアス加算の仕組みだ。

## アウトプット

このレッスンでは、幾何学的な直感を通じて行列演算を教えるためのプロンプトを作成する。`outputs/prompt-matrix-operations.md` を参照。

ここで構築した Matrix クラスは、フェーズ3のレッスン10で構築するミニニューラルネットワークフレームワークの基盤となる。

## 演習

1. **逆行列の検証。** `A @ A.inverse_2x2()` を計算し、単位行列が得られることを確認する。異なる3つの2x2行列で試してみよう。行列式がゼロの場合はどうなるか？

2. **3x3逆行列の実装。** 余因子行列を使って3x3行列の逆行列を計算できるよう Matrix クラスを拡張する。NumPy の `np.linalg.inv` と照合してテストする。

3. **2層ネットワークの構築。** Matrix クラスだけを使って（NumPy なしで）、2層ニューラルネットワークを作成する：入力(3) -> 隠れ層(4) -> 出力(2)。ランダムな重みで初期化し、順伝播を実行して、すべての形状が正しいことを確認する。

## 重要用語

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| ベクトル | 「矢印のようなもの」 | 数値の順序付きリスト。AI では：高次元空間上の点。 |
| 行列 | 「数値の表」 | 線形変換。あるベクトル空間から別の空間へベクトルを写像する。 |
| 行列積 | 「単に数値を掛けるだけ」 | 第1行列のすべての行と第2行列のすべての列のドット積。順序が重要。 |
| 転置 | 「ひっくり返す」 | 行と列を入れ替える。m x n 行列を n x m に変える。誤差逆伝播で重要。 |
| 行列式 | 「行列から得られる何らかの数値」 | 行列が面積（2次元）または体積（3次元）をどれだけスケーリングするかを測る。ゼロは変換がある次元を潰すことを意味する。 |
| 逆行列 | 「行列を元に戻す」 | 変換を逆にする行列。行列式がゼロでない場合にのみ存在する。 |
| 単位行列 | 「つまらない行列」 | 1を掛けることに相当する行列。残差接続（ResNets）で使われる。 |
| ブロードキャスト | 「魔法の形状調整」 | 欠けている次元方向に繰り返すことで、小さい配列を大きい配列に合わせて引き伸ばす。 |
| 要素ごとの演算 | 「普通の掛け算」 | 対応する位置同士を掛け算する。両方の配列は同じ形状（またはブロードキャスト可能）でなければならない。 |

## 参考資料

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - ここで扱うすべての演算の視覚的な直感
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy が従う正確なルール
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 特化の線形代数の簡潔なリファレンス
