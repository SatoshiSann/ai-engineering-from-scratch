# JAX入門

> PyTorchはテンソルを変更する。TensorFlowはグラフを構築する。JAXは純粋な関数をコンパイルする。最後のものがディープラーニングについての考え方を変える。

**タイプ:** 構築
**言語:** Python
**前提条件:** フェーズ03 レッスン01-10、基本的なNumPy
**所要時間:** 約90分

## 学習目標

- JAXの関数型API（jax.numpy、jax.grad、jax.jit、jax.vmap）を使って純粋関数ニューラルネットワークコードを書く
- PyTorchのeagerな変更とJAXの関数型コンパイルモデルの主要な設計上の違いを説明する
- jitコンパイルとvmap自動ベクトル化を適用して、ナイーブなPythonに比べて訓練ループを高速化する
- JAXでシンプルなネットワークを訓練し、明示的な状態管理とPyTorchのオブジェクト指向アプローチを対比させる

## 問題

PyTorchでニューラルネットワークを構築する方法を知っている。`nn.Module`を定義し、`.backward()`を呼び、オプティマイザをステップする。機能する。何百万人もの人が使っている。

しかしPyTorchにはそのDNAに焼き付けられた制約がある：Pythonで一度に1つずつ、eagerに演算をトレースする。`tensor + tensor`はすべて別々のカーネル起動だ。すべての訓練ステップが同じPythonコードを再解釈する。これは5400億パラメータのモデルを2048台のTPUで訓練する必要が出るまでは問題なく機能する。そのときオーバーヘッドが致命的になる。

Google DeepMindはGeminiをJAXで訓練している。AnthropicはClaudeをJAXで訓練した。これらは小規模な運用ではない—地球上で最大のニューラルネットワーク訓練実行だ。彼らがJAXを選んだのは、訓練ループをPythonの呼び出しシーケンスではなくコンパイル可能なプログラムとして扱うからだ。

JAXは3つの超能力を持つNumPyだ：自動微分、XLAへのJITコンパイル、自動ベクトル化。1つのサンプルを処理する関数を書く。JAXはバッチを処理し、勾配を計算し、機械語にコンパイルし、複数のデバイスで実行する関数を提供する。元の関数を変えることなく。

## コンセプト

### JAXの哲学

JAXは関数型フレームワークだ。クラスなし、可変な状態なし、`.backward()`メソッドなし。代わりに：

| PyTorch | JAX |
|---------|-----|
| 状態を持つ`nn.Module`クラス | 純粋関数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager実行 | XLAによるJITコンパイル |
| バッチ用の`for x in batch:`手動ループ | `jax.vmap(f)`自動ベクトル化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)`自動並列化 |
| 可変な`model.parameters()` | 不変なpytreeの配列 |

これはスタイルの好みではない。コンパイラの制約だ。JITコンパイルは純粋関数を必要とする—同じ入力は常に同じ出力を生成し、副作用なし。その制約こそが100倍の速度向上を可能にする。

### jax.numpy：親しみやすいサーフェス

JAXはアクセラレータ上でNumPy APIを再実装する：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

同じ関数名。同じブロードキャストルール。同じスライスのセマンティクス。しかし配列はGPU/TPUに存在し、すべての演算がコンパイラによってトレース可能だ。

一つの重要な違い：JAX配列は不変だ。`a[0] = 5`はできない。代わりに：`a = a.at[0].set(5)`。これは1週間は不便に感じるが、やがて理解できる—不変性こそが`grad`、`jit`、`vmap`のような変換を合成可能にするものだ。

### jax.grad：関数型自動微分

PyTorchは勾配をテンソルに付与する（`.grad`）。JAXは勾配を関数に付与する。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`は関数を受け取り、勾配を計算する新しい関数を返す。`.backward()`の呼び出しなし。テンソルに格納された計算グラフなし。勾配は呼び出し、合成、またはJITコンパイルできる別の関数に過ぎない。

これは任意に合成できる：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二次微分。三次微分。ヤコビアン。ヘシアン。すべて`grad`を合成することで。PyTorchもこれができる（`torch.autograd.functional.hessian`）が、後付けだ。JAXではそれが基盤だ。

制約：`grad`は純粋関数にのみ機能する。内部にprint文があるとダメだ（実行ではなくトレース中に動く）。外部状態の変更なし。明示的なキー管理なしの乱数生成なし。

### jit：XLAにコンパイルする

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

最初の呼び出し時、JAXは関数をトレースする—実行せずにどの演算が発生するかを記録する。次にそのトレースをXLA（Accelerated Linear Algebra）、GoogleのTPUとGPU向けコンパイラに渡す。XLAは演算を融合し、冗長なメモリコピーを排除し、最適化された機械語を生成する。

以降の呼び出しはPythonを完全にスキップする。コンパイルされたコードはC++の速度でアクセラレータ上で実行される。

JITが役立つ場合：
- 訓練ステップ（同じ計算が何千回も繰り返される）
- 推論（同じモデル、異なる入力）
- 類似したshapeの入力で複数回呼ばれるあらゆる関数

JITが害になる場合：
- 値に依存するPythonの制御フローを含む関数（トレースされた配列である`x > 0`のif文）
- 一度きりの計算（コンパイルのオーバーヘッドが実行時間を超える）
- デバッグ（トレーシングが実際の実行を隠す）

制御フローの制限は現実だ。`jax.lax.cond`が`if/else`を置き換える。`jax.lax.scan`が`for`ループを置き換える。これらはオプションではない—コンパイルの代価だ。

### vmap：自動ベクトル化

1つのサンプルを処理する関数を書く：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`がバッチを処理するようにリフトする：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`は：`params`はバッチ化しない（共有）、`x`の軸0でバッチ化する、という意味だ。手動の`for`ループなし。リシェイプなし。バッチ次元のスレッドなし。JAXがバッチ次元を把握して計算全体をベクトル化する。

これは構文糖衣ではない。`vmap`はPythonのループより10〜100倍速く動作する融合ベクトル化コードを生成する。そして`jit`と`grad`と合成できる：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

サンプルごとの勾配。1行で。PyTorchではハックなしにこれはほぼ不可能だ。

### pmap：デバイス間のデータ並列化

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`は利用可能なすべてのデバイス（GPU/TPU）に関数を複製してバッチを分割する。関数内部では、`jax.lax.pmean`と`jax.lax.psum`がデバイス間で勾配を同期する。

Googleは`pmap`（とその後継の`shard_map`）を使って数千のTPU v5eチップでGeminiを訓練している。プログラミングモデル：単一デバイスバージョンを書き、`pmap`でラップして完成。

### Pytree：ユニバーサルデータ構造

JAXは「pytree」—リスト、タプル、辞書、配列のネストされた組み合わせ—で動作する。モデルパラメータはpytreeだ：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

すべてのJAX変換—`grad`、`jit`、`vmap`—はpytreeをたどる方法を知っている。`jax.tree.map(f, tree)`はすべての葉に`f`を適用する。これがオプティマイザがすべてのパラメータを一度に更新する方法だ：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()`メソッドなし。パラメータ登録なし。ツリー構造がモデルだ。

### 関数型対オブジェクト指向

PyTorchはオブジェクト内に状態を格納する：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAXは明示的な状態を持つ純粋関数を使う：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

paramsが渡される。何も格納されない。何も変更されない。これですべての関数がテスト可能、合成可能、コンパイル可能になる。また、paramsを自分で管理しなければならない—またはFlaxやEquinoxのようなライブラリを使う。

### JAXエコシステム

JAXはプリミティブを提供する。ライブラリは人間工学を提供する：

| ライブラリ | 役割 | スタイル |
|---------|------|-------|
| **Flax** (Google) | ニューラルネットワーク層 | 明示的な状態を持つ`nn.Module` |
| **Equinox** (Patrick Kidger) | ニューラルネットワーク層 | pytreeベース、Pythonicスタイル |
| **Optax** (DeepMind) | オプティマイザ + 学習率スケジュール | 合成可能な勾配変換 |
| **Orbax** (Google) | チェックポインティング | pytreeの保存/復元 |
| **CLU** (Google) | メトリクス + ロギング | 訓練ループユーティリティ |

Optaxは標準オプティマイザライブラリだ。勾配変換（Adam、SGD、クリッピング）をパラメータ更新から切り離し、合成を容易にする：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX対PyTorchの使い分け

| 要因 | JAX | PyTorch |
|--------|-----|---------|
| TPUサポート | ファーストクラス（GoogleがどちらもGPUを構築） | コミュニティメンテナンス（torch_xla） |
| GPUサポート | 良好（XLA経由のCUDA） | ベストインクラス（ネイティブCUDA） |
| デバッグ | 難しい（トレーシング + コンパイル） | 簡単（Eager、行ごと） |
| エコシステム | 研究志向（Flax、Equinox） | 巨大（HuggingFace、torchvisionなど） |
| 採用 | ニッチ（Google/DeepMind/Anthropic） | メインストリーム（あらゆる場所） |
| 大規模訓練 | 優れている（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| プロトタイピング速度 | 遅い（関数型オーバーヘッド） | 速い（変更して実行） |
| 本番推論 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

正直な答え：JAXを使う特定の理由がない限りPyTorchを使う。その理由とは—TPUアクセス、サンプルごとの勾配の必要性、大規模なマルチデバイス訓練、またはGoogle/DeepMind/Anthropicで働くこと。

### JAXの乱数

JAXはグローバルなランダム状態を持たない。すべてのランダム演算に明示的なPRNGキーが必要だ：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

最初は不便だ。しかしデバイスとコンパイル間での再現性を保証する—PyTorchの`torch.manual_seed`がマルチGPU設定で保証できないプロパティだ。

## 構築する

### ステップ1：セットアップとデータ

JAXとOptaxを使ってMNISTで3層MLPを訓練する。784入力、256と128ニューロンの2つの隠れ層、10出力クラス。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### ステップ2：パラメータの初期化

クラスなし。pytreeを返す関数だけ：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He初期化、手動で。1つのシードから分割された3つのPRNGキー。すべての重みはネストされた辞書内の不変な配列だ。

### ステップ3：フォワードパス

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

純粋関数。paramsが入り、予測が出る。`self`なし、格納される状態なし。`loss_fn`はゼロから交差エントロピーを計算する—softmax、log、負の平均。

### ステップ4：JITコンパイルされた訓練ステップ

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`は損失値と勾配の両方を1パスで返す。`@jax.jit`デコレータは両方の関数をXLAにコンパイルする。最初の呼び出し後、各訓練ステップはPythonに触れることなく実行される。

### ステップ5：訓練ループ

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10エポック。約97%のテスト精度。最初のエポックは遅い（JITコンパイル）。エポック2〜10は速い。

欠けているものに注目：`.zero_grad()`なし、`.backward()`なし、`.step()`なし。更新全体が1つの合成された関数呼び出しだ。勾配が計算され、Adamによって変換され、パラメータに適用される—すべて`train_step`の内部で。

## 活用する

### Flax：Googleの標準

FlaxはJAXで最も一般的なニューラルネットワークライブラリだ。`nn.Module`を追加するが、明示的な状態管理を持つ：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorchと同じ構造だが、`params`はモデルから分離している。`model.init()`はparamsを作成する。`model.apply(params, x)`はフォワードパスを実行する。モデルオブジェクトは状態を持たない。

### Equinox：Pythonicな代替

Equinox（Patrick Kidgerによる）はモデルをpytreeとして表現する：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

モデル自体がpytreeだ。`.apply()`が不要。パラメータはモデルの葉に過ぎない。これはJAXの考え方に近い。

### Optax：合成可能なオプティマイザ

Optaxは勾配変換を更新から切り離す：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

勾配クリッピング、学習率ウォームアップ、重み減衰—すべて変換のチェーンとして合成される。各変換が勾配を見て、修正し、次に渡す。モノリシックなオプティマイザクラスなし。

## Ship It

**インストール：**

```bash
pip install jax jaxlib optax flax
```

GPUサポートの場合：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）の場合：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**パフォーマンスの注意事項：**

- 最初のJIT呼び出しは遅い（コンパイル）。ベンチマーク前にウォームアップする。
- JIT内部でJAX配列のPythonループを避ける。`jax.lax.scan`や`jax.lax.fori_loop`を使う。
- `jax.debug.print()`はJIT内部で機能する。通常の`print()`はしない。
- `jax.profiler`やTensorBoardでプロファイリングする。XLAコンパイルはボトルネックを隠すことがある。
- JAXはデフォルトでGPUメモリの75%を事前割り当てする。無効にするには`XLA_PYTHON_CLIENT_PREALLOCATE=false`を設定する。

**チェックポインティング：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**このレッスンが生成するもの：**
- `outputs/prompt-jax-optimizer.md` — 適切なJAXオプティマイザ設定を選択するプロンプト
- `outputs/skill-jax-patterns.md` — JAXの関数型パターンをカバーするスキル

## 演習

1. MLPにドロップアウトを追加する。JAXでは、ドロップアウトにPRNGキーが必要だ—フォワードパスを通じてキーをスレッドし、各ドロップアウト層のために分割する。ドロップアウトありなしでテスト精度を比較する。

2. `jax.vmap`を使って32枚のMNIST画像のバッチのサンプルごとの勾配を計算する。各サンプルの勾配ノルムを計算する。どのサンプルが最大の勾配を持ち、なぜか？

3. 前のforward関数を任意の数の層で機能する汎用の`mlp_forward(params, x)`に置き換える。`jax.tree.leaves`を使って深さを自動的に決定する。

4. `@jax.jit`ありとなしで訓練ステップをベンチマークする。それぞれ100ステップを計測する。お使いのハードウェアでの速度向上はどのくらいか？最初の呼び出しでのコンパイルオーバーヘッドはどのくらいか？

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`を合成して勾配クリッピングを実装する。クリッピングありとなしで訓練する。訓練を通じて勾配ノルムをプロットして効果を確認する。

## 用語集

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| XLA | 「JAXを速くするもの」 | Accelerated Linear Algebra—計算グラフから演算を融合して最適化されたGPU/TPUカーネルを生成するコンパイラ |
| JIT | 「ジャストインタイムコンパイル」 | JAXは最初の呼び出し時に関数をトレースし、XLAにコンパイルし、以降の呼び出しでコンパイル済みバージョンを実行する |
| 純粋関数 | 「副作用なし」 | 出力が入力のみに依存する関数—グローバル状態なし、変更なし、明示的なキーなしのランダム性なし |
| vmap | 「自動バッチ化」 | 1つのサンプルを処理する関数を書き直さずにバッチを処理するものに変換する |
| pmap | 「自動並列化」 | 複数のデバイスに関数を複製して入力バッチを分割する |
| Pytree | 「配列のネストされた辞書」 | JAXがたどって変換できるリスト、タプル、辞書、配列の任意のネストされた構造 |
| トレーシング | 「計算の記録」 | JAXが実際の結果を計算せずに抽象値で関数を実行して計算グラフを構築する |
| 関数型自動微分 | 「関数のgrad」 | テンソルに勾配ストレージを付与するのではなく関数を変換することで導関数を計算する |
| Optax | 「JAXのオプティマイザライブラリ」 | Adam、SGD、クリッピング、スケジューリングの合成可能な勾配変換ライブラリ。チェーンで繋がる |
| Flax | 「JAXのnn.Module」 | 状態を明示的に保ちながら層の抽象化を追加するGoogleのJAX用ニューラルネットワークライブラリ |

## 参考文献

- JAXドキュメント：https://jax.readthedocs.io/ — grad、jit、vmapの優れたチュートリアルを含む公式ドキュメント
- 「JAX: composable transformations of Python+NumPy programs」（Bradburyら、2018年）—設計哲学を説明する元の論文
- Flaxドキュメント：https://flax.readthedocs.io/ — GoogleのJAX用ニューラルネットワークライブラリ
- Patrick Kidger、「Equinox: neural networks in JAX via callable PyTrees and filtered transformations」（2021年）—FlaxのPythonicな代替
- DeepMind、「Optax: composable gradient transformation and optimisation」—標準オプティマイザライブラリ
- 「You Don't Know JAX」（Colin Raffel、2020年）—T5の著者の一人による実用的なJAXの落とし穴とパターンガイド
