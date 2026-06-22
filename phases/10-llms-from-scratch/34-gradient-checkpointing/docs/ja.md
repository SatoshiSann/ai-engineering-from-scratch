# グラデイエント・チェックポイント（勾配チェックポイント）と活性化の再計算

> バックプロパゲーションはすべての中間活性化を保持します。70Bパラメータで128Kコンテキストの場合、ランクあたり3TBの活性化があります。チェックポイントはFLOPSとメモリをトレードオフします。再計算の代わりに保存します。問題は、どのセグメントをドロップするかであり、答えは「すべてではない」です。

**タイプ:** ビルド
**言語:** Python（numpy、オプションでtorch）
**前提条件:** Phase 10 Lesson 04（事前学習Mini-GPT）、Phase 10 Lesson 05（スケーリング・分散学習）
**所要時間:** 約70分

## 問題

トランスフォーマーの学習では、各層について、逆伝播で微分されるすべての演算の入力を保存します。アテンション入力、Q/K/V投影、ソフトマックス出力、FFN入力、正規化出力、および残差ストリーム。隠れサイズ`d`、系列長`L`、バッチ`B`の層の場合、これは層あたり約`12 * B * L * d`の浮動小数点数です。

`d=8192, L=8192, B=1`の場合、BF16での層あたり800MB。64層モデルは51GBの活性化です。これはマイクロバッチサイズを掛ける前で、アテンション・ソフトマックスの中間値（ヘッドあたり`L^2`）を追加する前で、テンソル並列の部分的なコピーを考慮する前です。

双方向の請求：BF16重みとオプティマイザの状態で80GBに収まる可能性がありますが、活性化はそれを超えます。グラデイエント・チェックポイント（活性化の再計算とも呼ばれます）は標準的な解決策です。ほとんどの活性化を削除し、逆伝播中に順伝播を再実行してそれらを復元します。コスト：追加のFLOPS。メリット：メモリはチェックポイントセグメントと総層の比率で削減されます。

素朴に行うと、チェックポイントは1ステップあたり約33%の追加の順伝播FLOPSを要します。うまく行うと、Korthikanti et al.の「スマート選択」に従って、メモリで5倍を節約し、FLOP上のオーバーヘッドは5%未満です。FP8行列乗算、FSDPオフロード、エキスパート並列MoEでは、これは本当に重要です。メモリも無駄なコンピュートも両方に余裕がありません。

## 概念

### バックワードが実際に必要とすること

`output = layer(input)`。バックワードは`grad_input`と`grad_params`を求めます。これを計算するには以下が必要です：

- `input`（線形層に対して`grad_params = input.T @ grad_output`を計算する）
- アクティベーション導関数の中間値（ReLU/GELU/ソフトマックスの導関数はアクティベーション値に依存）

順伝播はこれらを自動的にオートグラフグラフに保存します。すべての`tensor.retain_grad()`とその入力を保持する必要のあるすべての演算は参照を保持します。

### 素朴な完全チェックポイント

ネットワークを`N`個のセグメントに分割します。順伝播中に、各セグメントの*入力*のみを保存します。バックワードが中間値を必要とするとき、セグメントの順伝播を再実行して具現化し、微分します。

例：32層トランスフォーマーを32個の1層セグメントに分割します。

- メモリ：32層入力（小）対32 *（層あたりの活性化量）（大）。
- 追加コンピュート：セグメントあたり1回の追加順伝播、つまり全体の約33%の追加順伝播FLOP（バックワードは順伝播の2倍なので、完全なステップは1 + 1 + 2 = 4単位ではなく1 + 2 = 3単位になります）。

これはオリジナルのChen et al. 2016レシピです：メモリとコンピュートのバランスをとるために、`sqrt(L)`層ごとに1つのチェックポイント。L=64の場合、それは8つのチェックポイントです。

### 選択的チェックポイント（Korthikanti 2022）

すべての活性化が同じコストではありません。アテンション・ソフトマックス出力は`B*L*L*heads`で、系列長に対して*二次的に*増加します。FFN隠れ活性化は`B*L*4d`で線形に増加します。長い系列の場合、ソフトマックスが支配的です。

選択的チェックポイントは、保存が安い活性化（線形投影、残差）を保持し、高い活性化（アテンション）のみを再計算します。再計算するFLOPSは最小限ですが、O(L^2)メモリを節約します。

Megatron-Coreは、これを「選択的」活性化再計算として実装します。ほとんどの2024年以降のフロンティア学習実行で使用されています。

### オフロード

再計算の代替：順伝播とバックワードの間にCPU RAMに活性化を送ります。PCIe帯域幅が必要です。アイドル帯域幅が再具現化のコストを超える場合に有益です。混合戦略は一般的です。一部の層をチェックポイント、他の層をオフロードします。

FSDP2は、オフロードを第一級オプションとして提供します。GPUがメモリでボトルネックになっているがCPU-GPU転送に余裕がある場合、オフロードは輝きます。

### 再計算コストモデル

`k`層ごとのナイーブなチェックポイント、`L`層外のFLOPS/ステップあたり：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # セグメント内の層あたり1回の追加順伝播
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

選択的チェックポイントでは、層全体ではなくアテンションカーネルのみを再計算します：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### メモリ節約モデル

層あたりの活性化量：`A`。`L`層の場合、総活性化メモリ：`L * A`。

完全チェックポイント（セグメントサイズ1）：`L * input_volume`のみを保存（標準トランスフォーマーでは約`L * 1/10 A`）。約`9 * L * A * 1/10`を節約します。

`k`層ごとのチェックポイント：`L/k * A`プラス、アクティブセグメント内の`k-1`層分を保存します。

`k = sqrt(L)`では、メモリと再計算コストの両方が`sqrt(L)`でスケールします。均一コスト層の最適なトレードオフです。

### チェックポイントしない場合

- パイプラインステージの最内層はすでに飛行中。とにかく終了する必要があります。
- 最初と最後の層。ステージのコンピュートに支配される場合（トランスフォーマーではまれ）。
- すでにFlashAttentionを使用しているアテンションカーネル。Flashはソフトマックスを高速で再計算するので、追加の層レベルのチェックポイントは上に少し追加します。

### 実装パターン

1. **関数ラッパー:** `torch.utils.checkpoint.checkpoint(fn, input)`でセグメントをラップします。PyTorchは`input`のみを保存し、逆伝播で他のすべてを再計算します。

2. **デコレータベース：** チェックポイント可能として層をラベル付けします。トレーナーは、どのセグメントを設定時にラップするかを決定します。

3. **手動明示的再計算：** バックワードパスを自分で書き、保存された入力で順伝播を複製するカスタム`recompute_forward`を呼び出します。

3つすべてが同じ機能結果を与えます。ラッパーは標準的な用語です。

### TP / PP / FP8との相互作用

- **テンソル並列:** チェックポイント入力は再計算時に収集または散乱する必要があります。通信コストを処理します。
- **パイプライン並列：** 典型的なパターンは、各パイプラインステージの順伝播をチェックポイントして、逆順マイクロバッチが活性化メモリを再利用できるようにすることです。
- **FP8再計算：** 再計算中に更新されるamax履歴は、オリジナルの順伝播と一致する必要があります。そうしないと、FP8スケールが漂流します。ほとんどのフレームワークはスケールをスナップショットします。

## ビルド

### ステップ1：セグメント付きのおもちゃモデル

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### ステップ2：ナイーブなバックワード。すべての活性化が必要

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### ステップ3：チェックポイント・エブリーK・メモリ

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### ステップ4：コストモデル

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### ステップ5：メモリエスティメータ

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### ステップ6：最適なセグメントサイズ

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### ステップ7：選択的チェックポイント決定

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用方法

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorchの標準的なラッパー。関数をラップします。入力のみを保存し、逆伝播で再計算します。
- **Megatron-Core活性化再計算**: `selective`、`full`、`block`モードをサポートします。2024年以降のフロンティア学習の標準。
- **FSDP2オフロード**: `module.to_empty(device="cpu")`、FSDP2の`offload_policy`。CPUに活性化を分割します。
- **DeepSpeed ZeRO-Offload**: オプティマイザ状態と活性化のCPUオフロード、チェックポイントを補完します。

## シップ

このレッスンは`outputs/prompt-activation-recompute-policy.md`を生成します。これはモデル設定（層、隠れ、系列、バッチ）と利用可能なGPUメモリを取り、層ごとの再計算ポリシー（なし/選択的/完全/オフロード）を発行するプロンプトです。

## 演習

1. 正確性を確認します。`model_forward` + `model_backward`（完全な活性化）対`model_forward_checkpointed` + `model_backward_checkpointed`（セグメント）を実行します。パラメータ勾配は機械精度と同じである必要があります。

2. セグメントサイズ`k`を1から`L`までスイープします。FLOPオーバーヘッドとメモリをプロットします。曲線の膝を見つけます。

3. 選択的チェックポイント化を実装します。アテンション・モジュール入力を保存しますが、その中間値は保存しません。seq=8192の32層モデルの場合、完全層チェックポイント化とのFLOPオーバーヘッドを測定します。

4. オフロードを追加します。セグメント入力をシミュレートされた「CPUバッファ」（別のリスト）に保存します。「PCIe帯域幅」をバイト/時間として測定し、オフロードと再計算の間の損益分岐点を見つけます。

5. 実装や実装を使用してオンオフの実装トランスフォーマーをベンチマークします。メモリを測定します（`torch.cuda.max_memory_allocated`経由）とステップ時間。

## キーターム

| 用語 | 人が言うこと | 実際の意味 |
|------|------------|---------|
| グラデイエント・チェックポイント | 「順伝播を再実行してメモリを節約」 | セグメント入力のみを保存します。バックワード中に中間値を再計算して勾配サポート張量を取得します |
| 活性化再計算 | 「チェックポイント化と同じ」 | HPC風味の同じテクニックの名前 |
| セグメントサイズ（k） | 「チェックポイントごとの層数」 | 中間値がドロップされ、一緒に再具現化される層数 |
| 選択的チェックポイント | 「Korthikanti のトリック」 | 保存が高い活性化（アテンション・ソフトマックス）のみを再計算します。保存が安い活性化を保持します |
| 完全チェックポイント | 「素朴なバージョン」 | セグメント内のすべての層の中間値を再計算します |
| ブロックチェックポイント | 「粗粒度」 | トランスフォーマーブロック全体をチェックポイントします。最大の粒度 |
| FLOPオーバーヘッド | 「コンピュート税」 | ステップあたりの追加FLOP = （再計算FLOP）/（順伝播+バックワードFLOP）;33%素朴、5%選択的 |
| 活性化オフロード | 「CPUに送信」 | 順伝播->バックワード間に活性化をCPU RAMに移動します。再計算の代替 |
| sqrt-L rule | 「古典的な最適値」 | 均一コスト層の場合、最適なチェックポイント間隔はsqrt(L)層です |
| アテンション・ソフトマックスボリューム | 「O(L^2)問題」 | L^2 * heads * batchの浮動小数点数。長いコンテキストでアクティベーションメモリを支配します |

## 参考文献

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- グラデイエント・チェックポイントを形式化した元のペーパー
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 選択的活性化再計算と形式的なコスト分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 逆モード再具現化による代替定メモリアプローチ
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- スケール時の活性化オフロード
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 標準API
- [Megatron-Core活性化再計算ドキュメント](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 選択的、完全、およびブロックモード
