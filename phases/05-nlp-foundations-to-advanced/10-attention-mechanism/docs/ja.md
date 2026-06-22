# アテンション機構 — ブレークスルー

> デコーダは圧縮された要約を目を細めて見るのをやめ、ソース全体を見始める。これ以降のすべては、アテンションにエンジニアリングを足したものだ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 09 (Sequence-to-Sequence Models)
**所要時間:** 約45分

## 問題

レッスン 09 は測定された失敗で締めくくられた。おもちゃのコピータスクで訓練された GRU エンコーダ・デコーダは、長さ 5 での 89% の精度から長さ 80 でのほぼ偶然レベルへと落ちる。理由は構造的なもので、訓練のバグではない。エンコーダが集めた情報のすべてが一つの固定サイズの隠れ状態に収まらねばならず、デコーダはそれ以外を決して見ないからだ。

Bahdanau、Cho、Bengio は 2014 年に 3 行の修正を発表した。デコーダに最終エンコーダ状態だけを与える代わりに、すべてのエンコーダ状態を保持する。各デコーダステップで、エンコーダ状態の加重平均を計算する。重みは「デコーダは今、エンコーダ位置 `i` をどれだけ見る必要があるか」を表す。その加重平均がコンテキストであり、それは各デコーダステップで変化する。

それがアイデアのすべてだ。トランスフォーマーはそれを拡張した。セルフアテンションはそれを単一の系列に適用した。マルチヘッドアテンションはそれを並列に走らせた。だが 2014 年版がすでにボトルネックを打ち破っており、それを手にすれば、トランスフォーマーへの転換は概念的ではなくエンジニアリングだ。

## コンセプト

![Bahdanau アテンション: デコーダがすべてのエンコーダ状態に問い合わせる](../assets/attention.svg)

各デコーダステップ `t` で:

1. 前のデコーダ隠れ状態 `s_{t-1}` を**クエリ**として使う。
2. それをすべてのエンコーダ隠れ状態 `h_1, ..., h_T` に対してスコア付けする。エンコーダ位置ごとに一つのスカラー。
3. スコアをソフトマックスして、合計が 1 になるアテンション重み `α_{t,1}, ..., α_{t,T}` を得る。
4. コンテキストベクトル `c_t = Σ α_{t,i} * h_i`。エンコーダ状態の加重平均。
5. デコーダは `c_t` に前の出力トークンを加えて、次のトークンを生成する。

加重平均が要点だ。デコーダが "Je" を "I" に翻訳する必要があるとき、"Je" 上のエンコーダ状態を高く、他を低く重みづける。"not" が必要なとき、"pas" を高く重みづける。コンテキストベクトルは各ステップで形を変える。

## 形状 (誰もが噛みつかれるもの)

これは、あらゆるアテンション実装が初回に間違う箇所だ。ゆっくり読むこと。

| もの | 形状 | 注記 |
|-------|-------|-------|
| エンコーダ隠れ状態 `H` | `(T_enc, d_h)` | BiLSTM なら `d_h = 2 * d_hidden` |
| デコーダ隠れ状態 `s_{t-1}` | `(d_s,)` | 一つのベクトル |
| アテンションスコア `e_{t,i}` | スカラー | エンコーダ位置ごとに一つ |
| アテンション重み `α_{t,i}` | スカラー | すべての `i` にわたるソフトマックス後 |
| コンテキストベクトル `c_t` | `(d_h,)` | エンコーダ状態と同じ形状 |

**Bahdanau (加算的) スコア。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` は形状 `(d_s,)`、`h_i` は形状 `(d_h,)` を持つ。
- `W_a` は形状 `(d_attn, d_s)`。`U_a` は形状 `(d_attn, d_h)`。
- tanh 内のそれらの和は形状 `(d_attn,)` を持つ。
- `v_α` は形状 `(d_attn,)` を持つ。`v_α` との内積はスカラーへと畳み込まれる。**これが `v_α` のすること**だ。魔法ではない。アテンション次元のベクトルをスカラースコアに変える射影だ。

**Luong (乗算的) スコア。** 3 つのバリアント:

- `dot`: `e_{t,i} = s_t^T * h_i`。`d_s == d_h` が必要。厳しい制約。エンコーダが双方向ならスキップ。
- `general`: `e_{t,i} = s_t^T * W * h_i`、`W` は形状 `(d_s, d_h)`。等次元制約を取り除く。
- `concat`: 本質的に Bahdanau 形式。最初の 2 つの方が安価なのでめったに使われない。

**名前を付ける価値のある Bahdanau / Luong の落とし穴。** Bahdanau は `s_{t-1}` (現在の語を生成する*前*のデコーダ状態) を使う。Luong は `s_t` (*後*の状態) を使う。これらを取り違えると、デバッグが極めて難しい微妙に誤った勾配が生じる。一つの論文を選んでその慣例に従うこと。

## ビルド

### ステップ 1: 加算的 (Bahdanau) アテンション

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

上の表に対して自分の形状を確認すること。`encoder_states` は形状 `(T_enc, d_h)` を持つ。`projected_enc` は形状 `(T_enc, d_attn)` を持つ。`projected_dec` は形状 `(d_attn,)` を持ち、ブロードキャストされる。`combined` は形状 `(T_enc, d_attn)` を持つ。`scores` は形状 `(T_enc,)` を持つ。`weights` は形状 `(T_enc,)` を持つ。`context` は形状 `(d_h,)` を持つ。出荷せよ。

### ステップ 2: Luong dot と general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

それぞれ 3 行。これが Luong の論文が成功した理由だ。ほとんどのタスクで同等の精度、ずっと少ないコード。

### ステップ 3: 数値例の実演

3 つのエンコーダ状態 (おおよそ "cat"、"sat"、"mat") と、最初のものに最もよく揃うデコーダ状態が与えられると、アテンション分布は位置 0 に集中する。デコーダ状態が最後のものに揃うようシフトすると、アテンションは位置 2 へ移る。コンテキストベクトルはそれを追跡する。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

最初の行が勝つ。次にデコーダ状態を 3 番目のエンコーダ状態に近づけて、重みが移るのを見よう。それで終わりだ。アテンションは明示的なアラインメントだ。

### ステップ 4: なぜこれがトランスフォーマーへの橋なのか

上の言葉を Q/K/V に翻訳する:

- **クエリ (Query)** = デコーダ状態 `s_{t-1}`
- **キー (Key)** = エンコーダ状態 (それに対してスコア付けする対象)
- **バリュー (Value)** = エンコーダ状態 (それを重みづけて合計する対象)

古典的アテンションでは、キーとバリューは同じものだ。セルフアテンションはそれらを分離する。系列を自分自身に対して問い合わせることができ、K と V に異なる学習済み射影を使える。マルチヘッドアテンションは異なる学習済み射影でそれを並列に走らせる。トランスフォーマーは段全体を何度も積み重ね、RNN を捨てる。

数学は同じだ。形状は同じだ。Bahdanau アテンションからスケールドドット積アテンションへの教育的な跳躍は、ほとんどが記法だ。

## 使う

PyTorch と TensorFlow はアテンションを直接同梱している。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

それがトランスフォーマーのアテンション層だ。5 位置のクエリバッチ、10 位置のキー/バリューバッチ、各 128 次元、8 ヘッド。`output` は新しいコンテキスト拡張されたクエリだ。`weights` は可視化できる 5x10 のアラインメント行列だ。

### 古典的アテンションが今なお重要なとき

- 教育。シングルヘッド、シングルレイヤー、RNN ベースの版はすべての概念を可視化する。
- トランスフォーマーが収まらないオンデバイス系列タスク。
- 2014〜2017 年のあらゆる論文。Bahdanau の慣例を知らないと誤読する。
- MT における細粒度のアラインメント分析。生のアテンション重みはトランスフォーマーモデル上でも解釈可能性ツールであり、それを読むには何であるかを知る必要がある。

### アテンション重みを説明とみなす罠

アテンション重みは解釈可能に見える。位置全体で合計が 1 になる重みだ。プロットできる。高ければ「ここを見た」だ。査読者はこれを好む。

それらは見た目ほど解釈可能ではない。Jain と Wallace (2019) は、一部のタスクでは、モデルの予測を変えずにアテンション分布を並べ替えたり任意の代替物で置き換えたりできることを示した。アブレーションや反事実チェックなしに、アテンション重みを推論の証拠として報告してはならない。

## 出荷

`outputs/prompt-attention-shapes.md` として保存する。

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 演習

1. **易しい。** エンコーダのパディングトークンがアテンション重みゼロになるよう `softmax` マスキングを実装せよ。可変長系列のバッチでテストせよ。
2. **中級。** Luong の `general` 形式にマルチヘッドアテンションを追加せよ。`d_h` を `n_heads` グループに分割し、ヘッドごとにアテンションを走らせ、連結せよ。シングルヘッドの場合が以前の実装と一致することを検証せよ。
3. **難しい。** レッスン 09 のおもちゃのコピータスクで、Bahdanau アテンションつきの GRU エンコーダ・デコーダを訓練せよ。精度対系列長をプロットせよ。アテンションなしのベースラインと比較せよ。長さが増えるにつれてギャップが広がるのが見えるはずで、アテンションがボトルネックを解消することを裏付ける。

## 重要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| アテンション | ものを見ること | バリュー系列の加重平均。重みはクエリ・キー類似度から計算される。 |
| クエリ、キー、バリュー | QKV | 3 つの射影。Q は問い、K は照合対象、V は返す対象。 |
| 加算的アテンション | Bahdanau | フィードフォワードスコア: `v^T tanh(W q + U k)`。 |
| 乗算的アテンション | Luong dot / general | スコアは `q^T k` または `q^T W k`。より安価で、ほとんどのタスクで同等の精度。 |
| アラインメント行列 | きれいな絵 | `(T_dec, T_enc)` のグリッドとしてのアテンション重み。読むとモデルが何に注目したかが分かる。 |

## 参考文献

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — その論文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 3 つのスコアバリアントとその比較。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 解釈可能性に関する但し書き。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorch での実行可能なウォークスルー。
