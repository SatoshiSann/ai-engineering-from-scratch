# 位置的符号化 — 正弦波、RoPE、ALiBi

> アテンションは並列移動不変。「猫は玄関に座った」と「玄関は座った猫に」は位置的信号なしで同じ出力を生成。3つのアルゴリズムはそれを修正 — あらゆるものが「位置」の意味についての異なる賭けを持つ。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 02 （自己注意）, Phase 7 · 03 （マルチ頭アテンション）
**所要時間:** 約45分

## 問題

スケーリングされたドット積アテンションは順序盲。アテンション行列 `softmax(Q K^T / √d) V` はペアワイズ類似度から計算。`X` の行をシャッフル、出力の行を同じ方法でシャッフル。注目の内側に何も位置を気にしない。

言語、コード、オーディオ、ビデオ — 順序が意味を運ぶ何か — それは致命的。

修正はどのように位置を埋め込むかだ。3つの時代の答え:

1. **絶対正弦波** （Vaswani 2017）。位置の `sin/cos` をエンベディング追加。シンプル、学習可能フリー、訓練された長さを超えて外推のに乏しい。
2. **RoPE — 回転位置エンベディング** （Su 2021）。Q と K ベクトルを位置に比例する角度で回転。相対位置をドット積で直接符号化。2026年に支配的。
3. **ALiBi — 線形バイアスを持つ注目** （Press 2022）。埋め込みすべて スキップ; 距離に基づくアテンションスコアに あたり頭線形ペナルティを追加。優れた長さ外推。

2026年まで、本質的にすべてのフロンティアオープンモデルは RoPE を使用: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi。少数の長いコンテキストモデルは ALiBi または最新の変種を使用。絶対正弦波は歴史的。

## コンセプト

![正弦波絶対 vs RoPE 回転 vs ALiBi 距離バイアス](../assets/positional-encoding.svg)

### 絶対正弦波

固定行列 `PE` を形 `(max_len, d_model)` の事前計算:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

その後 `X' = X + PE[:N]` 注目の前。あらゆる次元は異なる周波数での正弦波。モデルは位相パターンから位置を読むことを学ぶ。`max_len` の超越に失敗: 何も モデルに位置2048で何が起こるかが言ってなかった、位置0–2047のみを見たときモデルに。

### RoPE

Q と K ベクトルを回転（エンベディング）。次元のペア `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = デフォルト 10000
```

位置 `pos_k` でキーに同じ回転を応用。ドット積 `q'_m · k'_n` は `(m - n)` のみの関数になる。だ: **アテンション スコアは相対距離にのみ依存**、たとえ回転は絶対位置を キーオフしていた。優美なトリック。

RoPE 拡張: `base` はスケール可能（NTK-aware, YaRN, LongRoPE）再訓練なしで長いコンテキストに外推。Llama 3 は 8K から 128K コンテキストこの方法で拡張。

### ALiBi

埋め込みトリック スキップ。アテンション スコアを直接バイアス:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

ここで `m_h` は頭特有なスロープ（例 `1 / 2^(8·h/H)`）。近いトークンは増幅される; 遠いトークン罰せられ。訓練時間コスト。論文は長さ外推が正弦波を打つと RoPE をその元訓練長で一致を示す。

### 2026年で何を選ぶか

| バリエーション | 外推 | 訓練コスト | 使用者 |
|---------|---------------|---------------|---------|
| 絶対正弦波 | 乏しい | 無料 | 元トランスフォーマー、初期BERT |
| 学習絶対 | なし | 小さい | GPT-2, GPT-3 |
| RoPE | スケーリング良い | 無料 | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 優れた | ファインチューン段階 | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 優れた | 無料 | BLOOM, MPT, Baichuan |

RoPE は勝った これは注目にスロットイン、再帰性を変わることなく、相対位置符号化、`base` ハイパーパラメータが長いコンテキストファインチューンのため清潔なノブを与える。

## ビルド

### ステップ1: 正弦波符号化

`code/main.py` を見よ。4行の計算:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

最初のアテンション層の前に埋め込み行列にこれを追加。

### ステップ2: RoPE Q, K に応用

RoPE は Q と K でインプレイス操作。次元のあたりペア:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

決定的: Q を位置 `m` で、K を位置 `n` で同じ関数を適用。ドット積は`cos((m-n)·θ_i)` 係数を拾う あらゆる座標ペア上。注目は無料で相対位置を学ぶ。

### ステップ3: ALiBi スロープとバイアス

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # softmax の前にアテンション スコア追加
```

`bias[h]` を頭 `h` の `(seq_len, seq_len)` アテンション スコア行列に追加、その後 softmax。

### ステップ4: RoPE の相対距離性質確認

2つのランダムベクトル `a, b` を選択。`(pos_a, pos_b)` で回転。その後 `(pos_a + k, pos_b + k)` で。両ドット積は浮動小数点誤差内で一致する。その性質は RoPE の全ポイント — これは絶対オフセットに不変、相対隙間のみが重要。

## 使用

PyTorch 2.5+ は `torch.nn.functional` で RoPE ユーティリティを出荷。ほとんどの本番コードは `flash_attn` または `xformers` を使用 RoPE はアテンション カーネル内で応用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026年の長いコンテキストトリック:**

- **NTK-aware補間。** `base` を `base * (scale_factor)^(d/(d-2))` に再スケール 4K から 16K+ に拡張するとき。
- **YaRN。** よりスマートな補間は長いコンテキストで注目エントロピー保存。Llama 3.1 128K これを使う。
- **LongRoPE。** マイクロソフトの 2024 方法は進化探索を使用あたり次元スケール係数を選ぶ。Phi-3-Long これを使用。
- **位置補間 + ファインチューン。** ちょうど位置を拡張係数で縮め、1–5B トークンのためにファインチューン。驚くほど効果的。

## 展開

`outputs/skill-positional-encoding-picker.md` を見よ。スキルは符号化戦略を選ぶ新モデル向け ターゲット コンテキスト長、外推ニーズ、訓練予算を与えて。

## 演習

1. **簡単。** 正弦波 `PE` マトリックスをヒートマップとしてプロット `max_len=512, d=128` のため。「次元インデックス増幅するにつれてストライプはより広い」パターンを確認。
2. **中程度。** NTK-aware RoPE スケーリングを実装。小さいLM を長さ256のシーケンスで訓練、その後スケーリングでと なしで 長さ1024でテスト。Perplexityを測定。
3. **難しい。** ALiBi と RoPE を同じアテンションモジュール内で実装。長さ512のコピータスク上で4層トランスフォーマーを訓練。テスト時に2048に外推。劣化を比較。

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|-----------------|-----------------------|
| 位置符号化 | 「注目に順序を伝える」 | エンベディングまたは注目に追加される何か位置符号化。 |
| 正弦波 | 「元のもの」 | 幾何周波数で追加埋め込みに sin/cos; 外推しない。 |
| RoPE | 「回転埋め込み」 | 位置依存角度で Q, K を回転; ドット積相対距離符号化。 |
| ALiBi | 「線形バイアストリック」 | アテンションスコアに `-m·\|i-j\|` 追加; 埋め込み必要, 素晴らしい外推。 |
| base | 「RoPE のノブ」 | RoPE の周波数スケーラ; 推論でコンテキスト拡張するため増加。 |
| NTK-aware | 「RoPE スケーリングトリック」 | `base` を再スケールハイ周波次元がコンテキスト拡張するとき絞られない。 |
| YaRN | 「ファンシーなもの」 | あたり次元補間+外推は注目エントロピー保存。 |
| 外推 | 「訓練された長さを超えて機能」 | 位置スキーム正しい出力を `max_len` の超越に提供できるか？ |

## 参考文献

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 元正弦波。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE ペーパー。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先端 RoPE スケーリング。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta の Llama 2 長いコンテキストペーパー。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Microsoft 方法 Phi-3-Long と共に使用され参照。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — あらゆる RoPE スケーリング スキーム（デフォルト、線形、動的、YaRN、LongRoPE、Llama-3）の本番グレード実装。
