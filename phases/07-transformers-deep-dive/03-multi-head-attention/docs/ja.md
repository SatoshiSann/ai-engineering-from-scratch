# マルチ頭アテンション

> 1つのアテンション頭は1つの関係を学ぶ。8つの頭は8つを学ぶ。頭は無料。さらに多くを取る。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 02 （ゼロからの自己注意）
**所要時間:** 約75分

## 問題

単一の自己注意頭は1つのアテンション行列を計算。その行列は1種の関係をキャプチャ — 通常、訓練信号で何かの損失を最小化するもの。データが主語-動詞一致、共参照、長距離談話、構文チャンキングをすべてもつれている場合、単一の頭はそれを単一のソフトマックス分布に混ぜ込み、半分の信号を失う。

2017年Vaswaniペーパーからの修正: 複数のアテンション関数を並列実行、各自身のQ, K, V投影を持ち、出力を連結。各頭は、次元 `d_model / n_heads` の小さい部分空間で動作。全パラメータは同じままだ。表現力は上昇する。

マルチ頭アテンションは、2026年中のあらゆるトランスフォーマーが出荷するデフォルト。唯一の論拠は *どの程度多く* の頭およびキーと値が投影を共有するか（Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention）。

## コンセプト

![マルチ頭アテンション分割、注目、連結](../assets/multi-head-attention.svg)

**分割。** 形 `(N, d_model)` の `X` を取る。各Q, K, V を投影して形 `(N, d_model)`。`(N, n_heads, d_head)` に改形、ここで `d_head = d_model / n_heads`。転置して `(n_heads, N, d_head)`。

**並列で注目。** あらゆる頭の内側でスケーリングされたドット積アテンション実行。あらゆる頭は `(N, d_head)` を生成。頭はエンベディングの異なる部分空間で動作し、アテンション計算自体の間に話さない。

**連結し投影。** ヘッドを `(N, d_model)` に戻し、学習された出力行列 `W_o` で形 `(d_model, d_model)` を乗じる。`W_o` は頭が混合する場所。

**なぜ機能するか。** あらゆる頭は表現予算について他とライバルすることなく専門化できる。2019–2024からの調査研究は異なる頭の役割を示す: 位置的頭、前トークンに注目する頭、コピー頭、命名エンティティ頭、帰納頭（これは文脈内学習の根底）。

**2026バリエーションの系統:**

| バリエーション | Q 頭 | K/V 頭 | 使用者 |
|---------|---------|-----------|---------|
| マルチ頭 (MHA) | N | N | GPT-2, BERT, T5 |
| マルチクエリ (MQA) | N | 1 | PaLM, Falcon |
| グループクエリ (GQA) | N | G （例 N/8） | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| マルチ頭潜在 (MLA) | N | 低ランクに圧縮 | DeepSeek-V2, V3 |

GQAは最新デフォルト `N/G` の係数でKV-キャッシュメモリを切り、ほぼ完全な品質を保つ。MLAはさらに行き、K/Vを潜在空間に圧縮し、計算時にに投影し戻す — FLOPsを費やす、多くより多くのメモリを保存。

## ビルド

### ステップ1: すでに持つ単一頭アテンションから頭を分割

レッスン02から `SelfAttention` を取得し、分割/結合ペアで包む。`code/main.py` を見よnumpy実装のため; ロジック:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

1つの改形と1つの転置。ループなし。これは正確に PyTorch が `nn.MultiheadAttention` の下で。

### ステップ2: 頭あたりスケーリングドット積アテンション実行

各頭はQ, K, Vのその自身のスライスを得る。アテンションはバッチmatmulになる:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

実ハードウェア上 `Qh @ Kh.transpose(...)` は1つの `bmm`。GPUは単一バッチmatmulを見る形 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。頭を追加することは無料。

### ステップ3: グループクエリアテンション変種

キーと値の投影のみ変わる。Qは `n_heads` グループを得る; KとVは `n_kv_heads < n_heads` グループを得て、一致するため繰返さる:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推論でこれはメモリを保存する。なぜなら `n_kv_heads` コピーのみがKV-キャッシュに生きる、`n_heads` ではなく。Llama 3 70B は 64 クエリ頭を使う 8 KV 頭 — 8倍キャッシュ縮み。

### ステップ4: あらゆる頭が学んだことを調査

4つの頭を持つ短い文上で MHA 実行。あらゆる頭のため、`(N, N)` アテンション行列を印刷。ランダム初期化でも異なる頭は異なる構造を選ぶことを見るだろう — これは部分的に信号、部分的に部分空間の回転対称性。

## 使用

PyTorch では、ワンラインバージョン:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ から GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention は Flash Attention を CUDA で自動ディスパッチ。
# GQA のため、形 (B, n_heads, N, d_head) の Q および形 (B, n_kv_heads, N, d_head) の K,V を渡す。PyTorch は繰返を処理。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**何個の頭？** 2026年本番モデルからのいくつかのオルニール:

| モデル サイズ | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小 (~125M) | 768 | 12 | 64 |
| ベース (~350M) | 1024 | 16 | 64 |
| 大 (~1B) | 2048 | 16 | 128 |
| フロンティア (~70B) | 8192 | 64 | 128 |

`d_head` はほぼ常に 64 または 128 に上陸。これは1つの頭がどの程度「見ることができるか」の単位。32の下に下げ、頭はスケーリング係数 `sqrt(d_head)` と戦い始める; 256を越えて行き、「多くの小さい専門家」利点を失う。

## 展開

`outputs/skill-mha-configurator.md` を見よ。スキルは頭カウント、kv-頭カウント、新トランスフォーマー向け投影戦略を推奨、パラメータ予算、シーケンス長、展開ターゲットを与えて。

## 演習

1. **簡単。** `code/main.py` から MHA を取得し、`n_heads` を 1 から 16 に変更、`d_model=64` 固定。小さい単層モデルの合成コピータスク上の損失をプロット。より多くの頭が助ける、プラトーまたは傷つける？
2. **中程度。** MQA を実装（1つのKV頭がすべてのクエリ頭で共有）。パラメータカウントがどれほど低下するか vs 完全MHA を測定。N=2048 で推論向け KV-キャッシュサイズがどれほどしぼむかを計算。
3. **難しい。** マルチ頭潜在アテンションの小さいバージョンを実装: K,V を秩 r 潜在に圧縮、潜在を KV キャッシュに格納、注目時に圧縮解除。キャッシュメモリが完全 MHA の 1/8 以下で品質が妥当化の 1 ビット内にとどまる r ?

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|-----------------|-----------------------|
| 頭 | 「単一アテンション回路」 | `d_head = d_model / n_heads` 次元の1つのQ/K/V投影、自身のアテンション行列 |
| d_head | 「頭次元」 | あたり頭隠れた幅; 本番では ほぼ常に 64 または 128。 |
| 分割 / 結合 | 「改形トリック」 | `(N, d_model) ↔ (n_heads, N, d_head)` 改形+転置注目の周り。 |
| W_o | 「出力投影」 | `(d_model, d_model)` 行列は頭を連結した後に応用; 頭が混合する場所。 |
| MQA | 「1つのKV頭」 | マルチクエリアテンション: 単一共有K/V投影。最小KV-キャッシュ、いくつかの品質損失。 |
| GQA | 「Llama 2 以来デフォルト」 | グループクエリアテンション `n_kv_heads < n_heads` と; 一致するため繰返。 |
| MLA | 「DeepSeekのトリック」 | マルチ頭潜在アテンション: K,V は低秩潜在に圧縮、注目時に非圧縮。 |
| 帰納頭 | 「文脈内学習の後ろの回路」 | 前の出現検出し何が続いたかを複製する頭のペア。 |

## 参考文献

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 元のマルチ頭仕様。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA ペーパー。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 訓練後 MHA を GQA に変換する方法。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA キャッシュメモリで MHA/GQA を打つ理由。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 機械的な 頭が実際に何をするか。
