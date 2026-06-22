# KVキャッシュ、Flash Attention & 推論最適化

> 学習は並列でFLOP境界。推論は順序付けで記憶境界。異なるボトルネック、異なるトリック。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 02 (自己注意), Phase 7 · 05 (完全なトランスフォーマー), Phase 7 · 07 (GPT)
**所要時間:** 約75分

## 問題

素朴な自己回帰デコーダは`N`トークンを生成するのに`O(N²)`の作業を行う：各ステップで完全なプレフィックスの注意を再計算。4K トークン応答の場合、16Mの注意操作、ほとんどが冗長。プレフィックストークンのすべての隠れ状態は計算後決定的です — あなたは新しいトークンのクエリをすべての前の鍵と値に対して実行するだけで済みます。

その上、注意自体は多くのデータを移動させます。標準注意はN×N スコア行列、N×d ソフトマックス出力、N×d最終出力を具体化します — HBMへの多くの読み書き。N≥2Kの場合、注意はFLOP境界前に記憶境界になります。古典的な注意カーネルはモダンGPUを4-10倍低く利用。

Dao et al.からの2つの最適化がフロンティア推論を「遅い」から「高速」に押し上げ：

1. **KVキャッシュ。** すべてのプレフィックストークンのK と V ベクトルを保管。各新しいトークンの注意は、キャッシュされたキーに対する1つのクエリです。推論は`O(N²)`から`O(N)`に削減/生成ステップ。
2. **Flash Attention。** タイル注意計算により、完全なN×N行列は決してHBMを打つことはありません。ソフトマックス + matmul のすべてはSRAMで発生。A100で2-4倍の壁時計スピードアップ；H100でFP8で5-10倍。

2026年までに両方はユニバーサル。すべての本番推論スタック(vLLM、TensorRT-LLM、SGLang、llama.cpp)はそれらを想定。すべてのフロンティアモデルはFlash Attentionが有効な状態で出荷。

## コンセプト

![KVキャッシュ成長とFlash Attentionタイリング](../assets/kv-cache-flash-attn.svg)

### KVキャッシュ数学

デコーダレイヤーあたり、トークンあたり、ヘッドあたり：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K と V
```

32層、32ヘッド、d_head=128、fp16の7Bモデルの場合：

```
トークンあたりレイヤーあたり = 2 * 128 * 2 = 512バイト
トークンあたり(32層) = 16 KB
32Kコンテキストあたり = 512 MB
```

Llama 3 70B(80層、d_head=128、8 KV ヘッドのGQA)の場合：

```
トークンあたりレイヤーあたり = 2 * 8 * 128 * 2 = 4096バイト (4 KB)
32Kコンテキストあたり = 10.4 GB
```

その10 GBはなぜLlama 3 70Bが128Kコンテキストで40 GB A100のほとんどがバッチサイズ1でのKVキャッシュに必要です。

**GQAはKVキャッシュの勝利。** 64ヘッドのMHAは32 GBになります。MLA さえさらに圧縮。

### Flash Attention — タイリングトリック

標準注意：

```
S = Q @ K^T          (HBM読み取り、N×N、HBM書き込み)
P = softmax(S)       (HBM読み取り、HBM書き込み)
O = P @ V            (HBM読み取り、HBM書き込み)
```

3つのHBMラウンドトリップ。H100では、HBM帯域幅は3 TB/s；SRAMは30 TB/s。すべてのHBMトリップはチップ上に保つのと比較して10倍の低下。

Flash Attention：

```
Q(タイルサイズ~128 × 128)の各ブロック：
    Q_tileをSRAMにロード
    K、Vの各ブロック：
        K_tile、V_tileをSRAMにロード
        S_tile = Q_tile @ K_tile^T を計算(SRAM)
        ランニングソフトマックス集約             (SRAM)
        O_tileに累積                  (SRAM)
    O_tileをHBMに書き込み
```

タイルあたり1つのHBMトリップ。総メモリフットプリント`O(N²)`から`O(N)`に低下。逆パスはいくつかの値を前向きパスから再計算 — 別のメモリの勝利。

**数値トリック。** ランニングソフトマックスは最終正規化が正確なように、タイル全体で`(max、合計)`を維持。近似ではなく — Flash Attentionは標準注意に同一の出力を計算(fp16非結合性を除いて)。

**バージョン進化：**

| バージョン | 年 | 主な変更 | 参照ハードウェアのスピードアップ |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | タイルSRAMカーネル | A100で2倍 |
| Flash 2 | 2023 | より良い並列性、因果優先順序 | A100で3倍 |
| Flash 3 | 2024 | Hopperの非同期性、FP8 | H100で1.5-2倍(~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5段パイプライン、ソフトウェア exp2 | 推論優先(初期の前向きのみ) |

Flash 4は起動時に前向きパスのみです。学習はまだFlash 3を使用。GQAおよび可変長サポートFlash 4は保留中(2026年中旬)。

### 推論的デコーディング — もう1つの遅延の勝利

安価なモデルがNトークンを提案。大きいモデルがすべてのNを並列で検証。検証がkトークンを受け入れた場合、あなたはk生成のために1つの大きいモデル前向きパスを支払った。典的なk=3-5コードとプローズ。

2026年デフォルト：
- **EAGLE 2 / Medusa。** 検証者の隠れ状態を共有する統合ドラフト頭。品質損失なしで2-3倍スピードアップ。
- **推論的デコーディングドラフトモデル。** 消費者ハードウェアで2-4倍スピードアップ。
- **先読みデコーディング。** Jacobi反復；ドラフトモデル不要。ニッチですが無料。

### 継続的バッチ処理

古典的なバッチ推論：最も遅いシーケンスが終わるまで待つ、新しいバッチを開始。短い応答が早く終わると、短応答が早く終わるとGPUを浪費。

継続的バッチ処理(最初にOrcaで出荷、今vLLM、TensorRT-LLM、SGLangで)：古いシーケンスが終わったときに新しいシーケンスをバッチにスワップ、バッチをドレインせず。典的なチャットワークロードで5-10倍スループット利得。

### PagedAttention — 仮想メモリとしてのKVキャッシュ

vLLMの見出し機能。KVキャッシュは16トークンブロックで割り当てられます；ページテーブルは論理位置を物理ブロックにマップ。並列サンプル(ビーム検索、並列サンプリング)全体でKVを共有、プロンプトキャッシングのホットスワッププレフィックス、メモリを断片化解除させます。素朴な連続割り当てよりも4倍スループット改善。

## ビルド

`code/main.py`を見てください。実装：

1. 素朴な`O(N²)`増分デコーダ。
2. `O(N)` KVキャッシュデコーダ。
3. Flash Attentionのランニングマックスアルゴリズムをシミュレートするタイルソフトマックス。

### ステップ1：KVキャッシュ

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

シンプル：層、ヘッド単位でレイヤー、ヘッド単位でのグローイング/トークンK、Vベクトルのリストを維持。

### ステップ2：タイルソフトマックス

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-stileのsoftmax(qK^T)V、ランニングmax/sum付き。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

1ショット`softmax(qK) V`と同一の出力ですが、いつでも作業セットは完全なN × d_headではなく、tile × d_headブロック。

### ステップ3：100トークン生成で素朴対キャッシュデコーディングを比較

注意操作をカウント。素朴：`O(N²)` = 5050。キャッシュ：`O(N)` = 100。コードは両方を印刷。

## 使用

```python
# HuggingFace transformersはデコーダのみgenerate()でKVキャッシュを自動的に有効にします。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopperの場合FA3を使用
    torch_dtype="bfloat16",
)
# generate()は自動的にKVキャッシュを使用
```

vLLM本番：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

リクエスト全体のプレフィックスキャッシングは大きい2026年の勝利 — 同じシステムプロンプト、少数例、または長いコンテキストドキュメントが呼び出し全体でKVを再利用。エージェントのワークロード用の繰り返しツールプロンプト、プレフィックスキャッシング は通常5倍スループット利得。

## シップ

`outputs/skill-inference-optimizer.md`を見てください。スキルは新しい推論展開用の注意実装、KVキャッシュ戦略、量子化、推論的デコーディングを選択します。

## 演習

1. **簡単。** `code/main.py`を実行。素朴とキャッシュデコーダが同じ出力を生成することを確認；操作カウント違い。
2. **中程度。** プレフィックスキャッシング実装：プロンプトPと複数の完成、Pの1つの前向きパスを実行してKVキャッシュを埋める、完成でブランチ。re-encodingパースペクティブから高速化を測定。
3. **難しい。** 小さいPagedAttentionを実装：KVキャッシュ固定16トークンブロックで、ページテーブルのフリーリスト。シーケンスが終了したとき、ブロックをプールに返す。1,000チャット完成をシミュレート、可変長。連続割り当てと比較してメモリ断片化。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| KVキャッシュ | 「デコーディングを高速にするトリック」 | すべてのプレフィックストークンから保管済みK と V ；新しいクエリはそれらに対して注意を払う代わりに再計算。 |
| HBM | 「GPUメインメモリ」 | 高帯域幅メモリ；H100で80 GB、B200で192 GB。~3 TB/s帯域幅。 |
| SRAM | 「オンチップメモリ」 | SM単位ファストメモリ、H100で~256 KB。~30 TB/s帯域幅。 |
| Flash Attention | 「タイル注意カーネル」 | HBMにN×Nを具体化せずに注意を計算。 |
| 継続的バッチ処理 | 「待たずバッチ」 | 完了したシーケンスを削除、新しいものを、バッチを排出せず追加。 |
| PagedAttention | 「vLLMの見出し」 | KVキャッシュ固定ブロックに割り当て、ページテーブル；断片化を消除。 |
| プレフィックスキャッシング | 「長いプロンプトを再利用」 | 共有プレフィックスをキャッシュKVリクエスト全体；エージェント主要コスト削減。 |
| 推論的デコーディング | 「ドラフト + 検証」 | 安価なドラフトモデル提案トークン；大きいモデルはkを1パスで検証。 |

## 参考文献

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4リリースノート(Dao-AILab、2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5段パイプラインとソフトウェア exp2トリック；前向き専用起動警告についてはリポREADMEを読む。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM論文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推論的デコーディング。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 統合ドラフトアプローチのEAGLE-1/2論文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — EAGLEの横にあるMedusaアプローチ。
- [vLLMドキュメント — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 正規の16トークンブロックとページテーブル設計の深掘り。
