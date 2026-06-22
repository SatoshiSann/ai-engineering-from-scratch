# アテンション変種 — スライディングウィンドウ、スパース、微分

> 完全アテンションは円である。すべてのトークンがすべてのトークンを見ており、メモリが代償を払う。4つの変種は円の形を曲げて、コストの半分を回復させる。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 02 (Self-Attention)、Phase 7 · 03 (Multi-Head)、Phase 7 · 12 (KV Cache / Flash Attention)
**所要時間:** ~60分

## 問題

完全アテンションは系列長の `O(N²)` メモリと `O(N²)` 計算コストがかかる。128Kコンテキストを持つLlama 3 70Bの場合、1層あたり160億のアテンション入力があり、80層を掛ける。Flash Attention(Lesson 12)は `O(N²)` の活性化メモリを隠すが、算術コストは変わらない — すべてのトークンは依然として他のすべてのトークンに注意する。

アテンション行列のトポロジー自体を変える3つのクラスの変種がある:

1. **スライディングウィンドウアテンション (SWA)。** 各トークンは全接頭辞ではなく、固定ウィンドウ内の隣接トークンのみに注意する。メモリと計算は `O(N · W)` に減少する。ここで `W` はウィンドウサイズ。Gemma 2/3、Mistral 7Bの最初のレイヤー、Phi-3-Long。
2. **スパース/ブロックアテンション。** 選択された対 `(i, j)` のみがスコアを取得し、残りはゼロ重みに強制される。Longformer、BigBird、OpenAIスパーストランスフォーマー。
3. **微分アテンション。** 個別のQ/K投影を持つ2つのアテンションマップを計算し、一方から他方を減算する。最初のいくつかのトークンに重みが流出する「アテンションシンク」を排除する。MicrosoftのDIFF Transformer(2024)。

これらは共存する。2026年のフロンティアモデルはしばしばそれらを混在させる: ほとんどのレイヤーはSWA-1024、5番目ごとにグローバル完全アテンション、そしていくつかの微分ヘッドが検索をクリーンアップする。Gemma 3の5:1 SWA対グローバル比は現在の標準教科書デフォルト。

## コンセプト

### スライディングウィンドウアテンション (SWA)

位置 `i` の各クエリは `[i - W, i]` (因果SWA)または `[i - W/2, i + W/2]` (双方向)内の位置のみに注意する。ウィンドウ外のトークンはスコア行列で `-inf` を取得する。

```
完全因果:           スライディングウィンドウ (W=4):
位置 0-7          位置 0-7、W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

`N = 8192` と `W = 1024` の場合、スコア行列は平均で1024 × 8192の非ゼロ行を持ち、8倍の削減。

**KVキャッシュはSWAで縮小する。** 層ごとに最後の `W` トークンのKとVのみを保持する必要がある。Gemma-3風の設定(1024ウィンドウ、128Kコンテキスト)の場合、KVキャッシュは128倍削減される。

**品質コスト。** SWAのみのトランスフォーマーは長距離検索で苦労する。修正: SWAレイヤーを完全アテンションレイヤーと交互配置する。Gemma 3は5:1 SWA:グローバルを使用する。Mistral 7Bは因果SWAスタックを使用していたが、情報は重なり合ったウィンドウを通じて「前方に流れる」 — 各レイヤーは有効な受容野を `W` だけ拡張し、`L` レイヤー後、モデルは `L × W` トークン戻ることができる。

### スパース/ブロックアテンション

事前に `N × N` スパース性パターンを選択。3つの標準的な形状:

- **局所 + ストライド (OpenAIスパートランスフォーマー)。** 最後の `W` トークンと、その前の各 `stride` 番目のトークンに注意する。局所と長距離の両方を `O(N · sqrt(N))` 計算でキャプチャ。
- **Longformer / BigBird。** ローカルウィンドウ + 小さなグローバルトークンセット(例: `[CLS]`)が誰にでも注意を払い、全員から注意される + ランダムスパースリンク。経験的に一致した品質で2倍コンテキスト。
- **ネイティブスパースアテンション (DeepSeek、2025)。** 重要な `(Q, K)` ブロックを学習する; カーネルレベルでゼロブロックをスキップする。FlashAttention互換。

スパースアテンションはカーネルエンジニアリングの話である。数学は単純(スコア行列をマスク)だが、勝利はゼロ入力をSRAMにロードしないことから来る。FlashAttention-3と2026年のFlexAttention APIはカスタムスパースパターンをPyTorchで第一級にする。

### 微分アテンション (DIFF Transformer、2024)

通常のアテンションには「アテンションシンク」問題がある: softmaxはすべての行が1に合計されるように強制するため、特別なもの以外に注意したくないトークンは最初のトークン(または最初のいくつか)に重みをダンプする。これは実際のコンテンツに行くべき容量を盗む。

微分アテンションは2つのアテンションマップを計算して減算することでこれを修正する:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

ここで `λ` は学習可能なスカラ(通常0.5-0.8)。A1は実際のコンテンツ重みをキャプチャ; A2はシンクをキャプチャ。減算はシンクをキャンセルし、関連トークンに重みを再割り当てする。

報告された結果 (Microsoft 2024): 5-10% 低い困惑、同じ訓練長で1.5-2倍より長い有効コンテキスト、より鋭い針-干し草検索。

### 変種比較

| 変種 | 計算 | KVキャッシュ | 完全比品質 | 本番利用 |
|---------|---------|----------|-----------------|----------------|
| 完全アテンション | O(N²) | 層あたり O(N) | ベースライン | すべてのモデルのデフォルトレイヤー |
| SWA (ウィンドウ 1024) | O(N·W) | 層あたり O(W) | -0.1 ppl、グローバルレイヤーで良好 | Gemma 2/3、Phi-3-Long |
| 局所 + ストライドスパース | O(N·√N) | 混合 | SWAに類似 | OpenAIスパートランスフォーマー、Longformer |
| BigBird (局所 + グローバル + ランダム) | O(N) 約 | 混合 | 2倍コンテキストで完全と一致 | 初期の長コンテキストBERT |
| ネイティブスパース (DeepSeek-V3.2) | O(N · アクティブ分数) | O(N) | 0.05 ppl以内 | DeepSeek-V3.2、2025年 |
| 微分 | O(2·N²) | O(2N) | -5 から -10% ppl | DIFF Transformer、2026年初期モデル |

## ビルドする

`code/main.py` を見よ。完全、SWA、局所+ストライド、および微分アテンションをおもちゃのシーケンスで並べて表示するマスク比較ツールを実装する。

### ステップ 1: 完全因果マスク (ベースライン)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Lesson 07からのベースライン。下三角; 対角線上のゼロ重み。

### ステップ 2: スライディングウィンドウ因果マスク

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

1つのパラメータ — `window`。`window >= n` の場合、完全因果アテンションを回復する。`window = 1` の場合、各トークンは自分自身のみに注意する。

### ステップ 3: 局所 + ストライドスパースマスク

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密な局所ウィンドウ、プラス各 `stride` 番目のトークン、シーケンスの開始まで戻る。受容野は追加のレイヤーで対数ステップで増加する。

### ステップ 4: 微分アテンション

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

2つのアテンション処理、学習混合係数で減算する。コードでは、単一対微分のアテンション-シンクヒートマップを比較し、シンクの崩壊を監視する。

### ステップ 5: KVキャッシュサイズ

`N = 131072` で各変種の層ごとのキャッシュサイズを出力。SWAおよびスパース変種は10-100倍削減される。微分は2倍にする。意識的にメモリ請求書を支払う。

## 使用

2026年の本番パターン:

```python
from transformers import AutoModelForCausalLM
# Gemma 3は5:1でSWA (window=1024)とグローバルレイヤーを混在させる。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+のFlexAttentionはマスク関数を受け入れる:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

これはカスタムTritonカーネルにコンパイルされる。一般的なパターンではFlashAttention-3速の10%以内で、マスク関数はPythonコーラブル。

**各を選択する時期:**

- **純粋な完全アテンション** — ~16Kコンテキストまでのすべてのレイヤー、または検索品質が最優先の場合。
- **SWA + グローバル混合** — 長いコンテキスト (>32K)、訓練および推論メモリバウンド。2026年のデフォルト、32K以上。
- **スパースブロックアテンション** — カスタムカーネル、カスタムパターン。専門的なワークロード(検索、オーディオ)用に予約済み。
- **微分アテンション** — アテンション-シンク汚染が害する任意のワークロード(長コンテキストRAG、針-干し草検索)。

## 配布

`outputs/skill-attention-variant-picker.md` を見よ。このスキルは、ターゲットコンテキスト長、検索要求、および訓練/推論計算プロファイルを与えられたとき、新しいモデルのアテンショントポロジーを選択する。

## 演習

1. **簡単。** `code/main.py` を実行。`window=4` でのSWAが各行の最後の4トークンの外側をすべてゼロにすることを確認。`window=n` が完全因果アテンションをビット同一に再現することを確認。
2. **中程度。** Lesson 07キャプストーンの上に `window=1024` を持つ因果SWAを実装。tinyshakespeareで1,000ステップを訓練。完全アテンション対val損失はどれだけ後退? ピークメモリはどれだけ低下?
3. **難しい。** キャプストーンモデルにGemma-3風の5:1レイヤー混合(5 SWA、1グローバル)を実装。純SWAおよび純グローバルベースラインに対して損失、メモリ、および生成品質を比較し、一致したパラメータで。
4. **難しい。** ヘッドごとに学習可能な `λ` を持つ微分アテンションを実装。合成検索タスク(1つの針、2,000妨害者)で訓練。検索精度対一致したパラメータの単一アテンションベースラインを測定。

## 重要な用語

| 用語 | 人々が言う | 実際の意味 |
|------|-----------------|-----------------------|
| スライディングウィンドウアテンション (SWA) | 「ローカルアテンション」 | 各クエリはその最後の `W` トークンに注意; KVキャッシュは `O(W)` に縮小。 |
| 有効受容野 | 「モデルがどれだけ遠く見るか」 | ウィンドウ `W` を持つ `L` 層のSWAスタックで、最大 `L × W` トークン。 |
| Longformer / BigBird | 「ローカル + グローバル + ランダム」 | スパースパターン、常に注意を払う少数のグローバルトークン; 初期の長コンテキストアプローチ。 |
| ネイティブスパースアテンション | 「DeepSeekのカーネルトリック」 | ブロックレベルのスパース性を学習; 品質を保ちながらカーネルレベルでゼロブロックをスキップ。 |
| 微分アテンション | 「2つのマップ、1つが減算」 | DIFF Transformer: 学習可能な `λ` 倍の第2アテンションマップを第1から減算してアテンションシンクをキャンセル。 |
| アテンションシンク | 「重みはトークン0に流出」 | Softmax正規化は行を1に合計するように強制; 無情報クエリは位置0に重みをダンプ。 |
| FlexAttention | 「マスク-as-Python」 | PyTorch 2.5+ API、任意のマスク関数をFlashAttention形状カーネルにコンパイル。 |
| レイヤータイプ混合 | 「5:1 SWA-to-グローバル」 | スタックのスパースおよび完全アテンションレイヤーを交互に配置して、低メモリで品質を保つ。 |

## さらに読む

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 標準的なスライディング-ウィンドウ + グローバル-トークンペーパー。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局所 + グローバル + ランダム。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAIの局所+ストライドパターン。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:グローバル混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 現在の標準教科書デフォルトの window=1024 を持つ 5:1 混合。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformerペーパー。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2の学習スパース性アテンション。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — 「Use It」のマスク-as-コーラブルパターンのAPIリファレンス。
