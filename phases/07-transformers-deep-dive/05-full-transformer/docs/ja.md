# 完全トランスフォーマー — エンコーダ + デコーダ

> アテンションはスター。その他すべて — 残差、正規化、フィードフォワード、クロスアテンション — あなたがそれを深くスタックできる 足場。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 02 （自己注意）, Phase 7 · 03 （マルチ頭アテンション）, Phase 7 · 04 （位置符号化）
**所要時間:** 約75分

## 問題

単一のアテンション層は特性抽出器、モデルではない。層あたり1つの matmul は言語向けに十分な容量ではない。深さが必要 — 正しい配管なしで深さは打ち壊す。

2017年Vaswaniペーパーは6つの設計決定をパッケージ化、1つのアテンション層をスタック可能ブロックに変えた。以来すべてのトランスフォーマー — エンコーダのみ（BERT）、デコーダのみ（GPT）、エンコーダデコーダ（T5） — 同じ骨組みを継承。2026年ブロックが改良化（RMSNorm、SwiGLU、pre-norm、RoPE）、骨組みは同一。

このレッスンは骨組みだ。次レッスンはそれを専門化 — 06 エンコーダ向け、07 デコーダ向け、08 エンコーダデコーダ向け。

## コンセプト

![エンコーダとデコーダブロック内部、配線](../assets/full-transformer.svg)

### 6つのピース

1. **埋め込み + 位置的信号。** トークン → ベクトル。位置注入 RoPE（最新）または 正弦波（古典）を通じて。
2. **自己注意。** あらゆる位置があらゆる他に注目。デコーダでマスク。
3. **フィードフォワードネットワーク（FFN）。** 位置ごとの2層MLP: `W_2 · activation(W_1 · x)`。拡張比 4× デフォルト。
4. **残差接続。** `x + sublayer(x)`。これなし、勾配は消滅 ~6層超。
5. **層正規化。** `LayerNorm` または `RMSNorm`（最新）。残差流を安定化。
6. **クロスアテンション（デコーダのみ）。** クエリはデコーダから来る、キーと値はエンコーダ出力から。

### エンコーダ ブロック （BERT、T5エンコーダで使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── 残差 ──┘
```

エンコーダは双方向。マスクなし。すべての位置はすべての位置を見る。

### デコーダ ブロック （GPT、T5デコーダで使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

デコーダはブロックあたり3つの副層を持つ。中間1つ — クロスアテンション — 情報がエンコーダからデコーダに流れる唯一の場所。純デコーダのみアーキテクチャ（GPT）では、クロスアテンションを省かれ、マスクされた自己注意 + FFN のみを持つ。

### Pre-norm vs post-norm

元論文: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm は 2019年周りで好まれた失敗 — 注意深いウォームアップなしに深く訓練するのはより難しい。Pre-norm（`LN` *の前に* sublayer）は 2026年 デフォルト: Llama, Qwen, GPT-3+, Mistral はすべてそれを使用。

### 2026年現代化 ブロック

Vaswani 2017 は LayerNorm + ReLU を出荷。最新スタックは両方を置き換えた。本番ブロックが実際に見える:

| コンポーネント | 2017 | 2026 |
|-----------|------|------|
| 正規化 | LayerNorm | RMSNorm |
| FFN 活性化 | ReLU | SwiGLU |
| FFN 拡張 | 4× | 2.6× (SwiGLU は3つのマトリックスを使う、全パラメータ一致) |
| 位置 | 正弦波絶対 | RoPE |
| アテンション | 完全 MHA | GQA (または MLA) |
| バイアス用語 | はい | いいえ |

RMSNorm は LayerNorm の平均中心化を落とす（1つ少ない減算）、計算を保存し経験的に少なくとも同じほど安定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）は Llama、PaLM と Qwen ペーパーで ReLU/GELU FFN を ~0.5 ポイント ppl でコンスタント上回る。

### パラメータ カウント

1つのブロック `d_model = d` と FFN 拡張 `r`:

- MHA: `4 · d²` (Q, K, V, O 投影)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- 正規化: 無視できる

`d = 4096, r = 2.6, layers = 32` で (ほぼ Llama 3 8B), 全: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B パラメータ あたり層 × 32 ≈ 7B` (加算 エンベディング と head)。公開カウントと一致。

## ビルド

### ステップ1: ビルディング ブロック

レッスン03の小さい `Matrix` クラスを使用（このファイルに独立性向けコピー）:

- `layer_norm(x, eps=1e-5)` — 平均を減算、std で除算。
- `rms_norm(x, eps=1e-6)` — RMS で除算。平均減算なし。
- `gelu(x)` と `silu(x) * W3 x` (SwiGLU)。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` と `decoder_block(x, enc_out, params)`。

完全配線向け `code/main.py` を見よ。

### ステップ2: 2層エンコーダと2層デコーダをワイア

スタック。エンコーダ出力をあらゆるデコーダ クロスアテンションに渡す。最終 LN を出力投影の前に追加。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### ステップ3: おもちゃの例でフォワード実行

6トークン ソースと 5トークン ターゲットを通す。出力形が `(5, vocab)` である確認。訓練なし — このレッスンはアーキテクチャについて、損失ではない。

### ステップ4: RMSNorm + SwiGLU とスワップイン

LayerNorm と ReLU-FFN を RMSNorm と SwiGLU と置き換え。形はまだ一致を確認。これは1つの関数置換を持つ 2026年 現代化。

## 使用

PyTorch/TF 参照実装: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`。だがほとんどの 2026年 本番コードは独自ブロックを転がす なぜなら:

- Flash Attention はアテンション内で呼ばれ、`nn.MultiheadAttention` 経由ではない。
- GQA / MLA は stdlib 参照にない。
- RoPE, RMSNorm, SwiGLU は PyTorch デフォルトではない。

HF `transformers` は読むべき清潔な参照ブロックを持つ: `modeling_llama.py` は規範的な 2026年 デコーダのみブロック。それは ~500行で一度歩くするに値する。

**エンコーダ vs デコーダ vs エンコーダデコーダ — いつピックするか:**

| ニーズ | ピック | 例 |
|------|--------|---------|
| 分類、埋め込み、テキスト上 QA | エンコーダのみ | BERT, DeBERTa, ModernBERT |
| テキスト生成、チャット、コード、推論 | デコーダのみ | GPT, Llama, Claude, Qwen |
| 構造化入力 → 構造化出力（翻訳、要約） | エンコーダデコーダ | T5, BART, Whisper |

デコーダのみ言語で勝った これは最もクリーンにスケール、包括および生成を処理。エンコーダデコーダ はいまだ最良 入力が明確な「ソースシーケンス」アイデンティティ（翻訳、音声認識、構造化タスク）。

## 展開

`outputs/skill-transformer-block-reviewer.md` を見よ。スキルは新トランスフォーマー ブロック実装を 2026年 デフォルトに対してレビュー、欠けているピースをフラグ（pre-norm、RoPE、RMSNorm、GQA、FFN 拡張比）。

## 演習

1. **簡単。** `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` でのあなたのencoder_blockのパラメータをカウント。`sum(p.numel() for p in block.parameters())` を使用するブロックを実装によって確認。
2. **中程度。** post-norm から pre-norm へスイッチ。両方を初期化し、ランダム入力上で 12 スタック層後の活性化 norm を測定。Post-norm の活性化は爆発すべき; pre-norm の境界内にとどまるべき。
3. **難しい。** 4層エンコーダデコーダをおもちゃコピータスクで実装（コピー `x` 逆行）。100ステップを訓練。損失を報告。RMSNorm + SwiGLU + RoPE とスワップイン — 損失は下がる？

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|-----------------|-----------------------|
| ブロック | 「1つのトランスフォーマー層」 | 正規化 + アテンション + 正規化 + FFN スタック、残差接続で包装。 |
| 残差 | 「スキップ接続」 | `x + f(x)` 出力; 深いスタックを通じた勾配流を有効にする。 |
| Pre-norm | 「正規化前、後ではない」 | 最新: `x + sublayer(LN(x))`。ウォームアップ体操なしで深くトレーニング。 |
| RMSNorm | 「LayerNorm 平均なし」 | RMS で除算; 1つ少ないオプ、同じ経験的安定性。 |
| SwiGLU | 「FFN 誰もが切り替えた」 | `Swish(W1 x) ⊙ W3 x → W2`。LM ppl で ReLU/GELU を打つ。 |
| クロスアテンション | 「デコーダはどのようにエンコーダを見る」 | MHA クエリはデコーダから、K/V はエンコーダ出力から。 |
| FFN 拡張 | 「中程度のMLP 方法の広さ」 | d_model への隠れサイズの比率、通常 4（LayerNorm）または 2.6（SwiGLU）。 |
| バイアスフリー | 「+b 用語を落とす」 | 最新スタックは線形層でバイアスを省略; わずかな ppl 改善、より小さいモデル。 |

## 参考文献

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 元のブロック仕様。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm が post-norm を深く打つ理由。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU ペーパー。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 規範的な 2026年 デコーダのみブロック。
