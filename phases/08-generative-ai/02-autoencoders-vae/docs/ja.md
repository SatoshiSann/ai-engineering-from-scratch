# オートエンコーダ & 変分オートエンコーダ (VAE)

> プレーンなオートエンコーダは圧縮してから再構成。それは暗記する。生成しない。1つのトリックを追加 — コードをガウス的に見えるよう強制 — そしてあなたはサンプラーを取得。その単一のトリック、`z = μ + σ·ε` の再パラメータ化は、なぜ2026年で使うすべての潜在拡散とフロー-マッチングイメージモデルが入力にVAEを持つかである。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 3 · 02 (Backprop)、Phase 3 · 07 (CNNs)、Phase 8 · 01 (Taxonomy)
**所要時間:** ~75分

## 問題

784ピクセルのMNIST数字を16数コードに圧縮、その後再構成。プレーンなオートエンコーダは再構成MSEを達成するが、コード空間は塊っぽい混乱。コード空間内のランダムなポイントを選択、それをデコード、そしてあなたはノイズを得る。サンプラーがない。それは圧縮モデルの衣装を着ている。

あなたが実際に欲しいもの: (a)コード空間は、あなたがサンプルできるきれいで滑らかな分布 — 例えば同等ガウス `N(0, I)`、(b)デコードどのサンプルもモッサイ数字を生成、および(c)エンコーダとデコーダはそれでも圧縮する良好。3つの目標、1つのアーキテクチャ、1つの損失。

Kingmaの2013 VAEは、エンコーダを*分布* `q(z|x) = N(μ(x), σ(x)²)` を出力するよう訓練し、その分布をKLペナルティを通じて事前 `N(0, I)` に向けてプル、その後 `q(z|x)` からサンプル `z` してからデコードするこをでこれを解決。推論時にエンコーダをドロップ、`z ~ N(0, I)` をサンプル、デコード。KLペナルティはコード空間を構造化させるもの。

2026年VAEはめったに単独で配布されない — それら微分品質は拡散によって超えられている — しかしそれらはすべての潜在拡散モデル(SD 1/2/XL/3、Flux、AudioCraft)のエンコーダの選択である。VAEを学習、そしてあなたは2026年で使うすべてのイメージパイプラインの見えない最初のレイヤーを学習。

## コンセプト

![オートエンコーダ対VAE: 再パラメータ化トリック](../assets/vae.svg)

**オートエンコーダ。** `z = encoder(x)`、`x̂ = decoder(z)`、損失 = `||x - x̂||²`。コード空間未構造化。

**VAEエンコーダ。** 2つのベクトルを出力: `μ(x)` および `log σ²(x)`。これら `q(z|x) = N(μ, diag(σ²))` を定義。

**再パラメータ化トリック。** `q(z|x)` からのサンプリングは微分不可能。サンプルをサンプルを `z = μ + σ·ε` として書き換える、ここで `ε ~ N(0, I)`。いま `z` は `(μ, σ)` のの関数プラス非パラメータノイズ — 勾配 `μ` および `σ` を通じて流れる。

**損失。** 証拠下限 (ELBO)、2つのタームで:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

再構成は `x̂` を `x` に向けてプッシュ。KLは `q(z|x)` を事前に向けてプッシュ。それらはトレードオフ。小さい β (<1) = より鋭いサンプル、コード空間はより少なくガウス。大きい β (>1) = よりクリーンなコード空間、よりぼやけたサンプル。β-VAE (Higgins 2017)はこのノブを有名にして、分解ならずを開始した。

**サンプリング。** 推論では: `z ~ N(0, I)` から描画、デコーダを通じて前方。1つの順伝播 — 拡散のような反復サンプリング。

## ビルドする

`code/main.py` はnumpyまたはtorchなしの小さいVAEを実装。入力は8-Dガウス混合から抽出された8-次元の合成データ。エンコーダとデコーダは単一の隠し層MLP。tanh活性化、順伝播、損失、および手書き逆伝播を実装。非本番 — 教育。

### ステップ 1: エンコーダ順伝播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` の代わりに `σ` で、ネットワーク出力は無制約(softplus の `σ` はトラップである — `σ ≈ 0` で勾配は死ぬ)。

### ステップ 2: 再パラメータ化とデコード

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### ステップ 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

両分布がガウスであるため、正確な閉鎖形KL。数値的に統合しない。人々は2026年でモンテ-カルロKL推定値を持つコードを配布 — それは何の理由もなく3倍遅い。

### ステップ 4: 生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

それは生成モデル。5行。

## 落とし穴

- **事後崩壊。** KL項は `q(z|x) → N(0, I)` をそれほど積極的にドライブするため、`z` は `x` についての情報を持たない。修正: β-アニーリング(β=0で開始、1にランプ)、無料ビット、または非活性次元でKLをスキップ。
- **ぼやけたサンプル。** ガウスデコーダ尤度はMSE再構成を意味し、これはL2に対してベイズ最適 — モッサイ数字のセットの平均は模糊た数字。修正: 離散デコーダ(VQ-VAE、NVAE)、または潜在上にVAEのみを使用してスタック拡散(これはStable Diffusionが行うこと)。
- **β太大きい、早すぎる。** 事後崩壊を見よ。β≈0.01 で開始、ランプ。
- **潜在次元太小さい。** 16-Dはfor MNIST、256-Dはfor ImageNet 256²、2048-DはImageNet 1024²。Stable Diffusionの VAE は 512×512×3 → 64×64×4 を圧縮(空間領域で32x下サンプル係数、チャネルで32x)。

## 使用

2026年VAEスタック:

| 状況 | ピック |
|-----------|------|
| 拡散のイメージ潜在エンコーダ | Stable Diffusion VAE (`sd-vae-ft-ema`) またはFlux VAE |
| オーディオ潜在エンコーダ | Encodec (Meta)、SoundStream、またはDAC (Descript) |
| ビデオ潜在 | Soraの空間時間パッチ、Latte VAE、WAN VAE |
| 分解表現学習 | β-VAE、FactorVAE、TCVAE |
| 離散潜在 (トランスフォーマーモデリング) | VQ-VAE、RVQ (ResidualVQ) |
| 生成のための継続潜在 | プレーンVAE、その後その潜在空間で流 / 拡散モデルを条件 |

潜在拡散モデルはVAEに拡散モデルがエンコーダとデコーダの間に住む。VAEは粗い圧縮、拡散モデルは重い持ち上げをする。ビデオ(VAE + ビデオ-拡散DiT)およびオーディオ(Encodec + MusicGen変圧器)の同じパターン。

## 配布

`outputs/skill-vae-trainer.md` で保存。

スキルは: データセット プロフィール + 潜在-次元ターゲット + 下流使用(再構成、サンプリング、または潜在-拡散入力)を取り、出力: アーキテクチャ選択(プレーン/β/VQ/RVQ)、β スケジュール、潜在次元、デコーダ尤度(ガウス対カテゴリ)、および評価プラン(再構成MSE、KL毎次元、`q(z|x)` と `N(0, I)` 間のFréchet距離)。

## 演習

1. **簡単。** `code/main.py` で `β` を `0.01`、`0.1`、`1.0`、`5.0` に変更。最終的な再構成MSEおよびKLを記録。どの β がPareto-最高のあなたの合成データについて?
2. **中程度。** ガウスデコーダ尤度をベルヌーイ尤度(クロス-エントロピー損失)と置き換え。同じ合成データの二値化版でのサンプル品質を比較。
3. **難しい。** `code/main.py` を小さいVQ-VAEに拡張: 継続 `z` を K=32 エントリの コードブック内の最近隣ルックアップで置き換え。再構成MSEを比較してレポート、いくつの コードブック エントリが使用される(コードブック崩壊は実在)。

## 重要な用語

| 用語 | 人々が言う | 実際の意味 |
|------|-----------------|-----------------------|
| オートエンコーダ | エンコード-デコードネットワーク | `x → z → x̂`、MSEを学習。非生成。 |
| VAE | サンプラーのあるAE | エンコーダは分布を出力、KLペナルティはコード空間を形状。 |
| ELBO | 証拠下限 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; `q = p(z\|x)` の際にタイト。 |
| 再パラメータ化 | `z = μ + σ·ε` | 確率的ノードを決定論的 + 純粋ノイズとして書き換え。サンプリングを通じたバックプロップを有効化。 |
| 事前 | `p(z)` | 潜在のターゲット分布、通常 `N(0, I)`。 |
| 事後崩壊 | 「KL項は勝つ」 | エンコーダは `x` を無視、事前を出力; デコーダはハルシネート。 |
| β-VAE | チューナブルKL重み | `loss = recon + β·KL`。より高い β = より分解されたがより模糊。 |
| VQ-VAE | 離散潜在 | 継続 `z` を最近隣コードブック ベクトルで置き換え; トランスフォーマー モデリングを有効化。 |

## 本番ノート: VAEは拡散サーバーの最もホットパス

Stable Diffusion / Flux / SD3 パイプラインでは、VAEはリクエストあたり2回呼ばれる — 一度エンコード(img2img/inpainting に場合) および一度デコード。1024² では、デコード処理は潜在 `128×128×16` を `1024×1024×3` 背景にアップサンプルするため、しばしば全パイプラインの単一最大活性化-メモリ ピークである。2つの実用的結果:

- **デコードをスライスまたはタイル。** `diffusers` は `pipe.vae.enable_slicing()` および `pipe.vae.enable_tiling()` を公開。タイリングは小さいシーム成果物を `O(tile²)` メモリに交換する、`O(H·W)` の代わりに。1024²+ 消費者GPU上に必須。
- **bf16デコーダ、最終的なリサイズのためのfp32数値。** SD 1.x VAEはfp32で放出された、*静かにNaN生成* 1024²+ でfp16にキャストされた場合。SDXL は `madebyollin/sdxl-vae-fp16-fix` を配布 — 常に fp16-fix変異体を好むまたはbf16を使用。

## さらに読む

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAEペーパー。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 分解β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先端イメージVAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAEはエンコーダ。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec、オーディオVAE標準。
