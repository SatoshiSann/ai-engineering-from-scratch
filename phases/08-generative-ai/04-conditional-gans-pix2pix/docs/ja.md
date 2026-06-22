# 条件付けGAN & Pix2Pix

> 2014-2017の最初の大きいロック解除はGANが作るものを制御した。ラベル、またはイメージ、または文を添付。Pix2Pixはイメージ版を行い、依然として狭いイメージ-to-イメージタスク上ですべての一般的なテキスト-to-イメージモデルを打つ。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 8 · 03 (GANs)、Phase 4 · 06 (U-Net)、Phase 3 · 07 (CNNs)
**所要時間:** ~75分

## 問題

無条件GANは恣意的フェイスをサンプル。デモのために有用、本番では無用。あなたが欲する: *スケッチをフォトにマップ、*マップを航空フォトにマップ、*昼シーンを夜にマップ、*グレースケール イメージをカラー化。これらすべてで、あなたは入力イメージ `x` が与えられ、いくつかセマンティック対応を持つ出力 `y` をしなければならない。`x` あたり多くのもっともらしい `y` がある。平均-二乗誤差それらをマッシュにフラット。敵対的損失はなく、なぜなら「本物のように見える」は鋭い。

条件付けGAN (Mirza & Osindero、2014)は条件 `c` を `G` と `D` 両方への入力として追加。Pix2Pix (Isola et al.、2017)これを特化: 条件はフル入力イメージ、生成器は U-Net、判別器は*パッチベース*分類ツール(PatchGAN)、および損失は敵対的 + L1。そのレシピはスクラッチ-フロムテキスト-to-イメージモデルを、狭いイメージ-to-イメージドメインで2026年でも、それが*対サンプリング データ — あなたは正確に信号が必要正確に持つため、優れる。

## コンセプト

![Pix2Pix: U-Netジェネレータ、PatchGAN判別器](../assets/pix2pix.svg)

**条件付けG。** `G(x, z) → y`。Pix2Pixで、`z` は `G` の内部ドロップアウト(入力ノイズなし — Isola見つけた明示的ノイズは無視された)。

**条件付けD。** `D(x, y) → [0, 1]`。入力は*ペア*(条件、出力)。これはキー違い: `D` は `y` が本物に見えるかどうかなく、`y` が `x` と矛盾しているかどうか判断。

**U-Netジェネレータ。** ボトルネック全体のスキップ接続を持つエンコーダ-デコーダ。入力と出力が低レベル構造(エッジ、輪郭)を共有するタスク向けな批判的。スキップなし、高周波詳細は消失。

**PatchGAN判別器。** 単一の本物/偽物スコアを出力する代わりに、`D` は`N×N` グリッド出力で各セルが ~70×70ピクセルの受動野を判断。平均化。これはマルコフランダムフィールド仮説: 現実主義はローカル。訓練速度多く、より少ないパラメータ、より鋭い出力。

**損失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1項は訓練を安定、`G` を既知ターゲットに向けてプッシュ。L1はL2より鋭いエッジ(中央値、不平均)を与える。`λ = 100` はPix2Pixデフォルト。

## CycleGAN — ペアを持たない場合

Pix2Pixはペアード `(x, y)` データが必要。CycleGAN (Zhu et al.、2017)この要件をドロップして追加損失でコスト: *サイクル一貫性*損失。2つのジェネレータ `G: X → Y` および `F: Y → X`。訓練してので `F(G(x)) ≈ x` および `G(F(y)) ≈ y`。これはあなたが対なし馬をシマウマに、夏から冬に翻訳できる。

2026年では、非ペアードイメージ-to-イメージはほとんど拡散(ControlNet、IP-Adapter)を介して行われ、CycleGAN以上、しかしサイクル-一貫性アイデアはほぼすべての非ペアードドメイン適応ペーパーで生き残る。

## ビルドする

`code/main.py` は1-Dデータ上の小さい条件付けGANを実装。条件 `c` はクラスラベル(0または1)。タスク: 与えられたクラスのための条件付け分布からサンプルを生成。

### ステップ 1: G と D 入力に条件を追加

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

ワンホット符号化は最も単純。大きいモデルは学習埋め込み、FiLMモジュレーション、またはクロス-アテンション使用。

### ステップ 2: 条件付け訓練

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

ジェネレータは、周辺でなく、与えられた条件のための本物分布にマッチしなければならない。

### ステップ 3: クラスあたりの出力を確認

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 落とし穴

- **条件は無視。** G が周辺化に学習、D 決して罰する条件信号は弱いため。修正: `D` をより積極的に条件(早レイヤー、遅くない)、射影判別器を使用(Miyato & Koyama 2018)。
- **L1重量太低。** G は恣意的本物のように見える出力にドリフト、忠実なもの。スタートλ≈100 Pix2Pix風タスク。
- **L1重量太高。** G はぼやけた出力を生成、L1 はまだLp ノルム。訓練が安定したら、それとともに退火。
- **グラウンド-真実漏洩 D.**をで。`(x, y)` をD 入力として連結、単に `y` ではなく。この D ない一貫性を確認できない。
- **モード崩壊クラスあたり。** 各クラスはそれとともに独立して崩壊できる。実行クラス-条件付き多様性チェック。

## 使用

2026年イメージ-to-イメージタスクの状態:

| タスク | 最良アプローチ |
|------|---------------|
| スケッチ → フォト、同じドメイン、ペアードデータ | Pix2Pix / Pix2PixHD (高速であり、鋭く依然として) |
| スケッチ → フォト、非ペアード | Scribbleモデリング条件を持つControlNet |
| セマンティック seg → フォト | SPADE / GauGAN2 またはSD + ControlNet-Seg |
| スタイル転送 | LoRA またはIP-Adapter拡散; GANメソッドは遺産 |
| 深さ → フォト | Stable Diffusionの上のControlNet-Depth |
| スーパー-解像度 | Real-ESRGAN (GAN)、ESRGAN-Plus、またはSD-Upscale (拡散) |
| カラー化 | ColTran、拡散ベースカラー化、またはPix2Pix-色 |
| 昼 → 夜、季節、天気 | CycleGAN またはControlNet-ベース |

Pix2Pixは(a)数千のペアード例を持つ、(b)タスクはナロー且つ反復可能である、および(c)あなたが高速推論が必要である場合、残る正しいツール。一般的なオープンドメインタスク上、拡散勝つ。

## 配布

`outputs/skill-img2img-chooser.md` で保存。スキルはタスク説明、データ利用可能性(ペアード対非ペアード、Nサンプル)、およびレイテンシ/品質予算を取り、出力: アプローチ(Pix2Pix、CycleGAN、ControlNetバージョン、SDXL + IP-Adapter)、訓練データ要件、推論コスト、および評価プロトコル(LPIPS、FID、タスク-固有)。

## 演習

1. **簡単。** 3番目のクラスを追加するための `code/main.py` を修正。G は依然として各クラスのノイズを正しいモードへマップすることを確認。
2. **中程度。** L1を認識的-スタイル損失で1-Dセッティングで置き換え(例: 特徴抽出器として作動小さい凍結D)。シャープネスは変わるか?
3. **難しい。** 1-Dセッティングでジェネレータを描画CycleGAN: 2つの分布、2つのジェネレータ、サイクル損失。それら非ペアードデータを持つそれら間にマップを学習する示す。

## 重要な用語

| 用語 | 人々が言う | 実際の意味 |
|------|-----------------|-----------------------|
| 条件付けGAN | 「ラベル付きGAN」 | G(z, c)、D(x, c)。両ネットワークは条件を見る。 |
| Pix2Pix | 「イメージ-to-イメージGAN」 | ペアードcGAN U-Net G および PatchGAN D + L1損失で。 |
| U-Net | 「エンコーダ-デコーダ、スキップと」 | 対称conv ネットワーク; スキップは高周波を保持。 |
| PatchGAN | 「ローカル-現実主義分類器」 | グローバルスコアの代わりに、D パッチあたりスコア出力。 |
| CycleGAN | 「非ペアードイメージ翻訳」 | 2つのG + サイクル-一貫性損失; ペアードデータなし。 |
| SPADE | 「GauGAN」 | セマンティックマップで中程度活性化を正規化; セグメンテーション-to-イメージ。 |
| FiLM | 「特徴-ワイズ線形モジュレーション」 | 条件から-特性アファイン変換; チープ条件付け。 |

## 本番ノート: ペアードデータ基準としてのPix2Pixレイテンシ-バウンド

ペアードデータおよびナロータスク(スケッチ → レンダー、セマンティックマップ → フォト、昼 → 夜)を持つ場合、Pix2Pixのワンショット推論はレイテンシ上の拡散で桁を打つ。本番比較は通常:

| パス | ステップ | 単一L4 で 512² でのタイプレイテンシ |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net順伝播) | 1 | ~30 ms |
| SD-Inpaint またはSD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pixは静的バッチ内のスループットで勝つ(すべてのリクエストは同じFLOPs)。拡散は品質および一般化で勝つ。現代的プレイはしばしばナロータスクのための蒸留Pix2Pixスタイルモデルおよび尾部入力の拡散フォールバックを配布。

## さらに読む

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGANペーパー。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 射影D。
