# Vision Transformers (ViT)

> 画像はパッチのグリッドです。センテンスはトークンのグリッドです。同じトランスフォーマーが両方を食べます。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 05 (完全なトランスフォーマー), Phase 4 · 03 (CNN), Phase 4 · 14 (Vision Transformers入門)
**所要時間:** 約45分

## 問題

2020年前、コンピュータビジョンは畳み込みを意味していました。ImageNet、COCO、検出ベンチマークのすべてのSOTAはCNNバックボーンを使用しました。トランスフォーマーは言語用でした。

Dosovitskiy et al. (2020) — 「An Image is Worth 16x16 Words」 — 畳み込みを完全にドロップできることを示しました。画像を固定サイズパッチにスライス、各パッチを埋め込みに線形に投影、シーケンスをバニラトランスフォーマーエンコーダにフィード。十分なスケール(ImageNet-21k事前学習またはそれ以上)で、ViTはResNetベースのモデルと同等またはそれ以上に一致します。

ViTは2026年のより広いパターンの開始：1つのアーキテクチャ、多くのモダリティ。Whisperがオーディオをトークン化します。ViTが画像をトークン化します。ロボティクス向けのアクショントークン。ビデオのピクセルトークン。トランスフォーマーはケアしません — シーケンスをフィード、学習します。

2026年までに、ViTとその子孫(DeiT、Swin、DINOv2、ViT-22B、SAM 3)がビジョンのほとんどを所有しています。CNNはまだエッジデバイスと遅延に敏感なタスクで勝つ。他のすべてはスタックのどこかにViTを持っています。

## コンセプト

![画像 → パッチ → トークン → トランスフォーマー](../assets/vit.svg)

### ステップ1 — パッチ化

`H × W × C`画像を`N × (P·P·C)`フラットパッチのシーケンスに分割。典型的なセットアップ：`224 × 224`画像、`16 × 16`パッチ → 各768値の196パッチ。

```
画像 (224, 224, 3) → 16x16x3パッチの14 × 14グリッド → 長さ768の196ベクトル
```

パッチサイズはレバーです。小さいパッチ=より多くのトークン、より良い解像度、二次のアテンション費用。大きいパッチ=より粗く、安い。

### ステップ2 — 線形埋め込み

各フラットパッチを`d_model`に投影する単一の学習可能な行列。カーネルサイズ`P`とストライド`P`の畳み込みと同等。PyTorchではこれは文字通り`nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — 2行実装。

### ステップ3 — `[CLS]`トークンを先頭に加える、位置埋め込みを追加

- 学習可能な`[CLS]`トークンを先頭に加えます。その最終隠れ状態は分類に使用される画像表現です。
- 学習可能な位置埋め込み(ViT-元)またはサイン波2D(後のバリアント)を追加します。
- 2024年以降RoPEが位置に2D拡張、時々明示的埋め込みなし。

### ステップ4 — 標準トランスフォーマーエンコーダ

L個の`LayerNorm → 自己注意 → + → LayerNorm → MLP → +`ブロックをスタック。BERTと同一。ビジョン固有のレイヤーなし。これが論文の教育的なポイントです。

### ステップ5 — ヘッド

分類：`[CLS]`隠れ状態を取得 → 線形 → ソフトマックス。DINOv2またはSAMでは、`[CLS]`を破棄、パッチ埋め込みを直接使用。

### 重要なバリアント

| モデル | 年 | 変更 |
|-------|------|--------|
| ViT | 2020 | 元。固定パッチサイズ、完全なグローバルアテンション。 |
| DeiT | 2021 | 蒸留；ImageNet-1kのみで訓練可能。 |
| Swin | 2021 | シフトウィンドウを持つ階層的。固定部分二次費用。 |
| DINOv2 | 2023 | 自己教師(ラベルなし)。最良の一般ビジョン機能。 |
| ViT-22B | 2023 | 22Bパラメータ；スケーリング則が適用。 |
| SigLIP | 2023 | ViT + 言語ペア、シグモイドコントラッシブ損失。 |
| SAM 3 | 2025 | 何でもセグメント化；ViT-Large + プロンプト可能なマスクデコーダ。 |

### 時間がかかった理由

ViTは、CNNの誘導的バイアス(移動不変性、局所性)がないため、CNNに一致するために*多くの*データが必要です。>100M個のラベル付き画像またはほぼ自己教師学習なしでは、CNNは同等の計算で勝ちます。DeiT は2021年に蒸留トリックでこれを修正しました；DINOv2は2023年に自己教師学習で永久に修正しました。

## ビルド

`code/main.py`を見てください。純粋なstdlibパッチ化 + 線形埋め込み + 健全性チェック。学習なし — 現実的なスケールのViTはPyTorchとGPU時間を必要とします。

### ステップ1：偽の画像

24 × 24 RGB画像を`(R, G, B)`タプルの行のリストとして。私たちは6×6パッチを使用 → 16パッチ、各108次元埋め込みベクトル。

### ステップ2：パッチ化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

ラスター順：グリッド全体で行メジャー。すべてのViTがこの順序を使用します。

### ステップ3：線形埋め込み

各フラットパッチにランダムな`(patch_flat_size, d_model)`行列を乗算。`[CLS]`を先頭に加えた後、出力形状が`(N_patches + 1, d_model)`であることを検証します。

### ステップ4：現実的なViTのパラメータ数を数える

ViT-Base用のパラメータ数を印刷：12レイヤー、12ヘッド、d=768、パッチ=16。ResNet-50(~25M)と比較。ViT-Baseは~86Mに着地。ViT-Large ~307M。ViT-Huge ~632M。

## 使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196パッチ
cls_emb = out[:, 0]                       # 画像表現
```

**DINOv2埋め込みは2026年のデフォルトです。** バックボーンをフリーズ、小さいヘッドを訓練。分類、検索、検出、キャプション作成で機能。Metaのチェックポイントはすべての非テキストビジョンタスクでCLIPを上回ります。

**パッチサイズピッキング。** 小さいモデルは16×16を使用(ViT-B/16)。密な予測(セグメント化)は8×8または14×14を使用(SAM、DINOv2)。非常に大きいモデルは14×14を使用。

## シップ

`outputs/skill-vit-configurator.md`を見てください。スキルはビジョンタスク、解像度、計算予算が与えられた新しいビジョンタスク用のViTバリアントとパッチサイズを選択します。

## 演習

1. **簡単。** `code/main.py`を実行。パッチの数が`(H/P) * (W/P)`に等しく、フラットパッチ次元が`P*P*C`に等しいことを検証します。
2. **中程度。** 2Dサイン波位置埋め込みを実装 — 各パッチの`row`と`col`用の2つの独立したサイン波コード、連結。小さいPyTorch ViTにフィード、CIFAR-10で学習可能な位置埋め込みと精度を比較します。
3. **難しい。** 3層ViT(PyTorch)をビルド、1,000MNISTイメージで4×4パッチで訓練。テスト精度を測定。同じ1,000イメージでDINOv2事前学習を追加(簡略化：マスク済みパッチからパッチ埋め込みを予測するようにエンコーダを訓練)。精度が改善されていますか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| パッチ | 「vision-transformerトークン」 | `P × P × C`画像領域のピクセル値のフラットベクトル。 |
| パッチ化 | 「チョップ + フラット」 | 画像を重複しないパッチにスライス、各をベクトルにフラット。 |
| `[CLS]`トークン | 「画像要約」 | 先頭に加える学習可能なトークン；その最終埋め込みが画像表現。 |
| 誘導的バイアス | 「モデルが想定すること」 | ViTはCNNより少ないプリオを持つ；ギャップを埋めるためより多くのデータが必要。 |
| DINOv2 | 「自己教師ViT」 | 画像拡張 + モーメンタム教師を使用してラベルなしで訓練。2026年の最高の一般画像機能。 |
| SigLIP | 「CLIPの後継」 | ViT + テキストエンコーダ、シグモイドコントラッシブ損失で訓練；同等の計算でCLIPより優れている。 |
| Swin | 「ウィンドウ付きViT」 | ローカルアテンション + シフトウィンドウを持つ階層的ViT；部分二次。 |
| レジスタートークン | 「2023年トリック」 | いくつかの追加の学習可能なトークン、アテンションシンクを吸収；DINOv2機能を改善。 |

## 参考文献

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT論文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2のレジスタートークンフィックス。
