# Vision Transformer（ViT）

> 画像をパッチに切り分け、各パッチを単語として扱い、標準的なトランスフォーマーを実行する。振り返らない。

**タイプ:** 構築
**言語:** Python
**前提条件:** Phase 7 レッスン02（自己アテンション）、Phase 4 レッスン04（画像分類）
**所要時間:** 約45分

## 学習目標

- パッチ埋め込み、学習された位置埋め込み、クラストークン、トランスフォーマーエンコーダブロックをゼロから実装して最小限のViTを構築する
- DeiTとMAEが証明するまで、ViTが大規模な事前学習データを必要とすると考えられていた理由を説明する
- ViT、Swin、ConvNeXtをそれらのアーキテクチャの帰納バイアス（なし、ローカルウィンドウアテンション、畳み込みバックボーン）で比較する
- `timm` を使って小さなデータセットで事前学習済みViTをファインチューニングし、標準的な線形プローブ / ファインチューニングレシピを用いる

## 問題

10年間、畳み込みはコンピュータビジョンと同義だった。CNNは強い帰納バイアスを持っていた — 局所性、平行移動等変性 — 誰も置き換えられると思っていなかった。それからDosovitskiy ら（2020年）は、畳み込み機構をまったく持たない、フラット化された画像パッチに適用された普通のトランスフォーマーが、スケールでベストのCNNと同等かそれ以上の性能を示せることを示した。

注意点は「スケールで」だった。ImageNet-1kではViTはResNetに負けた。ImageNet-21kやJFT-300Mで事前学習してからImageNet-1kでファインチューニングするとViTが勝った。結論は、トランスフォーマーは有用な帰納バイアスを持たないが、十分なデータからそれを学習できるというものだった。後続の研究（DeiT、MAE、DINO）は、適切な訓練レシピ — 強力なデータ拡張、自己教師あり事前学習、蒸留 — があれば、ViTは小さなデータでもうまく訓練できることを示した。

2026年現在、純粋なCNNはエッジデバイスでは依然として競争力がある（ConvNeXtが最強）が、トランスフォーマーはその他すべてを支配している：セグメンテーション（Mask2Former、SegFormer）、物体検出（DETR、RT-DETR）、マルチモーダル（CLIP、SigLIP）、動画（VideoMAE、VJEPA）。ViTブロック構造が知っておくべきものだ。

## コンセプト

### パイプライン

```mermaid
flowchart LR
    IMG["画像<br/>(3, 224, 224)"] --> PATCH["パッチ埋め込み<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["フラット化して<br/>(196, 768)トークンに"]
    FLAT --> CAT["[CLS]トークンを<br/>先頭に追加"]
    CAT --> POS["学習された<br/>位置埋め込みを加算"]
    POS --> ENC["Nトランスフォーマー<br/>エンコーダブロック"]
    ENC --> CLS["[CLS]トークンの<br/>出力を取得"]
    CLS --> HEAD["MLP分類器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

7つのステップ。パッチ → トークン → アテンション → 分類器。すべてのバリアント（DeiT、Swin、ConvNeXt、MAE事前学習）は7つのうち1〜2つを変更し、残りはそのままにする。

### パッチ埋め込み

最初の畳み込みが秘密だ。カーネルサイズ16、ストライド16なので、224x224の画像は16x16パッチの14x14グリッドになり、各パッチは768次元の埋め込みに射影される。この単一の畳み込みはパッチ化と線形射影の両方を行う。

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196パッチ = 196トークン。各トークンの特徴量次元は768（ViT-B）、1024（ViT-L）、または1280（ViT-H）だ。

### クラストークン

シーケンスの先頭に追加された単一の学習ベクトル：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

Nトランスフォーマーブロックの後、`[CLS]` の出力がグローバルな画像表現だ。分類ヘッドはこの1つのベクトルだけを読む。

### 位置埋め込み

トランスフォーマーには組み込みの空間位置の概念がない。すべてのトークンに学習ベクトルを加算する：

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

埋め込みはモデルのパラメータだ；勾配ベースの訓練が2D画像構造に適応させる。正弦波2Dの代替も存在するが、実際にはほとんど使われない。

### トランスフォーマーエンコーダブロック

標準的。マルチヘッド自己アテンション、MLP、残差接続、pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16は12ヘッドの12ブロックをスタックして合計86Mパラメータを持つ。

### なぜpre-LNか

初期のトランスフォーマーはpost-LN（`x = LN(x + sublayer(x))`）を使用し、ウォームアップなしに6〜8層を超えて訓練することが難しかった。pre-LN（`x = x + sublayer(LN(x))`）はウォームアップなしでより深いネットワークを安定して訓練する。すべてのViTとすべての現代的なLLMはpre-LNを使用する。

### パッチサイズのトレードオフ

- 16x16パッチ -> 196トークン、標準。
- 32x32パッチ -> 49トークン、高速だが解像度が低い。
- 8x8パッチ -> 784トークン、細かいが2乗のアテンションコストがスケールしない。

パッチが大きい = トークンが少ない = 高速だが空間詳細が少ない。SwinV2は階層的ウィンドウで4x4パッチを使用する。

### DeiTのImageNet-1kでのViT訓練レシピ

オリジナルのViTはCNNを上回るためにJFT-300Mが必要だった。DeiT（Touvron ら、2020年）は4つの変更でImageNet-1kだけでViT-Bを81.8%トップ1まで訓練した：

1. 強いデータ拡張：RandAugment、Mixup、CutMix、Random Erasing。
2. Stochastic depth（訓練中にブロック全体をランダムにドロップ）。
3. Repeated augmentation（同じ画像をバッチごとに3回サンプリング）。
4. CNNティーチャーからの蒸留（オプション、さらに精度を上げる）。

すべての現代的なViT訓練レシピはDeiTから派生している。

### Swin対ConvNeXt

- **Swin**（Liu ら、2021年）— ウィンドウベースのアテンション。各ブロックはローカルウィンドウ内でアテンションを取り、交互のブロックがウィンドウをシフトしてウィンドウを超えた情報を混合する。アテンション演算子を保ちながらCNNのような局所性の帰納バイアスを取り戻す。
- **ConvNeXt**（Liu ら、2022年）— Swinのアーキテクチャ選択（深さ方向畳み込み、LayerNorm、GELU、逆ボトルネック）に合わせた再設計されたCNN。ギャップは「アテンション対畳み込み」ではなく「現代の訓練レシピ + アーキテクチャ」であることを示した。

2026年現在、ConvNeXt-V2とSwin-V2はどちらも本番品質だ；正しい選択は推論スタック（ConvNeXtはエッジ向けのコンパイルが良い）と事前学習コーパスに依存する。

### MAE事前学習

Masked Autoencoder（He ら、2022年）：ランダムに75%のパッチをマスク、見えている25%だけを処理するようにエンコーダを訓練し、エンコーダの出力からマスクされたパッチを再構築するために小さなデコーダを訓練する。事前学習後、デコーダを捨ててエンコーダをファインチューニングする。

MAEはViTをImageNet-1kだけで訓練可能にし、SOTAに達し、現在のデフォルトの自己教師ありレシピだ。

## 実装

### ステップ1：パッチ埋め込み

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

1つの畳み込み、1つのフラット化、1つの転置。これが画像→トークンのステップ全体だ。

### ステップ2：トランスフォーマーブロック

pre-LN、マルチヘッド自己アテンション、GELUを使ったMLP、残差接続。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` はヘッドへの分割、スケール付きドット積、出力射影を処理する。`batch_first=True` なので形状は `(N, seq, dim)` だ。

### ステップ3：ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

約280万パラメータ — CPUで扱いやすい小さなViT。実際のViT-Bは86M；同じクラス定義で `dim=768, depth=12, num_heads=12`。

### ステップ4：健全性チェック — 単一画像推論

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

エラーなしで実行されるべきだ。確率の和は1。

## 活用

`timm` はすべてのViTバリアントとImageNet事前学習済み重みを提供する。1行で：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` は2026年の視覚トランスフォーマーの本番デフォルトだ。ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormerなどを同じAPIでサポートする。

マルチモーダル作業（画像+テキスト）には、`transformers` がCLIP、SigLIP、BLIP-2、LLaVAを提供する。それらすべての画像エンコーダはViTバリアントだ。

## 成果物

このレッスンの成果物：

- `outputs/prompt-vit-vs-cnn-picker.md` — データセットサイズ、計算量、推論スタックに基づいてViT、ConvNeXt、Swinの中から選ぶプロンプト。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — ViTのパッチ埋め込みと位置埋め込みの形状がモデルの期待するシーケンス長と一致するかを確認し、最も一般的な移植バグを検出するスキル。

## 演習

1. **（簡単）** 上記の小さなViTを通じた順伝播のすべての中間テンソルの形状をプリントする。確認：入力 `(N, 3, 64, 64)` -> パッチ `(N, 16, 192)` -> CLSあり `(N, 17, 192)` -> 分類器入力 `(N, 192)` -> 出力 `(N, num_classes)`。
2. **（中級）** 事前学習済み `timm` ViT-S/16をレッスン4の合成CIFARデータセットでファインチューニングする。同じデータでのResNet-18ファインチューニングと比較する。訓練時間と最終精度を報告する。
3. **（上級）** 小さなViT用のMAE事前学習を実装する：75%のパッチをマスク、マスクされたパッチを再構築するためにエンコーダ+小さなデコーダを訓練する。事前学習前後で合成データの線形プローブ精度を評価する。

## 用語集

| 用語 | 人々が言うこと | 実際の意味 |
|------|----------------|------------|
| パッチ埋め込み | "最初の畳み込み" | カーネルサイズ = ストライド = パッチサイズの畳み込み；画像をトークン埋め込みのグリッドに変換する |
| クラストークン | "[CLS]" | トークンシーケンスの先頭に追加される学習ベクトル；その最終出力がグローバルな画像表現 |
| 位置埋め込み | "学習された位置" | すべてのトークンに加算される学習ベクトル；トランスフォーマーが各パッチの由来を知ることができる |
| pre-LN | "サブレイヤー前のLayerNorm" | 安定したトランスフォーマーバリアント：`LN(x + sublayer(x))` の代わりに `x + sublayer(LN(x))` |
| マルチヘッドアテンション | "並列アテンション" | num_headsの独立したサブスペースに分割した標準的なトランスフォーマーアテンション、後で連結 |
| ViT-B/16 | "ベース、パッチ16" | 標準サイズ：dim=768、depth=12、heads=12、patch_size=16、image=224；約86Mパラメータ |
| DeiT | "データ効率ViT" | 強いデータ拡張でImageNet-1kだけで訓練されたViT；大規模事前学習データセットが厳密に必要ではないことを証明 |
| MAE | "マスク付きオートエンコーダ" | 自己教師あり事前学習：75%のパッチをマスク、再構築する；主要なViT事前学習レシピ |

## 参考文献

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT論文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — ImageNet-1kだけでViTを訓練する方法
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE事前学習
- [timm documentation](https://huggingface.co/docs/timm) — 本番で使うすべての視覚トランスフォーマーのリファレンス
