# ゼロから学ぶ3D Gaussian Splatting

> シーンは数百万個の3DガウシアンのクラウドだŸ。それぞれが位置、向き、スケール、不透明度、そして視点方向に依存する色を持つ。それらをラスタライズし、ラスタライゼーションを通じてバックプロパゲーションする。完成だ。

**タイプ:** 構築
**言語:** Python
**前提条件:** Phase 4 レッスン 13 (3DビジョンとネRF)、Phase 1 レッスン 12 (テンソル演算)、Phase 4 レッスン 10 (拡散の基礎 任意)
**所要時間:** 約90分

## 学習目標

- 2026年の本番フォトリアリスティック3D再構成において、3D Gaussian SplattingがNeRFに替わってデフォルトになった理由を説明できる
- ガウシアンごとの6種類のパラメータ（位置、回転クォータニオン、スケール、不透明度、球面調和係数色、任意の特徴量）と各パラメータが何floatを使うかを述べることができる
- `alpha`合成を使って2Dガウシアンスプラッティングラスタライザをゼロから実装し、3Dの場合が同じループに射影されることを示せる
- `nerfstudio`、`gsplat`、または`SuperSplat`を使って20〜50枚の写真からシーンを再構成し、`KHR_gaussian_splatting` glTF拡張またはOpenUSD 26.03の`UsdVolParticleField3DGaussianSplat`スキーマにエクスポートできる

## 問題

NeRFはシーンをMLPの重みとして保存する。すべてのレンダリングピクセルはレイに沿った数百回のMLPクエリだ。学習に数時間かかり、レンダリングに数秒かかり、重みは編集できない — シーン内の椅子を動かしたければ、再学習が必要だ。

3D Gaussian Splatting（Kerbl、Kopanas、Leimkühler、Drettakis、SIGGRAPH 2023）はそれをすべて置き換えた。シーンは明示的な3Dガウシアンの集合だ。レンダリングは100fps以上のGPUラスタライゼーションだ。学習は数分で完了する。編集は直接できる：ガウシアンのサブセットを平行移動すれば椅子が動く。2026年までにKhronos Groupはガウシアンスプラット用のglTF拡張を批准し、OpenUSD 26.03はガウシアンスプラットスキーマを搭載し、ZillowとApartments.comはそれで不動産をレンダリングし、3D再構成に関するほとんどの新しい研究論文はコア3DGSアイデアのバリアントだ。

メンタルモデルはシンプルだが、数学には十分な動く部品があり、ほとんどの入門はラスタライゼーションから始まり、射影と球面調和をスキップする。このレッスンはすべてを構築する — 最初に2Dバージョン、次に3D拡張。

## コンセプト

### ガウシアンが持つもの

一つの3Dガウシアンは空間内のパラメトリックなブロブで、次の属性を持つ：

```
position         mu         (3,)    世界座標での中心
rotation         q          (4,)    向きをエンコードする単位クォータニオン
scale            s          (3,)    軸ごとの対数スケール（レンダリング時に指数化）
opacity          alpha      (1,)    ポストシグモイド不透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   視点依存色
```

回転 + スケールで3x3共分散を構築する：`Sigma = R S S^T R^T`。それが3DでのガウシアンのShapeだ。球面調和は色が視点方向によって変わるようにする — 鏡面ハイライト、微妙な光沢、視点依存の輝き — 視点ごとのテクスチャを保存せずに。SH次数3では色チャンネルごとに16係数、ガウシアンごとに色だけで48float。

シーンには通常100〜500万個のガウシアンがある。それぞれが約60float（3 + 4 + 3 + 1 + 48 + misc）を保存する。500万ガウシアンのシーンで240MB — ポイントごとのテクスチャを持つ同等のポイントクラウドよりはるかに小さく、高解像度で再レンダリングされたNeRFのMLP重みより桁違いに小さい。

### ラスタライゼーション、レイマーチングではない

```mermaid
flowchart LR
    SCENE["数百万の3Dガウシアン<br/>(位置、回転、スケール、<br/>不透明度、SH色)"] --> PROJ["2Dに射影<br/>(カメラ外部 + 内部パラメータ)"]
    PROJ --> TILES["タイルに割り当て<br/>(16x16スクリーン空間)"]
    TILES --> SORT["タイルごとの<br/>深度ソート"]
    SORT --> ALPHA["手前から奥への<br/>アルファ合成"]
    ALPHA --> PIX["ピクセルの色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五つのステップ、すべてGPUフレンドリー。ピクセルごとのMLPクエリなし。単一のRTX 3080 TiがXX万スプラットを147fpsでレンダリングする。

### 射影ステップ

世界位置`mu`と3D共分散`Sigma`を持つ3Dガウシアンは、スクリーン位置`mu'`と2D共分散`Sigma'`を持つ2Dガウシアンに射影される：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

2Dガウシアンのフットプリントは、`Sigma'`の固有ベクトルが軸である楕円だ。その楕円内のすべてのピクセルは、`exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`で重み付けされたガウシアンの寄与を受け取る。

### アルファ合成ルール

一つのピクセルに対して、それをカバーするガウシアンは前から後ろにソートされる（または逆の公式で後ろから前）。色は1980年代以来のすべての半透明ラスタライザと同じ方程式で合成される：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       i までの透過率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局所的な寄与
c_i = eval_SH(SH_i, view_direction)    視点依存色
```

これは**NeRFのボリュームレンダリングと同じ方程式**で、レイに沿った密なサンプルの代わりに明示的な疎なガウシアンの集合上で実行される。このアイデンティティがなぜレンダリング品質がNeRFに匹敵するかの理由だ — 両方とも同じ輝度場方程式を積分している。

### なぜこれが微分可能なのか

すべてのステップ — 射影、タイル割り当て、アルファ合成、SH評価 — はガウシアンパラメータに対して微分可能だ。正解画像が与えられたとき、レンダリングされたピクセル損失を計算し、ラスタライザを通じてバックプロパゲーションし、勾配降下法ですべての`(mu, q, s, alpha, c_lm)`を更新する。約30,000イテレーションで、ガウシアンは正しい位置、スケール、色を見つける。

### 密度化と枝刈り

固定されたガウシアンの集合は複雑なシーンをカバーできない。学習には二つの適応的メカニズムが含まれる：

- **クローン**：現在の位置でガウシアンの勾配の大きさが高いがスケールが小さいとき — 再構成はここでより多くの詳細を必要としている。
- **分割**：スケールの大きいガウシアンの勾配が高いとき、それを二つの小さいものに分割する — 一つの大きなガウシアンはその領域に対して滑らかすぎる。
- **枝刈り**：不透明度が閾値以下に下がったガウシアンを削除する — それらは寄与していない。

密度化はNイテレーションごとに実行される。シーンは通常、（SfMポイントからシードされた）約10万の初期ガウシアンから学習終了時の100〜500万まで成長する。

### 球面調和を一段落で

視点依存色は単位球上の関数`c(direction)`だ。球面調和は球のフーリエ基底だ。次数`L`で打ち切ると、チャンネルごとに`(L+1)^2`の基底関数が得られる。新しい視点の色を評価するのは、学習済みSH係数と視点方向で評価された基底の内積だ。次数0 = 1係数 = 定数色。次数3 = 16係数 = ランベルト陰影、鏡面、軽度の反射を捉えるのに十分。SD Gaussian Splattingの論文はデフォルトで次数3を使用する。

### 2026年の本番スタック

```
1. キャプチャ         スマートフォン / DJIドローン / ハンドヘルドスキャナ
2. SfM / MVS       COLMAPまたはGLOMAPがカメラポーズ + 疎な点を導出
3. 3DGS学習        nerfstudio / gsplat / inria official / PostShot（RTX 4090で約10〜30分）
4. 編集            SuperSplat / SplatForge（フローターのクリーニング、セグメント）
5. エクスポート      .ply -> glTF KHR_gaussian_splatting または .usd（OpenUSD 26.03）
6. 閲覧            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4Dと生成バリアント

- **4D Gaussian Splatting** — ガウシアンが時間の関数；ボリュームビデオに使用（Superman 2026、A$AP Rockyの「Helicopter」）。
- **生成スプラット** — シーン全体を幻覚するテキストからスプラットへのモデル（World LabsのMarble）。
- **3D Gaussian Unscented Transform** — 自動運転シミュレーション用のNVIDIA NuRecのバリアント。

## 構築

### ステップ1：2Dガウシアン

まず2Dラスタライザを構築する。3Dの場合は射影後に同じものに還元される。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum`はすべての（ガウシアン、ピクセル）ペアに対して二次形式`diff^T Sigma^-1 diff`を実行する。

### ステップ2：2Dスプラッティングラスタライザ

手前から奥へのアルファ合成。2Dでは深度は意味がないので、順序付けのために学習済みのガウシアンごとのスカラーを使用する。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

高速ではない — 実際の実装はタイルベースのCUDAカーネルを使用する — が、正確な数学で完全に微分可能だ。

### ステップ3：学習可能な2Dスプラットシーン

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit`、`colour_logits`はすべて無制約パラメータで、レンダリング時に適切な活性化関数を通じてマッピングされる。これはすべての3DGS実装の標準パターンだ。

### ステップ4：ターゲット画像に2Dガウシアンをフィット

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

200ステップで64個のガウシアンは二つの形に落ち着く。それがすべてのアイデアだ — 明示的な幾何学的プリミティブ上の勾配降下法。

### ステップ5：2Dから3Dへ

3Dの拡張は同じループを保持する。追加要素：

1. ガウシアンごとの回転は単一の角度ではなくクォータニオンになる。
2. 共分散は`R S S^T R^T`で、`R`はクォータニオンから構築され、`S = diag(exp(log_scale))`。
3. 射影`(mu, Sigma) -> (mu', Sigma')`はカメラ外部パラメータと`mu`での透視射影のヤコビアンを使用する。
4. 色は球面調和展開になる；視点方向で評価する。
5. 深度ソートは学習済みスカラーではなく実際のカメラ空間のzから行う。

すべての本番実装（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）は、タイルベースのCUDAカーネルを使ってGPUでこれを正確に実行する。

### ステップ6：球面調和評価

次数3までのSH基底は一チャンネルあたり16項を持つ。評価：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

学習済み`sh_coeffs`はそのガウシアンの「あらゆる方向の色」を保存する。レンダリング時に現在の視点方向に対して評価し、3ベクトルのRGBを得る。

## 活用

実際の3DGS作業には、`gsplat`（Meta）または`nerfstudio`を使用する：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto`はnerfstudioの3DGSトレーナーだ。典型的なシーンでRTX 4090で10〜30分かかる。

2026年で重要なエクスポートオプション：

- `.ply` — 生ガウシアンクラウド（ポータブル、最大ファイル）。
- `.splat` — PlayCanvas / SuperSplatの量子化フォーマット。
- glTF `KHR_gaussian_splatting` — Khronosスタンダード、ビューア間でポータブル（2026年2月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USDネイティブ、NVIDIA OmniverseとVision Proパイプライン用。

4D/動的シーンには、`4DGS`と`Deformable-3DGS`が時間変化する平均と不透明度で同じ機構を拡張する。

## 成果物

このレッスンで生成されるもの：

- `outputs/prompt-3dgs-capture-planner.md` — 特定のシーンタイプに対してキャプチャセッション（写真の枚数、カメラパス、照明）を計画するプロンプト。
- `outputs/skill-3dgs-export-router.md` — 下流のビューアまたはエンジンが与えられたとき、適切なエクスポートフォーマット（`.ply` / `.splat` / glTF / USD）を選択するスキル。

## 演習

1. **(易)** 上記の2Dスプラットトレーナーを別の合成画像で実行する。`num_splats`を`[16, 64, 256]`で変えて、それぞれのMSEとステップをプロットする。収益逓減の点を特定する。
2. **(中)** 2Dラスタライザを拡張して、スカラーの「視点角度」を通じた次数2の調和関数によるガウシアンごとのRGB色をサポートする。二つのターゲット画像のペアで学習させ、モデルが両方を再構成することを確認する。
3. **(難)** `nerfstudio`をクローンし、自分が持つシーン（机、植物、顔、部屋）の20枚の写真で`splatfacto`を学習させる。glTF `KHR_gaussian_splatting`にエクスポートし、ビューア（Three.jsの`GaussianSplats3D`、SuperSplat、Babylon.js V9）で開く。学習時間、ガウシアン数、レンダリングfpsを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| 3DGS | 「ガウシアンスプラット」 | 数百万の3Dガウシアンとしての明示的なシーン表現、ガウシアンごとの位置、回転、スケール、不透明度、SH色付き |
| 共分散 | 「ガウシアンの形状」 | `Sigma = R S S^T R^T`；一つのガウシアンの向きと異方性スケール |
| アルファ合成 | 「後ろから前へブレンド」 | NeRFのボリュームレンダリングと同じ方程式、今や明示的な疎な集合上で |
| 密度化 | 「クローンと分割」 | 再構成がアンダーフィットの場所への新しいガウシアンの適応的追加 |
| 枝刈り | 「低不透明度を削除」 | 学習中にほぼゼロ不透明度に崩壊したガウシアンを削除する |
| 球面調和 | 「視点依存色」 | 球上のフーリエ基底；視点方向の関数として色を保存する |
| Splatfacto | 「nerfstudioの3DGS」 | 2026年の3DGS学習の最も簡単なパス |
| `KHR_gaussian_splatting` | 「glTF標準」 | 3DGSをビューアとエンジン間でポータブルにするKhronos 2026拡張 |

## 参考文献

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 元の論文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 本番品質のCUDAラスタライザ
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — リファレンス学習レシピ
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026年のポータブルフォーマット
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat`スキーマ
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 業界概観
