# 評価 — FID、CLIP スコア、人間の好み

> すべての生成モデル リーダーボードはFID、CLIPスコア、および人間の好み競技場からの勝率をを引用。各数値は決定した研究員がゲームできる失敗モード。失敗モードを知らない場合、実改善と実行ゲーミングを区別できない。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 8 · 01 (タクソノミー)、Phase 2 · 04 (評価メトリック)
**所要時間:** 約45分

## 問題

生成モデルは*サンプル品質*および*条件付け準拠*で判断。どちらも閉形式測定ない。モデルは10,000イメージをレンダリングする；何かそれらに数字を割り当てる必要がある；モデルファミリーで、解像度で、アーキテクチャで信頼番号が必要。3つのメトリック 2014-2026にまで生き残った：

- **FID（Fréchet Inception距離）。** 2つの分布 — 実と生成 — Inception ネットワークの特徴空間の距離。低くがより良い。
- **CLIPスコア。** 生成画像のCLIP-イメージ埋め込みとプロンプトのCLIP-テキスト埋め込みのコサイン類似性。高くがより良い。プロンプト準拠を測定。
- **人間の好み。** 2つのモデルを同じプロンプト上で頭-ツー-ヘッド、人間（またはGPT-4クラスモデル）より良い1つを選ぶ、Elo スコアに勝利を集約。

また見るだろう：IS (inception スコア、主に引退)、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。各前のもの1つの失敗を修正。

## コンセプト

![FID、CLIP、および好み：3つの軸、異なる失敗モード](../assets/evaluation.svg)

### FID — サンプル品質

Heusel et al. (2017)。ステップ：

1. Inception-v3特徴（2048-D）をN実イメージと N 生成から抽出。
2. ガウシアンを各プール経由フィット：平均`μ_r、μ_g`と共分散`Σ_r、Σ_g`を計算。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解釈：特徴空間の2つの多変量ガウシアン間のFréchet距離。低 = より相似分布。

失敗モード：
- **小さいNで偏る。** FIDは特徴分布と平均二乗 — 小さいN共分散を過小推定、誤って低FID与える。常に N≥10,000を使用。
- **Inception依存。** Inception-v3はImageNetで訓練。ImageNetから遠いドメイン（顔、アート、テキスト画像）は無意味なFIDを生成。ドメイン特定特徴抽出器を使用。
- **ゲーミング。** Inception先行に過度に適合はFID低を与える視覚品質改善なし。CMMD（下）でそれを打つ。

### CLIPスコア — プロンプト準拠

Radford et al. (2021)。生成画像+プロンプト：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

30kを越えて生成画像を平均 → モデル間で比較可能なスカラー。

失敗モード：
- **CLIPのその盲点。** CLIPは弱い構成的推論を持つ（「青い球上の赤キューブ」はしばしば失敗）。モデルは複雑なプロンプト実際フォローなしにCLIPスコアで良好なランク。
- **短いプロンプトバイアス。** 短いプロンプトはより多くのCLIP-イメージ野生マッチを持つ。長いプロンプトは機械的にCLIPスコアが低くなる。
- **プロンプトゲーミング。** 「高品質、4k、傑作」をプロンプトに含むとCLIPスコアを膨らませ、改善画像-テキストバインディングなし。

CMMD（Jayasumana et al., 2024）のうち一部固定：Inception代わりにCLIP特徴使用、Fréchetの代わりに最大-平均不一致。FIDより品質違い検出により良い。

### 人間の好み — 地面真実

プロンプトのプール選ぶ。モデルA およびモデルB で生成。ペアを人間（またはストロングLLM判定）に見せる。勝利を Elo またはBradley-Terry スコアに集約。ベンチマーク：

- **PartiPrompts (Google)**: 1,600多様プロンプト、12カテゴリー。
- **HPSv2**: 107k人間アノテーション、広く使用自動プロキシとして。
- **ImageReward**: 137kプロンプト画像好み対、MITライセンス。
- **PickScore**: Pick-a-Pic 2.6M好みで訓練。
- **Chatbot-Arena-スタイル画像競技場**: https://imagearena.ai/ および他の。

失敗モード：
- **判定分散。** 非専門家は専門家と異なる好みを持つ。両方を使用。
- **プロンプト分布。** チェリーピック プロンプトは1つのファミリーを好む。常に文書。
- **LLM-判定報酬ハッキング。** GPT-4-判定は美しいが間違った出力でだまされる。人間と三角測定。

## 一緒に使用

本番評価報告はを含むべき：

1. FID保有外リアル分布に対して10-30kサンプルで（サンプル品質）。
2. CLIPスコア/ CMMD同じサンプル vs それらのプロンプト（準拠）。
3. 前のモデルに対する視認不可競技場の勝率（全体的好み）。
4. 失敗モード分析：50ランダムサンプル出力、既知問題でフラグ（手解剖学、テキストレンダリング、一貫したオブジェクト数）。

いかなる単一メトリックは嘘。3つの相乗メトリック+定性的レビューは主張。

## ビルドしよう

`code/main.py`合成「特徴ベクトル」上でFID、CLIPスコアのような、およびElo集約を実装（わたしたちは4-D ベクトルを Inception 特徴の代わりに使用）。見る：

- 小さいNおよび大きいN上でのFID計算 — バイアス。
- 「CLIPスコア」として特徴プール間のコサイン類似性。
- 合成好み ストリーム から Elo 更新ルール。

### ステップ1：4行でFID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### ステップ2：CLIPスタイルコサイン類似性

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### ステップ3：Elo 集約

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## ピットフォール

- **N=1000でのFID。** ヒューリスティック N<10kで信頼できない。低N FIDを報告する論文はゲーミング。
- **解像度間でFIDを比較。** Inceptionの299×299リサイズは特徴分布を変更。マッチ解像度のみで比較。
- **単一シードを報告。** 最小3つシードを実行。標準を報告。
- **負プロンプトによるCLIPスコアインフレーション。** ある種のパイプラインはプロンプトに過度に適合することでCLIPを後押し。視覚的飽和をチェック。
- **プロンプト重複からのEloバイアス。** 両方のモデルがベンチマーク プロンプトで訓練を見た場合、Eloは無意味。保有外プロンプトセットを使用。
- **人間の評価有料群衆スキュー。** Prolific、MTurkアノテータは若い/技術に優しい方へスキュー。採用アート/デザイン専門家と混在。

## 使ってみよう

2026年本番評価プロトコル：

| 柱 | 最小 | 推奨 |
|--------|---------|-------------|
| サンプル品質 | 保有外リアルで10kでのFID | + 5kで CMMD + カテゴリーあたりサブセット でFID |
| プロンプト準拠 | 30kでCLIPスコア | + HPSv2 + ImageReward + VQAスタイル質問回答 |
| 好み | 200視認不可ペア vs ベースライン | + 2000ペア人間+ LLM-判定+ Chatbot Arena |
| 失敗分析 | 50手フラグ | 500手フラグ+自動安全分類 |

すべて4柱が1つの報告 = 主張。独力いかなる1つ = マーケティング。

## 配布しよう

`outputs/skill-eval-report.md`を保存する。スキルはチェックポイント + ベースラインをあたえ、出力完全評価計画：サンプルサイズ、メトリック、失敗-モードプローブ、署名-オフ基準。

## 演習

1. **簡単。** `code/main.py`を実行。N=100 vs N=1000でそれぞれの合成分布でFIDを比較。バイアス大きさを報告。
2. **中程度。** 合成CLIPスタイル特徴からCMMDを実装（CMMD式を、Jayasumana et al.、2024参照）。感度と比較品質違いを測定 vs FID。
3. **難しい。** HPSv2セットアップを複製：Pick-a-Picサブセットから1000画像-プロンプト対、好みで小さいCLIPベース スコア ファインチューン、保有外セット上の同意を測定。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| FID | 「Fréchet Inception 距離」 | Fréchet 距離 Gaussian フィット 実 vs gen Inception 特徴。 |
| CLIPスコア | 「テキスト-イメージ類似性」 | CLIP イメージとテキスト埋め込み間のコサイン類似性。 |
| CMMD | 「FIDの置き換え」 | CLIP特徴MMD；少ないバイアス、ガウシアン仮説ない。 |
| IS | 「Inception スコア」 | Exp KL(p(y|x) || p(y))；現代モデルで貧相に相関、引退。 |
| HPSv2 / ImageReward / PickScore | 「学習好み プロキシ」 | 人間の好みで訓練小さいモデル；自動判定として使用。 |
| Elo | 「チェス評価」 | ペアごとの勝利の Bradley-Terry 集約。 |
| PartiPrompts | 「ベンチマーク プロンプト セット」 | 12カテゴリ全体 1,600 Google キュレート プロンプト。 |
| FD-DINO | 「自己監督置き換え」 | FD DINOv2特徴使用；ImageNet-ドメイン外に優れる。 |

## 本番注：評価は推論ワークロードでもある

10kサンプルでFIDを実行は10kイメージ生成を意味。50ステップSDXL ベース1024²で単一L4では、約11時間単一リクエスト推論。評価予算は実、およびフレーミングはまさにオフライン推論シナリオ（スループット最大化、TTFTを無視）：

- **ハード バッチ、レイテンシ忘れ。** オフライン評価 = 静的バッチング最大サイズが メモリ フィット。`pipe(...).images` with `num_images_per_prompt=8` 80GB H100では、単一リクエストより4-6倍高速ウォール時計。
- **リアル特徴をキャッシュ。** Inception（FID）またはCLIP（CLIPスコア、CMMD）特徴抽出リアル参照セット全体は*1度*実行、`.npz`として保存。評価ごと再計算しない。

CI /回帰ゲート：500-サンプルサブセットでFID+CLIPスコアを実行/PR（約30分）；完全10k FID+HPSv2+Elo夜間を実行。

## 参考文献

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID紙。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失敗-モード調査。
