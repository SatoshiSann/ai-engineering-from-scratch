# ControlNet、LoRA & 条件付け

> テキストだけでは不器用な制御信号だ。ControlNetで事前訓練された拡散モデルをクローンして、デプスマップ、ポーズスケルトン、落書き、またはエッジ画像で操作できる。LoRAで2Bパラメータモデルを1000万パラメータ訓練することでファインチューニングできる。ともに彼らはStable Diffusionをおもちゃから2026年にあらゆる機関で配布される画像パイプラインに変えた。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 8 · 07 (潜在拡散)、Phase 10 (ゼロから作るLLM — LoRA基盤のため)
**所要時間:** 約75分

## 問題

「赤いドレスを着た女性が忙しい通りで犬を散歩させている」というプロンプトは、犬が*どこにいるのか*、女性が*どんなポーズなのか*、または街の*視点*について、モデルに情報を与えない。テキストは画像を指定するのに必要なもの約10%を確定するだけだ。残りは視覚的で、言葉で効率的に説明できない。

すべての信号（ポーズ、デプス、canny、セグメンテーション）の新しい条件付きモデルをゼロから訓練するのは、禁止的である。2.6Bパラメータ SDXLバックボーンを凍結し、条件付けを読み取り小さいサイドネットワークを接続し、バックボーンの中間特徴を押して欲しい。それがControlNetだ。

また、完全なモデルを再訓練することなく、モデルに新しいコンセプト（あなたの顔、あなたの製品、あなたのスタイル）を教えたい。100倍小さいデルタが欲しい。それがLoRA — 既存アテンション重みに接続する低ランク適応器だ。

ControlNet + LoRA + テキスト = 2026年の実務家のツールキット。ほとんどの本番画像パイプラインはSDXL / SD3 / Fluxベースの上に2-5LoRA、1-3ControlNet、およびIP-Adapterを重ねている。

## コンセプト

![ControlNetはエンコーダをクローン；LoRAは低ランク デルタを追加](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

事前訓練されたSDを取る。U-Netのエンコーダ半分を*クローン*する。元のものを凍結する。クローンを訓練して、追加の条件付け入力（エッジ、デプス、ポーズ）を受け入れるようにする。クローンを*ゼロ畳み込み*スキップ接続（ゼロで初期化された1×1 畳み込み — no-opとして開始、デルタを学習）で元のデコーダ半分に接続する。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

ゼロ畳み込み初期化は、ControlNetがアイデンティティとして開始することを意味する — 訓練前でも悪くない。標準拡散損失で1M（プロンプト、条件、画像）トリプルで訓練する。

モダリティごとのControlNetは小さいサイドモデルとして配布される（SDXL で約360M、SD 1.5で約70M）。推論時にそれらを構成できる：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

モデル内の任意の線形層`W ∈ R^{d×d}`に対して、`W`を凍結して低ランク デルタを追加：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

`r << d`とする。ランク4-16はアテンション標準、ランク64-128は重いファインチューン。新しいパラメータ数：`2 · d · r`の代わりに`d²`。SDXLアテンション`d=640`、`r=16`：アダプタあたり20kパラメータの代わりに410k — 20倍削減。モデル全体：LoRAは通常20-200MBベース5GBと比較。

推論では、LoRA：`W' = W + α · B @ A`をスケールできる。`α = 0.5-1.5`は通常である。複数LoRAs加算的にスタック（非線形方法で相互作用する通常の注意事項）。

### IP-Adapter (Ye et al., 2023)

*画像*を条件付けとして受け入れる小さいアダプタ（テキストと並行）。CLIP画像エンコーダを使用して画像トークンを生成、テキストトークンと並行してクロスアテンションに注入。ベースモデルあたり約20MB。「このリファレンスのスタイルで画像を生成」をLoRAなしで実行できる。

## 合成可能マトリックス

| ツール | 制御対象 | サイズ | いつ使用するか |
|------|------------------|------|-------------|
| ControlNet | 空間構造（ポーズ、デプス、エッジ） | 70-360MB | 正確なレイアウト、構成 |
| LoRA | スタイル、サブジェクト、コンセプト | 20-200MB | パーソナライゼーション、スタイル |
| IP-Adapter | リファレンス画像からのスタイルまたはサブジェクト | 20MB | テキストが外見を説明できない |
| テキスト反転 | 新しいトークンとしての単一コンセプト | 10KB | レガシー、主にLoRAに置き換え |
| DreamBooth | サブジェクト上の完全ファインチューン | 2-5GB | 強い身元、高計算 |
| T2I-Adapter | 軽いControlNetの代替 | 70MB | エッジデバイス、推論予算 |

ControlNet ≈ 空間。LoRA ≈ セマンティック。両方を使う。

## ビルドしよう

`code/main.py`は1-D上の2つのメカニズムをシミュレート：

1. **LoRA。** 事前訓練された線形層`W`。凍結。低ランク`B @ A`を訓練して`W + BA`がターゲット線形層と一致するようにする。`r = 1`がランク1修正を完全に学習するのに十分であることを示す。

2. **ControlNet-lite。** 「凍結ベース」予測器と追加信号を読み取る「サイドネットワーク」。サイドネットワーク出力は、ゼロで初期化される学習可能なスカラーで制御される（ゼロ畳み込みのバージョン）。訓練し、ゲートがランプアップするのを見る。

### ステップ1：LoRA数学

```python
def lora(W, A, B, x, alpha=1.0):
    # Wは凍結；A、Bは訓練可能な低ランク因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### ステップ2：ゼロ初期化サイドネットワーク

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gateはゼロに初期化される
h = base(x) + gated
```

ステップ0では、出力はベースと同じだ。早期訓練は`gate`をゆっくり更新 — 破壊的なドリフトなし。

## ピットフォール

- **LoRAsをオーバースケール。** `α = 2`または`α = 3`は「より強く」よくあるハック、過度にスタイル化した/壊れた出力を生成。`α ≤ 1.5`を保つ。
- **ControlNet重み競合。** 重み1.0でポーズControlNetと重み1.0でデプスControlNetを使用すると、通常オーバーシュート。重みの合計 ≈ 1.0は安全なデフォルト。
- **間違ったベース上のLoRA。** SDXL LoRAsはSD 1.5で静かに動作しない、アテンション寸法が一致しないため。Diffusers 0.30+で警告する。
- **テキスト反転ドリフト。** 1つのチェックポイントで訓練されたトークンは別のもので悪くドリフト。LoRAはより携帯可能。
- **LoRA重みマージおよびストレージ。** LoRAを基本モデルウェイトに焼き込むことができ、より速い推論（ランタイム追加なし）を得ることができるが、実行時に`α`をスケール機能を失う。両方のバージョンを保持。

## 使ってみよう

| ゴール | 2026パイプライン |
|------|---------------|
| ブランドのアートスタイルを再現 | 約30キュレートされた画像でランク32で訓練されたLoRA |
| 生成画像に顔を入れる | DreamBoothまたはLoRA + IP-Adapter-FaceID |
| 特定のポーズ+プロンプト | ControlNet-Openpose + SDXL + テキスト |
| デプス認識構成 | ControlNet-Depth + SD3 |
| リファレンス+プロンプト | IP-Adapter + テキスト |
| 正確なレイアウト | ControlNet-ScribbleまたはControlnet-Canny |
| 背景交換 | ControlNet-Seg + インペイント（レッスン09） |
| 高速1ステップスタイル | SDXL-Turbo上のLCM-LoRA |

## 配布しよう

`outputs/skill-sd-toolkit-composer.md`を保存する。スキルはタスク（入力資産：プロンプト、オプションリファレンス画像、オプションポーズ、オプションデプス、オプション落書き）を受け取り、出力ツールスタック、重み、および再現可能なシード プロトコル。

## 演習

1. **簡単。** `code/main.py`で、LoRAランク`r`を1から4に変動させる。LoRAが正確にランク2ターゲット デルタと一致するランクは何か？
2. **中程度。** 2つの別個LoRAsを2つのターゲット変換で訓練。一緒にロードしてそれらの加算的相互作用を示す。相互作用がいつ線形性を破りますか？
3. **難しい。** diffusersを使用してスタック：SDXL-base + Canny-ControlNet（重み0.8）+スタイルLoRA（α 0.8）+ IP-Adapter（重み0.6）。スタック重みが変化するときのFID vs プロンプト準拠のトレードオフを測定。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| ControlNet | 「空間制御」 | クローンエンコーダ+ゼロ畳み込みスキップ；条件付け画像を読む。 |
| ゼロ畳み込み | 「アイデンティティとして開始」 | ゼロで初期化された1×1畳み込み；ControlNetは no-op として開始。 |
| LoRA | 「低ランク適応器」 | `W + B @ A`、`r << d`；完全ファインチューンより100倍少ないパラメータ。 |
| ランクr | 「ノブ」 | LoRA圧縮；4-16は典型的、64+は重いパーソナライゼーション。 |
| α | 「LoRA強度」 | LoRA デルタの実行時スケール。 |
| IP-Adapter | 「リファレンス画像」 | CLIP画像トークン経由の小さい画像条件付けアダプタ。 |
| DreamBooth | 「完全サブジェクトファインチューン」 | サブジェクトの約30枚の画像で完全なモデルを訓練。 |
| テキスト反転 | 「新しいトークン」 | 新しい単語埋め込みのみを学習；レガシー、主に置き換え。 |

## 本番注：LoRA スワップ、ControlNet レーン、マルチテナント配信

実テキスト画像SaaSは同じベースチェックポイント上で数百のLoRAsおよび 1ダースControlNetを配信。配信問題はLLMマルチテナンシーによく似ている（本番文献はLLMケースの下で継続的バッチ処理とLoRAX / S-LoRA をカバー）：

- **ホットスワップLoRAs、マージしない。** `W' = W + α·B·A`をベースにマージすると、ステップごとの推論が約3-5%高速だが、`α`とベースを凍結。LoRAsをVRAMにランク r デルタとして保持；diffusersは`pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`をリクエストごとの活性化に公開。スワップコストは`2 · d · r · num_layers`ウェイト — MB スケール、サブ秒。
- **ControlNet を2番目のアテンション レーン として。** クローンエンコーダはベースと並行して実行。各重み1.0の2つのControlNets = 1つのマージパスではなく、ステップあたり2つの追加前進パス。バッチサイズヘッドルームは二次的に低下。アクティブControlNetあたり約1.5倍ステップコスト予算。
- **量子化LoRAsも。** ベース（レッスン07、Flux on 8GB）を量子化した場合、LoRA デルタも8ビットまたは4ビットに綺麗に量子化。QLoRA-style読み込みは、メモリを爆発させることなく、4ビットFluxベースの上に5-10LoRAsをスタックできる。

Fluxスペシフィック：NielsのFlux-on-8GBノートブックベースを4ビットに量子化；スタイルLoRA（`pipe.load_lora_weights("user/style-lora")`）をその量子化ベースの上に`weight_name="pytorch_lora_weights.safetensors"`で読み込むと、相変わらず機能。これは2026年にほとんどのSaaS機関が配布するレシピだ。

## 参考文献

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（元々LLM；拡散にポート）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNetの軽い代替。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参照パイプライン。
