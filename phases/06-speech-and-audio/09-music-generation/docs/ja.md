# 音楽生成 — MusicGen、Stable Audio、Suno、ライセンス地震

> 2026年音楽生成: Suno v5とUdio v4は商用で支配; MusicGen、Stable Audio Open、ACE-Stepはオープンソースをリード。技術的問題はほぼ解決。法的問題(Warner Music $500M和解、UMG和解)は2025-2026年分野を再形成。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 6 · 02 (スペクトログラム)、Phase 4 · 10 (拡散モデル)
**所要時間:** 約75分

## 問題

テキスト → 30秒から4分のミュージッククリップ、歌詞、ボーカル、構造を使用。3つのサブ問題:

1. **楽器生成。** 「ウォームキーとロー-フィヒップホップドラム」のようなテキスト → オーディオ。MusicGen、Stable Audio、AudioLDM。
2. **曲生成(ボーカル+歌詞付き)。** 「多雨テキサスナイトについての田舎曲」 → 完全な曲。Suno、Udio、YuE、ACE-Step。
3. **条件/制御可能。** 既存クリップを拡張、ブリッジを再生成、ジャンル交換、干渉分離、または修復。Udio's修復+干渉分離は2026年機能をマッチ。

## コンセプト

![音楽生成: トークン-LM対拡散、2026年モデルマップ](../assets/music-generation.svg)

### ニューラルコーデックトークンの上のトークンLM

Meta's **MusicGen** (2023、MIT)およびたくさんの派生物: テキスト/メロディ埋め込みで条件、自動回帰的にEnCodecトークン(32 kHz、4コードブック)を予測、EnCodecでデコード。300M - 3.3Bパラム。強いベースライン; 30秒を超えて闘争。

**ACE-Step** (オープンソース、4B XLは2026年4月リリース)は完全な曲歌詞条件生成のための拡張。オープンコミュニティの Sunoに最も近い。

### メルまたは潜在の上の拡散

**Stable Audio (2023)** と **Stable Audio Open (2024)**: 圧縮オーディオの潜在拡散。ループ、サウンドデザイン、環境テクスチャで優れている。構造化完全曲に優れていない。

**AudioLDM / AudioLDM2**: T2Iスタイル潜在拡散を通じたテキスト-オーディオ、音楽、音響効果、音声に一般化。

### ハイブリッド(本番) — Suno、Udio、Lyria

クローズドウェイト。可能性ARコーデックLM +拡散ベースボコーダー特殊化ボイス/ドラム/メロディ頭。Suno v5 (2026)はELO 1293品質リーダー。Udio v4は修復+干渉分離を追加(低音、ドラム、ボーカル個別ダウンロード)。

### 評価

- **FAD (Fréchet Audio Distance)。** VGGishまたはPANNs機能を使用して実オーディオ分布に対する生成の埋め込みレベルの距離。より低いが良い。MusicGenスモール: MusicCapsで4.5 FAD; SOTA ~3.0。
- **音楽性(主観)。** 人間の好み。Suno v5 ELO 1293リード。
- **テキストオーディオアライメント。** CLAPスコアプロンプト間と出力。
- **音楽性アーチファクト。** オフビート遷移、ボーカルフレーズドリフト、構造損失30秒を超えて。

## 2026年モデルマップ

| モデル | パラム | 長さ | ボーカル | ライセンス |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | いいえ | MIT |
| Stable Audio Open | 1.2B | 47 s | いいえ | Stabilityの非商用 |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | はい | Apache-2.0 |
| YuE | 7B | &gt; 2 min | はい、多言語 | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | はい、ELO 1293 | 商用 |
| Udio v4 (closed) | ? | 4 min | はい +干渉分離 | 商用 |
| Google Lyria 3 (closed) | ? | リアルタイム | はい | 商用 |
| MiniMax Music 2.5 | ? | 4 min | はい | 商用API |

## 法的景観(2025-2026)

- **Warner Music対Suno和解。** $500M。WMGはnowAI類似、音楽権、Sunoのユーザー生成トラック上の監督を持つ。Udoで類似UMG和解。
- **EU AI法** + **California SB 942**: AI生成音楽は開示される必要があります。
- **Riffusion / MusicGen** MITの下は法的コンプライアンスバゲージをしかし商用ボーカルのなし。

安全な出荷パターン:

1. 楽器のみ(MusicGen、Stable Audio Open、MIT/CC0出力)を生成。
2. 商用API(Suno、Udio、ElevenLabsミュージック)を使用、生成ごとのライセンス付き。
3. 所有またはライセンスカタログで訓練(ほとんどの企業はここで終わる)。
4. 透かし+メタデータで世代にタグ。

## ビルドしてみよう

### ステップ1: MusicGenで生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

3つのサイズ: `small` (300M、高速)、`medium` (1.5B)、`large` (3.3B)。スモールは「アイデアが着地したか」に十分。

### ステップ2: メロディー条件付け

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melodyは色画像を取り、メロディーを保存しながら、音色を交換。「この楽器をストリングカルテットとして与える」に有用。

### ステップ3: FAD評価

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish埋め込み距離を計算。ジャンルレベルの回帰テストに有用; 人間のリスナーの代替ではない。

### ステップ4: LLM音楽ワークフローに追加

レッスン7-8からのアイデアと結合:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用方法

| 目標 | スタック |
|------|-------|
| 楽器サウンドデザイン | Stable Audio Open |
| ゲーム/適応音楽 | Google Lyria リアルタイム(閉)。 |
| ボーカル付き完全曲(商用) | Suno v5 または Udio v4明示ライセンス付き |
| ボーカル付き完全曲(オープン) | ACE-Step XL または YuE |
| 短い広告ジングル | ハミングされた参照でのMusicGenメロディー条件付け |
| ミュージックビデオ背景 | MusicGen + Stable Video Diffusion |

## 2026年でもまだ出荷するピットフォール

- **著作権ロンダリングプロンプト。** 「テイラースウィフトのスタイルの曲」 — 商用Suno/Udioはこれらをフィルタリング、オープンモデルはしない。独自のフィルターリストを追加。
- **30秒を超えた繰り返し/ドリフト。** ARモデルループ。複数世代を交差フェード、またはACE-Stepを使用して構造的首尾一貫性用。
- **テンポドリフト。** モデルはBPMを離れる。プロンプトでBPMタグを使用し、librosasで`beat_track`でポストフィルター。
- **ボーカル理解度。** Sunoは優秀; オープンモデルはしばしば単語でmushy。歌詞が重要な場合は、商用APIまたはファインチューン使用。
- **モノ出力。** オープンモデルはモノを生成またはフェイクステレオ。適切なステレオ再構成(ezst、Cartesia'sステレオ拡散)でアップグレード。

## 出荷しよう

`outputs/skill-music-designer.md`として保存。モデル、ライセンス戦略、長さ/構造計画、音楽ジェン配備の開示メタデータを選択。

## 演習

1. **簡単。** `code/main.py`を実行。ASCII記号として「生成」弦進行+ドラムパターン - 音楽生成の漫画。好みならMIDIレンダラーを通して再生。
2. **中程度。** `audiocraft`をインストール、MusicGen-smallで4ジャンルプロンプト全体で10秒クリップを生成、参照ジャンルセットに対してFADを測定。
3. **難しい。** ACE-Step(またはMusicGen-melody)を使用して、異なる音色プロンプトで同じ調べの3つのバリアションを生成。プロンプト対CLAPスコア計算して調整を検証。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| FAD | オーディオFID | 本物対生成の埋め込み分布間のFrecharetの距離。 |
| 色画像 | メロディとしてのピッチ | 12次元フレームごとベクトル; メロディー条件付けへの入力。 |
| 干渉分離 | 楽器トラック | 分離されたバス/ドラム/ボーカル/メロディーWAVとして。 |
| 修復 | 再生セクション | マスク時間ウィンドウ; モデルは再生だけその。 |
| CLAP | テキストオーディオCLIP | 対照的オーディオテキスト埋め込み; テキスト-オーディオアライメント評価。 |
| EnCodec | 音楽コーデック | Meta'sニューラルコーデック使用MusicGen; 32 kHz、4コードブック。 |

## 参考文献

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — オープン自動回帰ベンチマーク。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — サウンドデザインデフォルト。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — オープン4B完全曲ジェネレータ、2026年4月。
- [Suno v5プラットフォームドック](https://suno.com) — 商用品質リーダー。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 潜在拡散音楽+音響効果用。
- [WMG-Suno和解カバレッジ](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025年11月先例。
