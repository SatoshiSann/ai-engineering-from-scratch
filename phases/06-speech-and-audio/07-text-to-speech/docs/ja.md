# テキスト読み上げ (TTS) — TacotronからF5、Kokoroまで

> ASRは音声をテキストに逆変換する。TTSはテキストを音声に逆変換する。2026年のスタックは3つの部分から成る: テキスト → トークン、トークン → メル、メル → 波形。各部分にはラップトップに収まるデフォルトモデルがある。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 02 (Spectrograms & Mel)、Phase 5 · 09 (Seq2Seq)、Phase 7 · 05 (Full Transformer)
**所要時間:** 約75分

## 問題

文字列がある。「Please remind me to water the plants at 6 pm.」あなたは、自然に聞こえ、正しいプロソディ(ポーズ、強勢)を持ち、「plants」を正しい母音で発音し、ライブのボイスアシスタント向けにCPU上で300 ms未満で動く3秒のオーディオクリップが必要だ。さらに声を入れ替え、コードスイッチ入力(「remind me at 6 pm, daijoubu?」)を扱い、名前で恥をかかないようにする必要もある。

現代のTTSパイプラインは次のように見える:

1. **テキストフロントエンド。** テキストを正規化し(日付、数字、メール)、音素またはサブワードトークンに変換し、プロソディ特徴を予測する。
2. **音響モデル。** テキスト → メルスペクトログラム。Tacotron 2 (2017)、FastSpeech 2 (2020)、VITS (2021)、F5-TTS (2024)、Kokoro (2024)。
3. **ボコーダ。** メル → 波形。WaveNet (2016)、WaveRNN、HiFi-GAN (2020)、BigVGAN (2022)、2024年以降のニューラルコーデックボコーダ。

2026年には、音響 + ボコーダの分割は、エンドツーエンドの拡散モデルとフローマッチングモデルでぼやけてくる。しかし3部構成のメンタルモデルは、デバッグのために依然として有効だ。

## コンセプト

![Tacotron, FastSpeech, VITS, F5/Kokoroの並列比較](../assets/tts.svg)

**Tacotron 2 (2017)。** Seq2seq: 文字埋め込み → BiLSTMエンコーダ → 位置感応アテンション → 自己回帰LSTMデコーダがメルフレームを出力する。遅い(AR)、長いテキストで不安定。今もベースラインとして引用される。

**FastSpeech 2 (2020)。** 非自己回帰的。継続時間予測器が各音素が得るメルフレーム数を出力する。1パス、Tacotronより10倍高速。一部の自然さ(単調アラインメント)を失うが、どこでも出荷される。

**VITS (2021)。** エンコーダ + フローベースの継続時間 + HiFi-GANボコーダを、変分推論でエンドツーエンドに同時訓練する。高品質、単一モデル。2022〜2024年の支配的なオープンソースTTS。バリアント: YourTTS(マルチ話者ゼロショット)、XTTS v2(2024、Coqui)。

**F5-TTS (2024)。** フローマッチング上の拡散transformer。自然なプロソディ、5秒の参照オーディオによるゼロショット音声クローニング。2026年のオープンソースTTSリーダーボードのトップ。335Mパラメータ。

**Kokoro (2024)。** 小型(82M)、CPU実行可能、リアルタイム用途で最高クラスの英語TTS。クローズド語彙の英語のみ、apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商用の最先端。ElevenLabs v2.5の感情タグ(「[whispered]」、「[laughing]」)とキャラクターボイスは、2026年のオーディオブック制作を支配している。

### ボコーダの進化

| 時代 | ボコーダ | レイテンシ | 品質 |
|-----|---------|---------|---------|
| 2016 | WaveNet | オフラインのみ | リリース時SOTA |
| 2018 | WaveRNN | 約リアルタイム | 良好 |
| 2020 | HiFi-GAN | 100× リアルタイム | ほぼ人間 |
| 2022 | BigVGAN | 50× リアルタイム | 話者/言語をまたいで汎化 |
| 2024 | SNAC, DAC (ニューラルコーデック) | ARモデルと統合 | 離散トークン、ビット効率的 |

2026年までに、ほとんどの「TTS」モデルはテキストから波形までエンドツーエンドだ。メルスペクトログラムは内部表現となる。

### 評価

- **MOS (Mean Opinion Score)。** 1〜5のスケール、クラウドソース。今もゴールドスタンダード。痛々しく遅い。
- **CMOS (Comparative MOS)。** A対Bの選好。アノテーションごとにより狭い信頼区間。
- **UTMOS、DNSMOS。** 参照なしのニューラルMOS予測器。リーダーボードに使われる。
- **ASR経由のCER (Character Error Rate)。** TTS出力をWhisperに通し、入力テキストに対してCERを計算する。明瞭度の代理指標。
- **SECS (Speaker Embedding Cosine Similarity)。** 音声クローニングの品質。

LibriTTS test-cleanでの2026年の数値:

| モデル | UTMOS | CER (Whisper経由) | サイズ |
|-------|-------|-------------------|------|
| グラウンドトゥルース | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 作ってみる

### ステップ1: 入力を音素化する

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素は普遍的な橋渡しだ。VITSレベルの品質を下回るものには生のテキストを供給しないこと。

### ステップ2: Kokoroを実行する(2026年のCPUデフォルト)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

オフラインで動く、単一ファイル、82Mパラメータ。

### ステップ3: 音声クローニング付きでF5-TTSを実行する

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

5秒の参照クリップ + その文字起こしを渡す。F5はプロソディと音色をクローンする。

### ステップ4: スクラッチからのHiFi-GANボコーダ

チュートリアルスクリプトに収めるには大きすぎるが、形は以下の通り:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

訓練: 敵対的(短い窓上の識別器) + メルスペクトログラム再構成損失 + 特徴マッチング損失。コモディティ化されている — `hifi-gan`リポジトリまたはnvidia-NeMoの事前学習済みチェックポイントを使うこと。

### ステップ5: 完全なパイプライン(疑似コード)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| リアルタイム英語ボイスアシスタント | Kokoro (CPU)またはXTTS v2 (GPU) |
| 5秒参照からの音声クローニング | F5-TTS |
| 商用キャラクターボイス | ElevenLabs v2.5 |
| オーディオブックナレーション | ElevenLabs v2.5またはXTTS v2 + ファインチューニング |
| 低リソース言語 | 5〜20時間のターゲット言語データでVITSを訓練 |
| 表現豊か / 感情タグ | ElevenLabs v2.5またはStyleTTS 2のファインチューニング |

2026年時点のオープンソースリーダー: **品質ならF5-TTS、効率ならKokoro**。歴史家でない限りTacotronに手を伸ばさないこと。

## 落とし穴

- **テキスト正規化器なし。** 「Dr. Smith」は「Doctor」と読むか「Drive」と読むか? 「2026」は「twenty twenty six」か「two zero two six」か? 音素化器の前に正規化すること。
- **OOVの固有名詞。** 「Ghumare」→「ghyu-mair」? 未知のトークンのためにフォールバックの書記素-音素モデルを出荷すること。
- **クリッピング。** ボコーダ出力はめったにクリップしないが、推論時のメルスケーリングの不一致は±1.0を超えることがある。常に`np.clip(wav, -1, 1)`すること。
- **サンプルレートの不一致。** Kokoroは24 kHzを出力する。下流のパイプラインが16 kHzを期待している → リサンプルするか、エイリアシングを起こす。

## 出荷する

`outputs/skill-tts-designer.md`として保存する。与えられた声、レイテンシ、言語ターゲットに対してTTSパイプラインを設計する。

## 演習

1. **易しい。** `code/main.py`を実行する。トイ語彙から音素辞書を構築し、音素ごとの継続時間を推定し、偽の「メル」スケジュールを出力する。
2. **中級。** Kokoroをインストールし、同じ文を声`af_bella`と`am_adam`で合成せよ。オーディオの継続時間と主観的品質を比較せよ。
3. **難しい。** 自分自身の5秒の参照クリップを録音せよ。F5-TTSを使ってそれをクローンせよ。参照とクローン出力の間のSECSを報告せよ。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| Phoneme | 音の単位 | 抽象的な音のクラス。英語では39個(ARPABet)。 |
| Duration predictor | 各音素がどれくらい続くか | 非ARモデルの出力。音素ごとの整数フレーム数。 |
| Vocoder | メル → 波形 | メルスペクトログラムを生サンプルにマッピングするニューラルネット。 |
| HiFi-GAN | 標準ボコーダ | GANベース。2020〜2024年に支配的。 |
| MOS | 主観的品質 | 人間の評価者による1〜5の平均オピニオンスコア。 |
| SECS | 音声クローン指標 | ターゲットと出力の話者埋め込みの間のコサイン類似度。 |
| F5-TTS | 2024年オープンソースSOTA | フローマッチング拡散。ゼロショットクローニング。 |
| Kokoro | CPU英語リーダー | 82Mパラメータモデル、Apache 2.0。 |

## 参考文献

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seqベースライン。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — エンドツーエンドのフローベース。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 現在のオープンソースSOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026年でも出荷されているボコーダ。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024年のCPUフレンドリーな英語TTS。
