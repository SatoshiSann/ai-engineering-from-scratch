# オーディオの基本 — 波形、サンプリング、フーリエ変換

> 波形は生のシグナルだ。スペクトログラムは表現だ。メル特徴はML対応形式だ。すべてのモダンASRとTTSパイプラインはこのラダーを歩く、最初の段階はサンプリングとフーリエを理解することだ。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 1 · 06 (ベクトル・行列)、Phase 1 · 14 (確率分布)
**所要時間:** 約45分

## 問題

マイクは圧力対時間シグナルを生成する。ニューラルネットはテンソルを消費する。その間には、違反すると静かなバグを生成する規約のスタックがある: モデルは微調整だが、WERは2倍になり、またはTTSはヒスを出荷し、または音声クローニングシステムはスピーカーではなくマイクを記憶する。

音声システムのすべてのバグは3つの質問の1つに遡る:

1. データはどのサンプルレートで記録され、モデルは何を期待しているか?
2. シグナルはエイリアシングされているか?
3. 生のサンプルまたは周波数表現で動作しているか?

これらを正しくして、Phase 6の残りは扱いやすい。これらを間違えて、Whisper-Large-v4でさえ、ガベージを生成。

## コンセプト

![波形、サンプリング、DFT、周波数ビン可視化](../assets/audio-fundamentals.svg)

**波形。** `[-1.0、1.0]`の浮動小数点のものの1次元配列。サンプル番号でインデックス。秒に変換するには、サンプルレートで分割: `t = n / sr`。10秒のクリップ16 kHzで160,000浮動小数点の配列だ。

**サンプルレート(sr)。** 秒あたりのサンプル数。2026年の共通レート:

| レート | 使用 |
|------|-----|
| 8 kHz | テレフォニー、レガシーVOIP。4 kHzでのNyquistは子音を殺す。ASRを避ける。 |
| 16 kHz | ASR標準。Whisper、Parakeet、SeamlessM4T v2はすべて16 kHzを消費。 |
| 22.05 kHz | 古いモデルのTTS vocoder訓練。 |
| 24 kHz | モダンTTS (Kokoro、F5-TTS、xTTS v2)。 |
| 44.1 kHz | CDオーディオ、音楽。 |
| 48 kHz | フィルム、プロオーディオ、高忠実度TTS (VALL-E 2、NaturalSpeech 3)。 |

**Nyquist-Shannon。** サンプルレート`sr`は`sr/2`までの周波数を明確に表現できる。`sr/2`境界は*Nyquist周波数*だ。Nyquistを超えるエネルギーは*エイリアシング* — より低い周波数に折り返す — そしてシグナルを破損。常にダウンサンプリングする前にローパスフィルターを適用。

**ビット深度。** 16ビットPCM (符号付きint16、範囲±32,767)はユニバーサル交換フォーマット。音楽用24ビット、内部DSP用32ビット浮動。`soundfile`のようなライブラリは、int16を読むが、`[-1、1]`の浮動小数点32配列を露光。

**フーリエ変換。** すべての有限シグナルは、異なる周波数でシヌソイドの合計だ。離散フーリエ変換(DFT)は`N`サンプル、`N`複素系数を計算する — 周波数ビンあたり1つ。`ビンk`は周波数`k · sr / N` Hzにマップ。大きさは周波数での振幅、角度は位相。

**FFT。** 高速フーリエ変換: `N`がパワー2の場合、DFTの`O(N log N)`アルゴリズム。すべてのオーディオライブラリはボンネットの下でFFTを使用。1024サンプルFFT 16 kHzは0–8 kHzにまたがる512使用可能周波数ビンを提供し、15.6 Hz解像度。

**フレーミング+ウィンドウ。** 全体クリップをFFTしない。重複する*フレーム*(通常25 msを10 msホップで)に刻み、各フレームをウィンドウ関数(Hann、Hamming)で掛け、エッジ不連続を殺し、各フレームをFFTして。これは短時間フーリエ変換(STFT)だ。レッスン02はここから。

## ビルドしてみよう

### ステップ1: クリップを読み波形をプロット

`code/main.py`は依存関係なしでデモを保つために標準`wave`モジュールのみを使用。本番用には`soundfile`または`torchaudio.load`を使用する(両方は`(waveform、sr)`タプルを返す):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### ステップ2: 最初の原則からシーン波を合成

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz sine (concert A) 16 kHzで1秒は16,000浮動小数点。`wave.open(...、"wb")`でも16ビットPCMエンコーディングを使用して書く。

### ステップ3: 手でDFTを計算

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — `N=256`を正確性を確認するのに適切だが、本当のオーディオに無用。本当のコードは`numpy.fft.rfft`または`torch.fft.rfft`を呼ぶ。

### ステップ4: 支配周波数を見つける

大きさピークインデックス`k_star`は周波数`k_star * sr / N`にマップ。440 Hz sineでこれを実行すると、ビン`440 * N / sr`でピークを返すべき。

### ステップ5: エイリアシングを実証

10 kHz(Nyquist = 5 kHz)で7 kHz sineをサンプル。7 kHzトーンはNyquist以上で、`10−7 = 3 kHz`に折り返す。FFTピークは3 kHzに表示。これはクラシックエイリアシングデモとすべてのDAC/ADCが煉瓦壁ローパスフィルターで出荷される理由だ。

## 使用方法

2026年で実際に出荷するスタック:

| タスク | ライブラリ | 理由 |
|------|---------|-----|
| WAV/FLAC/OGG読み込み/書き込み | `soundfile` (libsndfileラッパー) | 最速、安定、float32を返す。 |
| リサンプル | `torchaudio.transforms.Resample`または`librosa.resample` | 正しいアンチエイリアシング組み込み。 |
| STFT / メル | `torchaudio`または`librosa` | GPU対応; PyTorchエコシステム。 |
| リアルタイムストリーミング | `sounddevice`または`pyaudio` | クロスプラットフォームPortAudioバインディング。 |
| ファイルを検査 | `ffprobe`または`soxi` | CLI、高速、sr/channels/codecを報告。 |

決定ルール: **他のものをマッチさせる前にサンプルレートをマッチさせてください**。Whisperは16 kHz単一トラックfloat32を期待。44.1 kHzステレオを与えて、モデルバグのように見えるガベージを取得。

## 出荷しよう

`outputs/skill-audio-loader.md`として保存。スキルはあなたがオーディオ入力がダウンストリームモデルの期待にマッチすることをチェックするのを助け、合致しないときに正しく再サンプルする。

## 演習

1. **簡単。** 16 kHzで220 Hz + 440 Hz + 880 Hzの1秒ミックスを合成。DFTを実行。予期ビンで3つのピークを確認。
2. **中程度。** 48 kHzで声の3秒WAVを記録。`torchaudio.transforms.Resample`(アンチエイリアシング付き)を使用して16 kHzにダウンサンプル、その後3番目のサンプルで単純な削減を使用して16 kHz。両方をFFT。エイリアシングはどこに表示されるか?
3. **難しい。** ステップ3のDFTのみを使用して、`math`からSTFTをゼロから構築。フレームサイズ400、ホップ160、Hannウィンドウ。`matplotlib.pyplot.imshow`で大きさをプロット。これはレッスン02のスペクトログラムだ。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| サンプルレート | 秒あたりいくつのサンプル | ADCがシグナルを測定する周波数Hz。 |
| Nyquist | あなたが表現できる最大周波数 | `sr/2`; Nyquist上のエネルギーはより低いビンに折り返す。 |
| ビット深度 | 各サンプルの解決度 | `int16` = 65,536レベル; `float32` = `[-1、1]`の24ビット精度。 |
| DFT | シーケンスのフーリエ変換 | `N`サンプル → `N`複素周波数係数。 |
| FFT | 高速DFT | `N` = パワー2必要な場合の`O(N log N)`アルゴリズム。 |
| ビン | 周波数列 | `k · sr / N` Hz; 解決度 = `sr / N`。 |
| STFT | ボンネットの下のスペクトログラム | フレーム+ウィンドウ付きFFTを時間全体。 |
| エイリアシング | 奇妙な周波数ゴースト | Nyquist上のエネルギーが低いビンに鏡。 |

## 参考文献

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — サンプリング定理の背後にあるペーパー。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 自由、正規DSPテキスト。
- [librosaドキュメント — オーディオプライマー](https://librosa.org/doc/latest/tutorial.html) — コード付き実践的ウォークスルー。
- [Heinrich Kuttruff — Room Acoustics (第6版)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 本当の世界のオーディオが清潔なシヌソイドではない理由のリファレンス。
- [Steve Eddins — FFT解釈ノートブック](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 周波数ビンの直感10分で明らか。
