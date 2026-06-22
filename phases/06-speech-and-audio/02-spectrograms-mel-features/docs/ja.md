# スペクトログラム、メルスケール、オーディオ特徴量

> ニューラルネットは生の波形をうまく消化できない。スペクトログラムなら消化できる。メルスペクトログラムならさらにうまく消化できる。2026年のあらゆるASR、TTS、オーディオ分類器は、この単一の前処理の選択によって生死が決まる。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 01 (Audio Fundamentals)
**所要時間:** 約45分

## 問題

10秒の16 kHzクリップを取り上げてみよう。それは160,000個の浮動小数点数であり、すべて`[-1, 1]`の範囲にあり、「犬の吠え声」や「catという単語」というラベルとはほぼ完全に無相関だ。生の波形には情報が含まれているが、モデルが簡単に抽出できない形になっている。100 ms離れて発話された2つの同一の音素は、生のサンプルとしてはまったく異なる。

スペクトログラムはこれを解決する。人間の知覚が無視する時間的詳細(マイクロ秒のジッタ)を畳み込み、知覚が注目する構造(約10〜25 msの時間窓にわたって、どの周波数にエネルギーがあるか)を保持する。

メルスペクトログラムはさらに踏み込む。人間はピッチを対数的に知覚する。100 Hz対200 Hzは、1000 Hz対2000 Hzと「同じだけ離れている」ように聞こえる。メルスケールは、これに合わせて周波数軸を歪める。メルスケールのスペクトログラムは、2010年から2026年に至るまで、音声MLにおいて最も重要な単一の特徴量である。

## コンセプト

![波形からSTFT、メルスペクトログラム、MFCCへのはしご](../assets/mel-features.svg)

**STFT (短時間フーリエ変換)。** 波形を重なり合うフレームにスライスする(典型的には、25 msの窓、10 msのホップ = 16 kHzで400サンプル / 160サンプル)。各フレームに窓関数を掛ける(Hannがデフォルト。Hammingはわずかに異なるトレードオフ)。各フレームをFFTする。振幅スペクトルを`(n_frames, n_freq_bins)`という形状の行列に積み重ねる。それがスペクトログラムだ。

**対数振幅。** 生の振幅は5〜6桁にわたる。`log(|X| + 1e-6)`または`20 * log10(|X|)`を取ってダイナミックレンジを圧縮する。あらゆる本番パイプラインは、生の振幅ではなく対数振幅を使う。

**メルスケール。** Hz単位の周波数`f`は、`m = 2595 * log10(1 + f / 700)`によってメル`m`にマッピングされる。このマッピングは1 kHz以下ではおおむね線形、それ以上ではおおむね対数的だ。0〜8 kHzをカバーする80メルビンが標準的なASR入力である。

**メルフィルタバンク。** メルスケール上で等間隔に配置された三角フィルタの集合。各フィルタは隣接するFFTビンの加重和だ。STFT振幅にフィルタバンク行列を掛けることで、1回の行列積でメルスペクトログラムが得られる。

**対数メルスペクトログラム。** `log(mel_spec + 1e-10)`。Whisperの入力。Parakeetの入力。SeamlessM4Tの入力。2026年の普遍的なオーディオフロントエンドだ。

**MFCC。** 対数メルスペクトログラムを取り、DCT(タイプII)を適用し、最初の13係数を保持する。特徴量を無相関化し、さらに圧縮する。CNN/Transformerが生の対数メル上で追いついた2015年頃まで支配的な特徴量だった。話者認識(x-vectors、ECAPA)では今も使われている。

**解像度のトレードオフ。** より大きなFFT = より良い周波数解像度だが、より悪い時間解像度。25 ms / 10 msがオーディオMLのデフォルト。音楽には50 ms / 12.5 ms。過渡音検出(ドラムのヒット、破裂音)には5 ms / 2 ms。

## 作ってみる

### ステップ1: 波形をフレーム化する

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160`の10秒16 kHzクリップは998フレームを生成する。

### ステップ2: Hann窓

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFTの前に要素ごとに掛ける。非ゼロの端点での切り詰めによって生じるスペクトル漏れを除去する。

### ステップ3: STFT振幅

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

本番では`torch.stft`または`librosa.stft`(FFTバック、ベクトル化済み)を使う。ここでのループは教育的なものだ。`code/main.py`では短いクリップ上で実行される。

### ステップ4: メルフィルタバンク

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400`で0〜8 kHzをカバーする80メルは、`(80, 201)`行列を与える。`(n_frames, 201)`のSTFT振幅にその転置を掛けると、`(n_frames, 80)`のメルスペクトログラムが得られる。

### ステップ5: 対数メル

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

一般的な代替手段: `librosa.power_to_db`(参照正規化されたdB)、`10 * log10(power + eps)`。Whisperはより込み入ったクリップ + 正規化ルーチンを使う(Whisperの`log_mel_spectrogram`を参照)。

### ステップ6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

各対数メルフレームにDCTを適用し、最初の13係数を保持する。それがMFCC行列だ。最初の係数は通常破棄される(全体のエネルギーをエンコードしている)。

## 使ってみる

2026年のスタック:

| タスク | 特徴量 |
|------|----------|
| ASR (Whisper、Parakeet、SeamlessM4T) | 80対数メル、10 msホップ、25 ms窓 |
| TTS音響モデル (VITS、F5-TTS、Kokoro) | 80メル、細かい時間制御のため5〜12 msホップ |
| オーディオ分類 (AST、PANNs、BEATs) | 128対数メル、10 msホップ |
| 話者埋め込み (ECAPA-TDNN、WavLM) | 80対数メルまたは生波形SSL |
| 音楽 (MusicGen、Stable Audio 2) | EnCodec離散トークン(メルではない) |
| キーワードスポッティング | 小型デバイス向け40 MFCC |

経験則: **音楽を扱っているのでなければ、80対数メルから始めよ。** いかなる逸脱にも立証責任がある。

## 2026年でもまだ出荷される落とし穴

- **メル数の不一致。** 80メルで訓練、128メルで推論。サイレント障害。両端で特徴量の形状をログに出すこと。
- **上流でのサンプルレート不一致。** 22.05 kHzで計算されたメルは16 kHzのものと違って見える。特徴量化の*前に*SRを修正すること。
- **dB対log。** WhisperはdBメルではなく対数メルを期待する。一部のHFパイプラインは自動検出するが、あなたのカスタムコードはしない。
- **正規化のドリフト。** 訓練時は発話ごとの正規化、推論時はグローバル正規化。WERを倍増させる本番バグ。
- **パディングからの漏れ。** クリップの末尾をゼロパディングすると、末尾のフレームでフラットなスペクトルが生じる。対称的にパディングするか、複製すること。

## 出荷する

`outputs/skill-feature-extractor.md`として保存する。このスキルは、与えられたモデルターゲットに対して、特徴量タイプ、メル数、フレーム/ホップ、正規化を選択する。

## 演習

1. **易しい。** `code/main.py`を実行する。チャープ(周波数を200 → 4000 Hzに掃引)を合成し、フレームごとのargmaxメルビンを出力する。(任意で)プロットして、掃引と一致することを確認せよ。
2. **中級。** `n_mels`を`{40, 80, 128}`、`frame_len`を`{200, 400, 800}`にして再実行する。時間軸にわたる鋭いピークの帯域幅を測定せよ。どの組み合わせがチャープを最もよく解像するか?
3. **難しい。** `power_to_db`を実装し、AudioMNIST上の小さなCNN分類器のASR精度を、(a) 生の対数メル、(b) `ref=max`のdBメル、(c) MFCC-13 + delta + delta-deltaを使って比較せよ。top-1精度を報告せよ。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| Frame | スライス | 1回のFFTに供給される25 msの波形チャンク。 |
| Hop | ストライド | 連続するフレーム間のサンプル数。10 msがASRデフォルト。 |
| Window | Hann/Hammingのやつ | フレームの端をゼロにテーパーさせる点ごとの乗数。 |
| STFT | スペクトログラム生成器 | フレーム化 + 窓掛けFFT。時間 × 周波数行列を生成。 |
| Mel | 歪んだ周波数 | 対数知覚スケール。`m = 2595·log10(1 + f/700)`。 |
| Filterbank | その行列 | STFTをメルビンに射影する三角フィルタ。 |
| Log-mel | Whisperの入力 | `log(mel_spec + eps)`。2026年に標準化。 |
| MFCC | 旧来の特徴量 | 対数メルのDCT。13係数、無相関化。 |

## 参考文献

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCCの論文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 元のメルスケール。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 参照実装を読む。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram`、hop/windowの参照。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canaryモデル向けの本番規模パイプライン。
