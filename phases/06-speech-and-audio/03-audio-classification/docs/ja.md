# オーディオ分類 — MFCC上のk-NNからAST、BEATsまで

> 「犬の吠え声対サイレン」から「これは何語か」まで、すべてがオーディオ分類だ。特徴量はメルである。アーキテクチャは10年ごとに移り変わる。評価はAUC、F1、クラスごとのリコールのままだ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 02 (Spectrograms & Mel)、Phase 3 · 06 (CNNs)、Phase 5 · 08 (CNNs & RNNs for Text)
**所要時間:** 約75分

## 問題

10秒のクリップを受け取る。あなたは「これは何か?」を知りたい。都市の音(サイレン、ドリル、犬)、音声コマンド(yes/no/stop)、言語ID(en/es/ar)、話者の感情(怒り/中立)、環境音(屋内/屋外、雑談)。これらすべてが*オーディオ分類*であり、2026年にはベースラインアーキテクチャは成熟している。対数メル → CNNまたはTransformer → softmaxだ。

核心的な困難はネットワークではない。データだ。オーディオデータセットは残酷なクラス不均衡、強いドメインシフト(クリーン対ノイジー)、ラベルノイズ(「都市の雑談」対「レストランのノイズ」を誰が決めたのか?)を抱えている。問題の80%は、CNNをTransformerに置き換えることではなく、キュレーション、データ拡張、評価にある。

## コンセプト

![オーディオ分類のはしご: MFCC上のk-NNからAST, BEATsへ](../assets/audio-classification.svg)

**MFCC上のk-NN(1990年代のベースライン)。** クリップごとにMFCCを平坦化し、ラベル付きバンクとのコサイン類似度を計算し、上位Kの多数決を返す。クリーンで小さなデータセット(Speech Commands、ESC-50)では驚くほど強い。GPUなしで動く。

**対数メル上の2D CNN(2015-2019)。** `(T, n_mels)`の対数メルを画像として扱う。ResNet-18またはVGGスタイルを適用する。時間軸をグローバル平均プーリングする。クラスにわたってsoftmax。2026年のほとんどのkaggleコンペティションでも依然としてベースラインだ。

**Audio Spectrogram Transformer、AST(2021-2024)。** 対数メルをパッチ化し(例: 16×16パッチ)、位置埋め込みを追加し、ViTに供給する。教師あり学習においてAudioSet上で最先端(mAP 0.485)。

**BEATsとWavLM-base(2024-2026)。** 数百万時間での自己教師あり事前学習。必要だったであろう教師ありデータの1〜10%で自分のタスクにファインチューニングする。2026年には、これが非音声オーディオのデフォルトの出発点だ。BEATs-iter3は、1/4の計算量を使いながらAudioSet上でASTを1〜2 mAP上回る。

**凍結バックボーンとしてのWhisperエンコーダ(2024)。** Whisperのエンコーダを取り、デコーダを捨て、線形分類器を取り付ける。オーディオデータ拡張ゼロで、言語IDと単純なイベント分類においてほぼSOTA。「無料ランチ」のベースラインだ。

### クラス不均衡こそが本当の課題

ESC-50: 50クラス、各40クリップ — バランスが取れていて簡単。UrbanSound8K: 10クラス、10:1の不均衡。AudioSet: 100,000:1のロングテールを持つ632クラス。効果のある技術:

- 訓練中のバランスサンプリング(評価では行わない)。
- Mixup: データ拡張として2つのクリップ(とそのラベル)を線形補間する。
- SpecAugment: ランダムな時間帯と周波数帯をマスクする。単純。決定的に重要。

### 評価

- 多クラス排他(Speech Commands): top-1精度、top-5精度。
- 多クラスマルチラベル(AudioSet、UrbanSoundスタイル): 平均適合率の平均(mAP)。
- 大きく不均衡: クラスごとのリコール + マクロF1。

知っておくべき2026年の数値:

| ベンチマーク | ベースライン | SOTA 2026 | 出典 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs論文 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEARリーダーボード 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2の結果 |

## 作ってみる

### ステップ1: 特徴量化する

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### ステップ2: 固定長の要約

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

単純だが強力: 時間にわたる平均 + 分散は、13係数MFCCに対して26次元の固定埋め込みを与える。瞬時に動く。つい2017年まで、ESC-50で最先端のNNベースラインを打ち負かしていた。

### ステップ3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### ステップ4: 対数メル上のCNNにアップグレード

PyTorchで:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300万パラメータ。単一のRTX 4090でESC-50を約10分で訓練。80%以上の精度。

### ステップ5: 2026年のデフォルト — BEATsをファインチューニングする

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATsの場合は、`beats`ライブラリ経由で`microsoft/BEATs-base`を使う。transformers APIは同じ形状だ。

## 使ってみる

2026年のスタック:

| 状況 | 何から始めるか |
|-----------|-----------|
| 小さなデータセット (<1000クリップ) | MFCC平均上のk-NN(ベースライン) + オーディオデータ拡張 |
| 中規模データセット (1K〜100K) | BEATsまたはASTのファインチューニング |
| 大規模データセット (>100K) | スクラッチから訓練、またはWhisperエンコーダのファインチューニング |
| リアルタイム、エッジ | int8に量子化した40-MFCC CNN(KWSスタイル) |
| マルチラベル (AudioSet) | BCE損失 + mixup + SpecAugmentによるBEATs-iter3 |
| 言語ID | MMS-LID、SpeechBrain VoxLingua107ベースライン |

判断ルール: **新しいモデルではなく、凍結バックボーンから始めよ**。BEATsヘッドのファインチューニングは、数週間ではなく数時間でSOTAの95%に到達させてくれる。

## 出荷する

`outputs/skill-classifier-designer.md`として保存する。与えられたオーディオ分類タスクに対して、アーキテクチャ、データ拡張、クラスバランス戦略、評価指標を選択する。

## 演習

1. **易しい。** `code/main.py`を実行する。4クラスの合成データセット(異なるピッチの純音)でk-NN MFCCベースラインを訓練する。混同行列を報告せよ。
2. **中級。** `summarize`を[平均、分散、歪度、尖度]に置き換える。4モーメントプーリングは、同じ合成データセットで平均+分散を上回るか?
3. **難しい。** `torchaudio`を使って、ESC-50のfold 1で2D CNNを訓練する。5分割交差検証の精度を報告せよ。SpecAugment(time mask = 20、freq mask = 10)を追加し、その差分を報告せよ。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| AudioSet | オーディオのImageNet | Googleの200万クリップ、632クラスの弱ラベル付きYouTubeデータセット。 |
| ESC-50 | 小さな分類ベンチマーク | 環境音の50クラス × 40クリップ。 |
| AST | Audio Spectrogram Transformer | 対数メルパッチ上のViT。2021年のSOTA。 |
| BEATs | 自己教師ありオーディオ | Microsoftのモデル。iter3が2026年時点でAudioSetをリード。 |
| Mixup | ペアデータ拡張 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | マスクベースのデータ拡張 | スペクトログラムのランダムな時間帯と周波数帯をゼロにする。 |
| mAP | 主要なマルチラベル指標 | クラスとしきい値にわたる平均適合率の平均。 |

## 参考文献

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021〜2024年の定番アーキテクチャ。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024年以降のデフォルト。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 支配的なオーディオデータ拡張。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 生き続ける50クラスベンチマーク。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632クラスのYouTube分類体系。今もゴールドスタンダード。
