# 音声スプーフィング対策 & オーディオ透かし — ASVspoof 5, AudioSeal, WaveVerify

> 音声クローニングは防御より早く展開された。2026年本番環境の音声システムには2つが必要である。実対話音声と合成音声を分類する検出器（AASIST, RawNet2）と、圧縮や編集に耐える透かし（AudioSeal）である。その両方を展開するか、音声クローニングを展開しないかのどちらかだ。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 6 · 06 （話者認識）, Phase 6 · 08 （音声クローニング）
**所要時間:** 約75分

## 問題

3つの関連する防御がある:

1. **スプーフィング対策/ディープフェイク検出。** オーディオクリップが与えられたとき、それは合成か実際か？ASVspoofベンチマーク（ASVspoof 2019 → 2021 → 5）は金準基。
2. **オーディオ透かし。** 生成されたオーディオに知覚できない信号を埋め込み、後で検出器が抽出できるようにする。AudioSeal（Meta）とWavMarkがオープンオプション。
3. **認証された出所。** オーディオファイル+メタデータの暗号署名。C2PA / Content Authenticity Initiative。

検出は協力しない敵に対処する。透かしはコンプライアンスを扱う — AI生成オーディオはそのように識別可能であるべき。2026年には両方が必須だ。

## コンセプト

![スプーフィング対策 vs 透かし vs 出所 — 3つの防御層](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025ベンチマーク

以前の版からの最大の変更:

- **クラウドソース データ** （スタジオクリーンではない） — 現実的な条件。
- **~2000話者** （以前は~100）。
- **32攻撃アルゴリズム。** TTS + 音声変換 + 敵対的摂動。
- **2つのトラック。** Countermeasure（CM）スタンドアロン検出; Spoofing-robust ASV（SASV）バイオメトリックシステム向け。

ASVspoof 5の最新技術: ~7.23% EER。古いASVspoof 2019 LA: 0.42% EER。実環境展開: in-the-wildクリップで5-10% EERを期待せよ。

### AASIST と RawNet2 — 検出モデルファミリー

**AASIST** （2021、2026まで更新）。スペクトル特性上のグラフアテンション。ASVspoof 5 countermeasureタスクでの現在のSOTA。

**RawNet2。** 生波形上のConvolutional前処理 + TDNNバックボーン。よりシンプルなベースライン; ファインチューニングでも競争力がある。

**NeXt-TDNN + SSL特性。** 2025バージョン: ECAPAスタイル + WavLM特性 + focal loss。ASVspoof 2019 LAで0.42% EERを達成。

### AudioSeal — 2024透かしデフォルト

Meta の **AudioSeal** （2024年1月、v0.2 2024年12月）。主な設計:

- **ローカライズ。** 16 kHzサンプルレゾリューション（1/16000秒）でフレーム単位で透かしを検出。
- **ジェネレータ+検出器の共同訓練。** ジェネレータは知覚できない信号埋め込みを学習; 検出器は増強を通じてそれを見つけることを学習。
- **ロバスト。** MP3 / AAC圧縮、EQ、速度シフト±10%、ノイズミックス+10 dB SNRに耐える。
- **高速。** 検出器は485倍の実時間で実行; WavMarkより1000倍高速。
- **容量。** 16ビットペイロード（モデルID、生成タイムスタンプ、ユーザーIDをエンコード可能）各発話に埋め込み可能。

### WavMark

AudioSeal前のオープンベースライン。可逆ニューラルネットワーク、32ビット/秒。問題:

- 同期ブルートフォースが遅い。
- ガウスノイズまたはMP3圧縮で削除可能。
- リアルタイムフレンドリーではない。

### WaveVerify （2025年7月）

AudioSealの弱点に対処 — 具体的には時間的操作（反転、速度）。FiLMベースのジェネレータ + Mixture-of-Expertsのためのジェネレータ + Mixture-of-Experts検出器を使用。標準攻撃で競争力がある; 時間的編集を処理。

### 敵が利用するギャップ

AudioMarkBenchから: 「ピッチシフト下では、すべての透かしはBit Recovery Accuracyが0.6以下を示し、ほぼ完全な削除を示す。」 **ピッチシフトは普遍的な攻撃。** 2026の透かしはいずれも、積極的なピッチ変更に完全にロバストではない。これが検出（AASIST）を透かしと並行して必要とする理由だ。

### C2PA / Content Authenticity Initiative

ML技術ではない — マニフェスト形式。オーディオファイルは作成ツール、著者、日付に関する暗号署名メタデータを携帯。Audobox / Seamlessが使用。出所に良い; 悪い行為者が再エンコードしてメタデータを削除した場合は何もしない。

## ビルド

### ステップ1: 簡単なスペクトル特性検出器（おもちゃ）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成音声は不通に平坦な高周波エネルギーをしばしば持つ。本番検出器はAASISTを使用し、これを使用しない。しかし直感は成り立つ。

### ステップ2: AudioSeal埋め込み + 検出

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1]内の浮動小数点 — 透かし存在の確率
# decoded_payload: 16ビット; 埋め込まれたペイロードと照合
```

### ステップ3: 評価 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### ステップ4: 本番環境統合

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

すべての生成は以下を含む: (1) 透かし、(2) 署名マニフェスト、(3) 保持方針準拠監査ログ。

## 使用

| ユースケース | 防御 |
|----------|---------|
| TTS / 音声クローニングの展開 | 出力のすべてにAudioSeal埋め込み（交渉不可） |
| バイオメトリック音声ロック解除 | AASIST + ECAPA集合; 生性チャレンジ |
| コールセンター詐欺検出 | 着信通話の20%サンプルでAASIST |
| ポッドキャスト真正性 | アップロード時C2PA署名、AI生成の場合AudioSeal |
| 研究 / 検出器訓練 | ASVspoof 5訓練/開発/評価セット |

## 落とし穴

- **検出器なしで実行される透かし。** 無意味。検出器をCIに展開せよ。
- **キャリブレーションなしの検出。** AASIST訓練ASVspoof LAオーバーフィット; 実環境精度低下。ドメインでキャリブレーション。
- **ピッチシフトギャップ。** 積極的なピッチシフトはほとんどの透かしを削除。検出フォールバックを持つ。
- **メタデータ削除と再ホスト。** C2PAは再エンコードで簡単に回避可能。常に暗号化+知覚防御（透かし）を一緒に追加。
- **生性として検出。** ユーザーにランダムフレーズを言わせてほしい。リプレイ攻撃を防ぐがリアルタイムクローニングではない。

## 展開

`outputs/skill-spoof-defender.md`として保存。検出モデル、透かし、出所マニフェスト、音声生成展開の運用プレイブックを選択。

## 演習

1. **簡単。** `code/main.py`を実行。合成オーディオ上のおもちゃ検出器 + おもちゃ透かし埋め込み/検出。
2. **中程度。** `audioseal`をインストール、TTS出力に16ビットペイロードを埋め込み、再デコード。Bit Recovery Accuracyノイズで音声を破損させ測定。
3. **難しい。** ASVspoof 2019 LAでRawNet2またはAASISTをファインチューニング。EER測定。F5-TTS生成クリップのホールドアウトセットでテスト — OOD検出どの程度低下するか。

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|-----------------|-----------------------|
| ASVspoof | ベンチマーク | 隔年チャレンジ; 2024 = ASVspoof 5。 |
| CM (countermeasure) | 検出器 | 分類器: 実音声 vs 合成/変換。 |
| SASV | 話者検証 + CM | 統合バイオメトリック + スプーフ検出。 |
| AudioSeal | Meta透かし | ローカライズ、16ビットペイロード、WavMarkより485倍高速。 |
| Bit Recovery Accuracy | 透かし生存 | 攻撃後に回収されるペイロードビットの分数。 |
| C2PA | 出所マニフェスト | 作成/著作に関する暗号メタデータ。 |
| AASIST | 検出器ファミリー | グラフアテンションベースのスプーフ対策SOTA。 |

## 参考文献

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 現在のベンチマーク。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 透かしデフォルト。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 時間的攻撃向けMoE検出器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTAスプーフ対策バックボーン。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — ロバスト評価。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 出所マニフェスト形式。
