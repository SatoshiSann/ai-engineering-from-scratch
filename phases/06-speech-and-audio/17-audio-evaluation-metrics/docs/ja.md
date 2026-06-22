# オーディオ評価 — WER, MOS, UTMOS, MMAU, FAD, オープンリーダーボード

> 測定できないものは展開できない。このレッスンは2026年のすべてのオーディオタスク向けメトリクスに名前をつける: ASR（WER, CER, RTFx）、TTS（MOS, UTMOS, SECS, WER-on-ASR-round-trip）、オーディオ言語（MMAU, LongAudioBench）、音楽（FAD, CLAP）、および話者（EER）。さらに比較するリーダーボード。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 （モデル評価）
**所要時間:** 約60分

## 問題

すべてのオーディオタスクは複数のメトリクスを持ち、各メトリクスは異なる軸を測定する。誤ったメトリクスを使用することは、ダッシュボードでは素晴らしく見えるが本番環境では ひどいモデルを展開する方法だ。2026年の規範的リスト:

| タスク | 主要 | 二次 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 音声クローニング | SECS （ECAPA cosine） | MOS · CER |
| 話者検証 | EER | minDCF · FAR / FRR（動作ポイント） |
| ダイアリゼーション | DER | JER · 話者混同 |
| オーディオ分類 | top-1 · mAP | macro F1 · per-class recall |
| 音楽生成 | FAD | CLAP · listening panel MOS |
| オーディオ言語モデル | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| ストリーミング S2S | latency P50/P95 | WER · MOS |

## コンセプト

![オーディオ評価マトリクス — メトリクス vs タスク vs 2026リーダーボード](../assets/eval-landscape.svg)

### ASRメトリクス

**WER（Word Error Rate）。** `(S + D + I) / N`。小文字、句読点削除、スコアリング前の正規化数値。`jiwer`またはOpenAIの`whisper_normalizer`を使用。 < 5% = 人間同等の読み取り音声。

**CER（Character Error Rate）。** 同じ公式、文字レベル。言語トーン（標準中国語、広東語）で使用、単語分割が曖昧。

**RTFx（逆リアルタイム係数）。** 壁時計秒あたり処理されたオーディオ秒。高いほど良い。Parakeet-TDTは3380×に達する。Whisper-large-v3は~30×。

**First-token latency。** オーディオ入力から最初のトランスクリプトトークンへの壁時計。ストリーミング向け重要。Deepgram Nova-3: ~150 ms。

### TTSメトリクス

**MOS（Mean Opinion Score）。** 1-5人間評価。金準基だが遅い。サンプルあたり20+リスナー、モデルあたり100+サンプルを集めよ。

**UTMOS（2022-2026）。** 学習されたMOS予測器。標準ベンチマークで人間MOSと~0.9相関。F5-TTS: UTMOS 3.95; ground truth: 4.08。

**SECS（Speaker Encoder Cosine Similarity）。** 音声クローニング向け。参照と複製出力の間のECAPAエンベディング余弦。 > 0.75 = 認識可能なクローン。

**WER-on-ASR-round-trip。** TTS出力上でWhisperを実行、入力テキストに対するWER計算。知能低下を逃さない。2026 SOTA: < 2% CER。

**TTFA（time-to-first-audio）。** 壁時計レイテンシー。Kokoro-82M: ~100 ms; F5-TTS: ~1 s。

### 音声クローニング特有

**SECS + MOS + CER**を3つの組として使用。高SECSだが低MOS クローニングは音色が正しいが不自然; 逆は自然音声だが話者が違う。

### 話者検証

**EER（Equal Error Rate）。** False Accept RateがFalse Reject Rateと等しい閾値。ECAPA VoxCeleb1-O: 0.87%。

**minDCF（min Detection Cost）。** 選択した動作ポイント（よく FAR=0.01）での加重コスト。EERより本番関連。

### ダイアリゼーション

**DER（Diarization Error Rate）。** `(FA + Miss + Confusion) / total_speaker_time`。見逃し音声 + 誤警報音声 + 話者混同、各々分数として。AMI会議: DER ~10-20%は現実的。pyannote 3.1 + Precision-2商用: <10% DER良質音声上。

**JER（Jaccard Error Rate）。** DERの代替、短セグメント偏向ロバスト。

### オーディオ分類

マルチラベル: すべてのクラスで**mAP（mean Average Precision）**。AudioSet: BEATs-iter3で 0.548 mAP。

マルチクラス排他: **top-1, top-5 accuracy**。Speech Commands v2: 99.0% top-1 （Audio-MAE）。

不均衡: **macro F1** + **per-class recall**。Per-classを報告 — 集約精度はどのクラスが失敗するかを隠す。

### 音楽生成

**FAD（Fréchet Audio Distance）。** 実対生成オーディオのVGGishエンベディング分布間の距離。MusicGen-small MusicCaps: 4.5。MusicLM: 4.0。低いほど良い。

**CLAP Score。** CLAPエンベディングを使用したテキストオーディオアライメントスコア。 > 0.3 = 妥当なアライメント。

**Listening panel MOS。** コンシューマーグレード音楽向けで最終的な言葉。Suno v5 ELO 1293 TTS Arena上（ペア人間選好から）。

### オーディオ言語ベンチマーク

**MMAU（Massive Multi-Audio Understanding）。** 10k オーディオQAペア。

**MMAU-Pro。** 1800ハード項目、4カテゴリ: 音声 / 音 / 音楽 / マルチオーディオ。ランダムチャンス4方法で25%。Gemini 2.5 Pro全体 ~60%; マルチオーディオ ~22%全モデル中。

**LongAudioBench。** セマンティッククエリを持つマルチ分クリップ。Audio Flamingo Next Gemini 2.5 Proを上回る。

**AudioCaps / Clotho。** キャプショニングベンチマーク。SPICE, CIDEr, FENSE メトリクス。

### ストリーミング音声-音声

**Latency P50 / P95 / P99。** ユーザー音声終了から最初の可聴応答への壁時計。Moshi: 200 ms; GPT-4o Realtime: 300 ms。

**WER / MOS**出力。

**Barge-in responsiveness。** ユーザー割り込みからアシスタント消音まで。ターゲット < 150 ms。

### 2026リーダーボード

| リーダーボード | トラック | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + 多言語 + 長形式 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELOペア投票から | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM推論 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 話者認識 | `voxsrc.github.io` |
| MMAU音楽サブセット | 音楽LALM | （MMAU内） |
| HEAR benchmark | 自己教師音声 | `hearbenchmark.com` |

## ビルド

### ステップ1: 正規化によるWER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### ステップ2: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### ステップ3: 音声クローニングのSECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### ステップ4: 音楽生成向けFAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### ステップ5: 話者検証向けEER（レッスン6と同じコード）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用

すべてのデプロイをモデル更新のたびに実行される固定評価ハーネスと配置。3つの基本ルール:

1. **スコアリング前の正規化。** 小文字、句読点削除、数値拡張。正規化ルールを報告。
2. **分布を報告、平均値ではない。** レイテンシーP50/P95/P99。分類のPer-class recall。MMauのPer-category。
3. **1つの規範的公開ベンチマークを実行。** 本番データが異なっても、Open ASR / TTS Arena / MMAUでの報告は、レビュアーがapples-to-applesで比較を許可。

## 落とし穴

- **UTMOS外推。** VCTKスタイル清潔音声で訓練; ノイズ/複製/情感オーディオを貧弱にスコア。
- **MOS パネルバイアス。** 20 Amazon Mechanical Turk 労働者 ≠ 20ターゲットユーザー。ステークが高い場合ドメインパネル向け支払い。
- **FADは参照セットに依存。** すべてのモデルで同じ参照分布と比較。
- **Aggregate WER。** 全体で5% WERは、アクセント音声上30% WERを隠す。人口統計スライスで報告。
- **公開ベンチマーク飽和。** ほとんどのフロンティアモデルは標準ベンチマーク上で天井近く。実トラフィックを反映するハウス保持セットを構築。

## 展開

`outputs/skill-audio-evaluator.md`として保存。メトリクス、ベンチマーク、オーディオモデルリリース向け報告形式を選択。

## 演習

1. **簡単。** `code/main.py`を実行。WER / CER / EER / SECS / FAD類似 / MMAU類似をおもちゃ入力で計算。
2. **中程度。** TTS round-trip WERハーネスを構築。Kokoro またはF5-TTS出力をWhisperに実行。50プロンプトでWER計算。WER > 10%プロンプトはフラグ。
3. **難しい。** レッスン10 LALMチョイスをMMUA-Pro 音声+マルチオーディオサブセット（各50項目）でスコア。Per-category accuracyを報告し、公開番号と比較。

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|-----------------|-----------------------|
| WER | ASRスコア | `(S+D+I)/N`正規化後の単語レベル。 |
| CER | 文字WER | トーン言語またはchar-level シスム。 |
| MOS | 人間意見 | 1-5評価; 20+ リスナー × 100 サンプル。 |
| UTMOS | ML MOS予測器 | 学習モデル; 人間MOSと~0.9相関。 |
| SECS | 音声クローン相似 | 参照とクローン間のECAPA余弦。 |
| EER | 話者検証スコア | FAR = FRRの閾値。 |
| DER | ダイアリゼーションスコア | (FA + Miss + Confusion) / total。 |
| FAD | 音楽生成品質 | VGGishエンベディング上のFréchet距離。 |
| RTFx | スループット | 壁時計秒あたりのオーディオ秒。 |

## 参考文献

- [jiwer](https://github.com/jitsi/jiwer) — 正規化ユーティリティ付きWER/CERライブラリ。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学習されたMOS予測器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音楽生成基準。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026ライブランキング。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人間投票TTSリーダーボード。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM推論リーダーボード。
- [HEAR benchmark](https://hearbenchmark.com/) — オーディオSSLベンチマーク。
