# 音声クローニングと音声変換

> 音声クローニングは誰かの音声でテキストを読む。音声変換は誰かの音声にあなたの音声を書き直しながら、あなたが言ったことを保存する。両方は同じ分解を吊す: 話者アイデンティティからコンテンツを分離。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 6 · 06 (話者認識)、Phase 6 · 07 (TTS)
**所要時間:** 約75分

## 問題

2026年で、5秒のオーディオクリップは誰かの音声の高品質クローンを消費者GPUで生成するのに十分だ。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox はすべてゼロショットまたは少ショットクローニングを出荷。テクノロジーは祝福(アクセシビリティTTS、ダビング、補助音声)と武器(詐欺呼び出し、政治的ディープフェイク、IP盗難)だ。

密接に関連する2つのタスク:

- **音声クローニング(TTS側):** テキスト+5秒参照音声 → その音声でのオーディオ。
- **音声変換(音声側):** ソースオーディオ(人A、X言う) + 人Bの参照音声 → B、X言うのオーディオ。

両方は波形を(コンテンツ、話者、プロソディ)に計算し、1つのソースからコンテンツと別のソースからの話者を再結合。

2026年で出荷する下でのキー制約: **透かしとコンセント門は法的にEU(AI法、2026年8月施行)およびカリフォルニア(AB 2905、2025年有効)で必須だ**。パイプラインは知覚不可能な透かしを発散し、非合意クローンを拒否する必要があります。

## コンセプト

![音声クローニング対変換: 計算、話者スワップ、再結合](../assets/voice-cloning.svg)

**ゼロショットクローニング。** 数千の話者で訓練されたモデルに5秒クリップを渡す。話者エンコーダーはクリップを話者埋め込みにマップ; TTSデコーダーはその埋め込みプラステキストを条件する。

使用: F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少ショットファインチューニング。** 対象音声の5-30分を記録。LoRA-ファインチューン時間。品質は「大丈夫」から「区別不可能」に跳躍。CoquiとElevenLabsの両方がこのパターンをサポート; コミュニティはF5-TTSで使用。

**音声変換(VC)。** 2つのファミリー:

- **認識合成。** ASRのようなモデルを実行してコンテンツ表現を抽出(例えば、ソフトフォノム後ろPPG)、その後対象話者埋め込みで再合成。言語とアクセントに対して強固。KNN-VC (2023)、Diff-HierVC (2023)で使用。
- **分離。** ボトルネックの潜在空間でコンテンツ、話者、プロソディを分離するオートエンコーダーを訓練。推論でスワップ話者埋め込み。より低い品質だが高速。AutoVC (2019)、VITS-VC バリアント。

**ニューラルコーデックベースクローニング(2024+)。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox — オーディオを離散トークン SoundStream / EnCodecから扱う、大きな自動回帰またはフロー照合モデルをコーデックトークンの上で訓練。短いプロンプトでElevenLabsに匹敵する品質。

### 倫理ビット、ボルトオン

**透かし。** PerTh (Perth)とSilentCipher (2024)はオーディオに知覚的に16-32ビットIDを埋め込む。再エンコーディング、ストリーミング、共通編集を生き残る。本番準備完了オープンソース。

**コンセント門。** 検証可能なコンセント記録とすべてのクローン出力をペアにする必要があります。「I、Rohit、2026-04-22、このボイスを許可すること X 目的のために。」改ざん証拠ログに保存。

**検出。** AASIST、RawNet2、Wav2Vec2-AASISTは検出器として出荷。ASVspoof 2025チャレンジはElevenLabs、VALL-E 2、Bark出力に対してSOTA検出器の0.8–2.3%のEERを公開。

### 番号(2026)

| モデル | ゼロショット? | SECS(対象sim) | WER (インテリ。) | パラム |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | はい | 0.72 | 2.1% | 335M |
| XTTS v2 | はい | 0.65 | 3.5% | 470M |
| OpenVoice v2 | はい | 0.70 | 2.8% | 220M |
| VALL-E 2 | はい | 0.77 | 2.4% | 370M |
| VoiceBox | はい | 0.78 | 2.1% | 330M |

SECS > 0.70は、ほとんどのリスナーにとって対象からほぼ区別不可能。

## ビルドしてみよう

### ステップ1: 認識合成で分解(code/main.pyのコードのみデモ)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念的にシンプル; 実装マスは`tts_model`と話者エンコーダーにある。

### ステップ2: F5-TTSでゼロショットクローン

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参照トランスクリプトは正確にオーディオに一致する必要があります; 不一致はアライメント壊す。

### ステップ3: KNN-VCでの音声変換

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VCは WavLMを実行してソースとターゲットプールのフレームごとの埋め込みを抽出、その後各ソースフレームをプール内の最も近いネイバーで置き換える。非パラメトリック、ターゲット音声の1分で機能。

### ステップ4: 透かしを埋め込む

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # ペイロードバイトを返す
```

約32ビットのペイロード、MP3再エンコードおよび軽いノイズ後に検出可能。

### ステップ5: コンセント門

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用方法

2026年スタック:

| 状況 | 選択 |
|-----------|------|
| 5秒ゼロショットクローン、オープンソース | F5-TTS または OpenVoice v2 |
| 商用本番クローニング | ElevenLabs Instant Voice Clone v2.5 |
| 音声変換(書き直し) | KNN-VC または Diff-HierVC |
| 多話者ファインチューン | StyleTTS 2 +話者アダプタ |
| クロスリンガルクローニング | XTTS v2 または VALL-E X |
| ディープフェイク検出 | Wav2Vec2-AASIST |

## ピットフォール

- **不一致参照トランスクリプト。** F5-TTSと類似は参照テキストが参照オーディオに正確に一致する必要、句読点を含めて。
- **残響参照。** エコーはクローンを殺す。乾燥して、近い-マイク記録。
- **感情不一致。** トレーニング参照「陽気」は陽気なクローンを生成すべてのこと。マッチ参照感情をターゲット使用。
- **言語漏洩。** 英語話者をクローンして、その後フランス語で話すようモデルに尋ねるとアクセントは多くの場合に実行; クロスリンガルモデル(XTTS、VALL-E X)を使用。
- **透かしなし。** 2026年8月からEUで出荷不可法的。

## 出荷しよう

`outputs/skill-voice-cloner.md`として保存。コンセント門+透かし+品質ターゲットでクローニングまたは変換パイプラインを設計。

## 演習

1. **簡単。** `code/main.py`を実行。スワップ前後の2つの「話者」の間のコサインを計算することで、話者埋め込みスワップを実証。
2. **中程度。** OpenVoice v2を使用して、独自の音声をクローン。SECSを参照対クローンの間で測定。Whisperを介したCERを測定。
3. **難しい。** 20クローンにSilentCipher透かしを適用、128 kbps MP3エンコード+デコードを通して実行、ペイロード検出。ビット精度を報告。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| ゼロショットクローン | 5秒十分 | 事前訓練モデル+話者埋め込み; 訓練なし。 |
| PPG | 音素後ろPPG | 言語不知論コンテンツ担当として使用されるフレームごとのASR後ろ。 |
| KNN-VC | 最も近いネイバー変換 | 各ソースフレームを最も近いターゲットプールフレームで置換。 |
| ニューラルコーデックTTS | VALL-Eスタイル | EnCodec/SoundStreamトークンの上のARモデル。 |
| 透かし | 知覚的署名 | オーディオに埋め込みビット、再エンコード後に生き残る。 |
| SECS | クローニング忠実度 | 対象とクローン話者埋め込みの間のコサイン。 |
| AASIST | ディープフェイク検出器 | アンチスプーフモデル; 合成音声を検出。 |

## 参考文献

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — オープンソースSOTA ゼロショットクローニング。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) および [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — ニューラル-コーデックTTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 分離ベース音声変換。
- [Baas、Waubert de Puiseau、Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 検索ベースVC。
- [SilentCipher (2024) — オーディオ透かし](https://github.com/sony/silentcipher) — 本番準備完了32ビット オーディオ透かし。
- [ASVspoof 2025結果](https://www.asvspoof.org/) — 検出器対合成機軍拡競争、2026年更新。
