# オーディオトランスフォーマー — Whisperアーキテクチャ

> オーディオは時間にわたる周波数の画像です。Whisperはメルスペクトログラムを食べて話し戻す ViT です。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 7 · 05 (完全なトランスフォーマー), Phase 7 · 08 (エンコーダ・デコーダ), Phase 7 · 09 (ViT)
**所要時間:** 約45分

## 問題

Whisper(OpenAI、Radford et al. 2022)の前、最先端の自動音声認識(ASR)はwav2vec 2.0とHuBERT — 自己教師機能抽出器とファインチューニングされたヘッドを意味していました。高品質ですが、コストのかかるデータパイプライン、ドメイン-脆弱。多言語音声認識は言語ファミリーごとに個別のモデルが必要でした。

Whisperは3つの賭けをしました：

1. **すべてに訓練。** インターネットから掻き集めた97言語にわたる弱くラベル付けされた680,000時間のオーディオ。クリーンな学術コーパスなし。音素ラベルなし。
2. **マルチタスク単一モデル。** タスクトークン経由で転写、翻訳、音声活動検出、言語ID、タイムスタンピングで共同訓練された1つのデコーダ。
3. **標準的なエンコーダ・デコーダトランスフォーマー。** エンコーダはログメルスペクトログラムを消費。デコーダは自己回帰的にテキストトークンを生成。ボコーダ、CTC、HMM なし。

結果：Whisper large-v3は多くのアクセント、ノイズ、クリーンなラベル付きデータがゼロな言語全体で堅牢です。2026年のすべてのオープンソース音声アシスタントとほとんどの商業用の既定の音声フロントエンドです。

## コンセプト

![Whisperパイプライン：オーディオ → mel → エンコーダ → デコーダ → テキスト](../assets/whisper.svg)

### ステップ1 — リサンプル + ウィンドウ

16 kHzでのオーディオ。30秒にクリップ/パッド。ログメルスペクトログラムを計算：80 mel ビン、10 ms ストライド → ~3,000フレーム × 80機能。これがWhisperが見る「入力画像」です。

### ステップ2 — 畳み込み茎

2つのConv1Dレイヤーでカーネル3とストライド2は3,000フレームを1,500に削減。大量のパラメータを追加しないでシーケンス長を半分にします。

### ステップ3 — エンコーダ

1,500タイムステップの24層(大規模)トランスフォーマーエンコーダ。サイン波位置エンコーディング、自己注意、GELU FFN。1,500 × 1,280隠れ状態を生成。

### ステップ4 — デコーダ

24層トランスフォーマーデコーダ。BPE語彙から自己回帰的にトークンを生成。これはGPT-2と少数のオーディオ固有の特殊トークンのスーパーセットです。

### ステップ5 — タスクトークン

デコーダプロンプトは、モデルに何をするか伝える制御トークンで始まります：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

または

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

モデルはこの規約で訓練されました。プレフィックスでタスクを制御。2026年の同等物の命令調整、ただし音声に適用。

### ステップ6 — 出力

ビーム検索(幅5)とログ確率閾値。`<|notimestamps|>`トークンが存在しない場合、タイムスタンプは0.02秒ごとのオーディオで予測されます。

### Whisperサイズ

| モデル | パラメータ | レイヤー | d_model | ヘッド | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4層デコーダ) |

Large-v3-turbo (2024)はデコーダを32層から4層に削減。WERポイント回帰なしで8倍高速デコード。そのデコード速度ロック解除は、Whisper-turboが2026年のリアルタイム音声エージェントのデフォルトである理由です。

### Whisperが行わないこと

- ジャレーション(誰が話しているか)なし。pyannoteとペアを組んでください。
- ネイティブではリアルタイムストリーミング — 30秒ウィンドウは固定。現代的なラッパー(`faster-whisper`、`WhisperX`)がVAD + オーバーラップでストリーミングを着ける。
- 外部チャンキングなしの30秒を超える長い形式のコンテキストなし。人間の音声は転写のための長範囲コンテキストが必要なため実践的に良好に機能します。

### 2026年のランドスケープ

| タスク | モデル | 注釈 |
|------|-------|-------|
| 英語ASR | Whisper-turbo、Moonshine | Moonshineはエッジで4倍高速 |
| 多言語ASR | Whisper-large-v3 | 97言語 |
| ストリーミングASR | faster-whisper + VAD | 150 msレイテンシターゲット達成可能 |
| TTS | Piper、XTTS-v2、Kokoro | エンコーダ・デコーダパターン、ただしWhisper形 |
| オーディオ + 言語 | AudioLM、SeamlessM4T | 1つのトランスフォーマー内のテキストトークン+オーディオトークン |

## ビルド

`code/main.py`を見てください。Whisperを訓練しません — ログメルスペクトログラムパイプライン + タスクトークンプロンプトフォーマッターを構築。これらは本番環境で実際に触る部分です。

### ステップ1：オーディオを合成

440 Hz で1秒のサイン波を16 kHzでサンプリング生成。16,000サンプル。

### ステップ2：ログメルスペクトログラム(簡略化)

完全なメルスペクトログラムはFFTが必要。`librosa`を必要とせずパイプラインを表示する簡略化されたフレーミング + フレーム単位エネルギー版を行います：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

フレーム = 25 ms、ホップ = 10 ms。Whisperのウィンドウと一致。フレーム単位エネルギーは教育目的のメルビンの代わり。

### ステップ3：30秒にパッド

Whisperは常に30秒チャンクを処理。スペクトログラムを3,000フレームにパッド(またはクリップ)。

### ステップ4：プロンプトトークンを構築

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

それが全体のタスク制御サーフェス。4トークンプレフィックス。

## 使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

より高速、OpenAI互換：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026年でWhisperを選ぶ場合：**

- 1つのモデルで多言語ASR。
- ノイズ、多様なオーディオの堅牢な転写。
- 研究/プロトタイプASR — 最速の開始点。

**何か他を選ぶ場合：**

- 超低遅延ストリーミングエッジ — Moonshineが同等品質で Whisper を上回る。
- リアルタイム対話型AI、<200 msが必要 — 専用ストリーミングASR。
- スピーカー分離 — Whisperはこれを行いません；pyannoteを着ける。

## シップ

`outputs/skill-asr-configurator.md`を見てください。スキルはASRモデル、デコーディングパラメータ、新しい音声アプリケーション用の前処理パイプラインを選択します。

## 演習

1. **簡単。** `code/main.py`を実行。16 kHzでの1秒信号のフレーム数が10 ms ホップで~100フレーム。30秒の場合：~3,000フレーム。
2. **中程度。** `numpy.fft`を使用して完全なログメルスペクトログラムを構築。80 mel ビンが`librosa.feature.melspectrogram(n_mels=80)`内で数値誤差と一致することを検証します。
3. **難しい。** ストリーミング推論を実装：オーディオを2秒オーバーラップで10秒ウィンドウにチャンク、各ウィンドウで Whisper を実行、トランスクリプトをマージ。5分ポッドキャストサンプルでシングルパスと単語エラー率を測定。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| メルスペクトログラム | 「オーディオ画像」 | 2D表現：1軸上の周波数ビン、他軸上の時間フレーム；セルごとのログスケールエネルギー。 |
| ログメル | 「Whisperが見るもの」 | メルスペクトログラムはログを通過；人間の騒音知覚に近似。 |
| フレーム | 「1つの時間スライス」 | サンプルの25 msウィンドウ；10 msストライドで重複。 |
| タスクトークン | 「音声のプロンプトプレフィックス」 | デコーダプロンプトの`<\|transcribe\|>`/`<\|translate\|>`のような特殊トークン。 |
| 音声活動検出(VAD) | 「音声を検出」 | ASR前の沈黙を削除するゲート；大幅にコストを削減。 |
| CTC | 「Connectionist Temporal Classification」 | ASRの従来の損失整列なし訓練；Whisperは使用しません。 |
| Whisper-turbo | 「小さいデコーダ、完全なエンコーダ」 | large-v3エンコーダ + 4層デコーダ；デコード8倍高速。 |
| Faster-whisper | 「本番ラッパー」 | CTranslate2再実装；int8量子化；OpenAIの参照より4倍高速。 |

## 参考文献

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper論文。
- [OpenAI Whisperリポ](https://github.com/openai/whisper) — 参照コード + モデルウェイト。`whisper/model.py`を読んで、Conv1D茎 + エンコーダ + デコーダトップツーボトムを~400行で確認。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — ステップ5-6で説明されているビーム検索 + タスクトークンロジック；500行、完全に可読。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前駆体；いくつかの設定でまだ SOTA 機能。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本番ラッパー、参照より4倍高速。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024年エッジ対応ASR、Whisper形ですがより小さい。
- [HuggingFace ブログ — 「Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers」](https://huggingface.co/blog/fine-tune-whisper) — メルスペクトログラムプリプロセッサとトークンタイムスタンプハンドリングを含む正規のファインチューニングレシピ。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完全な実装(エンコーダ、デコーダ、交差注意、生成)がレッスンのアーキテクチャ図をミラー。
