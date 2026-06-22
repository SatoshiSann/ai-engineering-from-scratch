# 音声区間検出とターンテイキング — Silero、Cobra、フラッシュトリック

> すべての音声エージェントは、2つの判断によって生かされも殺されもする: ユーザーは今話しているか、そして話し終えたか。VAD は前者に答える。ターン検出（VAD + 無音ハングオーバー + 意味的エンドポイントモデル）は後者に答える。どちらかを間違えると、アシスタントはユーザーの話を遮るか、いつまでも黙らないかのどちらかになる。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 11 (Real-Time Audio)、Phase 6 · 12 (Voice Assistant)
**所要時間:** 約45分

## 問題

音声エージェントが20ミリ秒のチャンクごとに行う3つの明確に異なる判断:

1. **このフレームは音声か？** — VAD。フレームごとのバイナリ。
2. **ユーザーは新しい発話を始めたか？** — オンセット検出。
3. **ユーザーは話し終えたか？** — エンドポインティング（ターン終了）。

素朴な答え（エネルギー閾値）は、あらゆるノイズ — 交通音、キーボード、群衆のざわめき — で失敗する。2026年の答え: Silero VAD（オープン、深層学習）+ ターン検出モデル（意味的エンドポインティング）+ VAD でキャリブレーションされた無音ハングオーバー。

## コンセプト

![VAD カスケード: エネルギー → Silero → ターン検出器 → フラッシュトリック](../assets/vad-turn-taking.svg)

### 3段階の VAD カスケード

**ティア1: エネルギーゲート。** 最も安価。RMS を -40 dBFS で閾値処理する。明らかな無音をフィルタするが、閾値を超えるあらゆるノイズで発火する。

**ティア2: Silero VAD**（2020-2026, MIT）。100万パラメータ。6000以上の言語で訓練済み。単一の CPU スレッド上で30ミリ秒チャンクあたり約1ミリ秒で動作する。FPR 5% で TPR 87.7%。オープンソースのデフォルト。

**ティア3: 意味的ターン検出器。** LiveKit のターン検出モデル（2024-2026）または独自の小さな分類器。「文の途中の一時停止」を「話し終えた」と区別する。単なる無音ではなく、言語的文脈（イントネーション + 直近の単語）を使う。

### 主要なパラメータとそのデフォルト

- **閾値。** Silero は確率を出力する。&gt; 0.5（デフォルト）または &gt; 0.3（高感度）で音声と分類する。低い閾値 = 最初の単語のクリッピングは減るが、誤検出は増える。
- **最小発話時間。** 250ミリ秒より短い音声を棄却する — 通常は咳や椅子の音。
- **無音ハングオーバー（エンドポインティング）。** VAD が0に戻った後、ターン終了を宣言する前に500〜800ミリ秒待つ。短すぎる → ユーザーを遮る。長すぎる → もたつきを感じる。
- **プリロールバッファ。** VAD が発火する前の300〜500ミリ秒の音声を保持する。「ねえ」がクリッピングされるのを防ぐ。

### フラッシュトリック（Kyutai 2025）

ストリーミング STT モデルには先読み遅延がある（Kyutai STT-1B は500ミリ秒、STT-2.6B は2.5秒）。通常は、発話終了後にその分だけ文字起こしを待つことになる。フラッシュトリック: VAD が発話終了を発火したとき、**STT にフラッシュ信号を送り**、即座の出力を強制する。STT は約4倍のリアルタイムで処理するため、500ミリ秒のバッファは約125ミリ秒で終わる。

エンドツーエンド: 125ミリ秒の VAD + フラッシュ STT = 会話的なレイテンシ。

### 2026年の VAD 比較

| VAD | TPR @ 5% FPR | レイテンシ | ライセンス |
|-----|--------------|---------|---------|
| WebRTC VAD（Google, 2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | 約1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | 約1 ms | 商用 |
| pyannote segmentation | 95% | 約10 ms | MIT 系 |

Silero が正しいデフォルトだ。Cobra はコンプライアンス / 精度のアップグレードだ。エネルギーのみの VAD は2026年の本番環境には居場所がない。

## 作ってみる

### ステップ1: エネルギーゲート

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### ステップ2: Python での Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### ステップ3: ターン終了のステートマシン

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### ステップ4: フラッシュトリックの骨格

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

これが機能するには STT（Kyutai、Deepgram、AssemblyAI）がフラッシュをサポートしている必要がある。Whisper ストリーミングはサポートしていない — ブロックベースで、常にチャンクを待つ。

## 使ってみる

| 状況 | VAD の選択 |
|-----------|-----------|
| オープン、高速、汎用 | Silero VAD |
| 商用コールセンター | Cobra VAD |
| オンデバイス（スマホ） | Silero VAD ONNX |
| 研究 / 話者分離 | pyannote segmentation |
| 依存ゼロのフォールバック | WebRTC VAD（レガシー） |
| ターン終了品質が必要 | Silero + LiveKit ターン検出器を重ねる |

経験則: 本当に他に選択肢がない場合を除き、エネルギーのみの VAD を絶対に出荷しないこと。

## 落とし穴

- **固定閾値。** 静かな環境では機能し、騒がしい環境では失敗する。オンデバイスでキャリブレーションするか、Silero に切り替えること。
- **無音ハングオーバーが短すぎる。** エージェントが文の途中で割り込む。会話的な発話では500〜800ミリ秒が最適点だ。
- **ハングオーバーが長すぎる。** もたつきを感じる。対象ユーザーで A/B テストすること。
- **プリロールバッファがない。** ユーザー音声の最初の200〜300ミリ秒が失われる。常にローリングのプリロールを保持すること。
- **意味的エンドポインティングを無視する。** 「うーん、考えさせて……」には長い一時停止が含まれる。ユーザーは考えの途中で遮られるのを嫌う。LiveKit のターン検出器などを使うこと。

## 出荷する

`outputs/skill-vad-tuner.md` として保存する。あるワークロードに対して VAD モデル、閾値、ハングオーバー、プリロール、ターン検出戦略を選択する。

## 演習

1. **易。** `code/main.py` を実行する。音声 + 無音 + 音声 + 咳のシーケンスをシミュレートし、3つの VAD ティアをテストする。
2. **中。** `silero-vad` をインストールし、5分の録音を処理し、最初の単語のクリッピングと誤発火の両方を最小化するよう閾値を調整する。適合率/再現率を報告する。
3. **難。** ミニターン検出器を構築する: Silero VAD + 最後の10単語の埋め込みに対する3層 MLP（sentence-transformers を使う）。手でラベル付けしたターン終了データセットで訓練する。Silero のみを F1 で10% 上回ること。

## 重要用語

| 用語 | 世間の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| VAD | 音声検出器 | フレームごとのバイナリ: これは音声か？ |
| ターン検出 | エンドポインティング | VAD + 無音ハングオーバー + 意味的エンドポイント。 |
| 無音ハングオーバー | 発話後の待機 | ターン終了を宣言する前の待機時間。500〜800ミリ秒。 |
| プリロール | 発話前バッファ | VAD 発火前の300〜500ミリ秒の音声を保持する。 |
| フラッシュトリック | Kyutai のハック | VAD → STT フラッシュ → 500ミリ秒の遅延の代わりに125ミリ秒。 |
| 意味的エンドポイント | 「止めるつもりだったか？」 | 単なる無音ではなく単語を見る ML 分類器。 |
| TPR @ FPR 5% | ROC 点 | 標準的な VAD ベンチマーク。Silero は87.7%、WebRTC は50%。 |

## 参考文献

- [Silero VAD](https://github.com/snakers4/silero-vad) — リファレンスのオープン VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商用の精度リーダー。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 200ミリ秒未満を実現するエンジニアリングトリック。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 本番環境での意味的エンドポインティング。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — レガシーのベースライン。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 話者分離グレードのセグメンテーション。
