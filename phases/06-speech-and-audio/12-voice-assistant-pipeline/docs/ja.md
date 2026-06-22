# 音声アシスタントパイプラインを構築する — Phase 6 のキャップストーン

> レッスン01-11のすべてを縫い合わせる。聞き、推論し、話し返す音声アシスタントを構築する。2026年において、これは研究上の問題ではなく、解決済みのエンジニアリング上の問題だ。だが、統合の細部が出荷できるかどうかを決める。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**所要時間:** 約120分

## 問題

エンドツーエンドのアシスタントを構築する。

1. マイク入力（16 kHz モノラル）を取り込む。
2. ユーザー発話の開始 / 終了を検出する。
3. ストリーミングで文字起こしする。
4. 文字起こしを、ツール（タイマー、天気、カレンダー）を呼び出せる LLM に渡す。
5. LLM のテキストを TTS にストリーミングする。
6. ユーザーに音声を再生する。
7. ユーザーが応答の途中で割り込んだら停止する。

レイテンシ目標: ラップトップ CPU 上で、ユーザーが発話を終えてから800ミリ秒以内に最初の TTS 音声バイトを出す。品質目標: 単語の欠落なし、静寂での字幕の幻覚なし、ボイスクローニングの漏洩なし、プロンプトインジェクションの成功なし。

## コンセプト

![音声アシスタントパイプライン: マイク → VAD → STT → LLM+ツール → TTS → スピーカー](../assets/voice-assistant.svg)

### 7つのコンポーネント

1. **音声キャプチャ。** マイク → 16 kHz モノラル → 20ミリ秒チャンク。通常は Python の `sounddevice`、本番環境ではネイティブの AudioUnit/ALSA/WASAPI。
2. **VAD（レッスン11）。** Silero VAD @ 閾値0.5、最小発話250ミリ秒、無音ハングオーバー500ミリ秒。「開始」と「終了」を通知する。
3. **ストリーミング STT（レッスン4-5）。** Whisper-streaming、Parakeet-TDT、または Deepgram Nova-3（API）。部分 + 最終の文字起こし。
4. **ツール呼び出しを伴う LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。ツール用の JSON スキーマ。トークンをストリーミングする。
5. **ストリーミング TTS（レッスン7）。** Kokoro-82M（最速のオープン）または Cartesia Sonic（商用）。LLM が20トークン出した後に TTS を開始する。
6. **再生。** スピーカー出力。低帯域ネットワーク向けには opus エンコードする。
7. **中断ハンドラ。** TTS 再生中に VAD が発火したら、再生を停止し、LLM をキャンセルし、STT を再起動する。

### 必ず遭遇する3つの失敗モード

1. **最初の単語のクリッピング。** VAD の開始がわずかに遅れる。ユーザーの「ねえ」が欠落する。開始閾値を0.5ではなく0.3にすること。
2. **応答途中の割り込みの混乱。** ユーザーが割り込んだ後も LLM が生成を続け、アシスタントがユーザーにかぶせて話す。VAD → LLM キャンセルを配線すること。
3. **静寂の幻覚。** Whisper が無音のウォームアップフレームで「ご視聴ありがとうございました」を出力する。常に VAD でゲートすること。

### 2026年の本番リファレンススタック

| スタック | レイテンシ | ライセンス | 備考 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商用 API | 2026年の業界標準 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | ほぼオープン | DIY 向き |
| Moshi（全二重） | 200-300 ms | CC-BY 4.0 | 単一モデル。異なるアーキテクチャ、レッスン15 |
| Vapi / Retell（マネージド） | 300-500 ms | 商用 | 最速で立ち上げ可能。カスタマイズは限定的 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | オフライン | オープン | プライバシー / エッジ |

## 作ってみる

### ステップ1: チャンク化したマイクキャプチャ（擬似コード）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### ステップ2: VAD ゲートによるターンキャプチャ

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### ステップ3: ストリーミング STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### ステップ4: LLM ループ内でのツール呼び出し

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### ステップ5: 中断処理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使ってみる

7つのコンポーネントすべてをスタブモデルで配線した実行可能なシミュレーションについては `code/main.py` を参照すること。ハードウェアがなくてもパイプラインの形を確認できる。実装するには、スタブを次のものに差し替える。

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` または `openai-whisper`
- `openai`（`gpt-4o`）または `anthropic`
- `kokoro` または `cartesia`
- I/O 用に `sounddevice`

## 落とし穴

- **PII を永久にロギングする。** 全ターンの音声は、ほとんどの法域で PII だ。30日保持、保存時暗号化を行うこと。
- **バージインがない。** ユーザーは割り込んでくる。アシスタントは話すのをやめなければならない。
- **ブロックする TTS。** 同期的な TTS はイベントループをブロックする。非同期または別スレッドを使うこと。
- **ツール呼び出しのエラーハンドリングがない。** ツールは失敗する。LLM はエラーを受け取って一度リトライし、その後は優雅に劣化しなければならない。
- **過剰に攻撃的な幻覚フィルタ。** フィルタが強すぎるとアシスタントが「それにはお答えできません」を繰り返す。弱すぎると何でも言ってしまう。ホールドアウトセットでキャリブレーションすること。
- **ウェイクワードのオプションがない。** 常時リスニングはプライバシー上の責任問題だ。ウェイクワードゲート（Porcupine または openWakeWord）を追加すること。

## 出荷する

`outputs/skill-voice-assistant-architect.md` として保存する。予算 + スケール + 言語 + コンプライアンス制約を与えられたら、フルスタックの仕様を作成する。

## 演習

1. **易。** `code/main.py` を実行する。スタブモジュールで1つの完全なターンをエンドツーエンドでシミュレートし、ステージごとのレイテンシを表示する。
2. **中。** STT スタブを、録音済みの `.wav` に対する実際の Whisper モデルに置き換える。WER とエンドツーエンドのレイテンシを計測する。
3. **難。** ツール呼び出しを追加する: `get_weather`（任意の API）と `set_timer` を実装する。LLM をツール経由でルーティングし、ユーザーが「5分のタイマーをセットして」と言ったときに正しい関数が発火し、音声応答がそれを確認することを検証する。

## 重要用語

| 用語 | 世間の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| ターン | ユーザー + アシスタントの往復 | 1つの VAD で区切られたユーザー発話 + 1つの LLM-TTS 応答。 |
| バージイン | 割り込み | アシスタントの発話中にユーザーが話す。アシスタントは停止する。 |
| ウェイクワード | 「ねえアシスタント」 | 短いキーワード検出器。Porcupine、Snowboy、openWakeWord。 |
| エンドポインティング | ターンの終了 | ユーザーが終えたと判断する VAD + 最小無音の判定。 |
| プリロール | 発話前バッファ | 最初の単語のクリッピングを避けるため、VAD 発火前の200〜400ミリ秒の音声を保持する。 |
| ツール呼び出し | 関数呼び出し | LLM が JSON を出力し、ランタイムがディスパッチし、結果がループ内にフィードバックされる。 |

## 参考文献

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 本番グレードのリファレンス。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY 向きのフレームワーク。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — マネージドの音声ネイティブな経路。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全二重のリファレンス（レッスン15）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — ウェイクワードゲーティング。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM の関数呼び出し。
