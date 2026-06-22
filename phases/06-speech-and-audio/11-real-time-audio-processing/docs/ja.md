# リアルタイム音声処理

> バッチパイプラインはファイルを処理する。リアルタイムパイプラインは、次の20ミリ秒が到着する前に、その20ミリ秒を処理する。すべての会話型 AI、放送スタジオ、テレフォニーボットは、このレイテンシ予算によって生かされも殺されもする。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 02 (Spectrograms)、Phase 6 · 04 (ASR)、Phase 6 · 07 (TTS)
**所要時間:** 約75分

## 問題

生きているように感じられる音声アシスタントが欲しい。人間の会話における話者交代のレイテンシは約230ミリ秒（沈黙から応答まで）だ。500ミリ秒を超えるとロボットのように感じられ、1500ミリ秒を超えると壊れているように感じられる。2026年における完全な**聞く → 理解する → 応答する → 話す**ループの予算は次のとおりだ。

| ステージ | 予算 |
|-------|--------|
| マイク → バッファ | 20 ms |
| VAD | 10 ms |
| ASR（ストリーミング） | 150 ms |
| LLM（最初のトークン） | 100 ms |
| TTS（最初のチャンク） | 100 ms |
| レンダリング → スピーカー | 20 ms |
| **合計** | **約400 ms** |

Moshi（Kyutai, 2024）は全二重で200ミリ秒を記録した。GPT-4o-realtime（2024）は約320ミリ秒を記録する。2022年のカスケード型パイプラインは2500ミリ秒で出荷されていた。この10倍の改善は3つの技術から生まれた。(1) あらゆる箇所でのストリーミング、(2) 部分結果を用いた非同期パイプライン化、(3) 中断可能な生成。

## コンセプト

![リングバッファ、VAD ゲート、中断を備えたストリーミング音声パイプライン](../assets/real-time.svg)

**フレーム / チャンク / ウィンドウ。** リアルタイム音声は固定サイズのブロックとして流れる。一般的な選択: 20ミリ秒（16 kHz で320サンプル）。下流のすべては、このリズムに追いつかなければならない。

**リングバッファ。** 固定サイズの循環バッファ。プロデューサスレッドが新しいフレームを書き込み、コンシューマスレッドが読み込む。ホットパスでのメモリ確保を防ぐ。サイズ ≈ 最大レイテンシ × サンプルレート。2秒の16 kHz リング = 32,000サンプル。

**VAD（Voice Activity Detection、音声区間検出）。** 誰も話していないときに下流の処理をゲートする。Silero VAD 4.0（2024）は CPU 上で30ミリ秒のフレームあたり1ミリ秒未満で動作する。`webrtcvad` は古い代替手段だ。

**ストリーミング ASR。** 音声が到着するにつれて部分的な文字起こしを出力するモデル。ストリーミングモードの Parakeet-CTC-0.6B（NeMo, 2024）は320ミリ秒のレイテンシで2〜5% の WER を達成する。Whisper-Streaming（Macháček et al., 2023）は Whisper をチャンク化し、約2秒のレイテンシでほぼストリーミングを実現する。

**中断。** ユーザーがアシスタントの発話中に話し始めたとき、あなたは (a) バージイン（割り込み）を検出し、(b) TTS を停止し、(c) 残りの LLM 出力を破棄しなければならない。すべて100ミリ秒以内に行わないと、ユーザーは耳が聞こえないアシスタントだと感じる。

**WebRTC Opus トランスポート。** 20ミリ秒フレーム、48 kHz、適応的ビットレート8〜128 kbps。ブラウザおよびモバイルの標準。LiveKit、Daily.co、Pion が2026年の音声アプリ構築スタックだ。

**ジッタバッファ。** ネットワークパケットは順不同 / 遅延して到着する。ジッタバッファは並べ替えと平滑化を行う。小さすぎると音声に隙間ができ、大きすぎるとレイテンシが増える。一般的には60〜80ミリ秒。

### よくある落とし穴

- **スレッド競合。** Python の GIL + 重いモデルは音声スレッドを枯渇させうる。C コールバック型の音声ライブラリ（sounddevice、PortAudio）を使い、Python をホットパスから外しておくこと。
- **サンプルレート変換のレイテンシ。** パイプライン内でのリサンプリングは5〜20ミリ秒を追加する。事前にリサンプリングするか、ゼロレイテンシのリサンプラ（PolyPhase、`soxr_hq`）を使うこと。
- **TTS のプライミング。** Kokoro のような高速な TTS でも、最初のリクエストで100〜200ミリ秒のウォームアップがある。モデルをキャッシュし、最初の本番ターンの前にダミー実行でウォームアップしておくこと。
- **エコーキャンセル。** AEC がないと、TTS 出力がマイクに再入力され、ボット自身の声で ASR が起動してしまう。WebRTC AEC3 がオープンソースのデフォルトだ。

## 作ってみる

### ステップ1: リングバッファ

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量が最大バッファリングレイテンシを決める。16 kHz で32,000サンプル = 2秒。

### ステップ2: VAD ゲート

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

本番環境では Silero VAD に置き換える。

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### ステップ3: ストリーミング ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### ステップ4: 中断ハンドラ

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

非同期 I/O とキャンセル可能な TTS ストリーミングが鍵となる。WebRTC の peerconnection.stop() を音声トラックに対して呼ぶのが定石の方法だ。

## 使ってみる

2026年のスタック:

| レイヤー | 選択 |
|-------|------|
| トランスポート | LiveKit（WebRTC）または Pion（Go） |
| VAD | Silero VAD 4.0 |
| ストリーミング ASR | Parakeet-CTC-0.6B または Whisper-Streaming |
| LLM の最初のトークン | Groq、Cerebras、vLLM-streaming |
| ストリーミング TTS | Kokoro または ElevenLabs Turbo v2.5 |
| エコーキャンセル | WebRTC AEC3 |
| エンドツーエンドネイティブ | OpenAI Realtime API または Moshi |

## 落とし穴

- **安全策として500ミリ秒バッファリングする。** バッファ*こそが*あなたのレイテンシの下限だ。縮小すること。
- **スレッドをピン留めしない。** UI より優先度の低いスレッドで音声コールバックを実行する = 負荷時のグリッチ。
- **TTS チャンクが小さすぎる。** 200ミリ秒未満のチャンクはボコーダのアーティファクトを可聴にする。320ミリ秒のチャンクが最適点だ。
- **ジッタバッファがない。** 実際のネットワークはジッタが多い。平滑化なしではポップ音が発生する。
- **一発限りのエラーハンドリング。** 音声パイプラインはクラッシュ耐性を持たねばならない。1つの例外がセッションを殺す。

## 出荷する

`outputs/skill-realtime-designer.md` として保存する。各ステージごとに具体的なレイテンシ予算を持つリアルタイム音声パイプラインを設計する。

## 演習

1. **易。** `code/main.py` を実行する。リングバッファ + エネルギー VAD をシミュレートし、フェイクの10秒ストリームのステージごとのレイテンシを表示する。
2. **中。** `sounddevice` を使い、20ミリ秒フレームでマイクを処理し、各フレームで VAD の状態を表示するパススルーループを構築する。
3. **難。** `aiortc` で全二重のエコーテストを構築する: ブラウザ → WebRTC → Python → WebRTC → ブラウザ。1 kHz のパルスでガラスからガラスまでのレイテンシを計測する。

## 重要用語

| 用語 | 世間の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| リングバッファ | 循環キュー | 音声フレーム用の固定サイズ、ロックフリー（または SPSC ロック）の FIFO。 |
| VAD | 無音ゲート | 音声 vs 非音声を判定するモデルまたはヒューリスティック。 |
| ストリーミング ASR | リアルタイム STT | 音声が到着するにつれ部分テキストを出力する。先読みが限定的。 |
| ジッタバッファ | ネットワーク平滑化 | 順不同のパケットを並べ替えるキュー。一般的には60〜80ミリ秒。 |
| AEC | エコーキャンセル | スピーカーからマイクへのフィードバック経路を差し引く。 |
| バージイン | ユーザー割り込み | システムが TTS の途中でユーザー音声を検出する。再生をキャンセルしなければならない。 |
| 全二重 | 同時双方向 | ユーザーとボットが同時に話せる。Moshi は全二重だ。 |

## 参考文献

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — チャンク化されたほぼストリーミングの Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全二重200ミリ秒レイテンシ。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 本番グレードの音声エージェントオーケストレーション。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 1ミリ秒未満の VAD、Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — オープンソースのエコーキャンセル。
