# ストリーミング音声対音声 — Moshi、Hibiki、全二重対話

> 2024-2026年は音声 AI を再定義した。Moshi は、200ミリ秒のレイテンシで同時に聞き、話す単一のモデルを出荷する。Hibiki は音声対音声の翻訳をチャンク単位で行う。どちらも ASR → LLM → TTS のパイプラインを捨て、Mimi コーデックトークン上の統合された全二重アーキテクチャを採用する。これが新しいリファレンス設計だ。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 6 · 13 (Neural Audio Codecs)、Phase 6 · 11 (Real-Time Audio)、Phase 7 · 05 (Full Transformer)
**所要時間:** 約75分

## 問題

レッスン11 + 12 から構築されたすべての音声エージェントには、300〜500ミリ秒前後という基本的なレイテンシの下限がある: VAD が発火し、STT が処理し、LLM が推論し、TTS が生成する。各ステージにはそれぞれの最小レイテンシがある。チューニングや並列化はできるが、パイプラインの形があなたを上限で縛る。

Moshi（Kyutai, 2024-2026）は別の問いを投げかける: もしパイプラインがなかったら？ 1つのモデルが音声を取り込み、音声を直接、連続的に出力し、テキストが必須のステージではなく中間的な「内なる独白」だったら？

その答えが**全二重音声対音声**だ。理論上のレイテンシは160ミリ秒（80ミリ秒の Mimi フレーム + 80ミリ秒の音響遅延）。単一の L4 GPU 上での実用レイテンシは200ミリ秒。これは最高クラスのパイプライン型音声エージェントが達成する値の半分だ。

## コンセプト

![Moshi アーキテクチャ: 2つの並列 Mimi ストリーム + 内なる独白テキスト](../assets/moshi-hibiki.svg)

### Moshi のアーキテクチャ

**入力。** 2つの Mimi コーデックストリーム、どちらも 12.5 Hz × 8コードブック:

- ストリーム1: ユーザー音声（Mimi エンコード、絶えず到着する）
- ストリーム2: Moshi 自身の音声（Moshi が生成する）

**トランスフォーマー。** 70億パラメータの Temporal Transformer が両方のストリームとテキストの「内なる独白」ストリームを処理する。80ミリ秒のステップごとに、それは次を行う:

1. 最新のユーザー Mimi トークン（8コードブック）を消費する。
2. 直近の Moshi Mimi トークン（生成された8コードブック）を消費する。
3. 次の Moshi テキストトークン（内なる独白）を生成する。
4. 次の Moshi Mimi トークン（小さな Depth Transformer 経由で8コードブック）を生成する。

3つすべてのストリーム — ユーザー音声、Moshi 音声、Moshi テキスト — が並列で動く。Moshi は話しながらユーザーを聞くことができ、ユーザーが割り込んだら自身を中断でき、主たる発話を壊さずに相づち（「うんうん」）を打つことができる。

**Depth Transformer。** フレーム内では、8つのコードブックは並列に予測されない — コードブック間に依存関係があるからだ。小さな2層の「depth transformer」が80ミリ秒以内にそれらを逐次予測する。これは AR コーデック LM の標準的な因数分解だ（VALL-E、VibeVoice でも使われる）。

### なぜ内なる独白テキストが役立つか

明示的なテキストがないと、モデルは音響ストリーム内で言語を暗黙的にモデル化しなければならない。Moshi の洞察: 音声と並んでテキストトークンを出力させること。テキストストリームは本質的に Moshi が話している内容の文字起こしだ。これは意味的な一貫性を向上させ、言語モデルヘッドの差し替えを容易にし、文字起こしを無料で手に入れさせてくれる。

### Hibiki: ストリーミング音声対音声翻訳

同じアーキテクチャを、翻訳ペアで訓練したもの。ソース音声を入力し、ターゲット言語の音声を連続的に出力する。Hibiki-Zero（2026年2月）は単語レベルで整合した訓練データの必要性を排除する — 文レベルのデータ + レイテンシ最適化のための GRPO 強化学習を使う。

当初は4つの言語ペアをサポートする。約1000時間で新しい言語に適応できる。

### より広い Kyutai スタック（2026年）

- **Moshi** — 全二重対話（フランス語が先行、英語も十分にサポート）
- **Hibiki / Hibiki-Zero** — 同時音声翻訳
- **Kyutai STT** — ストリーミング ASR（500ミリ秒または2.5秒の先読み）
- **Kyutai Pocket TTS** — CPU で動作する1億パラメータ TTS（2026年1月）
- **Unmute** — これらをパブリックサーバー上で組み合わせるフルパイプライン

L40S GPU 上のスループット: 3倍リアルタイムで64同時セッション。

### Sesame CSM — いとこ

Sesame CSM（2025）は似たアイデアを使う — Llama-3 バックボーンに Mimi コーデックヘッドを組み合わせたもの。ただし CSM は全二重ではなく単方向（文脈 + テキストを受け取り、音声を生成する）だ。市場で最良の「声の存在感」を持つ TTS だが、Moshi の全二重能力とは同じではない。

### 2026年の性能数値

| モデル | レイテンシ | ユースケース | ライセンス |
|-------|---------|----------|---------|
| Moshi | 200 ms（L4） | 全二重の英語 / フランス語対話 | CC-BY 4.0 |
| Hibiki | 12.5 Hz フレームレート | フランス語 ↔ 英語ストリーミング翻訳 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 5言語ペア、整合データ不要 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 文脈条件付き TTS | Apache-2.0 |
| GPT-4o Realtime | 約300 ms | クローズド、OpenAI API | 商用 |
| Gemini 2.5 Live | 約350 ms | クローズド、Google API | 商用 |

## 作ってみる

### ステップ1: インターフェース

Moshi は WebSocket サーバーを公開し、80ミリ秒の Mimi エンコード音声チャンクを受け取り、80ミリ秒の Mimi エンコード音声チャンクを返す。双方向に。絶えず。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### ステップ2: 全二重ループ

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

両方向が同時に動く。Python asyncio または Rust futures が標準的なトランスポートだ。

### ステップ3: 訓練の目的（概念的）

80ミリ秒のフレーム `t` ごとに:

- 入力: `user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 予測: `moshi_text[t]`、その後 `moshi_mimi[t, codebook_0..7]`

テキストが音声より先に予測される（内なる独白）。音声は depth transformer 内でコードブック逐次に予測される。

### ステップ4: Moshi が勝つ場面と勝たない場面

Moshi が勝つ:

- 安価なハードウェアでエンドツーエンド250ミリ秒未満。
- 自然な相づちと割り込み。
- パイプラインの接着コードがない。

Moshi が勝たない:

- ツール呼び出し（そのために訓練されていない。別の LLM 経路が必要）。
- 長い推論（Moshi は80億規模の対話モデルであり、Claude/GPT-4 ではない）。
- ニッチなトピックでの事実的正確性。
- ほとんどの本番エンタープライズユースケース（2026年でもパイプラインを使う）。

## 使ってみる

| 状況 | 選択 |
|-----------|------|
| 最低レイテンシの音声コンパニオン | Moshi |
| ライブ翻訳通話 | Hibiki |
| 音声デモ / 研究 | Moshi、CSM |
| ツールを持つエンタープライズエージェント | パイプライン（レッスン12）、Moshi ではない |
| 文脈中のカスタム音声 TTS | Sesame CSM |
| 音声対音声、任意の言語 | GPT-4o Realtime または Gemini 2.5 Live（商用） |

## 落とし穴

- **限定的なツール呼び出し。** Moshi は対話モデルであり、エージェントフレームワークではない。ツールにはパイプラインと組み合わせること。
- **特定の声への条件付け。** Moshi は単一の訓練済みペルソナを使う。クローニングは別の訓練実行だ。
- **言語カバレッジ。** フランス語 + 英語は優秀だが、その他は限定的。Hibiki-Zero は助けになるが、それでも訓練データは必要だ。
- **リソースコスト。** フル Moshi セッションは GPU スロットを占有する。安価なマルチテナント共有デプロイパターンではない。

## 出荷する

`outputs/skill-duplex-pipeline.md` として保存する。音声エージェントのワークロードに対して、理由とともにパイプライン vs 全二重アーキテクチャを選択する。

## 演習

1. **易。** `code/main.py` を実行する。2ストリーム + 内なる独白のアーキテクチャを記号的にシミュレートする。
2. **中。** HuggingFace から Moshi を取得し、サーバーを実行し、1つの会話をテストする。ユーザー発話終了から Moshi の応答開始までのウォールクロックレイテンシを計測する。
3. **難。** レッスン12のパイプラインエージェントを取り、20の一致したテスト発話で P50 レイテンシを Moshi と比較する。それでもパイプラインがアーキテクチャ的に勝つのはどんなときかをまとめる。

## 重要用語

| 用語 | 世間の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| 全二重 | 聞きながら話す | 同じモデル上で2つの音声ストリームが同時にアクティブ。 |
| 内なる独白 | モデルのテキストストリーム | Moshi が音声出力と並んでテキストトークンを出力する。 |
| Depth Transformer | コードブック間予測器 | 1つの80ミリ秒フレーム内で8コードブックを予測する小さなトランスフォーマー。 |
| Mimi | Kyutai のコーデック | 12.5 Hz × 8コードブック。セマンティック+音響。Moshi を支える。 |
| ストリーミング S2S | 音声 → 音声ライブ | パイプラインステージのない、チャンク単位の翻訳/対話。 |
| 相づち | 「うんうん」という反応 | Moshi が自身のターンを壊さずに小さな相づちを出せる。 |

## 参考文献

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 論文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 整合データなしのストリーミング翻訳。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 仕様。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — インストール + サーバー。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — クローズドな商用の同等品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — その背後にある STT/TTS フレームワーク。
