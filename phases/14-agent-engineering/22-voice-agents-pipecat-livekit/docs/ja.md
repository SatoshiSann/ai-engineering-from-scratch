# ボイスエージェント：Pipecat と LiveKit

> ボイスエージェントは 2026 年のファーストクラスの本番環境カテゴリです。Pipecat は Python フレームベースパイプライン（VAD → STT → LLM → TTS → トランスポート）を提供します。LiveKit Agents は AI モデルをユーザーに WebRTC 経由で橋掛けします。本番環境レイテンシ目標はプレミアムスタック用の 450–600ms エンドツーエンドに到達します。

**タイプ:** 学習
**言語:** Python（標準ライブラリ）
**前提条件:** Phase 14 · 01（エージェントループ）、Phase 14 · 12（ワークフローパターン）
**所要時間:** 約 60 分

## 学習目標

- Pipecat のフレームベースパイプラインを説明してください：DOWNSTREAM（source→sink）と UPSTREAM（制御）
- 正規的なボイスパイプラインステージと Pipecat がサポートするトランスポートを挙げてください
- LiveKit Agents の 2 つのボイスエージェントクラス（MultimodalAgent、VoicePipelineAgent）と各々がいつフィットするかを説明してください
- 2026 年の本番環境レイテンシ期待値と、それらが建築選択をどう運転するかをまとめてください

## 問題

ボイスエージェントは TTS がボルトオンされたテキストループではありません。レイテンシ予算は厳しい（~600ms）、部分的オーディオはデフォルトです、ターン検出はモデルです、トランスポートは電話通信 SIP から WebRTC まで範囲です。フレームベースパイプライン（Pipecat）を構築するか、プラットフォーム（LiveKit）に頼るか。

## コンセプト

### Pipecat（pipecat-ai/pipecat）

- Python フレームベースパイプラインフレームワーク
- `Frame` → `FrameProcessor` チェーン
- 2 つのフロー方向：
  - **DOWNSTREAM** — source → sink（オーディオイン、TTS アウト）
  - **UPSTREAM** — フィードバックと制御（キャンセル、メトリクス、barge-in）
- `PipelineTask` はライフサイクルをイベント（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）と observersでメトリクス/トレーシング/RTVI 用に管理

典型的なパイプライン：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

トランスポート：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows は構造化会話（ステートマシン）を追加します。Pipecat Cloud は管理ランタイムです。

### LiveKit Agents（livekit/agents）

- WebRTC 経由でユーザーに AI モデルを橋掛け
- キーコンセプト：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`
- 2 つのボイスエージェントクラス：
  - **MultimodalAgent** — OpenAI Realtime または同等経由のダイレクトオーディオ
  - **VoicePipelineAgent** — STT → LLM → TTS カスケード；テキストレベルの制御を与えます
- トランスフォーマーモデル経由のセマンティックターン検出
- ネイティブ MCP 統合
- SIP 経由の電話通信
- LiveKit Inference 経由の 50 以上のモデルに API キーなし；プラグイン経由の 200 以上

### 商用プラットフォーム

Vapi（最適化プレミアムスタック上で ~450–600ms）と Retell（~600ms エンドツーエンド、180 テストコール全体）はこれらの上に構築されます。WebRTC チームなしで管理ボイススタックが必要なとき、プラットフォームを選びます。

### このパターンがうまくいかない場所

- **barge-in ハンドリングなし。** ユーザーが割り込み；エージェントは話し続けます。Pipecat の UPSTREAM キャンセルフレーム、LiveKit の同等が必要です
- **STT 確実性無視。** 低信頼度トランスクリプトが gospel として LLM に供給されます。信頼度でゲートするか、確認をリクエストしてください
- **TTS 中盤カットオフ。** パイプラインが utterance の中盤でキャンセルするとき、TTS は知る必要があるか、オーディオをカットしてください
- **レイテンシ予算無視。** すべてのコンポーネントが 50–200ms を追加します。出荷する前にチェーンを合計してください

### 典型的な 2026 年レイテンシ

- VAD：20–60ms
- STT 部分：100–250ms
- LLM ファーストトークン：150–400ms
- TTS ファーストオーディオ：100–200ms
- トランスポート RTT：30–80ms

エンドツーエンド 450–600ms はプレミアム。800–1200ms は一般的。> 1500ms は何かが壊れているように見えます。

## ビルド

`code/main.py` はスクリプト化 processors を持つフレームベース toy パイプライン：

- `Frame` タイプ（オーディオ、トランスクリプト、テキスト、tts_audio、制御）
- `Processor` インターフェース `process(frame)` をした
- 5 ステージパイプライン（VAD → STT → LLM → TTS → トランスポート）をスクリプト化 processors として
- barge-in をデモンストレートする UPSTREAM キャンセルフレーム

実行：

```
python3 code/main.py
```

トレースは通常フローと TTS を utterance の中盤で停止する barge-in キャンセルを示します。

## 使用方法

- **Pipecat** 完全制御用 — カスタム processors、Python ファースト、プラガブルプロバイダー
- **LiveKit Agents** WebRTC ファーストデプロイメントと電話通信用
- **Vapi / Retell** WebRTC チームなしの hosted ボイスエージェント用
- **OpenAI Realtime / Gemini Live** ダイレクトオーディオイン/オーディオアウト（MultimodalAgent）用

## 出荷

`outputs/skill-voice-pipeline.md` は VAD + STT + LLM + TTS + トランスポート + barge-in ハンドリング付きの Pipecat 形状のボイスパイプラインをスキャフォールドします。

## 演習

1. toy パイプラインへのメトリクス observer を追加してください：ステージあたり秒あたりのフレームをカウント。レイテンシはどこに累積しますか？
2. 確実性ゲート STT を実装してください：閾値以下、「繰り返してもらえますか？」をリクエストしてください
3. セマンティックターン検出を追加してください：簡単なルール — トランスクリプトが「?」で終わる場合、ターンの終わり
4. Pipecat のトランスポートドックを読んでください。stdlib トランスポートを SmallWebRTCTransport 設定（stub）にスワップしてください
5. OpenAI Realtime 対同じクエリの STT+LLM+TTS カスケード を測定してください。テキストレベル制御がもたらすレイテンシコストは？

## キーターム

| 用語 | 人々が言うこと | 実際の意味 |
|------|----------------|----------|
| Frame | 「Event」 | パイプライン内のタイプ化データ単位（オーディオ、トランスクリプト、テキスト、制御） |
| Processor | 「Pipeline stage」 | process(frame) を持つハンドラー |
| DOWNSTREAM | 「Forward flow」 | Source から sink：オーディオイン、スピーチアウト |
| UPSTREAM | 「Feedback flow」 | 制御：キャンセル、メトリクス、barge-in |
| VAD | 「Voice activity detection」 | ユーザーが話しているとき を検出 |
| Semantic turn detection | 「Smart end-of-turn」 | ユーザーが完了しているという モデルベース決定 |
| MultimodalAgent | 「Direct audio agent」 | オーディオイン、オーディオアウト；中盤にテキストなし |
| VoicePipelineAgent | 「Cascade agent」 | STT + LLM + TTS；テキストレベル制御 |

## 参考文献

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — フレームベースパイプライン、processors、トランスポート
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + ボイスプリミティブ
- [Vapi](https://vapi.ai/) — 管理ボイスプラットフォーム
- [Retell AI](https://www.retellai.com/) — 管理ボイス、レイテンシ ベンチマーク
