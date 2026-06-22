# キャップストーン 19/03 — リアルタイム音声アシスタント (TypeScript)

`../docs/en.md` で解説されているストリーミング音声パイプラインのための、マルチファイル TypeScript Web クライアントハーネスです。オフラインのステートマシンシミュレーションと、`ws` パッケージをバックエンドとするライブ WebSocket サーバーを含みます。

## 構成

```text
src/
  index.ts        エントリーポイント。2つのオフラインセッションを実行し、ライブ ws を確認後、終了コード 0 で終了
  server.ts       hono /healthz + WebSocketServer 経由の ws アップグレード
  orchestrator.ts IDLE -> LISTENING -> WAITING -> THINKING -> SPEAKING (割り込み機能付き)
  vad.ts          ターン完了スコアラー + 合成 20ms フレームジェネレーター
  protocol.ts     zod 検証済みフレームエンベロープ (event / summary)
  types.ts        AudioChunk, Metrics, SessionOptions, SessionSummary
tests/
  vad.test.ts
  orchestrator.test.ts
  protocol.test.ts
```

## 実行方法

```bash
npm install
npm start                # 2つのオフラインセッションを実行 + ws セルフプローブ、終了コード 0 で終了
npm start -- --serve     # ws サーバーを起動し続ける。停止するには ctrl-c
npm test                 # tsx 経由の node --test ランナー
npm run typecheck        # tsc --noEmit
```

非インタラクティブな `npm start` の実行では、クリーンセッションが `first_audio_out` に到達すること、割り込みセッションで少なくとも1つの割り込みイベントが記録されること、ライブ WebSocket プローブがクローズ前に `summary` フレームを受信することを検証します。
