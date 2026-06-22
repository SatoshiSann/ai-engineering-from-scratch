# キャップストーン 08 - 本番対応 RAG チャットボット（TypeScript）

Server-Sent Events 経由で引用リンク付きレスポンスをストリーミングするチャット UI のスケルトンです。`../main.py` の Python パイプラインと連携します。会話状態は `sessionId` をキーとするプロセス内 Map に保持されるため、同じセッション ID でマルチターンの対話を行うことができます。

## レイアウト

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts        # entrypoint, demo + HTTP server
    server.ts      # hono app, /, /chat/stream (SSE), /sessions, /health
    session.ts     # SessionStore (Map<sessionId, Session>)
    stream.ts      # SSE frame encoder + parser + mock retrieval + tokenizer
    types.ts        # Session, Turn, Citation, KbEntry, SseEvent
  tests/
    session.test.ts
    stream.test.ts
    server.test.ts
```

## 実行方法

```bash
npm install
npm run typecheck
npm test
npm start          # one self-check pass, exits 0
npm run serve      # interactive HTTP server on 127.0.0.1:<port>
```

インタラクティブサーバーは `PORT` が未設定の場合に空きポートを自動選択し、`/` にチャット HTML クライアントをマウントして `GET /chat/stream?sessionId=...&q=...` 経由でストリーミングを行います。デモクライアントは `EventSource` を使用し、`session`、`citations`、`token`、`done` イベントをリッスンします。

## テスト

tsx 経由の `node --test` ランナーを使用。カバレッジ範囲:

- SessionStore: 作成、ルックアップ、追記、一覧取得、存在しない ID への無操作。
- SSE エンコーダー + パーサーのラウンドトリップ; 管轄タグによる検索ブースト;
  トークナイザーフォールバック + "See also" 末尾テキスト。
- サーバー: `/`、`/health`、`/chat/stream` の正常系（session + citations +
  token + done）、q パラメータ欠如時の 400 エラー、マルチターンのセッション永続化、
  `/sessions` 一覧取得。
