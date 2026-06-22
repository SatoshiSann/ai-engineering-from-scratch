# Capstone 19/02 — RAG over Codebase (TypeScript)

`../docs/en.md` に記載されたハイブリッド検索パイプラインを実装する、複数ファイル構成の TypeScript コード検索 API。オフライン動作、決定論的、6チャンクのサンプルコーパス、hono フェッチハンドラの背後で動作する node:http。

## ディレクトリ構成

```text
src/
  index.ts        エントリポイント; node:http を起動 + セルフプローブ + 終了コード0
  server.ts       hono ルート (/healthz, /query) と zod バリデーション済み POST ボディ
  retrieval.ts    runQuery + 密ベクトルと BM25 に対する RRF マージ
  index_store.ts  FNV-1a ハッシュ埋め込み器、コサイン類似度、フィールド重み付き BM25
  corpus.ts       6チャンクのサンプル (uploader / auth / client / catalog)
  types.ts        Chunk, RankedChunk, QueryResponse, anchor()
tests/
  index_store.test.ts
  retrieval.test.ts
  server.test.ts
```

## 実行方法

```bash
npm install
npm start                # API を起動し、3つのクエリをプローブして終了コード0
npm start -- --serve     # サーバーを起動したままにする; 停止は ctrl-c
npm test                 # tsx 経由で node --test ランナーを実行
npm run typecheck        # tsc --noEmit
```

非インタラクティブな `npm start` パスでは、`/healthz` が 200 を返すこと、およびすべてのプローブクエリが少なくとも1つの引用を返すことを検証します。ルート一覧:

- `GET /healthz` — `{ok, corpus}` を返す。
- `GET /query?q=...` — ハイブリッドクエリを実行する。
- `POST /query` — JSON `{q, topK?}`、zod でバリデーション（`topK` は50が上限）。
