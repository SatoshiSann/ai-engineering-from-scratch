# キャップストーン 04 - マルチモーダル文書QA（TypeScript）

ドキュメントのページ画像URLと引用バウンディングボックスのJSONリストを返すビューアスケルトンです。HTMLレスポンスには、ページ画像上に引用領域を描画する小さなキャンバスオーバーレイスクリプトがインライン化されています。`../main.py` のPythonパイプラインと組み合わせて使用します。

## レイアウト

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts        # entrypoint, demo + HTTP server
    server.ts       # hono app, /health, /, /document/:id
    fixtures.ts     # 10-K table + Nature figure fixtures
    render.ts       # HTML index + per-document overlay renderer
    types.ts        # DocumentFixture, EvidenceRegion, BoundingBox
  tests/
    fixtures.test.ts
    render.test.ts
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

インタラクティブサーバーは `PORT` が未設定の場合に空きポートを選択し、選択されたURLをstdoutに出力します。インデックスには `/` を、デモオーバーレイには `/document/10k-acme-2025` を参照してください。また、`accept: application/json` を設定すると構造化レスポンスを取得できます。

## テスト

tsxを使用した `node --test` ランナー。テストはフィクスチャ検索（正常系・異常系）、5つの危険な文字に対するHTMLエスケープ、ドキュメントHTMLペイロード構造、およびhonoルート（200、404、コンテントネゴシエーション）をカバーしています。
