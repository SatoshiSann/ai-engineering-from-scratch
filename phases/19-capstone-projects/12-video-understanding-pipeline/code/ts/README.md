# レッスン 12 - 動画理解パイプライン（TypeScript UI）

キャップストーンの TypeScript 側。Python 側（`code/main.py`）がマルチベクターインデックスと時間的グラウンディングを担当する。このプロジェクトはダッシュボード側を提供する：パイプラインの4ステージ（chunk、embed、index、qa）を管理する Hono アプリ。

## ディレクトリ構成

```text
src/
  index.ts     エントリポイント: デモ（デフォルト）または HTTP サーバー（--serve）
  server.ts    Hono ルート（/, /jobs, /job/:id）+ HTML インデックス
  jobs.ts     JobStore + フィクスチャシーダー
  stages.ts    ステージの進行 + 全体ステータス
  types.ts     Stage、StageState、Job
tests/
  stages.test.ts  ジョブ状態遷移 + ストア
```

## 実行方法

```bash
npm install
npm run typecheck
npm test
npm start              # 自動終了デモ
npm run serve          # :8123 で HTTP サーバー起動
```
