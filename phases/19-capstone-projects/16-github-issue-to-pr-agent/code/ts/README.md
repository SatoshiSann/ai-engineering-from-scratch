# レッスン16 - GitHub Issue から PR を作成するエージェント（TypeScript webhook 受信サーバー）

キャップストーンの TypeScript 担当部分。Python 側はエージェントループとディスパッチャーを提供し、YAML 側は Actions ワークフローを提供する。このプロジェクトは GitHub App の webhook 受信サーバーであり、生のリクエストボディを HMAC で検証し、イベントタイプに応じてルーティングし、`issues.opened` に対してスタブエージェントをディスパッチする。

## レイアウト

```text
src/
  index.ts    エントリーポイント: デモ（デフォルト）または HTTP サーバー（--serve）
  server.ts   Hono webhook 受信サーバー（POST /webhook）
  verify.ts   X-Hub-Signature-256 HMAC、タイミングセーフ
  router.ts   イベントタイプのルーティング（ping、issues、pull_request）
  agent.ts    スタブエージェント + 監査ログ
  types.ts    ペイロードと監査の型定義
tests/
  verify.test.ts  署名の検証成功・改ざん検出・ルーターのパステスト
```

## 実行方法

```bash
npm install
npm run typecheck
npm test
npm start            # 自動終了するデモ（プロセス内リプレイ）
npm run serve        # :8081 番ポートで HTTP サーバー起動
```

HMAC シークレットは `GH_WEBHOOK_SECRET` 環境変数から読み込まれる（デモ用デフォルト値は `demo-shared-secret`）。
