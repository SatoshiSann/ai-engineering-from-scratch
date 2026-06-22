# Capstone 06 - DevOps トラブルシューティングエージェント（TypeScript）

`../main.py` のオンコールエージェント向け Slack 統合スケルトン。
スラッシュコマンドエンドポイントとインタラクティビティ（ボタンクリック）エンドポイントを公開し、
どちらも Slack の HMAC-SHA256 リクエスト署名および 5 分間のリプレイウィンドウで保護されています。
破壊的な修復操作は、Slack カードが承認された後にのみ実行されます。

## レイアウト

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts          # エントリポイント、デモ + HTTP サーバー
    server.ts         # hono アプリ、/slack/command + /slack/interactivity
    slack_verify.ts   # HMAC v0 検証 + タイミングセーフ比較
    agent.ts          # モック仮説ランカー
    blocks.ts         # Block Kit レスポンスビルダー
    types.ts          # Hypothesis、AgentReport、SlackResponse、OutboundCall
  tests/
    slack_verify.test.ts
    agent.test.ts
    server.test.ts
```

## 実行方法

```bash
npm install
npm run typecheck
npm test
npm start          # セルフチェックを1回実行して終了コード 0 で終了
npm run serve      # 127.0.0.1:<port> でインタラクティブ HTTP サーバーを起動
```

プレースホルダーのシークレットを上書きするには `SLACK_SIGNING_SECRET=...` を設定してください。
インタラクティブサーバーは選択されたポートを表示します（`PORT` が未設定の場合はランダムなポート番号になります）。

## テスト

tsx 経由の `node --test` ランナー。カバレッジ内容:

- Slack 署名検証: 有効な署名は通過、改ざんされた署名は拒否、古いタイムスタンプ（5 分以上のずれ）は拒否、数値でないタイムスタンプは拒否、定数時間比較の前に長さ不一致パスを検証。
- モックエージェント: OOM キーワードパス、CrashLoop キーワードパス、フォールバックパス。
- サーバー: `/health`、`/slack/command` の正常/改ざん/古いパス、`/slack/interactivity` の承認アクション。
