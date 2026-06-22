# コード移行エージェントダッシュボード（TypeScript スケルトン）

コード移行エージェントキャップストーンのダッシュボード層を対象とした、マルチファイルの TypeScript スケルトンです。エージェント（Python）はサンドボックス内で動作し、このサーバーはオペレーター向けに進捗を表示します。

## レイアウト

- `src/index.ts` — エントリーポイント。ティックをシミュレートし、オプションで HTTP を提供します。
- `src/server.ts` — `/`、`/dashboard`、`/migrations`、`/migrations/:id` の Hono ルート。
- `src/migrations.ts` — ファイルごとのステートマシンとシードデータ。
- `src/cost.ts` — ターン数とドル予算の制限。
- `src/types.ts` — 共有型定義。
- `tests/*.test.ts` — `tsx` を使用した `node --test` スタイルのテスト。

## インストール

```bash
npm install
```

## 実行

```bash
npm start         # オフライン: 40 ティックをシミュレートしてサマリーを表示
npm run serve     # PORT（デフォルト 8009）で HTML ダッシュボードを提供
```

## 検証

```bash
npm run typecheck
npm test
```

## 仕様リファレンス

- ソースレッスン: `phases/19-capstone-projects/09-code-migration-agent/docs/en.md`
- レシピ: [OpenRewrite](https://docs.openrewrite.org), libcst.
