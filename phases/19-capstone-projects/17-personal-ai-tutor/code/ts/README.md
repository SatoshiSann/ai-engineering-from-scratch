# レッスン 17 - パーソナル AI チューター（TypeScript ウェブアプリ）

カプストーンの TypeScript 側。Python 側は学習者モデルとチューターポリシーを提供し、このプロジェクトはウェブアプリの表面部分を担う: カリキュラム DAG ウォーカー、BKT スタイルの学習者モデル、そして2つの HTTP ルートの裏で動く FSRS-lite 間隔反復スケジューラ。

## レイアウト

```text
src/
  index.ts       エントリポイント: デモ（デフォルト）または HTTP サーバー (--serve)
  server.ts      Hono ルート (GET /lesson/next, POST /lesson/:id/submit)
  curriculum.ts  DAG フィクスチャ + Kahn トポロジカルソート + 次レッスン選択
  mastery.ts     MasteryStore (レッスンごとの BKT 風更新)
  repetition.ts  scheduleNextDue (間隔の倍増・半減、クランプ付き)
  types.ts       Lesson, Mastery, Pick
tests/
  curriculum.test.ts  トポロジカル順序、BKT 更新、FSRS スケジューリング
```

## 実行方法

```bash
npm install
npm run typecheck
npm test
npm start            # 自己終了するカリキュラムウォーク
npm run serve        # :8090 で HTTP サーバーを起動
```
