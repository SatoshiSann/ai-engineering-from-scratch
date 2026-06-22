# LLM 可観測性ダッシュボード（TypeScript スケルトン）

LLM 可観測性ダッシュボード・キャップストーン用のマルチファイル TypeScript スケルトン。
Hono サーバーが OpenTelemetry GenAI スパンを受信し、10k リングバッファに保持した上で、p50/p95/p99 レイテンシおよびモデルごとのコストをレンダリングします。

## ファイル構成

- `src/index.ts` — エントリポイント。合成スパンをシードし、オプションで HTTP を提供します。
- `src/server.ts` — `/trace`、`/`、`/dashboard`、`/dashboard.json`、`/healthz` の Hono ルート。
- `src/spans.ts` — `RingBuffer` と `ObservabilityStore`（デフォルト 10k スパン）。
- `src/rollup.ts` — `percentile` と `rollUpByModel`。
- `src/pricing.ts` — 2026 年のモデルごとの価格とコスト計算ヘルパー。
- `src/types.ts` — 共通型定義。
- `tests/*.test.ts` — `tsx` 経由の `node --test` スタイルテスト。

## インストール

```bash
npm install
```

## 実行

```bash
npm start         # 1200 件の合成スパンをシードしてロールアップを表示します
npm run serve     # HTTP インジェスト＋ダッシュボードを PORT（デフォルト 8011）で起動します
```

## 検証

```bash
npm run typecheck
npm test
```

## 仕様リファレンス

- ソースレッスン: `phases/19-capstone-projects/11-llm-observability-dashboard/docs/en.md`
- [OpenTelemetry GenAI セマンティック規約](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
