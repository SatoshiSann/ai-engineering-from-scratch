# Capstone 19/01 — ターミナルネイティブ コーディングエージェント (TypeScript)

`../docs/en.md` で説明されている plan/act/observe ループを実装した、マルチファイル TypeScript ハーネスです。オフライン・決定論的・ネットワーク呼び出しなし。

## レイアウト

```text
src/
  index.ts     エントリポイント。スクリプト化されたデモと評価を実行し、0 で終了
  repl.ts      インタラクティブコマンドパーサー（run / eval / help / quit）
  harness.ts   フックバスを通じて接続された plan-act-observe ループ
  hooks.ts     8イベントフックバスと破壊的コマンドガード
  model.ts     デモを駆動するスクリプト化されたオフライン LLM
  tools.ts     zod バリデーション済み引数を持つ read_file + run_shell
  plan.ts     PlanState（todo 書き換え）+ Budget（ターン / トークン / ドル上限）
  eval.ts      3つのオフラインタスクにわたる小さな合否カウンター
  types.ts     共有型定義
tests/
  harness.test.ts
  tools.test.ts
```

## 実行方法

```bash
npm install
npm start                # スクリプト化されたデモ + オフライン評価を実行し、0 で終了
npm start -- --repl      # インタラクティブハーネス REPL を開く
npm test                 # tsx 経由で node --test ランナーを実行
npm run typecheck        # tsc --noEmit
```

非インタラクティブな `npm start` パスは、評価が `passed=3
failed=0` を報告すること、およびスクリプト化された実行がすべて完了したプランに収束することをアサートします。いずれかがずれた場合、実行は失敗します。
