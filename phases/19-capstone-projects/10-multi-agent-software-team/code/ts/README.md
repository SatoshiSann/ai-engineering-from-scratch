# マルチエージェント ソフトウェアチーム（TypeScript スケルトン）

マルチエージェント ソフトウェアチーム カプストーン用のマルチファイル TypeScript スケルトンです。
プランナー、コーダー、レビュアーの各エージェントはワークスペースを共有し、
コーディネーターを通じてローテーションします。ワークツリースタブは、
拒否リストとシェルメタキャラクター拒否機能を持つ execFile を介して子プロセスを起動します。

## レイアウト

- `src/index.ts` — デモランナー。
- `src/agent.ts` — ベース `Agent` クラスおよび `PlannerAgent`、`CoderAgent`、`ReviewerAgent`。
- `src/coordinator.ts` — ラウンドロビンループとローテーション追跡。
- `src/workspace.ts` — 共有インメモリファイルシステムとメッセージログ。
- `src/runtime.ts` — 拒否リスト付き `child_process.execFile` ワークツリースタブ。
- `src/types.ts` — 共有型定義。
- `tests/*.test.ts` — `tsx` を介した `node --test` スタイルのテスト。

## インストール

```bash
npm install
```

## 実行

```bash
npm start
```

## 検証

```bash
npm run typecheck
npm test
```

## 仕様リファレンス

- ソースレッスン: `phases/19-capstone-projects/10-multi-agent-software-team/docs/en.md`
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) ロールベースのマルチエージェントフレームワーク。
