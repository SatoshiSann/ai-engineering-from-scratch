# Agent Workbench Pack

エージェント作業を確実に行いたい任意のリポジトリに組み込めるワークベンチです。

## 含まれるもの

- `AGENTS.md` パックの他のファイルへの簡易ルーター。
- `docs/` ルール、信頼性ポリシー、ハンドオフプロトコル、レビュアールーブリック。
- `schemas/` 状態、ボード、スコープコントラクト用の JSON Schema。
- `scripts/` 初期化、フィードバック実行、検証ゲート、ハンドオフ生成スクリプト。
- `bin/install.sh` 冪等性のあるインストーラー。

## クイックスタート

```
bin/install.sh
$EDITOR task_board.json
python3 scripts/init_agent.py
```

## バージョン管理

`VERSION` ファイルがコントラクトです。メジャーバージョンアップ時は状態の移行が必要です。
