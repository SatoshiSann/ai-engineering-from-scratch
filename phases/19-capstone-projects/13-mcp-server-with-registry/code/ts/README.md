# レッスン 13 - 内部 MCP サーバー（TypeScript）

カプストーンの TypeScript パート。Python 側（`code/main.py`）がレジストリとポリシーゲートを提供し、このプロジェクトは MCP トランスポートを担当する。stdio 上で手書きの改行区切り JSON-RPC 2.0 を実装し、3つのモックインシデントツールを持つ。`@modelcontextprotocol/sdk` は使用せず、ワイヤー上の全バイトを直接確認できる。

## 構成

```text
src/
  index.ts      エントリポイント: フィクスチャデモ（デフォルト）または stdio ループ（--serve）
  transport.ts  stdin readline + フィクスチャリプレイ
  protocol.ts   initialize / tools/list / tools/call / shutdown
  tools.ts      3つのインシデントツール + エグゼキューター
  types.ts      JSON-RPC + ツール定義の型
tests/
  protocol.test.ts  ラウンドトリップ、リスト形式、ディスパッチ、パースエラー
```

## 実行

```bash
npm install
npm run typecheck
npm test
npm start            # 自己終了するフィクスチャデモ
npm run serve        # 実際の stdio ループ（stdin 待機）
```
