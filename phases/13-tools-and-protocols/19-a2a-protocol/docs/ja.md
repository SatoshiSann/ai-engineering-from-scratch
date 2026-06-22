# A2A — エージェント間プロトコル

> MCPはエージェント・ツール。A2A（Agent2Agent）はエージェント・エージェント — 異なるフレームワークで構築された不透明なエージェントが協力できるようにするためのオープンプロトコル。2025年4月にGoogleによってリリースされ、2025年6月にLinux Foundationに寄付され、AWS、Cisco、Microsoft、Salesforce、SAP、ServiceNowを含む150以上のサポーターと共に2026年4月にv1.0に達しました。IBMのACPを吸収し、AP2支払拡張機能を追加しました。このレッスンはエージェントカード、タスクライフサイクル、および2つのトランスポートバインディングを説明します。

**タイプ:** Build
**言語:** Python (stdlib, エージェントカード + タスクハーネス)
**前提条件:** Phase 13 · 06 (MCP ファンダメンタル), Phase 13 · 08 (MCP クライアント)
**所要時間:** 約75分

## 学習目標

- エージェント・ツール（MCP）とエージェント・エージェント（A2A）のユースケースを区別します。
- `/.well-known/agent.json`でエージェントカードをスキルとエンドポイントメタデータで公開します。
- タスクライフサイクル（submitted → working → input-required → completed / failed / canceled / rejected）を説明します。
- パーツ（テキスト、ファイル、データ）とアーティファクトを出力として持つメッセージを使用します。

## 問題

カスタマーサービスエージェントは、レポート作成タスクを専門の作成者エージェントに委譲する必要があります。A2A前のオプション：

- カスタムREST API。機能しますが、ペアリングごとに一度限りです。
- 共有コードベース。2つのエージェントが同じフレームワークを実行する必要があります。
- MCP。フィットしません：MCPはツール呼び出しですが、2つのエージェントが各エージェントの不透明な内部推論を保持しながら協力することではありません。

A2Aがギャップを埋めます。それは、ライフサイクル、メッセージ、アーティファクトを持つ1つのエージェントが別のエージェントにタスクを送信するようにインタラクションをモデル化します。呼び出されたエージェントの内部状態は不透明のままです — 呼び出し元はタスク状態遷移と最終的な出力のみを見えます。

A2Aは「異なるフレームワーク全体のエージェントが互いに話し合う」プロトコルです。MCPを置き換えません。2つは相補的です。

## コンセプト

### エージェントカード

すべてのA2A準拠エージェントは`/.well-known/agent.json`でカードを公開します：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

検出はURLベース：カードを取得し、A2AエンドポイントのURLを学習し、スキルを列挙します。

### 署名付きエージェントカード（AP2）

AP2拡張機能（2025年9月）は、暗号署名をエージェントカードに追加します。パブリッシャーはJWTでカードに署名します。消費者が検証します。なりすましを防ぎます。

### タスクライフサイクル

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (メッセージ経由でループ)
```

クライアントは`tasks/send`で開始します。呼び出されたエージェントは状態を通じて遷移します。クライアントはSSEまたはポーリングを通じて状態更新に登録します。

### メッセージとパーツ

メッセージは1つ以上のパーツを実行します：

- `text` — プレーンコンテンツ。
- `file` — mimeTypeを持つ base64 blob。
- `data` — 型付きJSONペイロード（呼び出されたエージェント用の構造化入力）。

例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### アーティファクト

出力は生の文字列ではなく、アーティファクトです。アーティファクトは命名された、型付きの出力です：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

アーティファクトはチャンクとしてストリーミングできます。呼び出し元が蓄積します。

### 2つのトランスポートバインディング

1. **JSON-RPCを HTTPで。** `/a2a`エンドポイント、リクエスト用のPOST、ストリーミング用のオプションSSE。デフォルトバインディング。
2. **gRPC。** gRPCがネイティブなエンタープライズ環境向け。

両方のバインディングは同じ論理的なメッセージ形状を実行します。

### 不透明性の保存

重要な設計原則：呼び出されたエージェントの内部状態は不透明です。呼び出し元はタスク状態とアーティファクトを見えます。呼び出されたエージェントの思考チェーン、ツール呼び出し、サブエージェント委譲 — すべて不可視。これはツール呼び出しが透明なMCPとは異なります。

理由：A2Aにより、競合他社が内部を明かさずに協力することができます。A2Aは「このカスタマーサービスエージェントを呼び出す」ことができ、呼び出し元がそのエージェントがサービスを実装する方法を学ぶことなく。

### タイムライン

- **2025-04-09。** Googleが A2A を発表。
- **2025-06-23。** Linux Foundationに寄付。
- **2025-08。** IBMのACPを吸収。
- **2025-09。** AP2拡張機能（エージェント支払い）がシップ。
- **2026-04。** v1.0は150以上の支援組織でリリース。

### MCPとの関係

| 次元 | MCP | A2A |
|------|------|------|
| ユースケース | エージェント・ツール | エージェント・エージェント |
| 不透明性 | 透明ツール呼び出し | 不透明な内部推論 |
| 通常の呼び出し元 | エージェント ランタイム | 別のエージェント |
| 状態 | ツール呼び出し結果 | ライフサイクル付きタスク |
| 認可 | OAuth 2.1 (Phase 13 · 16) | JWT署名エージェントカード（AP2） |
| トランスポート | Stdio / Streamable HTTP | JSON-RPC via HTTP / gRPC |

特定のツール呼び出しを行う場合はMCPを使用します。全体のタスクを別のエージェントに委譲する場合はA2Aを使用します。多くの本番環境システムは両方を使用します：エージェントはツールレイヤーにMCPを使用し、協業レイヤーにA2Aを使用します。

## 使用方法

`code/main.py`は最小限のA2Aハーネスを実装します：研究エージェントはカードを公開し、ライターエージェントはPDFとテキスト命令を含むパーツで`tasks/send`を受け取り、working → input_required → working → completedを通じて遷移し、テキストアーティファクトを返します。すべてstdlib。メッセージ形状に焦点を当てるためのメモリ内トランスポートを使用します。

確認すべき点：

- エージェントカードJSON形状。
- タスクID割り当てと状態遷移。
- 混合型パーツを含むメッセージ。
- タスク途中の入力が必要なブランチ。
- 完了時のアーティファクト返却。

## リリース

このレッスンは`outputs/skill-a2a-agent-spec.md`を生成します。他のエージェントで呼び出し可能な新しいエージェントが与えられた場合、スキルはエージェントカードJSON、スキルスキーマ、エンドポイントブループリントを生成します。

## 演習

1. `code/main.py`を実行します。呼び出されたエージェントが明確化を求める入力が必要な一時停止を含む、完全なタスクライフサイクルをトレースします。

2. 署名付きエージェントカードを追加します。カードの正規JSONをHMACで署名します。検証者を書き、反転されたカードで失敗することを確認します。

3. タスクストリーミングを実装します：ライターエージェントはSSE上で3つの段階的なアーティファクトチャンクを出力し、呼び出し元がそれらを蓄積します。

4. MCPサーバーをラップするA2Aエージェントを設計します。各MCPツールをA2Aスキルにマッピングします。トレードオフに注意してください — どの不透明性が失われますか？

5. A2A v1.0発表を読み、2026年4月現在、どのフレームワークでもまだ実装されていない1つの機能を特定します。（ヒント：マルチホップタスク委譲に関連しています。）

## キー用語

| 用語 | 人々が言うこと | それが実際に意味すること |
|------|------------------|---------------------------|
| A2A | "エージェント間プロトコル" | 不透明なエージェント協力のためのオープンプロトコル |
| Agent Card | "`.well-known/agent.json`" | エージェントのスキルとエンドポイントを説明する公開メタデータ |
| Skill | "呼び出し可能なユニット" | エージェントがサポートする命名付き操作（MCPツールのアナログ） |
| Task | "委譲のユニット" | ライフサイクルと最終的なアーティファクトを持つ作業アイテム |
| Message | "タスク入力" | パーツ（テキスト、ファイル、データ）を実行 |
| Part | "型付きチャンク" | メッセージの`text` / `file` / `data`要素 |
| Artifact | "タスク出力" | 完了時に返された命名された、型付きの出力 |
| AP2 | "エージェント支払プロトコル" | 信頼と支払のための署名付きエージェントカード拡張機能 |
| Opacity | "ブラックボックス協業" | 呼び出されたエージェントの内部は呼び出し元から隠されています |
| Input-required | "タスク一時停止" | エージェントがより多くの情報を必要とするライフサイクル状態 |

## 参考文献

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 正規 A2A 仕様
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — リファレンス実装とSDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025年6月ガバナンス移転
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — ロードマップとパートナーモメンタム
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0リリースノートと後方互換性ガイダンス
