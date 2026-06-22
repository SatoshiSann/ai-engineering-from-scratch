# カプストーン — 完全なツールエコシステムを構築

> Phase 13 はすべてのピースを教えました。このカプストーンはそれらを1つの本番形状システムにワイヤリングします：ツール+リソース+プロンプト+タスク+ UIを備えた MCPサーバー、エッジ側のOAuth 2.1、RBACゲートウェイ、マルチサーバークライアント、A2Aサブエージェント呼び出し、コレクタへのOTelトレース、CI内のツール中毒検出、AGENTS.md + SKILL.mdバンドル。終わりまでに、すべての建築の選択を防ぐことができます。

**タイプ:** Build
**言語:** Python (stdlib、エンドツーエンドエコシステムハーネス)
**前提条件:** Phase 13 · 01 から 21
**所要時間:** 約120分

## 学習目標

- ツール、リソース、プロンプト、`ui://`アプリを備えたタスクを公開するMCPサーバーを構成します。
- RBACおよび固定ハッシュを実装するOAuth 2.1ゲートウェイでサーバーをフロンター。
- エンドツーエンドのOTel GenAI属性を使用してトレースするマルチサーバークライアントを作成。
- ワークロードの一部をA2Aサブエージェントにデリゲーションし、不透明性が保存されることを確認します。
- 他のエージェントがワークフローを再現できるように、全スタックを AGENTS.md + SKILL.md でパッケージング。

## 問題

「研究とレポート」システムをリリース：

- ユーザーが尋ねる：「エージェントプロトコルに関する2026年の最も引用された3つのarXivペーパーを要約してください。」
- システム：MCPを介してarXivで検索。専門の作成者エージェントへのペーパー要約をA2Aを介してデリゲーション。結果を集約。MCPアプリ`ui://`リソースとしてインタラクティブレポートをレンダー。OTelにすべてのステップをログに記録。

Phase 13のすべてのプリミティブが表示されます。これはおもちゃではありません — 2026年にAnthropicで配信された本番研究アシスタントシステム（Claude Research 製品）、OpenAI（Apps SDK を備えたGPT）、サードパーティはこの正確な形状を持っています。

## コンセプト

### アーキテクチャ

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### トレース階層

```
agent.invoke_agent
 ├── llm.chat (キックオフ)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (不透明な内部)
 ├── mcp.call -> tools/call generate_report (task 拡張)
 │    └── tasks/status ポーリング
 │    └── tasks/result (完了、ui:// リソース返却)
 └── llm.chat (最終合成)
```

1つのトレース ID。すべてのスパンは正しい`gen_ai.*`属性を持っています。

### セキュリティポスチャ

- OAuth 2.1 + PKCEとオーディエンスをゲートウェイにピンするリソースインジケータ。
- ゲートウェイは上流認証情報を保持します。ユーザーは決して見ません。
- RBAC：`alice`は`research:read`、`research:write`を持ち、すべてのツールを呼び出すことができます。`bob`は`research:read`を持ち、`generate_report`を呼び出すことはできません。
- 固定説明マニフェスト：ツールハッシュが変更されたサーバーをドロップ。
- Rule of Two 監査：信頼されていない入力、機密データ、結果のある操作を組み合わせるツールはありません。

### レンダリング

最終的な`generate_report`タスクはコンテンツブロック+`ui://report/current`リソースを返します。クライアントのホスト（Claude Desktop など）はインタラクティブダッシュボードをサンドボックス iframe でレンダー。ダッシュボードには、ソート済みペーパーリスト、引用数、および`host.callTool('summarize_paper', {arxiv_id})`を呼び出し可能なボタンが含まれています。ユーザーがクリックするペーパーの場合。

### パッケージング

全体は以下のように配信されます：

```
research-system/
  AGENTS.md                     # プロジェクト規約
  skills/
    run-research/
      SKILL.md                  # トップレベルワークフロー
  servers/
    research-mcp/               # MCP サーバー
      pyproject.toml
      src/
  agents/
    writer/                     # A2A エージェント
  gateway/
    config.yaml                 # RBAC + 固定マニフェスト
```

ユーザーは`docker compose up`でデプロイします。Claude Code、Cursor、Codex、opencode ユーザーは`run-research`スキルを呼び出すことで、システムをドライブできます。

### 各 Phase 13 レッスンが貢献したもの

| レッスン | カプストーンが使用するもの |
|--------|--------------------------|
| 01-05 | ツールインターフェース、プロバイダ互換性、並列呼び出し、スキーマ、linting |
| 06-10 | MCPプリミティブ、サーバー、クライアント、トランスポート、リソース+プロンプト |
| 11-14 | サンプリング、ルート+引き出し、非同期タスク、`ui://`アプリ |
| 15-17 | ツール中毒、OAuth 2.1、ゲートウェイ+レジストリ |
| 18 | A2A サブエージェント委譲 |
| 19 | OTel GenAI トレース |
| 20 | LLMレイヤーのルーティングゲートウェイ |
| 21 | SKILL.md + AGENTS.md パッケージング |

## 使用方法

`code/main.py`は前のレッスンのパターンを1つの実行可能なデモにステッチします。すべてstdlib、すべてのインプロセスなのでエンドツーエンドで読むことができます。研究レポートシナリオの完全なフローを実行します：ゲートウェイとのハンドシェイク、OAuth 2.1シミュレーション、マージされた tools/list、タスク生成 generate_report、ライター A2A 呼び出し、返却される ui:// リソース、出力される OTel スパン。

確認すべき点：

- すべてのホップ全体で1つのトレース ID。
- ゲートウェイポリシーは2番目のユーザーが書き込みをブロック。
- タスクライフサイクルは working → completed となり、テキストと ui:// コンテンツの両方を返します。
- A2A 呼び出しの内部状態はオーケストレータに対して不透明。
- AGENTS.md と SKILL.md は、別のエージェントがワークフローを再現する必要がある唯一のファイル。

## リリース

このレッスンは`outputs/skill-ecosystem-blueprint.md`を生成します。製品の需要（研究、要約、自動化）が与えられた場合、スキルは完全なアーキテクチャを生成します：どのMCPプリミティブ、どのゲートウェイコントロール、どのA2A呼び出し、どのテレメトリー、どのパッケージング。

## 演習

1. `code/main.py`を実行します。単一のトレースIDと スパンのネスト方法に注意します。デモが Phase 13 から触れるプリミティブの数をカウント。

2. デモを拡張：2番目のバックエンド MCPサーバー（例：`bibliography`）を追加し、ゲートウェイがそのツールを同じ名前空間にマージすることを確認します。

3. 偽のA2A ライターエージェントをサブプロセスで実行する実際のエージェントで置き換えます。レッスン19ハーネスを使用。

4. ルーティングゲートウェイにPII編集ステップをオーケストレータとLLMの間に追加。ユーザークエリのメールがスクラブされることを確認します。

5. このシステムを保守するチームメイトのためにAGENTS.mdを作成。読むために5分未満で、Cursor または Codex でカプストーンをドライブするために必要なすべてを提供する必要があります。

## キー用語

| 用語 | 人々が言うこと | それが実際に意味すること |
|------|------------------|---------------------------|
| Capstone | "Phase 13 統合デモ" | すべてのプリミティブを使用するエンドツーエンドシステム |
| Research and report | "シナリオ" | 検索、要約、レンダーパターン |
| Ecosystem | "すべてのピースを一緒に" | サーバー+クライアント+ゲートウェイ+サブエージェント+テレメトリー+パッケージ |
| Trace hierarchy | "単一トレース ID" | すべてのホップのスパンはトレースを共有。親子がスパンIDを通じて |
| Gateway-issued token | "推移的認証" | クライアントはゲートウェイのトークンのみを見えます。ゲートウェイは上流認証情報を保持 |
| Merged namespace | "1つのフラットリスト内のすべてのツール" | ゲートウェイでマルチサーバーマージ、衝突時プリフィックス |
| Opacity boundary | "A2A 呼び出しは内部を隠す" | サブエージェントの推論がオーケストレータに不可視 |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | プロジェクトコンテキスト+ワークフロー+ツール |
| Defense-in-depth | "複数のセキュリティレイヤー" | 固定ハッシュ、OAuth、RBAC、Rule of Two、監査ログ |
| Spec compliance matrix | "仕様が必要な配信内容" | デリバラブルを2025-11-25要件にマッピングするチェックリスト |

## 参考文献

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 統合参照
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — プロトコルの向かっている場所
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参照
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 正規トレース規約
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 本番エージェントランタイムパターン
