# MCP基礎 — プリミティブ、ライフサイクル、JSON-RPCベース

> MCP前のすべての統合は1回限り。モデルコンテキストプロトコル、最初Anthropicが2024年11月に配布し、今Linux Foundation のエージェンティックAI基礎によってステバード、発見と起動を標準化しするため、任意のクライアント任意サーバーと話すことができます。2025-11-25仕様は6つのプリミティブ（3つのサーバー、3つのクライアント）、3フェーズライフサイクル、およびJSON-RPC 2.0ワイヤ形式に名前。これらを学び、このフェーズのMCPチャプターの残り読書になります。

**タイプ:** 学習
**言語:** Python（stdlib、JSON-RPCパーサー）
**前提条件:** フェーズ13・01～05（ツールインターフェースと関数呼び出し）
**所要時間:** 約45分

## 学習目標

- すべての6つのMCPプリミティブ名（サーバー上の道具、リソース、プロンプト；クライアント上のルーツ、サンプリング、引出）とそれぞれ1つのユースケースを与える。
- 3フェーズライフサイクル（初期化、操作、シャットダウン）を説明し、各フェーズで誰が何のメッセージを送るかを述べ。
- JSON-RPC 2.0要求、応答、通知エンベロープをパースし放出。
- キャパビリティネゴシエーション`initialize`ときは何か説明し、それなしで何が破損するか。

## 問題

MCP前に、ツール使用エージェントはそれ自身のプロトコルを持った。CursorはMCP型だが互換性なしツールシステムを持った。Claude Desktopは異なるツール配布。VS Code のCopilot 拡張3番目を持った。「Postgres クエリ」ツール構築チームが3回同じツール書きました。各々のホストAPIに異なり。再利用することはコード複製を要求。

結果は1回限り統合のカンブリア爆発と生態系速度上のセーリングでした。

MCPはワイヤ形式を標準化することで修正。1つのMCPサーバーは、各MCPクライアント機能：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf、2026年4月までに300+クライアント。110M月間SDKダウンロード。10,000+の公開サーバー。Linux Foundation は2025年12月にステバードを承認しました。新しいエージェンティックAI基礎下。

このフェーズで使用仕様リビジョンは**2025-11-25**。非同期タスク（SEP-1686）、URL モード引出（SEP-1036）、ツール（SEP-1577）でのサンプリング、インクリメンタルスコープ同意（SEP-835）、およびOAuth 2.1リソース指示セマンティクスを追加。フェーズ13・09～16はそれらの拡張をカバー。このレッスンは基地で停止。

## コンセプト

### 3つのサーバープリミティブ

1. **道具。** 呼び出し可能アクション。フェーズ13・01と同じ4ステップループ。
2. **リソース。** 露出データ。読み取り専用コンテンツ URI でアドレス可能：`file:///path`、`db://query/...`、カスタムスキーム。
3. **プロンプト。** 再利用テンプレート。ホストUI内スラッシュコマンド；サーバーはテンプレート提供、クライアント引数埋め。

### 3つのクライアントプリミティブ

4. **ルーツ。** サーバーが触れるのに許可されたURI セット。クライアント宣言；サーバーは尊重。
5. **サンプリング。** サーバーは、クライアントのモデルに補完実行をリクエスト。サーバーがホストされたエージェントループサーバー側APIキー無し。
6. **引出。** サーバーはクライアントのユーザー飛行中構造化入力をリクエスト。フォームまたはURLs （SEP-1036）。

MCPの各キャパビリティはこれら6つのうち正確に1つに属す。フェーズ13・10～14は各々深くカバーします。

### ワイヤ形式：JSON-RPC 2.0

すべてのメッセージは以下フィールドを持つJSONオブジェクト：

- リクエスト：`{jsonrpc: "2.0", id, method, params}`。
- 応答：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — ID なし、応答予期されない。

基地仕様は~15メソッド持つ、プリミティブによってグループ。重要なもの：

- `initialize` / `initialized`（ハンドシェイク）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（サーバー対クライアント）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 3フェーズライフサイクル

**フェーズ1：初期化。**

クライアントはそれ自身の`capabilities`と`clientInfo`送信`initialize`。サーバーはそれ自身の`capabilities`、`serverInfo`、および仕様バージョンそれが話す応答。クライアントは`notifications/initialized`送信したときそれは応答を消化。ここから、いずれかの側は交渉キャパビリティあたりリクエスト送信ことができます。

**フェーズ2：操作。**

双方向。クライアント発見を呼び出す`tools/list`、次`tools/call`起動。サーバー送信することが`sampling/createMessage`宣言那カパビリティ。サーバー送信ことが`notifications/tools/list_changed`それのツールセット変異したとき。クライアント送信ことが`notifications/roots/list_changed`ユーザーがルートスコープ変更するとき。

**フェーズ3：シャットダウン。**

いずれかの側トランスポート閉じます。MCPに構造化シャットダウンメソッドなし；トランスポート（stdio またはStreamable HTTP、フェーズ13・09）接続終了シグナル運ぶ。

### キャパビリティネゴシエーション

`initialize`ハンドシェイク内`capabilities`はコントラクト。サーバーから例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

サーバーが`tools/list_changed`通知を放出でき`resources/subscribe`をサポート宣言。クライアントはそれ自身宣言することで同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

クライアントが`sampling`宣言しない場合、サーバーは`sampling/createMessage`呼び出してはいけません。シメトリック：サーバーが`resources.subscribe`宣言しない場合、クライアントは購読してはいけません。

これが生態系ドリフト防止。サンプリングサポートしないクライアントは依然有効MCPクライアント；`sampling`呼び出さないサーバーは依然有効MCPサーバー。彼ら単にそのフィーチャーを一緒に使用しない。

### 構造化コンテンツとエラー形

`tools/call`は`content`配列型ブロック返す：`text`、`image`、`resource`。フェーズ13・14は MCPアプリ（`ui://`インタラクティブUI）をそのリストに追加。

エラーはJSON-RPC エラーコード使用。仕様定義追加：`-32002`「リソースが見つかりません」、`-32603`「内部エラー」、プラスMCP固有エラーデータとして`error.data`。

### クライアントキャパビリティ対ツール呼び出し詳細

一般的混同：`capabilities.tools`はクライアントがツール-リスト-変更通知サポートするかどうか。クライアントが特定ツール呼び出す かどうかはランタイム選択は、そのモデル駆動で、キャパビリティフラグではない。キャパビリティフラグは仕様レベルコントラクト。モデル選択は直交。

### なぜJSON-RPC そしてREST ではなく。

JSON-RPC 2.0（2010）は軽量双方向プロトコル。RESTはクライアント始まり。MCPはサーバー始まりメッセージ（サンプリング、通知）必要でしたから、JSON-RPC持つ対称リクエスト/応答形状は自然フィット。JSON-RPCも複合cleanly stdioと WebSocket/Streamable HTTP HTTP要求形をREインベント無しで。

## 使用方法

`code/main.py` 最小限JSON-RPC 2.0パーサーと放出、次`initialize` → `tools/list` → `tools/call` → `shutdown`順序説明して手で、プリント各メッセージ。実トランスポートなし；単なるメッセージ形。参照の仕様並列をさらに読むため確認各エンベロープ。

見るべきもの：

- `initialize`それぞれ両方宣言キャパビリティ；応答は`serverInfo`と`protocolVersion: "2025-11-25"`を持つ。
- `tools/list`は`tools`配列を返す；各入力は`name`、`description`、`inputSchema`を持つ。
- `tools/call` `params.name`と`params.arguments`を使用。
- 応答`content`は`{type, text}`ブロックの配列。

## 出荷

このレッスンは`outputs/skill-mcp-handshake-tracer.md`を生成します。MCPクライアント-サーバー接続のpcap スタイルトランスクリプトが与えられた場合、スキルは各メッセージに注釈を付けます。プリミティブ、ライフサイクルフェーズ、キャパビリティは依存します。

## 演習

1. `code/main.py`を実行。キャパビリティネゴシエーション起こるラインを特定し、サーバーが`tools.listChanged`宣言しなかった場合何が変わるか説明。

2. 拡張パーサーを`notifications/progress`処理。メッセージ形：`{method: "notifications/progress", params: {progressToken, progress, total}}`。長実行中`tools/call`中それを放出したことを確認、クライアントハンドラーはプログレスバー表示。

3. MCP 2025-11-25仕様を上から下から読む — 全文書は大体80ページ。サーバーがしない必要なこと1つキャパビリティフラグを特定。ヒント：リソース購読に関する。

4. 紙上に仮説「cronジョブ」フィーチャー属するプリミティブをスケッチ。（ヒント：サーバーはクライアントが予定された時間それを起動してほしい。なし6つのプリミティブは今日適合。）MCPの2026ロードマップはこのための草稿SEP持つ。

5. オープンMCPサーバーを1つセッションログ GitHubから解析。リクエスト対応答対通知メッセージ数。トラフィックはライフサイクル対操作。

## キーターム

| 用語 | 人々が言うこと | それが実際に意味するもの |
|------|----------------|------------------------|
| MCP | 「モデルコンテキストプロトコル」 | モデルからツール発見と起動への開放プロトコル |
| サーバープリミティブ | 「サーバーが露出」 | 道具（アクション）、リソース（データ）、プロンプト（テンプレート） |
| クライアントプリミティブ | 「クライアントがサーバー使用させる」 | ルーツ（スコープ）、サンプリング（LLM コールバック）、引出（ユーザー入力） |
| JSON-RPC 2.0 | 「ワイヤ形式」 | 対称リクエスト/応答/通知エンベロープ |
| `initialize`ハンドシェイク | 「キャパビリティネゴシエーション」 | 最初メッセージペア；サーバー、クライアント機能宣言サポート |
| `tools/list` | 「発見」 | クライアント質問サーバーそれの現在ツールセット |
| `tools/call` | 「起動」 | クライアント質問サーバーツール実行引数で |
| `notifications/*_changed` | 「変異イベント」 | サーバーはクライアントにプリミティブリストが変更通知 |
| コンテンツブロック | 「型付き結果」 | `{type: "text" \| "image" \| "resource" \| "ui_resource"}`ツール結果内 |
| SEP | 「仕様進化提案」 | 名づけ草稿提案（例えば非同期タスク向けSEP-1686） |

## 参考文献

- [モデルコンテキストプロトコル — 仕様2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 標準仕様文書
- [モデルコンテキストプロトコル — アーキテクチャ概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 6プリミティブメンタルモデル
- [Anthropic — モデルコンテキストプロトコル導入](https://www.anthropic.com/news/model-context-protocol) — 2024年11月ローンチポスト
- [MCPブログ — 最初MCP記念日](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 1年レトロスペクティブおよび2025-11-25仕様変更
- [WorkOS — MCP 2025-11-25仕様更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835、1724概要
