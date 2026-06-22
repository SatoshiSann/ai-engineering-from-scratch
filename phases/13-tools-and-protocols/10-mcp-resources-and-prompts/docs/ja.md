# MCP リソースとプロンプト — ツール以外のコンテキスト公開

> ツールが MCP 注目の 90% を占めています。他の 2 つのサーバープリミティブは異なる問題を解決します。リソースはデータ読み取り用に公開；プロンプトはスラッシュコマンドとして再利用可能なテンプレートを公開します。多くのサーバーは、ツールでラップする代わりにリソースを使用するべき、クライアントプロンプトでハードコードされたワークフロー代わりにプロンプトを使用するべきです。このレッスンは決定ルールと `resources/*` および `prompts/*` メッセージの実装を示します。

**タイプ:** Build
**言語:** Python（stdlib、リソース + プロンプトハンドラー）
**前提条件:** Phase 13 · 07（MCP サーバー）
**所要時間:** 約45分

## 学習目標

- 与えられたドメインに対して、機能をツール、リソース、またはプロンプトとして公開するかを決定する。
- `resources/list`、`resources/read`、`resources/subscribe` を実装し、`notifications/resources/updated` を処理する。
- 引数テンプレートを使用した `prompts/list` と `prompts/get` を実装する。
- ホストがプロンプトをスラッシュコマンドとして表示する vs 自動挿入コンテキストとして表示するかを認識する。

## 問題

ノートアプリ向けの素朴な MCP サーバーはすべてをツールとして公開します：`notes_read`、`notes_list`、`notes_search`。これはすべてのデータアクセスをモデル駆動ツール呼び出しでラップします。結果：

- モデルは関連するコンテキストから利益を得る可能性があるすべてのクエリで `notes_read` を呼び出すかどうかを決定する必要があります。
- 読み取り専用コンテンツをホストのサイドパネルにサブスクライブまたはストリーミングできません。
- クライアント UI（Claude Desktop のリソース添付パネル、Cursor の「Include file」ピッカー）はデータを表示できません。

正しい分割：リソースとしてデータを公開、ツールとして変更又は計算アクションを公開、プロンプトとして再利用可能なマルチステップワークフローを公開します。各プリミティブは UX アフォーダンスとアクセスパターンを持ちます。

## コンセプト

### ツール vs リソース vs プロンプト — 決定ルール

| 機能 | プリミティブ |
|------------|-----------|
| ユーザーがデータを検索、フィルター、変換したい | tool |
| ユーザーがホストにこのデータをコンテキストとして含めたい | resource |
| ユーザーが再利用できるテンプレートワークフロー | prompt |

ガイドライン：モデルが関連するすべてのクエリで呼び出す可能性が高い場合、それはツール。ユーザーが会話に添付したい場合、それはリソース。ユーザーが再利用したい単位が全体のマルチステップワークフローの場合、それはプロンプト。

### リソース

`resources/list` は `{resources: [{uri, name, mimeType, description?}]}` を返す。`resources/read` は `{uri}` を受け取り、`{contents: [{uri, mimeType, text | blob}]}` を返す。

URI は何でもアドレス指定可能：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（カスタムスキーム）
- `memory://session-2026-04-22/recent`（サーバー固有）

`contents[]` はテキストと バイナリをサポート。バイナリは `blob` を Base64 エンコード文字列 plus `mimeType` として使用。

### リソースサブスクリプション

`{resources: {subscribe: true}}` をケーパビリティで宣言。クライアントが `resources/subscribe {uri}` を呼び出す。リソースが変わる際、サーバーが `notifications/resources/updated {uri}` を送信。クライアントが再読み込み。

ユースケース：ディスク上のファイルがリソースであるノートサーバー；ファイルウォッチャーが更新通知をトリガー；Claude Desktop がホスト外で編集された場合にファイルをコンテキストに再プル。

### リソーステンプレート（2025-11-25 追加）

`resourceTemplates` は `notes://{id}` のようにパラメータ化された URI パターンを公開し、`id` を完了ターゲットとします。クライアントはリソースピッカーで ID を自動完了できます。

### プロンプト

`prompts/list` は `{prompts: [{name, description, arguments?}]}` を返す。`prompts/get` は `{name, arguments}` を受け取り、`{description, messages: [{role, content}]}` を返す。

プロンプトはホストがモデルにフィードするメッセージのリストに記入するテンプレート。例えば、`code_review` プロンプトは `file_path` 引数を受け取り、3 メッセージシーケンスを返す：システムメッセージ、ファイルボディ付きユーザーメッセージ、推論テンプレート付きアシスタントキックオフ。

### ホストとプロンプト

Claude Desktop、VS Code、Cursor はプロンプトをチャット UI のスラッシュコマンドとして公開します。ユーザーが `/code_review` を入力し、フォームから引数を選択します。サーバーのプロンプトは「ユーザーショートカット」と「モデルに送信される全プロンプト」の契約。

すべてのクライアントがプロンプトをサポートしていません — ケーパビリティネゴシエーションをチェック。プロンプトケーパビリティ宣言があるサーバーとプロンプトサポートなしのクライアントは、単にスラッシュコマンドを見ません。

### 「リスト変更」通知

リソースとプロンプトの両方が、セットが変更される際に `notifications/list_changed` を発行します。20 個の新しいノートをインポートしたばかりのノートサーバーが `notifications/resources/list_changed` を発行；クライアントが `resources/list` を再呼び出して追加を取得。

### コンテンツタイプ慣例

テキスト用：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
バイナリ用：`image/png`、`application/pdf`、plus `blob` フィールド。
MCP Apps（Lesson 14）用：`ui://` URI で `text/html;profile=mcp-app`。

### 動的リソース

リソース URI は静的ファイルに対応する必要はありません。`notes://recent` は読み取りのたびに最新 5 つのノートを返せます。`db://query/users/active` はパラメータ化クエリを実行できます。サーバーは自由にコンテンツを動的に計算。

ルール：クライアントが URI でキャッシュできる場合、URI は安定している必要があります。計算がワンショットの場合、URI はタイムスタンプまたは nonce を含めてクライアントキャッシュが stale にならないようにすべき。

### サブスクリプション vs ポーリング

サブスクリプション対応クライアントは `notifications/resources/updated` 経由のサーバープッシュを取得。事前サブスクリプションクライアントまたはそれをサポートしていないホストはポーリングで再読み込み。両方ともスペック準拠。サーバーのケーパビリティ宣言はどちらをサポートするかクライアントに伝えます。

サブスクリプションのコスト：サーバーのセッションごとの状態（誰がサブスクライブしているか）。サブスクライブセットを制限に保つ；接続を切られたクライアントはタイムアウトすべき。

### プロンプト vs システムプロンプト

MCP のプロンプトはシステムプロンプトではありません。ホストのシステムプロンプト（独自の動作指示）と MCP プロンプト（ユーザーが起動するサーバー提供テンプレート）は並行して存在。適切に動作するクライアントはサーバープロンプトが独自のシステムプロンプトをオーバーライドさせません；レイヤー化します。

## Use It

`code/main.py` は Lesson 07 のノートサーバーを以下で拡張：

- ノートごとのリソース（`notes://note-1` など）と `resources/subscribe` サポート。
- `review_note` プロンプトが 3 メッセージテンプレートにレンダリング。
- ノートが編集されるとき `notifications/resources/updated` を発行するファイルウォッチャーシミュレーション。
- 常に最新 5 つのノートを返す `notes://recent` 動的リソース。

デモを実行してフルフローを見てください。

## Ship It

このレッスンは `outputs/skill-primitive-splitter.md` を作成します。提案された MCP サーバーが与えられた場合、スキルは各機能を tool / resource / prompt で理由付きで分類します。

## 演習

1. `code/main.py` を実行。初期リソースリストを観察、ノート編集をトリガーし、`notifications/resources/updated` イベントが発行されることを確認。

2. `resources/list_changed` エミッターを追加：新しいノートが作成されるとき、クライアントが新規発見を再呼び出しするようにする通知を送信。

3. GitHub MCP サーバー用に 3 つのプロンプトを設計：`summarize_pr`、`triage_issue`、`release_notes`。各々は引数スキーマ付き。プロンプト本体は編集なしで実行可能。

4. Lesson 07 サーバーの既存ツールを 1 つ取得し、ツールのままか resource + tool ペアに分割するかを分類。1 文で正当化。

5. スペックの `server/resources` と `server/prompts` セクションを読む。`resources/read` で滅多に設定されないが仕様サポートされたフィールドを 1 つ識別。ヒント：リソースコンテンツの `_meta` を見て。

## 主要用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| Resource | 「公開データ」 | ホストが読み込める URI アドレス指定可能コンテンツ |
| リソース URI | 「データへのポインタ」 | スキーム接頭辞付き識別子（`file://`、`notes://` など） |
| `resources/subscribe` | 「変更を監視」 | 特定の URI についてクライアント オプトインサーバープッシュ更新 |
| `notifications/resources/updated` | 「リソース変更」 | サブスクライブされたリソースが新しいコンテンツであることをクライアントに通知 |
| リソーステンプレート | 「パラメータ化 URI」 | ホストピッカー完了ヒント付き URI パターン |
| プロンプト | 「スラッシュコマンドテンプレート」 | 引数スロット付き名前付きマルチメッセージテンプレート |
| プロンプト引数 | 「テンプレート入力」 | ホストが レンダリング前に収集する型付きパラメータ |
| `prompts/get` | 「テンプレートレンダリング」 | サーバーが記入されたメッセージリストを返す |
| コンテンツブロック | 「型付きチャンク」 | `{type: text \| image \| resource \| ui_resource}` |
| スラッシュコマンド UX | 「ユーザーショートカット」 | ホストが `/` で開始するコマンドとしてプロンプトを表示 |

## 参考文献

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — リソース URI、サブスクリプション、テンプレート
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — プロンプトテンプレートとスラッシュコマンド統合
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完全な `resources/*` メッセージリファレンス
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完全な `prompts/*` メッセージリファレンス
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 公式ドキュメントを拡張するコミュニティガイド
