# MCPサーバー構築 — Python + TypeScript SDK

> ほとんどのMCPチュートリアルは stdio ハローワールド のみ表示。実サーバーは道具と露出リソース「プロンプト」、処理キャパビリティネゴシエーション、放出構造化エラー、そして SDK 全体で同じ機能。このレッスンはノートサーバーをエンドツーエンド構築：stdlibのstdio トランスポート、JSON-RPC派遣、3つのサーバープリミティブ、純粋関数形式が準備ができたら Python SDK FastMCP または TypeScript SDKにドロップ。

**タイプ:** ビルド
**言語:** Python（stdlib、stdio MCPサーバー）
**前提条件:** フェーズ13・06（MCP基礎）
**所要時間:** 約75分

## 学習目標

- `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`、および`prompts/get`メソッドを実装。
- JSON-RPCメッセージをstdinから読み、応答をstdoutに書く派遣ループを記述。
- JSON-RPC 2.0仕様とMCPの追加コードあたり構造化エラー応答を放出。
- stdlibの実装をFastMCP（Pythonの SDK）またはTypeScript SDKに卒業できるツールロジック書き換えなし。

## 問題

リモートトランスポート（フェーズ13・09）または認証層（フェーズ13・16）をあなたは使用ことができます前に、クリーンローカルサーバーが必要。ローカルは stdio を意味：サーバーはクライアントが子プロセスとして起動されます、メッセージ stdin/stdout 改行分け上流。

2025-11-25仕様は stdio メッセージが明示的`\n`セパレータの持つJSONオブジェクトとしてエンコードされることに指定。SSEなし：SSEは古いリモートモードでした、2026年中(Atlassian Rovo MCPサーバーは2026年6月30日に廃止、Keboolaは2026年4月1日)削除されている。stdio の場合、1行ごと1つのJSONオブジェクトが全ワイヤ形式。

ノートサーバーは良い形です、なぜなら3つのサーバープリミティブすべてを実行。道具は変異実行（`notes_create`）。リソースは露出データ（`notes://{id}`）。プロンプトは配布テンプレート（`review_note`）。このレッスン形状は任意のドメインに汎用化。

## コンセプト

### ディスパッチループ

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

3つのルール：

- JSON-RPCエンベロープではない stdoutに何も出力しない。デバッグログはstderr に移動。
- すべてのリクエスト合致 必須 である同じ`id`を持つ応答と。
- 通知は応答してはいけない。

### `initialize`を実装

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

あなたがサポートするのみ宣言。クライアントはキャパビリティセット使用ゲート機能。

### `tools/list`と`tools/call`を実装

`tools/list`戻す`{tools: [...]}`各入力を持つ`name`、`description`、`inputSchema`。`tools/call`取る`{name, arguments}`と戻す`{content: [blocks], isError: bool}`。

コンテンツブロックは型付き。最も一般的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

ツールエラーが2つの形式。プロトコルレベルエラー（未知メソッド、悪いパラメータ）はJSON-RPC エラー。ツールレベルエラー（有効呼び出ししかし道具失敗）は`{content: [...], isError: true}`として返。それがモデルを允許見る失敗そのコンテキスト内。

### リソース実装

リソースは設計により読み取り専用。`resources/list`マニフェスト返す；`resources/read`コンテンツ返。URIが`file://...`、`http://...`、またはカスタムスキーム（`notes://`のような）ことができます。

リソースとしてデータ露出する場合ツール代わりに：

- モデルは「呼び出さない」；クライアントはユーザー要求対コンテキストの組入可能。
- 購読はサーバーをリソースがあるときは更新を プッシュできます（フェーズ13・10）。
- フェーズ13・14はこれを拡張`ui://`インタラクティブリソース。

### プロンプト実装

プロンプトは名引数を持つテンプレート。ホストはスラッシュコマンドとしてそれらを表面。`review_note`プロンプトは`note_id`引数を取るかもしれません、クライアントがそのモデルをフィードするマルチメッセージプロンプトテンプレート生成。

### Stdio トランスポート微妙

- 改行区切りJSON。長さプレフィックスフレーミングなし。
- バッファしない。`sys.stdout.flush()`各書きの後。
- クライアントはライフタイム制御。stdin が閉じ（EOF）したとき、クリーンに終了。
- SIGPIPEをサイレント処理しない；ログしたり終了。

### 注釈

各道具`annotations`を運べます安全特性記述：

- `readOnlyHint: true` — 純粋読、再試行セーフ。
- `destructiveHint: true` — 逆行不可能副作用；クライアント確認すべき。
- `idempotentHint: true` — 同入力同出力を生成。
- `openWorldHint: true` — 外部システムと接する。

クライアント使用これらUXを決定（確認ダイアログ、状態インジケータ）とルーティング（フェーズ13・17）。

### 卒業パス

`code/main.py`内stdlibサーバーは大体180行。FastMCP（Python）同じロジックをデコレータ式に折りたたむ：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDKは等価形を持つ。卒業パスはあなたが準備ができたときドロップイン；概念（キャパビリティ、派遣、コンテンツブロック）は同。

## 使用方法

`code/main.py`は stdio 上の完全ノートMCPサーバー、stdlibのみ。`initialize`、`tools/list`、`tools/call` 3つのツール（`notes_list`、`notes_search`、`notes_create`）、`resources/list`と各ノート`resources/read`、`review_note`プロンプトを処理。パイプJSON-RPCメッセージJSONでドライブ：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

見るべきもの：

- ディスパッチャーは`dict[str, Callable]`メソッド名でキー付け。
- すべてのツール実行はコンテンツブロックのリスト返却、裸文字列ではなく。
- `isError: true`実行が上げるときセット。

## 出荷

このレッスンは`outputs/skill-mcp-server-scaffolder.md`を生成します。ドメインが与えられた場合（ノート、チケット、ファイル、データベース）、スキルは道具/リソース/プロンプト分割とSDK卒業パス持つMCPサーバーにスキャフォルド。

## 演習

1. `code/main.py`を実行しそれを手作りJSON-RPCメッセージでドライブ。`notes_create`、次`resources/read`新ノート検索。

2. `annotations: {destructiveHint: true}`を持つ`notes_delete`道具を追加。クライアントが確認ダイアログを表面することを検証（これは実ホスト要求；Claude Desktopは機能）。

3. 実装`resources/subscribe`ですからサーバーは`notifications/resources/updated`を プッシュできます いつノートが変更。キープアライブタスク追加。

4. FastMCP にサーバーをポート。Pythonファイルは80行以下に縮小。ワイヤ動作は同じである必要があります；同じJSON-RPC テスト ハーネスで検証。

5. 仕様`server/tools`セクションを読みます、このレッスンのサーバー内で実装ツール定義の1つのフィールド。（ヒント：いくつかがあります；1つ選択追加。）

## キーターム

| 用語 | 人々が言うこと | それが実際に意味するもの |
|------|----------------|------------------------|
| MCPサーバー | 「道具を露出する物」 | stdin または HTTP 上MCPを話すプロセス |
| stdio トランスポート | 「子プロセスモデル」 | サーバーはクライアントが起動；stdin/stdout 通信 |
| ディスパッチャー | 「メソッドルータ」 | JSON-RPCメソッド名からハンドラー関数へのマップ |
| コンテンツブロック | 「道具結果チャンク」 | ツール応答内`content`配列の型付けされた要素 |
| `isError` | 「ツール レベル失敗」 | ツール失敗シグナル；JSON-RPCエラーから区別 |
| 注釈 | 「安全ヒント」 | readOnly / destructive / idempotent / openWorld フラグ |
| FastMCP | 「Python SDK」 | MCP プロトコル上部デコレータベース高レベルフレームワーク |
| リソース URI | 「アドレス可能データ」 | `file://`、`db://`、またはカスタムスキームリソースを確認 |
| プロンプトテンプレート | 「スラッシュコマンド概要」 | サーバー供給テンプレート引数スロット持つホスト UI 用 |
| キャパビリティ宣言 | 「フィーチャートグル」 | `initialize`内プリミティブ単位フラグ サーバーがサポート機能宣言 |

## 参考文献

- [モデルコンテキストプロトコル — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参照Pythonの実装
- [モデルコンテキストプロトコル — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 並列 TS 実装
- [FastMCP — サーバーフレームワーク](https://gofastmcp.com/) — MCPサーバー向けデコレータスタイルPython API
- [MCP — クイックスタート サーバーガイド](https://modelcontextprotocol.io/quickstart/server) — 各SDK 使用エンドツーエンドチュートリアル
- [MCP — サーバー道具仕様](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — 道具/* メッセージ完全リファレンス
