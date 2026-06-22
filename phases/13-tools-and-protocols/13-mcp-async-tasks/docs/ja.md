# 非同期タスク（SEP-1686）— Call-Now、Fetch-Later 長時間実行作業用

> 実際のエージェント作業は数分から数時間：CI runs、deep-research synthesis、batch exports。同期 tool 呼び出しは接続をドロップ、timeout、UI を block。SEP-1686 は 2025-11-25 でマージ、Tasks プリミティブを add：任意のリクエストは task になるよう augment でき、結果は後でまたは state notifications 経由ストリームで fetch。ドリフトリスク注記：タスクは H1 2026 を通じて実験的；SDK surface は依然スペック周りで design 中。

**タイプ:** Build
**言語:** Python（stdlib、async task state machine）
**前提条件:** Phase 13 · 07（MCP サーバー）、Phase 13 · 09（transports）
**所要時間:** 約75分

## 学習目標

- 同期から task-augmented へ tool を promote する時 identify（>30 秒サーバー側作業）。
- task lifecycle を walk：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- task state を persist、crash が in-flight work を lose しない。
- `tasks/status` poll と `tasks/result` fetch を正しく。

## 問題

`generate_report` ツール、multi-minute 抽出パイプライン実行。同期モデル下のオプション：

1. 接続を 3 分間 open に保つ。リモートトランスポートが drop；client が timeout；UI freeze。
2. すぐに placeholder で return；クライアントが custom endpoint poll する require。MCP uniformity を break。
3. Fire-and-forget；no result。

なし良い。SEP-1686 が 4 番目を add：task augmentation。任意のリクエスト（typically `tools/call`）は task として tag できる。サーバーは task id を immediately return。クライアント `tasks/status` poll、done の際 `tasks/result` fetch。サーバー側 state は restart survive。

## コンセプト

### Task augmentation

リクエストが `params._meta.task.required: true`（または `optional: true`、サーバー decide）setting で task になる。サーバーは immediately以下で respond：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` はサーバーの promise、state を retain；ttl after task result は discard。

### Per-tool opt-in

Tool annotations は task support を declare できる：

- `taskSupport: "forbidden"` — このツール常に synchronously 実行。Fast tools には safe。
- `taskSupport: "optional"` — クライアントが task-augmentation をリクエスト可能。
- `taskSupport: "required"` — クライアント task augmentation を使用する必須。

`generate_report` ツールは `required` であろう。`notes_search` ツールは `forbidden` であろう。

### 状態

```
working  -> input_required -> working  (elicitation 経由ループ)
working  -> completed
working  -> failed
working  -> cancelled
```

状態マシンは append-only：`completed`、`failed`、または `cancelled` 一度、タスクは terminal。

### メソッド

- `tasks/status {taskId}` — 現在の state と progress hint return。
- `tasks/result {taskId}` — block または not-yet-done の場合 404 return。
- `tasks/cancel {taskId}` — idempotent；terminal states ignore。
- `tasks/list` — optional；active と recently-completed tasks を enumerate。

### State 変更ストリーミング

サーバーが support する際、クライアント state notifications にサブスクライブ可能：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

Stream ではなく poll するクライアントは better UX を get。Poll は常に minimal surface として support。

### Durable state

Task support を declare するスペック require サーバー state を persist。Crash は ttl 内で completed results を lose しないべき。Store は SQLite から Redis からファイルシステムまで range。Lesson 13 harness はファイルシステムを使用。

### Cancellation セマンティクス

`tasks/cancel` は idempotent。タスクが mid-execution の場合、サーバー stop を attempt（check executor-cooperative cancellation）。Already terminal の場合、リクエストは no-op。

### Crash recovery

サーバープロセスが restart の場合：

1. Persisted task states をすべて load。
2. Prrocess が死んだ `working` task を `failed` with error `CRASH_RECOVERY` でマーク。
3. `completed` / `failed` / `cancelled` を ttl に preserve。

### 非同期タスク plus サンプリング

タスク自体が `sampling/createMessage` を呼び出し可能。これが long-running research tasks がどう動作するか：サーバー task thread が必要に応じてクライアントのモデルをサンプル、クライアント UI は task を `working` として task progress update とともに表示。

### なぜこれが experimental か

SEP-1686 は 2025-11-25 で ship、しかし broader roadmap は 3 つの open issues を call out：durable subscription primitives、subtasks（parent-child task relationships）、result-TTL standardization。スペックは 2026 を通じて evolve することを expect。Production コードは common case に対してのみ Tasks を stable として treat し、future SDK changes for subtasks に guard。

## Use It

`code/main.py` は durable task store（filesystem-backed）と `generate_report` ツール、background thread で実行を実装。クライアント tool を呼び出し、task id を immediately get、worker が progress update する間 `tasks/status` poll、done の際 `tasks/result` fetch。Cancellation works；crash recovery は worker thread を kill、state を reload することで simulate。

見るべきこと：

- Task state JSON は `/tmp/lesson-13-tasks/<id>.json` に persist。
- Worker thread が `progress` フィールドを update；poll がそれを advance で表示。
- Client side からの cancellation が event を set；worker がチェック early exit。
- State reload on「crash」は in-flight task を `failed` with `CRASH_RECOVERY` としてマーク。

## Ship It

このレッスンは `outputs/skill-task-store-designer.md` を作成。Long-running tool（research、build、export）が与えられた場合、スキルが task store を design（state shape、ttl、durability）、正しい taskSupport flag を pick、progress notifications を sketch。

## 演習

1. `code/main.py` 実行。`generate_report` task を kick off、status poll、result を fetch。

2. Mid-run で `tasks/cancel` 呼び出しを add。Worker がそれを honor し state が `cancelled` になることを verify。

3. Crash recovery をシミュレート：worker thread を kill、loader をリスタート、`CRASH_RECOVERY` failure mode を observe。

4. SQLite に store を extend。Durability wins は同じ；query options が open up（session X からのすべてのタスクをリスト）。

5. 2026 年の MCP roadmap post を読む。Task-related open issue を identify、最も可能性がある次年度 SDK API design に影響。

## 主要用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| Task | 「Long-running tool call」 | Async 実行用 `_meta.task` でaugment request |
| SEP-1686 | 「Tasks spec」 | 2025-11-25 にタスク追加 Spec Evolution Proposal |
| `_meta.task` | 「Task envelope」 | Id、state、ttl を含む per-request metadata |
| taskSupport | 「Tool flag」 | Per tool `forbidden` / `optional` / `required` |
| `tasks/status` | 「Poll メソッド」 | 現在 state と optional progress hint fetch |
| `tasks/result` | 「Fetch result」 | Completed payload を return または not-yet-done の場合 404 |
| `tasks/cancel` | 「Stop it」 | Idempotent cancellation request |
| ttl | 「Retention budget」 | ミリ秒、サーバーが task state を keep する promise |
| `notifications/tasks/updated` | 「State push」 | サーバー起動 state-change event |
| Durable store | 「Crash-safe state」 | ファイルシステム / SQLite / Redis persistence layer |

## 参考文献

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — originating proposal と full discussion
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 理由付きデザイン walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics と state machine
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK-level task implementation patterns
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — open issues と subtasks を含む 2026 優先度
