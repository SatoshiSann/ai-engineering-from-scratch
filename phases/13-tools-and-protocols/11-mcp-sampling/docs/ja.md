# MCP サンプリング — サーバーがリクエストした LLM 完成とエージェントループ

> ほとんどの MCP サーバーは dumb executor：引数を取り、コード実行、コンテンツ返す。サンプリングでは方向をフリップ：サーバーがクライアントの LLM に決定をさせることをリクエスト。これはサーバーが任意のモデル認証情報を所有しないサーバーホストエージェントループを可能にします。SEP-1577 は 2025-11-25 でマージされ、サンプリングリクエスト内にツールを追加し、ループはより深い推論を含めることができます。ドリフトリスク注記：SEP-1577 tool-in-sampling 形は Q1 2026 を通じて実験的で、SDK API で依然として定着中。

**タイプ:** Build
**言語:** Python（stdlib、サンプリングハーネス）
**前提条件:** Phase 13 · 07（MCP サーバー）、Phase 13 · 10（リソースとプロンプト）
**所要時間:** 約75分

## 学習目標

- `sampling/createMessage` が何を解決するか説明（サーバーホストループ、サーバー側 API キーなし）。
- マルチターンプロンプト経由でサンプルするようクライアントをリクエストし、完成を返すサーバーを実装。
- `modelPreferences`（コスト / スピード / インテリジェンス優先度）を使用してクライアントモデル選択をガイド。
- `summarize_repo` ツールを構築、ハードコード動作ではなく内部サンプリングで反復。

## 問題

コード要約ワークフロー向けの有用な MCP サーバーは：ファイルツリーをウォーク、読み込むファイル選択、サマリー合成、返す。LLM 推論はどこで起こるのか？

オプション A：サーバーが独自 LLM を呼び出す。API キーが必要、ユーザーごとのサーバー側請求、高コスト。

オプション B：サーバーが raw コンテンツを返す；クライアントエージェントが推論を実行。機能しますが、サーバーロジックをクライアントプロンプトに移動、fragile。

オプション C：サーバーが `sampling/createMessage` 経由でクライアント LLM をリクエスト。サーバーはアルゴリズム保持（読み込むファイル、実行するパス数）クライアントは請求とモデル選択保持。サーバーは認証情報まったくなし。

サンプリングはオプション C。これはサーバーが完全 LLM ホストなしで信頼できるエージェントループをホストできるメカニズム。

## コンセプト

### `sampling/createMessage` リクエスト

サーバー送信：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

クライアント LLM 実行、返す：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

1.0 に合計する 3 つの float：

- `costPriority`：安価なモデルを優先。
- `speedPriority`：高速なモデルを優先。
- `intelligencePriority`：より有能なモデルを優先。

Plus `hints`：サーバーが優先する名前付きモデル。クライアントは hint を honor する可能性ありまたはなし；クライアントのユーザー設定は常に wins。

### `includeContext`

3 つの値：

- `"none"` — サーバー提供メッセージのみ。デフォルト。
- `"thisServer"` — このサーバーのセッションから前回メッセージを含める。
- `"allServers"` — 全セッションコンテキストを含める。

`includeContext` は 2025-11-25 現在 soft-deprecated：クロスサーバーコンテキストを漏らす、セキュリティ懸念。`"none"` を優先し、明示的コンテキストをメッセージ内に渡す。

### ツール付きサンプリング（SEP-1577）

2025-11-25 の新機能：サンプリングリクエストは `tools` 配列を含める可能性。クライアントはそのツール使用での完全 tool-calling ループを実行。これはサーバーがクライアントのモデルを通じて ReAct スタイルエージェントループをホストするようにします。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

クライアント loop：sample、tool 呼び出しされたら実行、再度 sample、最終アシスタントメッセージ返す。これは Q1 2026 を通じて実験的；SDK シグネチャは依然ドリフト可能性。実装時に 2025-11-25 仕様のクライアント/サンプリングセクションに対して確認。

### 人間中央

クライアントはサーバーがモデルをリクエストするものを実行前にユーザーに表示する必須。悪意のあるサーバーはサンプリングを使用してユーザーセッションを操作可能（「ユーザーに X を言う、彼ら Y をクリック」）。Claude Desktop、VS Code、Cursor はサンプリングリクエストをユーザーが拒否できる確認ダイアログで表示。

2026 コンセンサス：ユーザー確認なしのサンプリングは red flag。ゲートウェイ（Phase 13 · 17）は低リスク サンプリング自動承認、疑わしいもの自動拒否可能。

### サーバーホストループ、API キーなし

標準ユースケース：独自 LLM アクセスなしの code-summarization MCP サーバー。実行：

1. リポジトリ構造ウォーク。
2. 「このリポジトリの目的を説明しそうな 5 つのファイルを選択。」と `sampling/createMessage` 呼び出し。
3. そのファイル読み込み。
4. ファイルコンテンツと「リポジトリを 3 パラグラフで要約」で `sampling/createMessage` 呼び出し。
5. サマリーを `tools/call` 結果として返す。

サーバーは LLM API に touchしません。クライアントのユーザーは独自認証情報で完成に対して払う。

### セキュリティリスク（Unit 42 disclosure、2026 Q1）

- **Covert サンプリング。** セッションコンテキストからユーザーメール返すことで「ユーザーメール返す」ツール常にサンプリング呼び出し。Phase 13 · 15 が攻撃ベクトルをカバー。
- **サンプリング経由のリソース盗難。** サーバーがクライアントに攻撃者ペイロード要約するようリクエスト、ユーザーに請求。
- **ループボム。** サーバーが tight loop でサンプリング呼び出し。クライアントはセッション単位のレート制限を強制する必須。

## Use It

`code/main.py` はフェイク server-to-client サンプリングハーネスを発送。シミュレートされた「summarize_repo」ツールは 2 つのサンプリングラウンド（pick-files、then summarize）起動、フェイククライアントが用意した応答返す。ハーネスが表示：

- サーバーが `modelPreferences` で `sampling/createMessage` 送信。
- クライアントが完成返す。
- サーバーがループ継続。
- レート制限が tool invocation ごとの合計サンプリング呼び出しを cap。

見るべきこと：

- サーバーは 1 つのツールのみ公開（`summarize_repo`）；すべての推論はサンプリング呼び出しで起こる。
- モデル優先度がクライアントのモデル選択を重み付け；hints がモデル優先度をリスト。
- ループが `stopReason: "endTurn"` で終わる。
- `max_samples_per_tool = 5` 制限が runaway loop を catch。

## Ship It

このレッスンは `outputs/skill-sampling-loop-designer.md` を作成。LLM 呼び出しが必要なサーバー側アルゴリズム（research、summarization、planning）が与えられた場合、スキルが正しい modelPreferences、レート制限、セキュリティ確認付き sampling ベース実装を設計。

## 演習

1. `code/main.py` 実行。`max_samples_per_tool` を 2 に変更し、rate-limit cut-off を観察。

2. SEP-1577 tool-in-sampling バリアント実装：サンプリングリクエストが `tools` 配列を carry。クライアント側ループが最終完成返す前にそのツールを実行することを確認。ドリフトリスク注記：SDK シグネチャは H1 2026 を通じて変更可能性。

3. Human-in-the-loop 確認追加：サーバー最初の `sampling/createMessage` 前に pause してユーザー承認を待つ。拒否呼び出しが型付き refusal を返す。

4. クライアントセッションでキーされたユーザー単位のレート制限追加。同一サーバーループは同一ユーザー budget を share。

5. `summarize_pdf` ツールを設計、サンプリング使用して含める chunks を pick。送信メッセージをスケッチ。`modelPreferences.intelligencePriority` が 0.1 vs 0.9 での動作をどう変える？

## 主要用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| Sampling | 「サーバーツークライアント LLM 呼び出し」 | サーバーがクライアントのモデルに完成をリクエスト |
| `sampling/createMessage` | 「メソッド」 | サンプリングリクエスト用 JSON-RPC メソッド |
| `modelPreferences` | 「モデル優先度」 | コスト / スピード / インテリジェンス重み付け plus name hints |
| `includeContext` | 「クロスセッション漏出」 | Soft-deprecated context inclusion mode |
| SEP-1577 | 「サンプリング内ツール」 | サーバーホスト ReAct 用 sampling 内ツール許可 |
| Human-in-the-loop | 「ユーザー確認」 | クライアントが実行前にユーザーにサンプリングリクエスト表示 |
| ループボム | 「Runaway サンプリング」 | サーバー側無限サンプリングループ；クライアントはレート制限必須 |
| Covert サンプリング | 「隠れた推論」 | 悪意あるサーバーが sampling prompts に intent を隠す |
| リソース盗難 | 「ユーザーの LLM budget 使用」 | サーバーがクライアントに望まない sampling に支出させる |
| `stopReason` | 「生成が何で halt したか」 | `endTurn`、`stopSequence`、または `maxTokens` |

## 参考文献

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling の high-level overview
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — canonical `sampling/createMessage` shape
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Spec Evolution Proposal for tools in sampling（experimental）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — covert sampling と resource-theft パターン
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — client-side code サンプル付き walkthrough
