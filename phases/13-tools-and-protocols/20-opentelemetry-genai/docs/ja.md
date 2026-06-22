# OpenTelemetry GenAI — エンドツーエンドのツール呼び出しトレース

> エージェントが5つのツール、3つのMCPサーバー、2つのサブエージェントを呼び出します。すべてにわたって1つのトレースが必要です。OpenTelemetry GenAIセマンティック規約（v1.37以上の安定属性）は2026年の標準であり、Datadog、Langfuse、Arize Phoenix、OpenLLMetry、AgentOpsによってネイティブにサポートされています。このレッスンは必要な属性に名前を付け、スパン階層（エージェント → LLM → ツール）を説明し、任意のOTelエクスポーターにプラグインできるstdlibスパン出力側を配信します。

**タイプ:** Build
**言語:** Python (stdlib, OTel スパン出力側)
**前提条件:** Phase 13 · 07 (MCP サーバー), Phase 13 · 08 (MCP クライアント)
**所要時間:** 約75分

## 学習目標

- LLMスパンおよびツール実行スパンに必要なOTel GenAI属性に名前を付けます。
- エージェントループ、LLM呼び出し、ツール呼び出し、MCPクライアント送信をカバーするトレース階層を構築します。
- キャプチャすべきコンテンツ（オプトイン）対 編集（デフォルト）を決定します。
- ツールコードを書き直すことなく、スパンをローカルコレクタ（Jaeger、Langfuse）に出力側します。

## 問題

2026年2月のデバッグ：ユーザーが「私のエージェント時々30秒で応答する。時々3秒」と報告します。トレースなし。ログはLLM呼び出しを表示しますが、ツール送信、MCPサーバーラウンドトリップ、サブエージェントを表示しません。あなたは推測します。最終的に：あるMCPサーバーは時々コールドスタートで一時停止します。

エンドツーエンドのトレースなしでは、これを見つけることができません。OTel GenAIがそれを修正します。

規約は2025-2026年にOpenTelemetry semantic-conventions グループの下で解決しました。彼らは安定した属性名を定義し、Datadog、Langfuse、Phoenix、OpenLLMetry、AgentOpsすべてが同じスパンを解析します。1回計測します。任意のバックエンドに配信します。

## コンセプト

### スパン階層

```
agent.invoke_agent  (上、INTERNAL スパン)
 ├── llm.chat       (CLIENT スパン)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT スパン)
 ├── llm.chat       (CLIENT スパン)
 └── subagent.invoke (INTERNAL)
```

全体は1つのトレースIDの下にネストされます。スパンIDは親子関係をリンクします。

### 必要な属性

2025-2026 semconv ごと：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — リクエストされたモデル文字列（例：`"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 実際に提供されたモデル。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 相関のためのプロバイダ応答ID。

ツールスパンの場合：

- `gen_ai.tool.name` — ツール識別子。
- `gen_ai.tool.call.id` — 特定の呼び出し ID。
- `gen_ai.tool.description` — ツール説明（オプション）。

エージェントスパンの場合：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### スパンタイプ

- `SpanKind.CLIENT`プロセス境界を超えた呼び出し（LLMプロバイダ、MCPサーバー）。
- `SpanKind.INTERNAL`エージェント自体のループステップとツール実行。

### オプトインコンテンツキャプチャ

デフォルトでは、スパンはメトリクスとタイミングを実行します — プロンプトまたは完了ではなく。大きなペイロードとPIIはデフォルトでオフです。`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`と特定のコンテンツキャプチャ環境変数を設定して、コンテンツを含めます。本番環境で有効にする前に慎重に確認してください。

### スパンのイベント

トークンレベルのイベントはスパンイベントとして追加できます：

- `gen_ai.content.prompt` — 入力メッセージ。
- `gen_ai.content.completion` — 出力メッセージ。
- `gen_ai.content.tool_call` — 記録されたツール呼び出し。

イベントはスパン内で時系列順序付けられ、詳細な再生用。

### エクスポーター

OTelスパンは以下にエクスポートします：

- **Jaeger / Tempo。** OSS、オンプレ。
- **Langfuse。** LLM可観測性特異。トークン使用量を可視化。
- **Arize Phoenix。** Evals + トレーシング組み合わせ。
- **Datadog。** 商用。ネイティブに`gen_ai.*`属性を解析。
- **Honeycomb。** 列指向。クエリフレンドリー。

すべてOTLP、ワイヤ形式を話します。あなたのコードは気になりません。

### MCP全体の伝播

MCPクライアントがサーバーを呼び出す場合、W3C traceparent ヘッダーをリクエストに注入します。Streamable HTTPは標準ヘッダーをサポートします。Stdio はHTTPヘッダーをネイティブに実行しません。仕様の2026年ロードマップは、JSON-RPC呼び出しで`_meta.traceparent`フィールドの追加を論じます。

それが配信されるまで：すべてのリクエストで`_meta`に traceparent を手動で含めます。サーバーがトレースIDをログします。

### メトリクス

スパンの横に、GenAI semconv はメトリクスを定義します：

- `gen_ai.client.token.usage` — ヒストグラム。
- `gen_ai.client.operation.duration` — ヒストグラム。
- `gen_ai.tool.execution.duration` — ヒストグラム。

呼び出しごとの詳細を必要としないダッシュボード用にこれらを使用します。

### AgentOps レイヤー

AgentOps（2024年設立）は GenAI 可観測性に特化しています。人気のあるフレームワーク（LangGraph、Pydantic AI、CrewAI）をラップして、OTelスパンを自動的に出力します。スタックがサポートされているフレームワークを使用する場合に便利です。そうでなければ、手動計測を使用します。

## 使用方法

`code/main.py`は、エージェントがLLMを呼び出し、2つのツールを送信し、1つのMCPラウンドトリップを行う場合、stdout（OTLP-JSON形式で）のOTel形状スパンを出力します。実エクスポーターなし — レッスンはスパン形状と属性セットに焦点を当てます。出力をOTLP互換ビューアーに貼り付けるか、それを読むだけです。

確認すべき点：

- トレースIDはすべてのスパンで共有されています。
- 親子リンクは`parentSpanId`を通じてエンコードされます。
- 必要な`gen_ai.*`属性が取得されます。
- コンテンツキャプチャはデフォルトでオフです。1つのシナリオは環境変数を通じてそれをオンにします。

## リリース

このレッスンは`outputs/skill-otel-genai-instrumentation.md`を生成します。エージェントコードベースが与えられた場合、スキルは計測計画を生成します：スパンの追加場所、どの属性を取得するか、どのエクスポーターを対象にするか。

## 演習

1. `code/main.py`を実行します。スパンをカウントし、CLIENTか INTERNALかを特定します。

2. コンテンツキャプチャ（環境変数）をオンにし、`gen_ai.content.prompt`と`gen_ai.content.completion`イベントが表示されることを確認します。PIIの意味合いに注意してください。

3. ツール実行メトリック`gen_ai.tool.execution.duration`を追加し、それを呼び出しごとのヒストグラムサンプルとして出力します。

4. 親エージェントスパンから MCP リクエストの`_meta.traceparent`フィールドに traceparent を伝播させます。MCPサーバーが同じトレースIDを見えることを確認します。

5. OTel GenAI semconv仕様を読みます。semconv にリストされている1つの属性を特定し、このレッスンのコードが出力しません。追加します。

## キー用語

| 用語 | 人々が言うこと | それが実際に意味すること |
|------|------------------|---------------------------|
| OTel | "OpenTelemetry" | トレース、メトリクス、ログのためのオープン標準 |
| GenAI semconv | "GenAI セマンティック規約" | LLM / ツール / エージェントスパンの安定属性名 |
| `gen_ai.*` | "属性名前空間" | すべてのGenAI属性はこのプレフィックスを共有 |
| Span | "時間付き操作" | 開始、終了、属性を持つ作業のユニット |
| Trace | "クロススパン血統" | トレースIDを共有するスパンのツリー |
| SpanKind | "CLIENT / SERVER / INTERNAL" | スパン方向に関するヒント |
| OTLP | "OpenTelemetry Line Protocol" | エクスポーター用ワイヤ形式 |
| Opt-in content | "プロンプト / 完了キャプチャ" | デフォルトでオフ。環境変数で有効化 |
| traceparent | "W3C ヘッダ" | サービス全体でトレースコンテキストを伝播 |
| Exporter | "バックエンド固有の配送側" | Jaeger / Datadog / など にスパンを送信するコンポーネント |

## 参考文献

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI スパン、メトリクス、イベントの正規規約
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLMおよびツール実行スパン属性リスト
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — エージェントレベル`invoke_agent`スパン
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub ホスト事実上の源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 本番環境統合ウォークスルー
