# OpenTelemetry GenAI セマンティック規約

> OpenTelemetry の GenAI SIG（2024 年 4 月起動）はエージェントテレメトリの標準スキーマを定義します。スパン名、属性、およびコンテンツキャプチャルールは Datadog、Grafana、Jaeger、Honeycomb 全体で収束して、エージェントトレースが同じ意味を持つようにします。

**タイプ:** 学習 + ビルド
**言語:** Python（標準ライブラリ）
**前提条件:** Phase 14 · 13（LangGraph）、Phase 14 · 24（可観測性プラットフォーム）
**所要時間:** 約 60 分

## 学習目標

- GenAI スパンカテゴリを挙げてください：model/client、agent、tool
- `invoke_agent` CLIENT 対 INTERNAL スパンと各々がいつ適用されるかを区別してください
- トップレベル GenAI 属性を挙げてください：プロバイダー名、リクエストモデル、データソース ID
- コンテンツキャプチャ契約を説明してください：オプトイン、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部参照推奨

## 問題

すべてのベンダーは独自のスパン名を発明します。Ops チームは各フレームワークダッシュボードを構築することになります。OpenTelemetry の GenAI SIG はエコシステム全体がターゲットにする 1 つの標準を定義することで、これを修正します。

## コンセプト

### スパンカテゴリ

1. **モデル / クライアントスパン。** 生の LLM 呼び出しをカバー。プロバイダー SDK（Anthropic、OpenAI、Bedrock）とフレームワークモデルアダプターで発行されます
2. **エージェントスパン。** `create_agent`（エージェントが構築されるとき）と `invoke_agent`（それが実行されるとき）
3. **ツールスパン。** ツール起動あたり 1 つ；親子関係によってエージェントスパンに接続

### エージェントスパン命名

- スパン名：命名されている場合は `invoke_agent {gen_ai.agent.name}`；デフォルトは `invoke_agent` にフォールバック
- スパン種類：
  - **CLIENT** — リモートエージェントサービス用（OpenAI Assistants API、Bedrock Agents）
  - **INTERNAL** — インプロセスエージェントフレームワーク用（LangChain、CrewAI、ローカル ReAct）

### キー属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`
- `gen_ai.request.model` — モデル ID
- `gen_ai.response.model` — 解決されたモデル（ルーティングのため要求から異なる可能性がある）
- `gen_ai.agent.name` — エージェント識別子
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`
- `gen_ai.data_source.id` — RAG 用：どのコーパスまたはストアが参照されたか

技術固有の規約は Anthropic、Azure AI Inference、AWS Bedrock、OpenAI に対して存在します。

### コンテンツキャプチャ

デフォルトルール：instrumentations はデフォルトで入力/出力をキャプチャ すべきではありません。キャプチャは以下を経由してオプトインされます：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推奨本番環境パターン：外部にコンテンツを保存（S3、ログストア）、スパンで参照を記録（ポインター ID、プローズではなく）。これは Lesson 27 コンテンツポイズニング防御は可観測性に配線されています。

### 安定性

ほとんどの規約は 2026 年 3 月時点で実験的です。安定した preview をオプトイン：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ は GenAI 属性をネイティブに LLM Observability スキーマにマップします。他のバックエンド（Grafana、Honeycomb、Jaeger）は生の属性をサポートしています。

### このパターンがうまくいかない場所

- **スパンで完全なプロンプトをキャプチャする。** PII、シークレット、ops が読める可能性のあるトレースの顧客データ。外部に保存してください
- **`gen_ai.provider.name` なし。** マルチプロバイダーダッシュボードは帰属が欠落しているとき壊れます
- **親リンクなしのスパン。** 孤立したツールスパン。常にコンテキストを伝播してください
- **安定性オプトイン設定を設定していない。** バックエンドアップグレード上で属性が改名される可能性があります

## ビルド

`code/main.py` は GenAI 規約と一致する stdlib スパンエミッターを実装します：

- GenAI 属性スキーマを持つ `Span`
- `start_span`、ネストコンテキストを持つ `Tracer`
- スクリプト化エージェント実行：`create_agent`、`invoke_agent`（INTERNAL）、ツールごとスパン、LLM コール用の `chat` スパンを発行
- 外部にプロンプトを保存してスパンで ID を記録するコンテンツキャプチャモード

実行：

```
python3 code/main.py
```

出力：すべての必要な GenAI 属性を持つスパンツリー、および外部参照オプトイン表示するオプトイン「外部ストア」

## 使用方法

- **Datadog LLM Observability**（v1.37+）属性をネイティブにマップ
- **Langfuse / Phoenix / Opik**（Lesson 24）— auto-instrument エコシステム
- **Jaeger / Honeycomb / Grafana Tempo** — 生 OTel トレース；GenAI 属性からダッシュボード構築
- **自己ホスト型** — GenAI processor を持つ OTel Collector を実行

## 出荷

`outputs/skill-otel-genai.md` は既存エージェントへの OTel GenAI スパンをコンテンツキャプチャデフォルトと外部参照ストレージで配線します。

## 演習

1. Lesson 01 ReAct ループを `invoke_agent`（INTERNAL）+ ツールごとスパンで instrument してください。Jaeger インスタンスに送信してください
2. 「参照のみ」モードでコンテンツキャプチャを追加してください：SQLite へのプロンプト、スパン属性のみ行 ID を運びます
3. `gen_ai.data_source.id` のスペックを読んでください。Lesson 09 Mem0 検索に配線してください
4. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` を設定してコレクターが属性を改名しないことを検証してください
5. ダッシュボード構築：「どのツールエラーがどのモデルと相関しているか」を GenAI 属性だけから

## キーターム

| 用語 | 人々が言うこと | 実際の意味 |
|------|----------------|----------|
| GenAI SIG | 「OpenTelemetry GenAI group」 | スキーマを定義する OTel ワーキンググループ |
| invoke_agent | 「Agent span」 | エージェント実行を表すスパンの名前 |
| CLIENT span | 「Remote call」 | リモートエージェントサービスへの呼び出しのためのスパン |
| INTERNAL span | 「In-process」 | インプロセスエージェント実行のためのスパン |
| gen_ai.provider.name | 「Provider」 | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | 「RAG source」 | 取得ヒットがどのコーパス/ストア |
| Content capture | 「Prompt logging」 | メッセージのオプトインキャプチャ；本番環境で外部に保存 |
| Stability opt-in | 「Preview mode」 | 実験的規約をピンする Env var |

## 参考文献

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — スペック
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — デフォルトで GenAI スパン
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel スパン組み込み
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C トレースコンテキスト伝播
