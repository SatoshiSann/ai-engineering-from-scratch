# 関数呼び出し詳細解説 — OpenAI、Anthropic、Gemini

> 3つの最先端プロバイダーは2024年に同じツール呼び出しループに収束し、その後他のすべてのもので分岐しました。OpenAIは`tools`と`tool_calls`を使用します。Anthropicは`tool_use`と`tool_result`ブロックを使用します。Geminiは`functionDeclarations`と一意のID相関を使用します。このレッスンは、3つの側面を対比させるため、あるプロバイダーで出荷するコードが別のプロバイダーに移植されるときに破損しません。

**タイプ:** ビルド
**言語:** Python（stdlib、スキーマトランスレータ）
**前提条件:** フェーズ13・01（ツールインターフェース）
**所要時間:** 約75分

## 学習目標

- OpenAI、Anthropic、Gemini関数呼び出しペイロード（宣言、呼び出し、結果）間の3つの形状差を述べる。
- 1つのツール宣言をすべての3つのプロバイダー形式にわたって変換し、厳密モード制約がどこで異なるかを予測する。
- 各プロバイダーで`tool_choice`を使用して、ツール呼び出しを強制、禁止、または自動選択します。
- プロバイダーごとのハード制限（ツール数、スキーマの深さ、引数の長さ）と、各制限を超過したときに放出される各プロバイダーのエラー署名を知る。

## 問題

関数呼び出しリクエストの形状はプロバイダーによって異なります。2026年本番スタックからの3つの具体例：

**OpenAI Chat Completions / Responses API。** `tools: [{type: "function", function: {name, description, parameters, strict}}]`を渡します。モデルの応答には`choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`が含まれ、`arguments`はパース必須のJSON文字列です。厳密モード（`strict: true`）は制約付きデコーディング経由でスキーマ準拠を適用します。

**Anthropic Messages API。** `tools: [{name, description, input_schema}]`を渡します。応答は`content: [{type: "text"}, {type: "tool_use", id, name, input}]`で戻ります。`input`は既にパースされています（文字列ではなくオブジェクト）。一致する`tool_use_id`と`content`を持つ`{type: "tool_result", tool_use_id, content}`ブロックを含む新しい`user`メッセージで返信します。

**Google Gemini API。** `tools: [{functionDeclarations: [{name, description, parameters}]}]`を渡します（`functionDeclarations`の下にネストされています）。応答は`candidates[0].content.parts: [{functionCall: {name, args, id}}]`として到着し、Gemini 3以降の平行呼び出し相関にはIDが一意です。一致する`{functionResponse: {name, id, response}}`で返信します。

同じループ。異なるフィールド名、異なるネスト、異なる文字列対オブジェクト規則、異なる相関メカニズム。OpenAIで天気エージェントを書いたチームはAnthropicへの移植に2日間、Geminiへの別の1日を払います。単なる配管のためです。

このレッスンは、3つの形式を1つの標準的なツール宣言に統一し、エッジでルーティングする変換器を構築します。フェーズ13・17は、同じパターンをLLMゲートウェイに汎用化します。

## コンセプト

### 共通構造

すべてのプロバイダーに5つが必要です：

1. **ツールリスト。** ツール単位の名前、説明、および入力スキーマ。
2. **ツール選択。** 特定のツールを強制するか、ツールを禁止するか、モデルが決定するようにします。
3. **呼び出し放出。** ツールと引数の名前を付けた構造化出力。
4. **呼び出しID。** 応答を正しい呼び出しと相関させる（平行に重要）。
5. **結果注入。** 結果を呼び出しに結び付けるメッセージまたはブロック。

### 形状差、フィールド単位

| 側面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 宣言エンベロープ | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| スキーマフィールド | `parameters` | `input_schema` | `parameters` |
| 応答コンテナ | アシスタントメッセージの`tool_calls[]` | 型`tool_use`の`content[]` | 型`functionCall`の`parts[]` |
| 引数型 | 文字列化JSON | パースされたオブジェクト | パースされたオブジェクト |
| ID形式 | `call_...`（OpenAIが生成） | `toolu_...`（Anthropic） | UUID（Gemini 3以降） |
| 結果ブロック | ロール`tool`、`tool_call_id` | `tool_result`を持つ`user`、`tool_use_id` | 一致する`id`を持つ`functionResponse` |
| ツール強制 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| ツール禁止 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 厳密スキーマ | `strict: true` | スキーマはスキーマ（常に適用） | リクエストレベルの`responseSchema` |

### 実際に達するプロバイダー制限

- **OpenAI。** リクエスト単位128ツール。スキーマの深さ5。引数文字列≤8192バイト。厳密モードは`$ref`、`oneOf`/`anyOf`/`allOf`オーバーラップなし、`required`にリストされたすべてのプロパティが必要です。
- **Anthropic。** リクエスト単位64ツール。スキーマの深さは実際には無制限ですが、実際の制限10。厳密モードフラグなし；スキーマはコントラクトであり、モデルは準拠する傾向があります。
- **Gemini。** リクエスト単位64関数。スキーマ型はOpenAPI 3.0サブセット（JSONスキーマ2020-12からの軽微な相違）。Gemini 3以降平行呼び出し一意のID。

### `tool_choice`動作

3つのモード全員がサポートする、異なる名前：

- **自動。** モデルがツールまたはテキストを選択します。デフォルト。
- **必須 / 任意。** モデルは少なくとも1つのツールを呼び出す必要があります。
- **なし。** モデルはツールを呼び出してはいけません。

さらに各プロバイダーに固有の1つのモード：

- **OpenAI。** 特定のツールを名前で強制する。
- **Anthropic。** 特定のツールを名前で強制する；`disable_parallel_tool_use`フラグが単一対マルチを分離します。
- **Gemini。** `mode: "VALIDATED"`はモデルの意図に関係なく、すべての応答をスキーマ検証器を通してルーティングします。

### 並列呼び出し

OpenAIの`parallel_tool_calls: true`（デフォルト）は1つのアシスタントメッセージで複数の呼び出しを放出します。それらすべてを実行し、各`tool_call_id`ごとに1つのエントリを含む一括ツールロールメッセージで返信します。Anthropicは歴史的にシングル呼び出しでした。`disable_parallel_tool_use: false`（Claude 3.5以降のデフォルト）はマルチを有効にします。Gemini 2は並列呼び出しを許可しましたが、安定したIDを与えませんでした。Gemini 3は順序外の応答が正確に相関するようにUUIDを追加します。

### ストリーミング

3つすべてはストリーミングされたツール呼び出しをサポートしています。ワイヤ形式は異なります：

- **OpenAI。** `tool_calls[i].function.arguments`のデルタチャンク段階的に到着。`finish_reason: "tool_calls"`まで蓄積します。
- **Anthropic。** ブロック開始/ブロックデルタ/ブロック停止イベント。`input_json_delta`チャンク部分的引数を運びます。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3の新機能）は`functionCallId`を持つチャンクを放出し、複数の並列呼び出しはインターリーブできます。

フェーズ13・03は平行+ストリーミング再組立に深く掘り込みます。このレッスンは宣言とシングル呼び出し形状に焦点を当てます。

### エラーと修復

無効引数エラーも見た目が異なります。

- **OpenAI（非厳密）。** モデルは`arguments: "{bad json}"`を返します、JSONパースが失敗します、エラーメッセージを注入し、再呼び出しします。
- **OpenAI（厳密）。** 検証はデコーディング中に行われます；無効なJSONは不可能ですが`refusal`が表示される場合があります。
- **Anthropic。** `input`は予期しないフィールドを含めることができます；スキーマはアドバイザリーです。サーバー側で検証します。
- **Gemini。** OpenAPI 3.0の癖：オブジェクトフィールドの`enum`は黙って無視されます；自分で検証します。

### トランスレータパターン

コードの標準的なツール宣言は次のようになります（形状は選択します）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

3つの小さな関数がそれを3つのプロバイダー形状に変換します。`code/main.py`内ハーネスは正確にこれを行い、その後、偽のツール呼び出しを各プロバイダーの応答形状を通じてラウンドトリップします。ネットワークは必要なし — このレッスンは形状を教えており、HTTPではありません。

本番チームは、このトランスレータを`AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）、または`BaseTool`（LlamaIndex）でラップします。フェーズ13・17は、3つのいずれかの前にOpenAIシェイプドAPIを公開するゲートウェイを配布します。

## 使用方法

`code/main.py`は1つの正規`Tool`データクラスと、OpenAI、Anthropic、Gemini宣言JSONを放出する3つのトランスレータを定義します。その後、各形状の手作り作成プロバイダー応答を同じ正規呼び出しオブジェクトにパースし、皮膚の下のセマンティクスが同じであることを示します。実行し、3つの宣言を側面比較します。

見るべきもの：

- 3つの宣言ブロックはエンベロープとフィールド名でのみ異なります。
- 3つの応答ブロックは、呼び出しが存在する場所で異なります（トップレベル`tool_calls`、`content[]`ブロック、`parts[]`エントリ）。
- 1つの`canonical_call()`関数は`{id, name, args}`をすべての3つの応答形状から抽出します。

## 出荷

このレッスンは`outputs/skill-provider-portability-audit.md`を生成します。1つのプロバイダーに対する関数呼び出し統合が与えられた場合、スキルはポータビリティ監査を生成します：どのプロバイダー制限それが依存するか、どのフィールドを名前変更する必要があるか、各他プロバイダーに移植するときに何が破損するか。

## 演習

1. `code/main.py`を実行し、3つのプロバイダー宣言JSONが同じ基本的な`Tool`オブジェクトをすべてシリアライズすることを確認します。列挙パラメータを追加するために標準的なツールを変更し、Gemini トランスレータのみがOpenAPI癖を処理する必要があることを確認します。

2. モデルが`list_tools`または発見呼び出し後に返すツールリストを抽出する各プロバイダーに対して`ListToolsResponse`パーサーを追加します。OpenAIはネイティブに1つを持っていません；この非対称性に注意してください。

3. `tool_choice`変換を実装する：正規`ToolChoice(mode="force", tool_name="x")`をすべての3つのプロバイダー形状にマップします。次に`mode="any"`と`mode="none"`をマップしてください。レッスンの差分テーブルを確認してください。

4. 3つのプロバイダーのうち1つを選び、そのエンドツーエンド関数呼び出しガイドを読んでください。他の2つがサポートしていないスキーマ仕様内で1つのフィールドを見つけてください。候補：OpenAI`strict`、Anthropic`disable_parallel_tool_use`、Gemini`function_calling_config.allowed_function_names`。

5. テストベクトルを記述する：宣言されたスキーマに違反する引数を持つツール呼び出し。レッスン01の標準的なツール呼び出しを通してそれを実行し、各プロバイダーのバリデータのエラーが点火を記録してください。本番で使用するプロバイダーを厳密性で記録してください。

## キーターム

| 用語 | 人々が言うこと | それが実際に意味するもの |
|------|----------------|------------------------|
| 関数呼び出し | 「ツール使用」 | 構造化ツール呼び出し放出用のプロバイダーレベルAPI |
| ツール宣言 | 「ツール仕様」 | 名前+説明+JSON入力スキーマペイロード |
| `tool_choice` | 「強制/禁止」 | 自動/必須/なし/特定名モード |
| 厳密モード | 「スキーマ適用」 | スキーマにマッチするようにデコーディングを制約するOpenAIフラグ |
| `tool_use`ブロック | 「Anthropicの呼び出し形状」 | id、name、inputを持つインラインコンテンツブロック |
| `functionCall`部分 | 「Geminiの呼び出し形状」 | name、args、idを含む`parts[]`エントリ |
| 文字列としての引数 | 「文字列化JSON」 | OpenAIが引数をオブジェクトではなくJSON文字列として返す |
| 並列ツール呼び出し | 「1ターンでのファンアウト」 | 1つのアシスタントメッセージの複数ツール呼び出し |
| 拒否 | 「モデルが辞退」 | 呼び出しの代わりに厳密モード拒否ブロック |
| OpenAPI 3.0サブセット | 「Geminiスキーマの癖」 | Geminiは軽微な相違を伴うJSONスキーマのような方言を使用 |

## 参考文献

- [OpenAI — 関数呼び出しガイド](https://platform.openai.com/docs/guides/function-calling) — 厳密モードと並列呼び出しを含む標準参照
- [Anthropic — ツール使用概要](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use`と`tool_result`ブロックセマンティクス
- [Google — Gemini関数呼び出し](https://ai.google.dev/gemini-api/docs/function-calling) — 並列呼び出し、一意のID、OpenAPIサブセット
- [Vertex AI — 関数呼び出し参照](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Geminiのエンタープライズサーフェス
- [OpenAI — 構造化出力](https://platform.openai.com/docs/guides/structured-outputs) — 厳密モードスキーマ適用の詳細
