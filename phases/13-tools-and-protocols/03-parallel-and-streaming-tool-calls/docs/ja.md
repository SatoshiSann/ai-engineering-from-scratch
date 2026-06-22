# 並列ツール呼び出しとストリーミング

> 3つの独立した気象ルックアップが順序付きされた場合は3回のラウンドトリップです。平行で実行すればTotal時間は最も遅い単一呼び出しに折りたたまれます。すべての最先端プロバイダーは現在、1つのターン内で複数のツール呼び出しを放出します。ペイロフは本物です；配管は微妙です。このレッスンは両方の半分を説明します：平行ファンアウトとストリーム引数の再組立、ID相関トラップに重点を置きます。

**タイプ:** ビルド
**言語:** Python（stdlib、スレッドプール+ストリーミングハーネス）
**前提条件:** フェーズ13・02（関数呼び出し詳細解説）
**所要時間:** 約75分

## 学習目標

- `parallel_tool_calls: true`が存在する理由とそれを無効にするべき場合を説明します。
- 平行ファンアウト中に、ストリーム引数チャンクを正しいツール呼び出しIDと相関させる。
- 部分的な`arguments`文字列を完全なJSONに再組立し、早期にパースしない。
- 3都市気象ベンチマークを実行して、順序付きビル並列レイテンシーを示します。

## 問題

平行呼び出しなしで、「ベンガルール、東京、チューリッヒの気象は何ですか」と答えるエージェントはこれを行います：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

3つのLLMラウンドトリップ、各々も実行計算のレイテンシーを支払う。理想的な壁の時計時間の大体4倍。

並列呼び出しで：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

1つのLLMラウンドトリップ。実行時間は合計ではなく3つの最大です。OpenAI、Anthropic、Geminiの本番ベンチマークは、ファンアウトワークロードで60～70％の壁の時計削減を示しています。

価格は相関複雑性です。3つの呼び出しが順序外で完了する場合、結果はモデルがそれらを並べ直すことができるように一致する`tool_call_id`を運ぶ必要があります。結果がストリーミングされる場合、執行前に部分的な引数フラグメントを完全なJSONに組立てる必要があります。Gemini 3は実世界の問題を解決するために、2つの平行呼び出しが同じツールへの区別がつかない部分を解決するために一意のIDを追加しました。

## コンセプト

### 平行を有効にする

- **OpenAI。** `parallel_tool_calls: true`デフォルトでオン。シリアルを強制するには`false`を設定します。
- **Anthropic。** `disable_parallel_tool_use: false`（Claude 3.5以降のデフォルト）経由平行。シリアルには`true`を設定します。
- **Gemini。** 常に平行対応；`tool_config.function_calling_config.mode = "AUTO"`モデルが決定するようにします。

平行ツールが順序付き依存（`create_file`その後`write_file`）がある場合、1つの呼び出しの出力が別の入力を通知する場合、またはレート制限器がファンアウトを処理できない場合は無効にします。

### ID相関

モデルが放出するすべての呼び出しには`id`があります。ホストが返すすべての結果は同じIDを含める必要があります。これなしで、結果は曖昧です。

- **OpenAI。** 各ツールロールメッセージの`tool_call_id`。
- **Anthropic。** 各`tool_result`ブロックの`tool_use_id`。
- **Gemini。** 各`functionResponse`の`id`（Gemini 3以降；Gemini 2は同じ名前の並列呼び出しのために破損しました。

### 並発呼び出しの実行

ホストは各呼び出しの実行関数を独自のスレッド、コルーチン、またはリモートワーカーで実行します。最も簡単なハーネスはスレッドプールを使用します。本番はasyncioと`asyncio.gather`または構造化并行性を使用します。完了の順序は予測不可能です — IDは識別子です。

1つの一般的なバグ：呼び出しリスト順序で結果で返信します。これは通常機能します（モデルは`tool_call_id`のみを気にしているため）が、結果がドロップされたか複製された場合は、順序外の提出がデバッグを難しくします。完了順序で明示的なIDで返信することが好ましいです。

### ストリーミングツール呼び出し

モデルがストリーミングされる場合、`arguments`はピースで到着します。3つの平行呼び出しの3つの個別ストリーム ワイヤ上でインターリーブします。IDごとに1つのアキュムレータが必要です。

プロバイダーごとの形状：

- **OpenAI。** 各チャンクは`choices[0].delta.tool_calls[i].function.arguments`（部分文字列）です。チャンク`index`を運ぶ（呼び出しリスト内の位置）。インデックスごとに蓄積し、最初の表示時にIDを読み、`finish_reason = "tool_calls"`のときJSONをパース。
- **Anthropic。** ストリームイベントは`message_start`で、次に型`tool_use`を持つ1つの`content_block_start`（id、name、空のinput含む）。`content_block_delta`イベント`input_json_delta`チャンク。`content_block_stop`各ブロックを閉じます。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3以降）呼び出しはインターリーブできるように`functionCallId`を持つチャンクを放出します。Gemini 3前に、ストリーミングは1回に1つの完全な呼び出しを返しました。

### 部分的なJSON及びパース早期トラップ

完全になるまで`arguments`をパースすることはできません。`{"city": "Beng`のような部分的なJSONは無効で増加します。正しいゲートはプロバイダーの呼び出し終了信号です：OpenAIの`finish_reason = "tool_calls"`、Anthropicの`content_block_stop`、またはGeminiのストリーム終了イベント。それでのみ`json.loads`を試みます。より堅牢なアプローチは、構造が完成するとイベントが生じるインクリメンタルJSONパーサーを使用します。OpenAIのストリーミングガイドは、ライブ「思考」インジケータを示すUXに対してこれを推奨しています。括弧カウントは完全性テスト（括弧は引用された文字列またはエスケープコンテンツ内の誤検知を引き起こす）として信頼性はなく、非公式なデバッグヒューリスティックとしてのみ使用される必要があります。

### 順序外の完了

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

ホスト返信は依然としてIDを引用する必要があります：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

返信内の順序はOpenAIまたはAnthropicの正確性に対して重要ではありません。Geminiはすべての順序を受け入れIDは一致する限り。

### ベンチマーク：順序付けビル平行

`code/main.py`内ハーネスは400、600、800ミリ秒のレイテンシーを持つ3つの実行プログラムをシミュレートします。順序付きそれを1800ミリ秒合計で実行します。平行max(400, 600, 800) = 800ミリ秒で実行します。差は定数であり、比例ではないため、ツール数で節約が増加します。

実世界の注意：平行呼び出しはダウンストリームAPI。レート制限サービスへの10ウェイファンアウトは失敗するでしょう。フェーズ13・17はゲートウェイレベルのバックプレッシャーをカバーしています；再試行セマンティクスは将来のフェーズのために計画されています。

### ストリーミングファンアウト壁の時計

モデル自体がストリーミングされる場合、すべての呼び出しが最終化するまで待機するのではなく、1つの呼び出しの引数が完全になったらすぐに実行を開始できます。これはOpenAIが文書化する最適化ですが、すべてのSDKは露出しません。このレッスンの内部ハーネスはそれをします：シミュレートされたストリームが引数オブジェクトを完全に出現させるとすぐに、ホストはその呼び出しをキックオフします。

## 使用方法

`code/main.py`は2つの半分があります。最初は3つのシミュレートされた気象呼び出しを`concurrent.futures.ThreadPoolExecutor`を使用して順序付きで実行し、壁の時計時間を印刷します。2番目の半分は偽のストリーミング応答を再生 — 1つのストリーム上で3つの平行呼び出しのインターリーブされた`arguments`チャンク — と`StreamAccumulator`でそれらをID単位で再組立します。LLM、ネットワーク、単なる再組立ロジックなし。

見るべきもの：

- 順序付きタイマーは1.8秒に達します。平行タイマーは同じ偽のレイテンシー上で0.8秒に達します。
- アキュムレータは各ID別にバッファリングしてチャンクが順序外で到着し、各呼び出しのJSONが完全になったときのみパースすることで処理します。
- 実行はすべてのストリームが終了した後でなく、IDの引数が最終化するとすぐにキックオフします。

## 出荷

このレッスンは`outputs/skill-parallel-call-safety-check.md`を生成します。ツールレジストリが与えられた場合、スキルはどのツールがパラレルで安全か監査し、どのツールに順序付き依存があり、どちらがダウンストリームレート制限を圧倒するか — 改訂されたレジストリをツール単位の`parallel_safe`フラグで返す。

## 演習

1. `code/main.py`を実行し、シミュレートされたレイテンシーを変更します。平行対順序付き比率がほぼ`max/sum`（実際の実行はスレッドスケジューリング、シリアライゼーション、ハーネスオーバーヘッドのため理想から逸脱）であることを確認します。平行がどのレイテンシー分布で重要でなくなるでしょうか。

2. アキュムレータを拡張してキャンセル中ストリーム「呼び出しケース」を処理し、そのバッファをドロップし`cancelled`イベントを放出します。どのプロバイダーがこのケースを明示的に文書化しているか。Anthropicの`content_block_stop`セマンティクスおよびOpenAIの`finish_reason: "length"`動作をチェック。

3. スレッドプールを`asyncio.gather`で置き換えます。ベンチマーク両方。実行がI/Oを行う場合のみ、asyncioが低文脈スイッチコストのため小さな勝利が見えるはずですが。

4. 平行化してはいけない2つのツールを選ぶ（例`create_file`その後`write_file`）。`ordering_dependency`グラフをレジストリに追加し、そのグラフ上の平行ファンアウトゲート。これは、将来のエージェント工学フェーズが形式化する依存関係対応スケジューリングの最小限の仕組みです。

5. OpenAIの平行関数呼び出しセクションおよびAnthropicの`disable_parallel_tool_use`ドキュメント読みます。1つの実世界ツールタイプAnthropicが平行性を無効にすることを推奨することを特定します。（ヒント：同じリソースの重大な変更。）

## キーターム

| 用語 | 人々が言うこと | それが実際に意味するもの |
|------|----------------|------------------------|
| 並列ツール呼び出し | 「1ターンでのファンアウト」 | 1つのアシスタントメッセージで複数ツール呼び出しを放出 |
| `parallel_tool_calls` | 「OpenAIのフラグ」 | マルチ呼び出し放出を有効または無効 |
| `disable_parallel_tool_use` | 「Anthropicの逆」 | オプトアウトフラグ；デフォルトは平行が有効 |
| ツール呼び出しID | 「相関ハンドル」 | ツール単位の識別子結果メッセージはエコーする必要があります |
| アキュムレータ | 「ストリームバッファ」 | 部分的な`arguments`チャンクのためのIDごと文字列バッファ |
| 順序外の完了 | 「最初最速」 | 平行呼び出しは予測不可能な順序で完了；IDが糊です |
| 依存性グラフ | 「順序付き制約」 | 出力が別のツールの入力にフィードしますツール；パラレル化できません |
| パース早期トラップ | 「JSON.parseは爆発しました」 | 不完全な`arguments`文字列をパースしようとする |
| `streamFunctionCallArguments` | 「Gemini 3機能」 | 一意のID呼び出しごとのストリーム引数チャンク |
| 完了順序返信 | 「すべてを待機しません」 | 結果をID別にキー設定して到着するとき返信 |

## 参考文献

- [OpenAI — 並列関数呼び出し](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — デフォルト動作とオプトアウトフラグ
- [Anthropic — ツール使用：ツール使用を実装する](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use`および結果バッチング
- [Google — Gemini関数呼び出し平行セクション](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3からのID相関平行呼び出し
- [OpenAI — ツール付きストリーミング応答](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAIストリーム用チャンク引数再組立
- [Anthropic — ストリーミングメッセージ](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta`を`input_json_delta`で
