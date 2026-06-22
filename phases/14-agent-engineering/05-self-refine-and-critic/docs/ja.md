# Self-RefinとCRITIC：反復出力改善

> Self-Refine (Madaan et al., 2023)は1つのLLMを3つの役割 — 生成、フィードバック、洗練 — ループで使用します。平均利得：7つのタスクで+20絶対。CRITIC (Gou et al., 2023)は検証ステップを外部ツール経由でルーティングすることでフィードバックステップを強化します。2026年ではこのパターンすべてのフレームワークで「evaluator-optimizer」(Anthropic)またはガードレールループ(OpenAI Agents SDK)として出荷されます。

**タイプ:** ビルド
**言語:** Python (stdlib)
**前提条件:** Phase 14 · 01 (Agent Loop)、Phase 14 · 03 (Reflexion)
**所要時間:** 約60分

## 学習目標

- Self-Refineの3つのプロンプト(生成、フィードバック、洗練)を述べ、洗練プロンプトに履歴が重要である理由を説明する。
- CRITICの重大な洞察を説明：LLMは外部根拠なしで自己検証に信頼できない。
- 履歴とオプションの外部検証器を備えた標準ライブラリSelf-Refineループを実装する。
- このパターンをAnthropicの「evaluator-optimizer」ワークフローとOpenAI Agents SDKの出力ガードレールにマッピングする。

## 問題

エージェントはほぼ正しい答えを生成します。多くの場合、コード行に構文エラーがあります。多くの場合、要約が長すぎます。多くの場合、プランがエッジケースを見落とします。欲しいのは：エージェントが独自の出力を批判し、その後修正することです。

Self-Refineはこれが単一のモデル、トレーニングデータなし、RLなしで機能することを示しています。しかし、ここに落とし穴があります：LLMは困難な事実に関する自己検証に悪いです。CRITICは修正名を付けます — 検証ステップを外部ツール(検索、コード解釈、計算機、テストランナー)経由でルートします。

これら2つの論文を一緒に、反復改善のための2026年デフォルトを定義します：生成、検証(可能な場合は外部)、洗練、検証器がパスした場合に停止。

## コンセプト

### Self-Refine (Madaan et al., NeurIPS 2023)

1つのLLM、3つの役割：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

キーディテール：`refine`は完全な履歴を見ます — すべての先行する出力と批判 — それで誤った繰り返しません。論文はこれを廃止します：履歴を削除すると品質が急落します。

見出し：7つのタスク(数学、コード、頭字語、ダイアログ)全体で平均+20絶対改善(GPT-4を含む)。トレーニングなし、外部ツールなし、単一モデル。

### CRITIC (Gou et al., arXiv:2305.11738、v4 Feb 2024)

Self-Refineの弱点：フィードバックステップはLLMが自分自身をスコア化しています。事実的請求ではこれは信頼できない(幻覚はしばしば、それを生成したモデルに説得力を持っています)。CRITIC `feedback(task, output)`を`verify(task, output, tools)`で置き換えます。ここで`tools`は以下を含みます：

- 事実的請求用の検索エンジン。
- コード正確性用のコード解釈。
- 算術用の計算機。
- ドメイン固有の検証器(ユニットテスト、型チェッカー、リンター)。

検証器はツール結果に根付いた構造化された批判を生成します。リファイナーはこの批判を条件とします。

見出し：CRITICはSelf-Refineを事実的タスクで上回ります。なぜなら批判は根拠があるからです。外部検証器のないタスク(創意的ライティング、フォーマッティング)では、CRITICはSelf-Refineに低下します。

### 停止条件

2つの一般的な形：

1. **検証器がパス。** 外部テストは成功を返します。利用可能な場合に推奨(ユニットテスト、型チェッカー、ガードレールアサーション)。
2. **フィードバックは発行されない。** モデルは「出力は問題ありません。」と言う。安いですが信頼できない；max-iterationキャップとペアにします。

2026年デフォルト：それらを組み合わせます。「検証器がパスした場合、またはモデルが問題ありません。かつiteration >= 2、またはiteration >= max_iterationの場合に停止」。

### Evaluator-Optimizer (Anthropic, 2024)

Anthropicの2024年12月投稿はこれを5つのワークフローパターンの1つとして名前付けます。2つの役割：

- 評価器：出力をスコア化し、批判を生成します。
- オプティマイザ：批判を与えられた出力を改訂します。

評価器がパスするまでループします。これはAnthropicのフレーミングではSelf-Refine/CRITICです。Anthropicが追加する重大なエンジニアリングディテール：評価器とオプティマイザプロンプトはモデルが単にゴム印を押すだけではないように実質的に異なるべきです。

### OpenAI Agents SDKの出力ガードレール

OpenAI Agents SDKは「出力ガードレール」としてこのパターンを出荷します。ガードレールはエージェントの最終出力で実行される検証器です。ガードレールがトリップした場合(OutputGuardrailTripwireTriggeredを上げる)、出力は拒否されエージェントは再試行できます。ガードレールはツール(CRITIC形式)または純粋関数(Self-Refine形式)を呼び出すことができます。

### 2026年のピットフォール

- **ゴム印ループ。** 同じモデルが同じプロンプトスタイルで生成と批判を行うことは「私には良さそうです」に収束します。構造的に異なるプロンプト、または批判用の小さい安いモデルを使用します。
- **オーバー洗練。** 各洗練パスはレイテンシとトークンを追加します。1-3パスを予算；その後、人間レビューにエスカレートします。
- **些細なタスクでのCRITIC。** 外部検証器がない場合、CRITICはSelf-Refineに低下します；スタブ検証器のレイテンシを支払わないでください。

## ビルド

`code/main.py`はおもちゃのタスクでSelf-RefinとCRITICを実装します：トピックを与えられた短いbulletリストを生成します。検証器はフォーマット(3つのbullet、各々60文字以下)をチェックします。CRITICは既知の幻覚にペナルティを課す外部「fact verifier」を追加します。

コンポーネント：

- `generate` — スクリプト化されたプロデューサー。
- `feedback` — LLM形式自己批判。
- `verify_external` — CRITIC形式根拠検証器。
- `refine` — 履歴を与えられた出力を書き換え。
- 停止条件 — 検証器はパス、または最大4反復。

実行：

```
python3 code/main.py
```

Self-RefinとCRITIC実行を比較します。CRITICは事実的エラーを捕捉します。Self-Refineはそれを見逃しました。なぜなら外部検証器は自己批判が持たない根拠を持っているからです。

## 使う

Anthropicのevaluator-optimizerはこのパターンをClaude友好的な言語です。OpenAI Agents SDKの出力ガードレールはCRITC形(ガードレールはツールを呼び出すことができます)。LangGraphは自己批判のような読み取りの反射ノードを出荷します。Googleの Gemini 2.5 Computer UseはCRITICバリアント：毎ステップのセーフティ評価器を追加します。すべてのアクションはコミット前に検証されます。

## 出荷

`outputs/skill-refine-loop.md`はタスク形、検証器可用性、および反復予算を与えられたevaluator-optimizerループを構成します。生成器、評価器/検証器、およびオプティマイザのプロンプト、プラス停止ポリシーを発行します。

## 演習

1. max_iterations=1でおもちゃを実行します。CRITICはまだ役に立ちますか？
2. 外部検証器をノイズの多いもので置き換え(ランダム30%偽陽性)。ループは何をしますか？これは2026年ほとんどのガードレールスタックの現実です。
3. 「別のモデルで生成器批判」バリアントを実装：大きいモデルは生成、小さいモデルは批判。同じモデルを上回りますか？
4. CRITIC Section 3 (arXiv:2305.11738 v4)を読みます。3つの検証ツールカテゴリを名前付けし、各々に例を与えます。
5. OpenAI Agents SDKの`output_guardrails`をCRITICの検証器の役割にマッピングします。SDKが何を間違い、何を正しくするのか？

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|------------|----------|
| Self-Refine | 「自分自身を修正するLLM」 | 1つのモデルでの生成 -> フィードバック -> 洗練ループ、履歴付き |
| CRITIC | 「ツール根拠検証」 | フィードバックを外部検証器(検索、コード、計算、テスト)で置き換え |
| Evaluator-Optimizer | 「Anthropicワークフローパターン」 | 2つの役割 — 評価器スコア、オプティマイザ改訂 — 収束にループ |
| Output guardrail | 「事後確認」 | OpenAI Agents SDK検証器はエージェント出力後に実行 |
| Verify step | 「批判フェーズ」 | ロードベアリング決定：根拠、または自己評価 |
| Refine history | 「モデルは既に試したもの」 | 先行する出力+批判は洗練プロンプトに前置き；削除すると品質が崩壊 |
| Rubber-stamp loop | 「自己合意失敗」 | 同じプロンプト批判は「良さそうです」を返します；構造的に異なるプロンプトで修正 |
| Stop condition | 「収束テスト」 | 検証器がパス、またはフィードバックなし、かつ反復キャップ；決して単一条件 |

## 参考文献

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 標準論文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — ツール根拠検証
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizerワークフローパターン
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — CRITIC形出力ガードレールとしての検証器
