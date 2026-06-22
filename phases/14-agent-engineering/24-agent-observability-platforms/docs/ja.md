# エージェント可観測性：Langfuse、Phoenix、Opik

> 3 つのオープンソースエージェント可観測性プラットフォームが 2026 年を支配します。Langfuse（MIT）— 月間 6M+ インストール、トレーシング + プロンプト管理 + evals + セッション再生。Arize Phoenix（Elastic 2.0）— 深いエージェント固有の evals、RAG 関連性、OpenInference auto-instrumentation。Comet Opik（Apache 2.0）— 自動プロンプト最適化、ガードレール、LLM-judge ハルシネーション検出。

**タイプ:** 学習
**言語:** Python（標準ライブラリ）
**前提条件:** Phase 14 · 23（OTel GenAI）
**所要時間:** 約 45 分

## 学習目標

- 3 つのトップオープンソースエージェント可観測性プラットフォームとそれらのライセンスを挙げてください
- 各々が最高の何であるかを区別してください：Langfuse（プロンプト mgmt + セッション）、Phoenix（RAG + auto-instrumentation）、Opik（最適化 + ガードレール）
- 2026 年までに、89% の組織がエージェント可観測性を所有することを報告している理由を説明してください
- LLM-judge 評価を持つ stdlib トレース－ツー－ダッシュボードパイプラインを実装してください

## 問題

OTel GenAI（Lesson 23）はスキーマを提供します。スパンを取り込む、evals を実行する、プロンプトバージョンを保存する、回帰を表面にするプラットフォームが必要です。3 つの競争者は各々ライフサイクルの異なる部分を強調します。

## コンセプト

### Langfuse（MIT）

- 月間 6M+ SDK インストール、GitHub スター 19k+
- 機能：トレーシング、バージョン付けを持つ + playground、evals（LLM-as-judge、ユーザーフィードバック、カスタム）、セッション再生をプロンプト管理
- 2025 年 6 月：以前は商用モジュール（LLM-as-a-judge、注釈キュー、プロンプト実験、Playground）MIT 下でオープンソース化
- 最高：緊密なプロンプト管理ループを持つエンドツーエンド可観測性

### Arize Phoenix（Elastic License 2.0）

- より深いエージェント固有の評価：トレース clustering、異常検出、RAG のための取得関連性
- ネイティブ OpenInference auto-instrumentation
- 本番環境のため managed Arize AX と対
- プロンプトバージョンなし — ドリフト/動作回帰ツールとして広いプラットフォームの横に位置付け
- 最高：RAG 関連性、動作ドリフト、異常検出

### Comet Opik（Apache 2.0）

- A/B 実験を通した自動プロンプト最適化
- ガードレール（PII 削除、話題の制約）
- LLM-judge ハルシネーション検出
- Comet の独自測定からのベンチマーク：Opik ログ + evals 23.44s 対 Langfuse 327.15s（~14x ギャップ）— ベンダー公開数を方向性として取得してください
- 最高：最適化ループ、自動実験、ガードレール実施

### 業界データ

Maxim（2026 年フィールド分析）によると、89% の組織がエージェント可観測性を所有；品質問題はトップ本番環境障害（32% の回答者が引用）です。

### 1 つを選ぶ

| 必要 | 選ぶ |
|------|-----|
| すべての 1 つのプロンプト管理 | Langfuse |
| 深い RAG 評価 + ドリフト | Phoenix |
| 自動最適化 + ガードレール | Opik |
| オープンライセンス、ELv2 なし | Langfuse（MIT）または Opik（Apache 2.0）|
| Datadog / New Relic 統合 | 何か — すべて OTel をエクスポート |

### このパターンがうまくいかない場所

- **eval 戦略なし。** 評価なしのトレーシングは高価なロギングだけです
- **グラウンディングなしで自分－ロール LLM-judge。** CRITIC パターン（Lesson 05）が適用 — 判官は因数分解検証のための外部ツールが必要です
- **トレースに結びついたプロンプトバージョンなし。** 本番環境が回帰するとき、プロンプトをそれに実装できません

## ビルド

`code/main.py` は stdlib トレースコレクター + LLM-judge 評価者を実装します：

- GenAI 形スパンを取り込む
- セッションでグループ化、失敗した実行をタグ（ガードレールトリップ、低信頼度 evals）
- ルーブリックに対するスコアを付けるスクリプト化 LLM-judge
- ダッシュボードのような概要：失敗率、トップ失敗理由、eval スコア分布

実行：

```
python3 code/main.py
```

出力：セッションごとの eval スコアと失敗分類、Langfuse/Phoenix/Opik が示すもの マッチング

## 使用方法

- **Langfuse** 自己ホスト型または クラウド；OTel またはそれらの SDK 経由で配線
- **Arize Phoenix** 自己ホスト型；OpenInference auto-instrument
- **Comet Opik** 自己ホスト型またはクラウド；自動最適化ループ
- **Datadog LLM Observability** 既に Datadog を実行している混合 ops+ML チーム用

## 出荷

`outputs/skill-obs-platform-wiring.md` プラットフォームを選び、トレース + evals + プロンプトバージョンを既存エージェントに配線します。

## 演習

1. Langfuse クラウド（無料層）に 1 週間の OTel トレースをエクスポートしてください。どのセッションが失敗しましたか？なぜですか？
2. 自分のドメイン用の LLM-judge ルーブリック（因数分解正確性、トーン、スコープ遵守）を書いてください。50 トレースで試してください
3. Langfuse プロンプトバージョン対 Phoenix のトレース clustering を比較してください。どちらが何が壊れたかを高速に示しますか？
4. Opik のガードレール doc を読んでください。1 つのエージェント実行に PII 削除ガードレールを配線してください
5. 3 つを自分の corpus 上で ベンチマークしてください。ベンダー発行数を無視；自分の測定してください

## キーターム

| 用語 | 人々が言うこと | 実際の意味 |
|------|----------------|----------|
| Tracing | 「Spans collector」 | OTel / SDK スパンを取り込む；セッションでインデックス |
| Prompt management | 「Prompt CMS」 | バージョン付けプロンプト結びついたトレース |
| LLM-as-judge | 「Automated eval」 | 分離 LLM ルーブリックに対するエージェント出力をスコア |
| Session replay | 「Trace playback」 | デバッグのための過去実行をステップスルー |
| RAG relevancy | 「Retrieval quality」 | 取得コンテキストは質問にマッチ |
| Trace clustering | 「Behavioral grouping」 | ドリフト検出のための類似実行クラスター |
| Guardrail enforcement | 「Policy at log time」 | ログされたコンテンツの PII/毒性/スコープチェック |

## 参考文献

- [Langfuse docs](https://langfuse.com/) — トレーシング、evals、プロンプト mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation、ドリフト
- [Comet Opik](https://www.comet.com/site/products/opik/) — 最適化 + ガードレール
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 3 つすべてが消費するスキーマ
