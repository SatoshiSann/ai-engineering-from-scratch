# モデレーションシステム — OpenAI、Perspective、Llama Guard

> 本番環境のモデレーションシステムは、レッスン12～16で定義された安全方針を実装化する。OpenAI Moderation API：`omni-moderation-latest`（2024年）はGPT-4oで構築され、テキストと画像を1回の呼び出しで分類します。多言語テストセットで前バージョンより42%改善。レスポンススキーマは13のカテゴリ真偽値を返す — harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic。ほとんどの開発者は無料。レイヤード パターン：入力モデレーション（生成前）、出力モデレーション（生成後）、カスタムモデレーション（ドメイン固有ルール）。非同期並列呼び出しでレイテンシを隠蔽。フラグ立た時のプレースホルダレスポンス。Llama Guard 3/4（レッスン16）：14のMLCommons危険性、Code Interpreter Abuse、8言語（v3）、マルチ画像（v4）。Perspective API（Google Jigsaw）：LLM時代以前の毒性スコアリング。主に1次元の毒性を持つが、severe-toxicity/insult/profanity の変種あり。コンテンツモデレーション研究のベースライン。廃止予定：Azure Content Moderator は2024年2月廃止、2027年2月終了予定、Azure AI Content Safety に置き換わり。

**タイプ:** ビルド
**言語:** Python（標準ライブラリ、3層モデレーションハーネス）
**前提条件:** フェーズ18・16（Llama Guard / Garak / PyRIT）
**所要時間:** 約60分

## 学習目標

- OpenAI Moderation APIのカテゴリ分類体系を説明し、Llama Guard 3のMLCommons セットとどう異なるかを説明する。
- 3層モデレーションパターン（入力、出力、カスタム）を説明し、各層の障害モードを1つ名前を挙げる。
- Perspective APIのLLM以前時代のベースラインとしての位置づけと、それが研究でも使われ続ける理由を説明する。
- Azure廃止タイムラインを述べる。

## 問題

レッスン12～16は攻撃と防御ツールを説明する。レッスン29は、ユーザーが製品に触れる表面で防御を実装化した本番環境のモデレーションシステムをカバーする。3層パターンは2026年のデフォルト設定である。

## コンセプト

### OpenAI Moderation API

`omni-moderation-latest`（2024年）。GPT-4oで構築。テキストと画像を1回の呼び出しで分類。ほとんどの開発者は無料。

カテゴリ（レスポンススキーマの13の真偽値）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

マルチモーダルサポートは`violence`、`self-harm`、`sexual`に適用されますが、`sexual/minors`には適用されません。その他はテキストのみ。

`code/main.py`のコードハーネスでは、教育的簡潔性のため、`/threatening`、`/intent`、`/instructions`、`/graphic`のサブカテゴリをそれぞれの最上位親に統合する。本番コードは完全な13カテゴリスキーマを使用すべき。

多言語テストセットで前世代のモデレーションエンドポイントより42%改善。カテゴリごとのスコア。アプリケーションがしきい値を設定。

### Llama Guard 3/4

レッスン16でカバー。14のMLCommons危険性カテゴリ（OpenAIの13のレスポンススキーマ真偽値と異なる組織方法）。8言語対応（v3）。Llama Guard 4（2025年4月）はネイティブマルチモーダル、12B。

OpenAIとLlama Guardの分類体系は重複していますが相違もあります。OpenAIは「illicit」を広いカテゴリとして持つ一方、Llama Guardは「violent crimes」と「non-violent crimes」を別々に持つ。デプロイメントはポリシー分類体系の適合性に基づいて選択。

### Perspective API（Google Jigsaw）

毒性スコアリングシステムで、LLM時代前（2020年以前）のもの。カテゴリ：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。1次元の主スコア（TOXICITY）とサブ次元バリアント。

APIが安定し、ドキュメント化され、多年のキャリブレーションデータがあるため、コンテンツモデレーション研究のベースラインとして広く使用されている。最新のLLM関連ユースケースでは、Llama GuardまたはOpenAI Moderation が通常、より適切。

### 3層パターン

1. **入力モデレーション。** 生成前にユーザーのプロンプトを分類。フラグが立ったら拒否。レイテンシ：1つの分類器呼び出し。
2. **出力モデレーション。** 配信前にモデルの出力を分類。フラグが立ったら拒否で置き換え。レイテンシ：生成後の1つの分類器呼び出し。
3. **カスタムモデレーション。** ドメイン固有ルール（正規表現、許可リスト、ビジネスポリシー）。入力または出力のいずれかで実行。

3層は設計により順序立てられている：入力モデレーションは生成前に完了する必要があり、出力モデレーションは生成後に実行される。並列処理は層内で適用される — 複数の分類器（例：OpenAI Moderation + Llama Guard + Perspective）を同じテキストで並行実行するとクラス分類器ごとのレイテンシが隠蔽される。オプション最適化として、入力モデレーション完了中にプレースホルダレスポンス（「確認中です...」）を表示し、トークン-1ストリーミングを遅延させることができます。フラグ動作は設定可能：拒否、サニタイズ、人間によるレビューへのエスカレーション。

### 障害モード

- **入力のみ。** 出力ハルシネーションをキャッチできない（レッスン12-14エンコーディング攻撃は入力分類器をバイパス）。
- **出力のみ。** あらゆる入力がモデルに到達可能。コスト増加。内部推論を攻撃者に公開。
- **カスタムのみ。** カテゴリ全体で堅牢でない。正規表現は脆弱。

レイヤード（3層すべて）がデフォルト。二重防御。

### Azure廃止予定

Azure Content Moderator：2024年2月廃止予定、2027年2月終了予定。Azure AI Content Safety に置き換わり。これはLLMベースで、Azure OpenAIと統合。移行はAzureデプロイメント向けの2024～2027年フィールドレベルプロジェクト。

### フェーズ18での位置づけ

レッスン16はレッドチームコンテキストでモデレーションツールをカバー。レッスン29はデプロイされたモデレーション。レッスン30は現在の二重用途能力証拠で終わり。

## これを使う

`code/main.py`は3層モデレーションハーネスをビルド：入力モデレータ（キーワード + カテゴリスコア）、出力モデレータ（同じ分類器を出力に）、カスタムモデレータ（ドメインルール）。入力を実行してどの層が何をキャッチするか観察できます。

## デプロイする

このレッスンは`outputs/skill-moderation-stack.md`を生成。デプロイメントが与えられたとき、モデレーションスタック設定を推奨：入力時のどの分類器、出力時のどの分類器、どのカスタムルール、エッジケースのどの判定者。

## 演習

1. `code/main.py`を実行。無害な、ボーダーライン、有害な入力を3層すべてを通して実行。各層が何に反応するか報告。

2. 特定カテゴリーのPerspective-API形式の毒性スコアリングでハーネスを拡張。そのしきい値動作をカテゴリスコアと比較。

3. OpenAI Moderation API ドキュメントとLlama Guard 3カテゴリリストを読む。各OpenAIカテゴリを最も近いLlama Guardカテゴリにマップ。きれいにマップしない3つのカテゴリを特定。

4. コードアシスタントデプロイメント（例：GitHub Copilot）向けのモデレーションスタックを設計。最も関連が高いカテゴリと最も低いカテゴリを特定し、カスタムルールを提案。

5. Azure Content Moderator は2027年2月に終了予定。Azure AI Content Safety への移行を計画。移行の最高リスク要素を特定。

## キーターム

| 用語 | 人が言う | 実際の意味 |
|------|---------|----------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4oベースの13カテゴリ（テキスト）分類器、部分的マルチモーダルサポート |
| Perspective API | "Google Jigsaw 毒性" | LLM以前時代の毒性スコアリングベースライン |
| Llama Guard | "MLCommons 14カテゴリ" | Metaの危険性分類器（v3：8B テキスト、8言語；v4：12B マルチモーダル） |
| 入力モデレーション | "生成前フィルタ" | モデル呼び出し前のユーザープロンプトの分類器 |
| 出力モデレーション | "生成後フィルタ" | 配信前のモデル出力の分類器 |
| カスタムモデレーション | "ドメインルール" | デプロイメント固有ルール（正規表現、許可リスト、ポリシー） |
| レイヤードモデレーション | "3層すべて" | 標準本番環境デプロイメントパターン |

## 参考文献

- [OpenAI Moderation API ドキュメント](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation エンドポイント
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard リポジトリ
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性スコアリング
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 代替
