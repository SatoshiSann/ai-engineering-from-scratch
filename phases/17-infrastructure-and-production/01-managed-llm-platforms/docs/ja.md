# マネージド LLM プラットフォーム — Bedrock、Vertex AI、Azure OpenAI

> 3 つのハイパースケーラー、3 つの異なる戦略。AWS Bedrock はモデルマーケットプレイス — Claude、Llama、Titan、Stability、Cohere が 1 つの API の背後に。Azure OpenAI は OpenAI との排他的なパートナーシップとプロビジョニング スループット ユニット（PTU）による専用キャパシティ。Vertex AI は Gemini ファースト、最高レベルの長コンテキストとマルチモーダル。2026 年の Artificial Analysis の測定では、Azure OpenAI が Llama 3.1 405B 相当品で中央値約 50 ms、Bedrock が約 75 ms — PTU がこのギャップを説明するのは、専用キャパシティがオンデマンド共有に勝るから。判断基準は「どれが最速か」ではなく「どのモデルカタログと FinOps 表面が自分の製品に合致するか」。このレッスンでは感覚的でなく、トレードオフを書き込んで選択する方法を教えます。

**タイプ:** 学習
**言語:** Python（stdlib、簡単なコストと遅延の比較ツール）
**前提条件:** フェーズ 11（LLM エンジニアリング）、フェーズ 13（ツール & プロトコル）
**所要時間:** 約 60 分

## 学習目標

- 3 つのプラットフォーム戦略（マーケットプレイス vs 排他的 vs Gemini ファースト）の名前を挙げ、各戦略を製品ユースケースにマッチングさせる。
- Azure OpenAI でプロビジョニング スループット ユニット（PTU）が何をもたらすか説明し、405B スケールでオンデマンド Bedrock が通常 25 ms 遅い理由を説明する。
- 各プラットフォームの FinOps 属性表面（Bedrock Application Inference Profiles vs Vertex プロジェクト・パー・チーム vs Azure スコープ + PTU 予約）を図解する。
- 「2 プロバイダー最小値」方針を書き込み、2026 年のシングルベンダー ロックインが高くつく間違いである理由を説明する。

## 問題

Claude 3.7 Sonnet を製品用に選択しました。これで配信が必要です。Anthropic API を直接呼び出すか、AWS Bedrock 経由で呼び出すか、ゲートウェイ経由で呼び出すことができます。ダイレクト API が最もシンプル。Bedrock は BAA、VPC エンドポイント、IAM、CloudWatch 属性を追加します。ゲートウェイはフェイルオーバー、統一請求、プロバイダー間のレート制限を追加します。

さらに深い問題はカタログです。Claude と Llama と Gemini を同じ製品で必要とするなら、それらをすべて 1 つの場所から購入することはできません。その場所が同時に Bedrock と Vertex と Azure OpenAI でない限り。ハイパースケーラーは交換可能ではありません — 各自がモデルレイヤーの所有権について異なる賭けをしました。

このレッスンは 3 つの賭け、遅延ギャップ、FinOps ギャップ、ロックインリスクをマップします。

## コンセプト

### 3 つの戦略

**AWS Bedrock** — マーケットプレイス。Claude（Anthropic）、Llama（Meta）、Titan（AWS ファースト パーティ）、Stability（画像）、Cohere（埋め込み）、Mistral、および画像と埋め込みのサブカタログ。1 つの API、1 つの IAM 表面、1 つの CloudWatch エクスポート。Bedrock の賭けは、顧客が単一のモデルより選択肢を望むことです。

**Azure OpenAI** — 排他的なパートナーシップ。GPT-4 / 4o / 5 / o シリーズ、DALL·E、Whisper、および Azure データセンター内の OpenAI モデルのファインチューニングを取得します。「Azure OpenAI Service」カタログに非 OpenAI モデルはありません — それらは Azure AI Foundry（別製品）に行きます。Azure の賭けは OpenAI が最前線に留まり、顧客がその特定の関係に対してエンタープライズコントロールを望むことです。

**Vertex AI** — Gemini ファースト、その他はセカンド。Gemini 1.5 / 2.0 / 2.5 Flash と Pro、プラス Model Garden（サードパーティ）。Vertex の賭けはマルチモーダル長コンテキスト — 1M トークン Gemini コンテキストが差別化要因です。

### スケール時の遅延ギャップ

Artificial Analysis は継続的なベンチマークを実行しています。同等の Llama 3.1 405B デプロイメント（共有オンデマンド）では、Azure OpenAI 中央値初トークン遅延は約 50 ms。Bedrock は約 75 ms。ギャップは AWS の失敗ではなく — キャパシティモデルの違いです。Azure は PTU（プロビジョニング スループット ユニット）を販売し、GPU キャパシティをテナント用に予約します。Bedrock の同等物（プロビジョニング スループット）は存在しますが、ユニット当たり約 $21/時間から始まり、ほとんどの顧客は共有オンデマンドにとどまります。

オンデマンド共有キャパシティは他のすべての顧客のトラフィックと競合します。専用キャパシティはそうではありません。製品 SLA が TTFT < 100 ms at P99 なら、Azure で PTU を買うか、Bedrock プロビジョニング スループットを買うか、デフォルト分散を受け入れます。

### プロビジョニング スループット経済

Azure PTU：推論コンピュートの予約ブロック。予測可能なワークロードで約 70% のオンデマンド節約。トラフィックに関わらず、固定で時間当たりコスト — アイドル時でも予約に支払います。損益分岐点は通常、約 40-60% の持続的利用率です。

Bedrock プロビジョニング スループット：モデルとリージョンに応じて $21-$50/時間。同様の計算 — 損益分岐点はピーク利用率の約半分。月単位のコミットメント必須。

Vertex プロビジョニングキャパシティは Gemini SKU ごとに販売。価格はモデルとリージョンで異なり、公表されていません。

### FinOps 表面 — 真の差別化要因

**Bedrock Application Inference Profiles** がマーケットプレイスで最もクリーンな属性。プロファイルを `team`、`product`、`feature` でタグ付けし、すべてのモデル呼び出しをそれを通じてルーティング。CloudWatch は後処理なしでプロファイル当たりのコストを分割。2025 年追加、依然として最もきめ細かいハイパースケーラー ネイティブ。

**Vertex** の属性はプロジェクト・パー・チーム・プラス・ラベル・エブリウェア。各チームを GCP プロジェクトでモデル化し、すべてのリソースにラベルを付け、BigQuery 請求エクスポート + DataStudio をロールアップに使用。より多くの作業ですが、BigQuery はコストデータに対する任意の SQL を提供。

**Azure** は サブスクリプション / リソースグループスコープ + タグに依存し、PTU 予約を 1 級のコストオブジェクトとして。タグはリクエストからではなくリソースグループから継承されるため、リクエスト単位の属性には Application Insights カスタムメトリクスまたはヘッダーをスタンプするゲートウェイが必要です。

パターン：Bedrock がネイティブで最もクリーン、Vertex は BigQuery で最も柔軟、Azure は計測しない限り最も不透明。

### ロックインは 2026 年のリスク

シングルハイパースケーラーコミットメントは 1 つのモデルが支配していたときは問題ありませんでした。2026 年、最前線は毎月動きます — Claude 3.7 ある四半期、Gemini 2.5 次の四半期、GPT-5 その次の四半期。1 つのプラットフォームへのロックインは最前線の 3 分の 2 から逃れる。

パターンは働くチームが採用：製品に重要な LLM 呼び出しには 2 プロバイダー最小値。Bedrock と Azure OpenAI が一般的なペア — 1 つから Claude、別から GPT、その間のフェイルオーバー、同じゲートウェイ。コスト上昇はゲートウェイが最適ルートのため無視できます。停止中の可用性上昇（2025 年 1 月 Azure OpenAI インシデント、AWS us-east-1 停止など）は決定的です。

### データレジデンシー、BAA、規制対象業界

Bedrock：ほとんどのリージョンで BAA。VPC エンドポイント。ガードレール。一般的なフィンテック デフォルト。
Azure OpenAI：HIPAA、SOC 2、ISO 27001。EU データレジデンシー。エンタープライズ規制デフォルト。
Vertex：HIPAA、GDPR、リージョン当たりのデータレジデンシー。Google Cloud のコンプライアンススタック。

3 つすべてが基本的なチェックボックスを満たします。違いはデータ保持方針、ログの処理方法、虐待監視がトラフィックを読むかどうか（ほとんどはデフォルト オプトイン。エンタープライズではオプトアウト可能）です。

### 覚えるべき数値

- Azure OpenAI Llama 3.1 405B 相当品での中央値 TTFT：約 50 ms（PTU 付き）。
- Bedrock オンデマンド中央値 TTFT：約 75 ms。
- Bedrock プロビジョニング スループット：$21-$50/時間/ユニット。
- Azure PTU 損益分岐点：約 40-60% の持続的利用率。
- 高利用率での PTU 節約 vs オンデマンド：最大 70%。

## 使用

`code/main.py` は 3 つのプラットフォームを合成ワークロード上で比較 — オンデマンド vs PTU 経済、TTFT 分散、コスト属性忠実度をモデル化。これを実行して PTU が支払う場所と、マーケットプレイスのモデル幅がどこで TTFT ギャップを上回るかを見てください。

## 配布

このレッスンは `outputs/skill-managed-platform-picker.md` を生成します。ワークロード プロファイル（必要なモデル、TTFT SLA、日量、コンプライアンス要件）を考えると、主プラットフォーム、フォールバック、FinOps 計測計画を推奨。

## 演習

1. `code/main.py` を実行。Azure PTU がオンデマンドを上回る 70B クラスモデルの持続的利用率はどのくらい？損益分岐点を計算し、提示される 40-60% 帯と比較。
2. 製品に Claude 3.7 Sonnet と GPT-4o が必要。2 プロバイダーデプロイメントを設計 — どれが どのハイパースケーラーに行くか、何のゲートウェイが前に座るか、フェイルオーバー方針は？
3. 規制対象医療顧客が BAA、米国東部データレジデンシー、P99 TTFT < 100ms を要求。プラットフォームを選択し、3 つの特定機能で正当化。
4. Bedrock 請求が今月 4 倍になったことを発見。トラフィック変更なし。Application Inference Profiles なしでは、犯人をどのように見つけますか？プロファイルなら、かかる時間は？
5. Azure OpenAI と Bedrock の価格ページを読み。100M トークン/月の Claude ワークロードについて、どれが安いか — ダイレクト Anthropic API、Bedrock オンデマンド、Bedrock プロビジョニング スループット？

## 重要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-------------|----------|
| Bedrock | 「AWS LLM サービス」 | Claude、Llama、Titan、Mistral、Cohere にまたがるモデルマーケットプレイス |
| Azure OpenAI | 「Azure の ChatGPT」 | Azure データセンター内のエンタープライズコントロル付き排他的 OpenAI モデル |
| Vertex AI | 「Google の LLM」 | Gemini ファースト プラットフォーム、Model Garden でサードパーティモデル |
| PTU | 「専用キャパシティ」 | プロビジョニング スループット ユニット — 予約推論 GPU、時間当たり価格 |
| Application Inference Profile | 「Bedrock タグ付け」 | プロダクト当たりのコスト/利用プロファイル、タグ付け、CloudWatch ネイティブ |
| Model Garden | 「Vertex カタログ」 | Vertex AI の サードパーティモデル セクション、Gemini から分離 |
| 2 プロバイダー最小値 | 「LLM 冗長性」 | すべての重要な LLM パスを >= 2 ハイパースケーラーで実行するポリシー |
| BAA | 「HIPAA ペーパーワーク」 | ビジネス アソシエート契約。PHI に必須。3 つすべてが提供 |
| 虐待監視 | 「ログウォッチャー」 | プロンプト/出力のプロバイダー側安全スキャン。エンタープライズではオプトアウト可能 |

## 参考文献

- [AWS Bedrock 価格](https://aws.amazon.com/bedrock/pricing/) — 正式レートカード、プロビジョニング スループット価格。
- [Azure OpenAI Service 価格](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 経済、レートカード。
- [Vertex AI 生成 AI 価格](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini ティア、Model Garden 追加料金。
- [Artificial Analysis LLM リーダーボード](https://artificialanalysis.ai/) — プロバイダー間の継続的遅延・スループットベンチマーク。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO ガイド 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — エンタープライズ決定フレームワーク。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 属性メカニクス並列。
