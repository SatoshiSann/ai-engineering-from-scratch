# LLM Observability スタック選択

> 2026 年の observability 市場は 2 つのカテゴリに分割されます。開発プラットフォーム（LangSmith、Langfuse、Comet Opik）は observability を evals、プロンプト管理、セッション リプレイとバンドルします。Gateway/インストルメンテーション ツール（Helicone、SigNoz、OpenLLMetry、Phoenix）はテレメトリに焦点を当てます。Langfuse は MIT ライセンス コア を持つ強力な OSS バランス（月 50K イベント無料クラウド）。Phoenix は Elastic License 2.0 下で OpenTelemetry ネイティブ — ドリフト/RAG ビジュアライゼーションに優れ、永続的本番環境バックエンドではありません。Arize AX は zero-copy Iceberg/Parquet 統合を使用して月 100 倍以上安いと主張（モノリシック observability より）。LangSmith は LangChain/LangGraph でリード、$39/ユーザー/月、Enterprise のみで自己ホスト。Helicone はプロキシ ベースで 15-30 分セットアップ、100K req/mo 無料、ただしエージェント トレースでの深度は少ないです。一般的な本番パターン：Gateway（Helicone/Portkey）+ eval プラットフォーム（Phoenix/TruLens）を OpenTelemetry で接着。

**タイプ:** Learn
**言語:** Python (stdlib、おもちゃのトレース サンプリング シミュレータ)
**前提条件:** Phase 17 · 08（推論メトリクス）、Phase 14（エージェント エンジニアリング）
**所要時間:** 約 60 分

## 学習目標

- 開発プラットフォーム（バンドル：evals + プロンプト + セッション）を gateway/テレメトリ ツール（トレース + メトリクスのみ）と区別してください。
- 6 つのメジャーツール（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）をそれらのライセンシング、価格設定、スイートスポット ユースケースにマップしてください。
- gateway ツールを個別の eval プラットフォームと組み合わせるのを可能にする OpenTelemetry 接着パターンを説明してください。
- 2026 年コスト差別化因子（Arize AX の zero-copy アプローチ対 モノリシック ingest）を名前で述べ、ラフ 100 倍乗数を述べてください。

## 問題

LLM 機能を出荷しました。機能します。プロンプト障害、ツール ループ、レイテンシ リグレッション、コストスパイク、またはプロンプト キャッシュ ヒット率への可視性がありません。「LLM observability」をググって、同じ問題をすべて主張する 3 つの異なる価格ポイントで 8 つのツール が得られます。

彼らは同じ問題を解決しません。LangSmith は「このLangGraph 実行がなぜ失敗したか？」に答えます。Phoenix は「RAG パイプラインがドリフトしているか？」に答えます。Helicone は「どのアプリがトークンを焼いているか？」に答えます。Langfuse は「自分でホストできるか？」に答えます。異なるツール、異なるオーディエンス。

選択は 4 つの軸に関わります：スタック（LangChain？raw SDK？マルチベンダー？）、ライセンス許容（MIT のみ？Elastic OK？商用 OK？）、予算（無料ティア？$100/mo？$1000/mo？）、自己ホスト（必須？好ましい？なし？）。

## コンセプト

### 2 つのカテゴリ

**開発プラットフォーム** は observability を evals、プロンプト管理、データセット バージョニング、セッション リプレイとバンドル。実験を実行、どのプロンプトが機能したか確認、古いウィナーに対して新しいプロンプトを dataset-regression します。LangSmith、Langfuse、Comet Opik。

**Gateway/テレメトリ ツール** は推論呼び出しをインストルメント化 — プロンプト、レスポンス、トークン、レイテンシ、モデル、コスト。Helicone、SigNoz、OpenLLMetry、Phoenix。ミニマリスト。OpenTelemetry 経由で個別 eval ツールと組み合わせることが可能。

### Langfuse — OSS バランス

- コア Apache / MIT ライセンス；Docker 経由で自己ホスト。
- クラウド無料ティア：月 50K イベント。有料：チーム向け $29/mo。
- Evals、プロンプト管理、トレース、データセット。すべての 4 つの開発プラットフォーム機能の適切なカバレッジ。
- スイートスポット：LangSmith クラス機能が必要だが自己ホストまたは OSS ライセンスのままである必要がある。

### Phoenix（Arize） — テレメトリファースト、OpenTelemetry ネイティブ

- Elastic License 2.0；自己ホスト自明。
- RAG および ドリフト ビジュアライゼーションで優れています。埋め込み空間散布図はファースト クラスとして出荷。
- 永続的本番環境バックエンドとして設計されていない — 主に開発時 observability。
- スイートスポット：RAG パイプライン開発、ドリフト デバッグ、本番用個別 gateway と対にします。

### Arize AX — スケール プレイ

- 商用。Iceberg/Parquet 経由 zero-copy データレイク 統合。
- モノリシック observability（Datadog クラス）より 約 100 倍以上安いと主張。数学：S3 上で独自の Parquet にトレースを保存；Arize は直接読み取り。
- スイートスポット：>10M トレース/日、既存データレイク、Datadog 価格なしで LLM 固有ダッシュボードが必要。

### LangSmith — LangChain/LangGraph ファースト

- 商用、$39/ユーザー/月。Enterprise のみで自己ホスト。
- LangChain および LangGraph スタック向けベストインクラス。どちらにも乗っていない場合、それは説得力がありません。
- スイートスポット：チームは LangChain に コミット、支払い意思あり。

### Helicone — プロキシベース 最小実行可能

- `OPENAI_API_BASE` を Helicone プロキシに交換することで 15-30 分セットアップ。
- MIT ライセンス；月 100K req 無料、有料 $20/mo+。
- フェイルオーバー、キャッシング、レート制限を含む — gateway としても機能。
- エージェント / 複数ステップ トレース での深度が少ない。
- スイートスポット：素早いスタート、単一スタック アプリ、gateway + observability が 1 つに必要。

### Opik（Comet） — OSS 開発プラットフォーム

- Apache 2.0、完全 OSS。
- Langfuse に類似する機能セット（Comet ヘリテッジ）。
- スイートスポット：ML チームは既に Comet にいて、同じペインで LLM observability を望む。

### SigNoz — OpenTelemetry ファースト 全 APM

- Apache 2.0。OpenTelemetry 経由でサービスと LLM をハンドル。
- スイートスポット：サービスと LLM コールにわたる統一 observability。

### 接着剤：OpenTelemetry + GenAI セマンティック規約

OpenTelemetry は 2025 年後期に GenAI セマンティック規約を発行（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。OTel を消費するツールは相互運用可能。出現している本番パターン：

1. あらゆる LLM 呼び出しから GenAI 規約を使用した OTel を発行。
2. gateway（Helicone / Portkey）にルーティング 日々用。
3. デュアルシップ eval プラットフォーム（Phoenix / Langfuse）にリグレッション用。
4. データレイク（Iceberg）でアーカイブ 長期分析用 Arize AX または DuckDB 経由。

### 罠：間違ったレイヤーでのインストルメンテーション

エージェント フレームワーク内でのインストルメンテーション（例えば LangSmith トレース追加）はフレームワークに結合されます。HTTP/OpenAI SDK レイヤー（OpenLLMetry または gateway 経由）でのインストルメンテーションはポータブル。

### サンプリング — すべてを保つことはできない

>1M リクエスト/日で、完全トレース保持は LLM 呼び出しよりコストが多くかかります。ルールでサンプリング：100% エラー、100% 高コスト、5% 成功。集計は常に保つ；ロングテイルのロー は保つ。

### 覚えるべき数字

- Langfuse 無料クラウド：月 50K イベント。
- LangSmith：$39/ユーザー/月。
- Helicone 無料：月 100K req。
- Arize AX 主張：スケール時モノリシックより 約 100 倍以上安い。
- OpenTelemetry GenAI 規約：2025 年出荷、2026 年広く採用。

## 使用方法

`code/main.py` は 100 万トレース日をシミュレート（100% ingest、サンプリング、サンプリング + エラー）。保管コストと各々で失われるものを報告。

## 出荷方法

このレッスンは `outputs/skill-observability-stack.md` を生成します。スタック、スケール、予算、ライセンス 姿勢が与えられた場合、ツール を選択。

## 演習

1. LangChain 上のチーム は OSS 自己ホスト observability が必要。Langfuse または Opik を選択し、正当化。
2. 5M トレース/日で Datadog 見積もり $150K/月、Arize AX 向けの break-even を計算。
3. OpenTelemetry GenAI 属性セットを設計 組織のガイドラインがあらゆる LLM 呼び出しで必須すべき。
4. Phoenix のみが本番環境対応か議論。いつ不足するか？
5. Helicone は 20ms プロキシ オーバーヘッド。P99 TTFT 300 ミリ秒で、それはアクセプト可能？SLA が 100 ミリ秒の場合？

## 主要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|----------------|------------------------|
| OpenLLMetry | "LLM 用 OTel" | LLM 用オープンソース OpenTelemetry インストルメンテーション |
| GenAI 規約 | "OTel 属性" | LLM 呼び出し用標準 OTel 属性名 |
| LangSmith | "LangChain observability" | LangChain エコシステムとバンドルされた商用プラットフォーム |
| Langfuse | "OSS LangSmith" | 同様の機能セット付き MIT OSS |
| Phoenix | "Arize 開発ツール" | OpenTelemetry ネイティブ 開発/eval プラットフォーム |
| Arize AX | "スケール observability" | 商用 zero-copy Iceberg/Parquet observability |
| Helicone | "プロキシ observability" | LLM テレメトリを収集する HTTP プロキシ + gateway 機能 |
| Opik | "Comet LLM" | Comet からの Apache 2.0 OSS 開発プラットフォーム |
| セッション リプレイ | "トレース リラン" | ツール呼び出し付きで完全エージェント セッションを再生 |
| Eval | "オフライン テスト" | ラベル付きデータセット上で候補モデル/プロンプトを実行 |

## 参考文献

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
