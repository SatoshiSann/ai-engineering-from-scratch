# Agno と Mastra：本番環境用ランタイム

> Agno（Python）と Mastra（TypeScript）は 2026 年の本番環境用ランタイムペアリングです。Agno はマイクロ秒単位のエージェント起動と、ステートレスな FastAPI バックエンドを目指しています。Mastra は Vercel AI SDK 上にエージェント、ツール、ワークフロー、統一的なモデルルーティング、複合ストレージを提供します。

**タイプ:** 学習
**言語:** Python、TypeScript
**前提条件:** Phase 14 · 01（エージェントループ）、Phase 14 · 13（LangGraph）
**所要時間:** 約 45 分

## 学習目標

- Agno のパフォーマンス目標と、それらがいつ重要かを認識する
- Mastra の 3 つのプリミティブ（Agents、Tools、Workflows）と対応するサーバーアダプターを挙げる
- なぜステートレスなセッションスコープの FastAPI バックエンドが Agno 本番環境の推奨パスなのかを説明する
- 与えられたスタック（Python ファースト vs TypeScript ファースト）に対して Agno vs Mastra を選択する

## 問題

LangGraph、AutoGen、CrewAI はフレームワークヘビーです。「エージェントループだけが欲しい、高速で、私のランタイムで」を望むチームは Agno（Python）または Mastra（TypeScript）を選びます。どちらもフレームワーク所有のいくつかのプリミティブをトレードオフして、生の速度と周囲のスタックへのより密接なフィット感を実現します。

## コンセプト

### Agno

- Python ランタイム、以前は Phi-data
- 「グラフ、チェーン、複雑なパターンなし — ただの純粋な Python」
- ドキュメントからのパフォーマンス目標：約 2μs エージェント起動、1 エージェントあたり約 3.75 KiB メモリ、約 23 のモデルプロバイダー
- 本番環境パス：ステートレスなセッションスコープの FastAPI バックエンド。各リクエストが新しいエージェントを開始し、セッション状態は DB に存在します
- ネイティブなマルチモーダル（テキスト、画像、オーディオ、ビデオ、ファイル）と agentic RAG

速度目標は、1 秒あたり数千の短命なエージェント（チャット fan-in、評価パイプライン）を持つときに重要です。1 つのエージェントが 10 分間実行されるときはあまり重要ではありません。

### Mastra

- TypeScript、Vercel AI SDK 上に構築
- 3 つのプリミティブ：**Agents**、**Tools**（Zod 型）、**Workflows**
- 統一的なモデルルーター — 94 のプロバイダー全体で 3,300 以上のモデル（2026 年 3 月）
- 複合ストレージ：メモリ、ワークフロー、可観測性を異なるバックエンドに。大規模での可観測性には ClickHouse が推奨
- Apache 2.0、ソース利用可能なエンタープライズライセンスの下の `ee/` ディレクトリを含む
- Express、Hono、Fastify、Koa 用のサーバーアダプター；Next.js と Astro への第一級統合
- Mastra Studio（localhost:4111）をデバッグ用に同梱
- GitHub スター 22k 以上、1.0 時点で週次 npm ダウンロード 300k 以上（2026 年 1 月）

### ポジショニング

どちらも LangGraph になろうとしていません。競争は次の点にあります：

- **言語フィット。** Python ファーストチームには Agno；TypeScript ファーストには Mastra
- **ランタイム人間工学。** Agno = ほぼゼロオーバーヘッド；Mastra = Vercel エコシステムに統合
- **可観測性。** 両者は Langfuse/Phoenix/Opik（Lesson 24）と統合しますが、Mastra Studio はファーストパーティ

### どちらかを選ぶ場合

- **Agno** — Python バックエンド、多くの短命なエージェント、強いパフォーマンス要件、FastAPI ショップ
- **Mastra** — TypeScript バックエンド、Next.js / Vercel デプロイ、統一的なマルチプロバイダーモデルルーティング、Zod 型ツール
- **LangGraph**（Lesson 13）— 耐久的な状態と明示的なグラフ推論が生の速度より重要な場合
- **OpenAI / Claude Agent SDK** — プロバイダーの商品化された形状が必要な場合（Lessons 16–17）

### このパターンがうまくいかない場所

- **パフォーマンスのためのパフォーマンス。** 「2μs」が良く聞こえるからという理由で Agno を選ぶ場合、ワークロードは 1 つの遅いエージェント呼び出しでリクエストあたり。オーバーヘッドはボトルネックではありません
- **エコシステムロックイン。** Mastra の Vercel フレーバーの統合は Vercel ではプラス、他のところではマイナス
- **エンタープライズライセンスの混乱。** Mastra の `ee/` ディレクトリはソース利用可能ですが、Apache 2.0 ではありません。フォークする予定の場合はライセンスを読んでください

## ビルド

このレッスンは主に比較的であり、単一のコード成果物が両方のフレームワークに正当性をもたらすことはありません。`code/main.py` で side-by-side toy を参照してください：最小限の「エージェントを実行、出力をストリーム、セッションを永続化」フローは 2 回実装されています（1 回は Agno 形、1 回は Mastra 形）。

実行：

```
python3 code/main.py
```

2 つの構造的に異なるが機能的に同等のトレース。

## 使用方法

- **Agno** — 速度が必要で FastAPI 形状の Python バックエンド
- **Mastra** — 多くのプロバイダーとワークフロープリミティブを持つ TypeScript バックエンド
- 両者はファーストパーティの可観測性フックを同梱します。どちらも Langfuse と統合します

## 出荷

`outputs/skill-runtime-picker.md` はスタック、レイテンシ予算、運用形状に基づいて Agno、Mastra、LangGraph、またはプロバイダー SDK を選びます。

## 演習

1. Agno のドキュメントを読んでください。stdlib ReAct ループ（Lesson 01）を Agno に移植してください。何が消えて、何が残りましたか？
2. Mastra のドキュメントを読んでください。同じループを Mastra に移植してください。ツール型（Zod vs nothing）で何が変わりましたか？
3. ベンチマーク：スタック上でエージェント起動レイテンシを測定してください。Agno の 2μs はあなたのワークロードに重要ですか？
4. マイグレーションを設計してください：Python で CrewAI を実行している場合、Agno に移動すると何が壊れますか？
5. Mastra の `ee/` ライセンス条件を読んでください。オープンソースフォークにどのような制限が影響しますか？

## キーターム

| 用語 | 人々が言うこと | 実際の意味 |
|------|----------------|----------|
| Agno | 「Fast Python agents」 | ステートレスなセッションスコープのエージェントランタイム |
| Mastra | 「TypeScript agents on Vercel AI SDK」 | Agents + Tools + Workflows + Model Router |
| Unified Model Router | 「Multi-provider access」 | 94 のプロバイダー全体で 3,300 以上のモデルの単一クライアント |
| Composite storage | 「Multiple backends」 | メモリ/ワークフロー/可観測性の各々が異なるストアへ |
| Mastra Studio | 「Local debugger」 | エージェントを内省するための localhost:4111 UI |
| Source-available | 「Not OSS」 | ライセンスはソース読取を許可しますが商用利用を制限します |

## 参考文献

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — パフォーマンス目標、FastAPI 統合
- [Mastra docs](https://mastra.ai/docs) — プリミティブ、サーバーアダプター、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — ステートフルグラフ代替手段
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 統合で引用されている可観測性比較
