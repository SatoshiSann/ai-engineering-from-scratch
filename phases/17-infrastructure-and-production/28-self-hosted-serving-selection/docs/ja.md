# セルフホスト型サービング選択 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026年のセルフホスト推論は 4 つのエンジンが支配。ハードウェア、スケール、エコシステムに基づいて選択。**llama.cpp** は CPU 上で最速 — 最広いモデルサポート、量子化とスレッドの完全制御。**Ollama** は開発用ノートパソコンのワンコマンドインストール、~15-30% llama.cpp より遅い（Go + CGo + HTTP シリアル化）、本番のような負荷下で 3 倍のスループット ギャップ。**TGI は 2025年12月11日にメンテナンスモードに移行** — バグ修正のみ、~10% 低いスループット（生ではvLLM より、歴史的には最高の観測性と HF エコシステム統合）。そのメンテナンス ステータスはリスクの高い長期ベット — SGLang または vLLM は新規プロジェクトのより安全なデフォルト。**vLLM** は一般用途本番環境デフォルト — v0.15.1（2026年2月）は PyTorch 2.10、RTX Blackwell SM120、H200 最適化を追加。**SGLang** はエージェント マルチターン / プリフィックス ヘビーの専門家 — 本番環境で 400,000+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。ハードウェア制約：CPU のみ → llama.cpp のみ。AMD / 非 NVIDIA → vLLM のみ（TRT-LLM は NVIDIA ロック）。2026年パイプライン パターン：開発 = Ollama、ステージング = llama.cpp、本番 = vLLM または SGLang。全体で同じ GGUF/HF ウェイト。

**タイプ:** Learn
**言語:** Python（stdlib、エンジン決定木ウォーカー）
**前提条件:** Phase 17 のエンジンカバーリングすべてのレッスン（04、06、07、09、18）
**所要時間:** ~45分

## 学習目標

- ハードウェア（CPU / AMD / NVIDIA Hopper / Blackwell）、スケール（1 ユーザー / 100 / 10,000）、ワークロード（一般チャット / エージェント / 長コンテキスト）を指定してエンジンを選択。
- 2026年の TGI メンテナンスモード ステータス（2025年12月11日）と新規プロジェクトを vLLM または SGLang にバイアスさせる理由を名前で述べる。
- 同じ GGUF または HF ウェイトを全体で使用する開発/ステージング/本番パイプラインを説明。
- なぜ「CPU のみ」が llama.cpp を強制し、「AMD」が TRT-LLM を除外するかを説明。

## 問題

チームが新規セルフホスト型 LLM プロジェクトを開始。1 人のエンジニアが Ollama を言い、別が vLLM を言い、3 番目が「TGI はそのままで機能しない？」と言う。3 つすべてが異なるコンテキストで正しい。どれもすべてが正しくない。

2026年の選択ツリーが問題：ハードウェア最初、スケール 2 番目、ワークロード 3 番目。そして 1 つの特定の 2025年イベント — TGI が 2025年12月11日にメンテナンスモードに移行 — 新規プロジェクトのデフォルトを変更。

## コンセプト

### 5 つのエンジン

| エンジン | 最適 | メモ |
|--------|----------|-------|
| **llama.cpp** | CPU / エッジ / 最小デプ / 最広モデルサポート | CPU 上で最速、完全な制御 |
| **Ollama** | 開発ノートパソコン、単一ユーザー、ワンコマンドインストール | llama.cpp より 15-30% 遅い、本番のような負荷下で 3 倍ギャップ |
| **TGI** | HF エコシステム、規制業界 | **メンテナンスモード 2025年12月11日** |
| **vLLM** | 一般用途本番、100+ ユーザー | 広い本番デフォルト、v0.15.1 2026年2月 |
| **SGLang** | エージェント マルチターン、プリフィックス ヘビー ワークロード | 本番環境で 400,000+ GPU |

### ハードウェアファースト決定

**CPU のみ** → llama.cpp。Ollama も機能するが遅い。他のエンジンは CPU 上で競争的ではない。

**AMD GPU** → vLLM（AMD ROCm サポート）。SGLang も機能。TRT-LLM は NVIDIA ロック、つまり外。

**NVIDIA Hopper（H100 / H200）** → vLLM または SGLang または TRT-LLM。3 つすべてトップティア。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM がスループット リーダー（Phase 17 · 07）。vLLM と SGLang が近い。

**Apple Silicon（M シリーズ）** → llama.cpp（Metal）。Ollama がこれをラップ。

### スケール セカンド決定

**1 ユーザー / ローカル開発** → Ollama。1 コマンド、最初トークン秒で。

**10-100 ユーザー / 小さいチーム** → vLLM 単一 GPU。

**100-10k ユーザー / 本番** → vLLM 本番スタック（Phase 17 · 18）または SGLang。

**10k+ ユーザー / エンタープライズ** → vLLM 本番スタック + 分散化（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### ワークロード サード決定

**一般チャット / Q&A** → vLLM がデフォルトで勝つ。

**エージェント マルチターン（ツール、計画、メモリ）** → SGLang の RadixAttention（Phase 17 · 06）が支配。

**プリフィックス再利用ヘビーの RAG** → SGLang。

**コード生成** → vLLM 良、SGLang はキャッシュで少し良。

**ロングコンテキスト（128K+）** → vLLM + チャンク プリフィル、SGLang + 階層化 KV。

### TGI メンテナンス トラップ

Hugging Face TGI は 2025年12月11日にメンテナンスモード移行 — 以降はバグ修正のみ。歴史的：トップティア観測性、ベストインクラス HF エコシステム統合（モデルカード、安全ツール）、スループット上は少し vLLM より背後。

2026年の新規プロジェクト：TGI からデフォルト離脱。既存 TGI デプロイメント継続可能だが最終的に移行すべき。SGLang と vLLM がより安全なデフォルト。

### パイプライン パターン

開発（Ollama）→ ステージング（llama.cpp）→ 本番（vLLM）。全体で同じ GGUF または HF ウェイト。エンジニアはノートパソコンで素早く反復、ステージングが本番量子化をミラー、本番がサービング対象。

### Ollama 注意事項

Ollama は開発に優れている。共有本番には優れていない：Go HTTP シリアル化が overhead を追加、同時実行管理は vLLM より単純、OpenTelemetry サポートが遅延。Ollama が輝く場所で使用 — 1 ユーザー、1 コマンド — 共有に対して vLLM に切り替え。

### セルフホスト vs 管理は別の決定

Phase 17 · 01（管理ハイパースケーラー）、· 02（推論プラットフォーム）が管理をカバー。このレッスンはセルフホストをすでに決定したと仮定。セルフホストする理由：データ レジデンシー、カスタム ファインチューン、総保有コスト スケール、ドメイン モデルがホスト上で利用不可。

### 覚えるべき数字

- TGI メンテナンスモード：2025年12月11日。
- vLLM v0.15.1：2026年2月、PyTorch 2.10、Blackwell SM120 サポート。
- SGLang 本番フットプリント：400,000+ GPU。
- Ollama スループット ギャップ vs llama.cpp：15-30% 遅い、本番のような負荷下で 3 倍。

## これを使用

`code/main.py` は決定木ウォーカー：ハードウェア + スケール + ワークロード指定で、エンジンを選択し理由を説明。

## 配送

このレッスンは `outputs/skill-engine-picker.md` を生成。制約指定で、エンジンを選択してマイグレーション計画を書く。

## 演習

1. `code/main.py` をハードウェア / スケール / ワークロードで実行。出力が直感にマッチ？
2. インフラは 12 H100s および 8 MI300X AMD。どのエンジン？TRT-LLM がなぜテーブルから外？
3. チームが 2026年で TGI を使用したい理由「知っているのは」。マイグレーション ケースを議論。
4. Ollama 開発から vLLM 本番：量子化、設定、観測性で何が変わる？
5. P99 プリフィックス長 8K で高再利用のテナント間の RAG 製品。エンジンを選択して Phase 17 · 11 + 18 でスタック。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|----------------|--------|
| llama.cpp | 「CPU 1」 | 最広モデル サポート、CPU 上で最速 |
| Ollama | 「ノートパソコン 1」 | ワンコマンド インストール、開発グレード スループット |
| TGI | 「HF のサービング」 | 2025年12月以降メンテナンスモード |
| vLLM | 「デフォルト」 | 広い本番ベースライン 2026年 |
| SGLang | 「エージェント 1」 | プリフィックス ヘビー、RadixAttention |
| TRT-LLM | 「NVIDIA ロック」 | Blackwell スループット リーダー、NVIDIA のみ |
| GGUF | 「llama.cpp フォーマット」 | バンドルされた K クォント バリアント |
| 本番スタック | 「vLLM K8s」 | Phase 17 · 18 参考デプロイメント |
| パイプライン パターン | 「開発→ステージ→本番」 | 同じウェイト上の Ollama → llama.cpp → vLLM |

## 参考文献

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI メンテナンスアナウンス](https://github.com/huggingface/text-generation-inference) — リリース ノート。
- [vLLM v0.15.1 リリース ノート](https://github.com/vllm-project/vllm/releases)
