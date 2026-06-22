# シャドウトラフィック、カナリア ロールアウト、および LLM のプログレッシブ デプロイメント

> LLM ロールアウトはソフトウェアデプロイメントの最難パーツを結合：ユニットテストなし、拡散障害モード、遅延シグナル。シーケンスは (1) シャドウ モード — candidate モデルに prod リクエストを複製、ログ、ユーザー影響ゼロで比較。明白な分布問題をキャッチするが品質保証ではない。(2) カナリア ロールアウト — プログレッシブ トラフィック シフト 10% → 25% → 50% → 75% → 100%、各ステップでゲート。レイテンシ パーセンタイル、コスト/リクエスト、エラー/拒否率、出力長分布、ユーザーフィードバック率をトラック。(3) A/B テスト、安定性確認後の異なる代替。非決定性は不可約 — GPU FP 非結合性 plus バッチサイズ分散のため同一入力での実行間で最大 15% 精度変動。コストは変数、定数ではない — 20% 優れたモデルは 3 倍のコスト/呼び出し。ロールバック速度は決定的：ロールバックが redeploy を必要とするなら遅すぎ。ポリシーが config/フラグに存在。モデルがレジストリにあり、pinned ダイジェスト。ロールバック = フラグ フリップ + しきい値の復帰 + 秒でモデル pin。

**タイプ:** Learn
**言語:** Python (stdlib, toy canary-progression simulator)
**前提条件:** Phase 17 · 13 (Observability)、Phase 17 · 21 (A/B Testing)
**所要時間:** 約 60 分

## 学習目標

- シャドウ モード（ゼロ影響比較）、カナリア（ライブトラフィック プログレッシブ）、A/B（安定確認比較）を区別。
- 5 つの LLM 固有カナリア メトリクスを列挙（レイテンシ、コスト/リクエスト、エラー/拒否、出力長分布、ユーザーフィードバック）。
- LLM 非決定性（最大 15%）がロールアウトで「安定」の意味を変える理由を説明。
- 秒を要する（ポリシーフリップ）ロールバック パス、時間ではない（redeploy）を設計。

## 問題

新しいモデルを出荷。Offline evals が 3% 精度改善を示す。Production で有効化。24 時間内にコストが 40% アップ、ユーザー thumbs-down が 8% アップ、3 つのカスタマーチケットが「weird answers」を報告。ロールバック。Redeploy は 3 時間を要す。週末が台無し。

すべてのピースは回避可能だった。シャドウ モード 40% コストスパイクをキャッチした、ユーザーが見る前。カナリア は 10% で thumbs-down が移動するとき停止した。ポリシーフラグ ロールバックは 30 秒を要した。ディシプリンが「offline evals 良く見える」と「ユーザーが幸せ」の間のギャップを埋める。

## コンセプト

### シャドウ モード

Candidate は production と同じリクエストを受け取る。出力はログされ、ユーザーには返却されない。ゼロ ユーザー影響。ログ：

- 出力コンテンツ（production と比較差分）。
- トークン数（コスト delta）。
- レイテンシ。
- 拒否とエラー。

キャッチ：コスト吹上、長さ 回帰、明白な拒否変更、ハード エラー。キャッチしない：品質 delta ユーザーが認識。シャドウは smoke テスト、品質テストではない。

### カナリア ロールアウト

プログレッシブ トラフィック シフト、ゲート付き。典型的プログレッション：1% → 10% → 25% → 50% → 75% → 100%。各ステップで 5 メトリクスをゲート：

1. **レイテンシ パーセンタイル** — P50、P95、P99。侵害：カナリア P99 > 1.5x ベースライン。
2. **リクエストあたりコスト** — blended $。侵害：>20% ベースライン上。
3. **エラー / 拒否率** — 5xx plus 明白な拒否。侵害：2x ベースライン。
4. **出力長分布** — mean + P99。侵害：分布 shift。
5. **ユーザーフィードバック率** — thumbs-down / チケット filing。侵害：1.5x ベースライン。

### 非決定性は新しい分散

同一入力は非同一出力を作成。理由：

- GPU FP 非結合性（浮動小数点 reduction order はバッチにより異なる）。
- バッチサイズ分散（同じ prompt バッチ 128 vs バッチ 16）。
- サンプリング（temperature > 0）。

測定：同一 eval セット上での実行間で最大 15% 精度変動。「安定」ロールアウト では、メトリクスがベースラインと同一ではなく期待分散内。ノイズフロア上のゲート設定。

### コストは変数

20% 優れたモデルは 3 倍の呼び出しあたりコスト。コスト/リクエストは 5 つゲートの 1 つ。「優れた」モデル出荷が unit エコノミクスを破棄する、ロールバック ケース。

### ロールバックは武器

- ポリシー フラグ（feature フラグ システム）：config でパーセンテージ フリップ。秒を要す。
- モデル pinning（registry ダイジェスト）：pinned モデルは自動アップグレードしない。
- ロールバック = フラグ復帰 + pinned ダイジェスト を previous に設定。秒、時間ではない。

スタックが redeploy を ロールバックに要求するなら、rolling 前に修正。

### ツーリング

**Argo Rollouts** / **Flagger** — Kubernetes プログレッシブ デリバリ コントローラー。Istio/Linkerd 重み付けルーティングと統合。

**Istio 重み付けルーティング** — service-mesh-level トラフィック split。

**KServe / Seldon Core** — ビルトイン カナリア を持つモデル サービング。

**Feature フラグ** — LaunchDarkly、Flagsmith、Unleash。ポリシーレベル フリップ、redeploy なし。

### メトリクス cadence

カナリア ゲートはトラフィック量に応じて 5-15 分ごとにチェック。1% トラフィック 10 req/min は 50-150 データポイント/ウィンドウ — レイテンシに十分だが noisy ユーザーフィードバック。10% は ~10x より。プログレッション各ステップで十分なサンプルを蓄積する長さだけ一時停止すべき。

### A/B ステップはオプション

新しいモデルが異なる（異なる behavior、異なる cost curve、異なる tone）なら、カナリア通過後 50% で A/B テスト。単なる改善版なら、カナリア ゲート通過時 100% にスキップ。

### 記憶すべき数字

- カナリア プログレッション：1% → 10% → 25% → 50% → 75% → 100%。
- 非決定性上限：同一入力での実行間の最大 15% 分散。
- 5 つカナリア メトリクス：レイテンシ、コスト、エラー/拒否、出力長、ユーザーフィードバック。
- コスト ゲート：>20% ベースライン上は侵害。
- ロールバック：秒、時間ではない。

## 使用方法

`code/main.py` はカナリア ロールアウトを injected 回帰でシミュレート。ロールアウトが halt するステージと trigger ゲートを報告。

## 配送

このレッスンは `outputs/skill-rollout-runbook.md` を作成。candidate モデル、ベースライン、risk tolerance を指定して shadow→canary→100% プランを設計。

## 演習

1. `code/main.py` 実行。25% コスト 回帰を注入。どのステージでカナリア halt するか？
2. 新しいモデルが offline で 3% 精度改善だが コスト/リクエスト +18%。これは ship？ポリシーに依存 — 両方パスを書く。
3. 60 秒以下エンドツーエンドを要するロールバックを設計。リスト必要インフラ。
4. 非決定性は eval で ±7% を示す。false-alarm しないカナリア ゲート設定。どの乗数を使う？
5. シャドウ モード が 40% コスト スパイクを canary 前にキャッチ。シャドウで発火するアラート ルールを書く。

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| Shadow mode | "duplicate to new" | ゼロ影響 candidate への送信ログ用 |
| Canary | "プログレッシブ トラフィック" | ゲート付き段階的ユーザー露出ロールアウト |
| Gates | "ロールアウト チェック" | メトリクス しきい値がプログレッション ブロック |
| Non-determinism | "LLM 分散" | 不可約な実行間の差分 |
| Policy flag | "フラグ フリップ ロールバック" | Config-level ロールバック、秒 time ではなく hours |
| Model pin | "registry ダイジェスト" | モデル バージョンへの不変参照 |
| Argo Rollouts | "K8s プログレッシブ" | Kubernetes ネイティブ canary/ロールバック コントローラー |
| KServe | "inference K8s" | canary プリミティブ付きモデル サービング |
| Istio weighted | "mesh split" | Service-mesh トラフィック splitter |

## 参考文献

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
