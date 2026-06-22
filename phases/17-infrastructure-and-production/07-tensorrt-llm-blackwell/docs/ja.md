# Blackwell での TensorRT-LLM（FP8 と NVFP4 付き）

> TensorRT-LLM は NVIDIA のみですが Blackwell で勝ちます。Dynamo オーケストレーション付き GB200 NVL72 上で、SemiAnalysis InferenceX は 2026 年 Q1-Q2 で 120B モデル で 1M トークン当たり $0.012 を計測、H100 + vLLM での $0.09/M に対し — 7x 経済ギャップ。スタックは 3 つの浮動小数点レジームが複合：FP8 は KV キャッシュとアテンション カーネルで重要（動的範囲の必要性から）。NVFP4（4-ビット マイクロスケーリング）は重みと活性化を処理。マルチトークン予測（MTP）と分離プリフィル/デコードは別 2-3x を上。Day-0 モデルサポートは FP4 重みを直接ロード、ポスト訓練変換なし。2026 年エンジニアリングチーム用のキャッチ：TensorRT-LLM はクローズド NVIDIA スタック、採用はスループット用に移植性をトレード。モデルとハードウェアミックスで数学を実行してコミットする前に。

**タイプ:** 学習
**言語:** Python（stdlib、簡単な FP8/NVFP4 メモリとコスト計算機）
**前提条件:** フェーズ 17 · 04（vLLM サービング インターナル）、フェーズ 10 · 13（量子化）
**所要時間:** 約 75 分

## 学習目標

- FP8 が重みが NVFP4 でも KV キャッシュとアテンション で重要である理由を説明。
- 3 つのスタック間で最前線モデルの HBM フットプリントを計算：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM、セービングの出所を理由。
- Blackwell 特定特徴 TRT-LLM が活用すること に名前を付ける（day-0 FP4、MTP、分離サービング、全対全プリミティブ）。
- TensorRT-LLM の NVIDIA ロックが Hopper 上の vLLM との 7x コストギャップの価値があるかどうかを決定。

## 問題

推論経済学の最前線は 2026 年 「1 ドルあたり何トークンか」。答えはスタックされた 4 つの選択肢に依存：ハードウェア世代（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、サービングエンジン（vLLM vs SGLang vs TRT-LLM）、オーケストレーション（平素 vs 分離 vs Dynamo）。

Hopper 上の vLLM では、120B MoE は 1M トークンあたり約 $0.09 で実行。Blackwell 上 TRT-LLM + Dynamo では、同じモデルが約 $0.012 — 7x 安い。ギャップの一部はハードウェア（Blackwell は Hopper より 11-15x LLM スループット/GPU）。一部はスタック：FP4 重み、MTP ドラフト、分離プリフィル/デコード、MoE 専門家通信用の NVLink 5 全対全。

NVIDIA のスタック外でこれを複製することはできません。それがトレードオフ — 経済学のための移植性。どのスタック選択がギャップの何シェアを与えるかを理解することがこのレッスンの要点。

## コンセプト

### FP8 はなぜまだ KV キャッシュのフロアである

2026 年の一般的な間違い：NVFP4 がどこでも適用されるという仮定。そうではない。KV キャッシュは FP8（8-ビット浮動小数点）を必要（アテンション キーと値を幅広い動的範囲にまたがって保存）。KV を FP4 に量子化すると、壊滅的な精度損失 — 分布の尾はドロップ、アテンション スコアは崩壊。FP8 の指数ビットは KV キャッシュに必要な範囲を与える。

NVFP4（2025-2026）は重みと活性化に適用。マイクロスケーリング：重みの各ブロックは独自のスケールファクタを持つため、小さいブロックは テンソルごとスケール損失なしで異なる動的範囲にまたがることができる。活性化用、FP4 はレイヤー内で活性化が小さい範囲のため保持。

典型的な Blackwell 設定：

- 重み：NVFP4（4-ビット マイクロスケーリング）。
- 活性化：NVFP4。
- KV キャッシュ：FP8。
- アテンション アキュムレータ：FP32（softmax 安定性）。

### Blackwell 特定プリミティブ TensorRT-LLM が使用

- **Day-0 FP4 重み**：モデルプロバイダーが FP4 で直接重みを配布。TensorRT-LLM はポスト訓練変換なしでロード。AWQ / GPTQ ステップなし FP4。
- **マルチトークン予測（MTP）**：EAGLE と同じ考え（フェーズ 17 · 05）ですが TensorRT-LLM ビルドに統合。
- **分離サービング**：プリフィルと別 GPU プール でデコード、KV キャッシュ NVLink または InfiniBand 上で転送。Dynamo と同じ考え（フェーズ 17 · 20）。
- **全対全通信プリミティブ**：NVLink 5 は MoE 専門家通信レイテンシを Hopper より 3x 削減。TensorRT-LLM の MoE カーネルはこれ用にチューン。
- **NVFP4 + MXFP8 マイクロスケーリング**：Blackwell Tensor Core でハードウェア加速スケールファクタ処理。

### 覚えるべき数字

- HGX B200 at $0.02/M トークン GPT-OSS-120B 経由 TensorRT-LLM。
- GB200 NVL72 at $0.012/M トークン Dynamo 経由（TensorRT-LLM オーケストレーション）。
- H100 + vLLM ≈ $0.09/M トークン同等ワークロードで。
- 3 ヶ月の TensorRT-LLM 更新での 2.8x スループット利得（2026）。
- 11-15x LLM スループット/GPU、Blackwell vs Hopper。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell が提出されたすべてのタスクで優位。

### FP4 が実際に品質に何をコスト

NVFP4 は積極的。推論重いワークロード（連鎖思考、数学、長コンテキスト付きコード生成）、FP4 重みは目に見える劣化。ブロック当たり較正は軽減しますが排除しない。推論モデルをシップするチームはしばしば、FP8 重み + FP4 活性化を妥協として使用、または H200 を FP8 全体で使用に固持。

ルール：NVFP4 重みへのコミットの前に、自分の評価セット上で常にタスク品質を検証。

### これが NVIDIA ロック判定である理由

TensorRT-LLM は C++ + CUDA + クローズド ソースカーネル。モデルは特定 GPU SKU 用にコンパイル される必要。AMD なし、Intel なし、ARM なし。インフラ戦略がマルチベンダーなら、TensorRT-LLM は TRT-LLM 配信ティア用に非スターター — vLLM から混合ハードウェアで配信できます。NVIDIA のみなら、7x ギャップはロック支払い。

### 2026 年実用レシピ

年 $100M+ 推論請求の場合、Hopper + vLLM で実行は 7-10x を左にしているままにします。Blackwell + TensorRT-LLM + Dynamo へコスト支配ワークロードを移行。モデル反復速度用に H100 + vLLM で実験ティアを保つ。本番前に各 NVFP4 変換モデル上で品質を検証。

### 分離ボーナス

TensorRT-LLM の分離サービング（分離プリフィル/デコード プール）はフェーズ 17 · 20 で深く覆われます。Blackwell 上で、乗数スタック：FP4 重み × MTP スピードアップ × 分離配置 × キャッシュ対応ルーティング。7x 数字はこの完全スタックを仮定。

## 使用

`code/main.py` は 3 つのスタック間で HBM フットプリント、デコード スループット（メモリ境界レジーム）、$/M トークンを計算：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TensorRT-LLM。複合効果を見るために、ギャップの各変更がどのシェアを貢献するかを実行。

## 配布

このレッスンは `outputs/skill-trtllm-blackwell-advisor.md` を生成。ワークロード、モデルサイズ、年トークンボリュームを考えると、Blackwell + TensorRT-LLM スタックが NVIDIA ロックの価値があるかどうかを決定。

## 演習

1. `code/main.py` を実行。30% アクティブパラメータの 120B MoE で、メモリ帯域幅限定デコード スループット を計算（H100 BF16、H100 FP8、B200 NVFP4/FP8）。最大ジャンプはどこから？
2. 顧客は H100 + vLLM で年 $2M を支出。2026 年で $100M 支出の場合、TensorRT-LLM への移行を償却するために必要な Blackwell GPU の損益分岐点の数は何か？7x ギャップを考えると。
3. NVFP4 重みの変換後に MATH で 3 ポイント精度低下を見る。2 つの復旧パス に名前を付け：1 つは品質ファースト（FP8 重み保つ）、1 つはコストファースト（ドメイン内データで較正）。
4. MLPerf v6.0 推論結果を読む。Blackwell-over-Hopper ギャップが最小のタスクは何で、なぜ？
5. NVFP4 重み + FP8 KV キャッシュ at 128k コンテキスト で 405B モデル用の HBM を計算。単一 GB200 NVL72 ノードに適合するか？

## 重要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-------------|----------|
| FP8 | 「8-ビット浮動小数点」 | 8-ビット浮動小数点。動的範囲用 KV キャッシュ、アテンション で使用 |
| NVFP4 | 「4-ビット マイクロ」 | NVIDIA の 4-ビット マイクロスケーリング FP フォーマット。Blackwell で重み・活性化 |
| MXFP8 | 「MX 8」 | マイクロスケーリング FP8 変種。Blackwell Tensor Core でハードウェア加速 |
| Day-0 FP4 | 「FP4 重み配布」 | モデルプロバイダー がポスト訓練変換ステップなしで直接 FP4 で重みをリリース |
| MTP | 「マルチトークン予測」 | TensorRT-LLM 統合推測デコード ドラフト（フェーズ 17 · 05） |
| 分離サービング | 「分離プリフィル/デコード」 | 分離 GPU プール 上でプリフィル/デコード、NVLink/IB 上で転送 KV キャッシュ |
| 全対全 | 「MoE 専門家通信」 | トークン ルーティングパターン 専門家 GPU へ。NVLink 5 cuts 3x |
| InferenceX | 「SemiAnalysis inference bench」 | 2026 業界許容コスト・トークン・ベンチマーク |

## 参考文献

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 結果。
- [NVIDIA — Blackwell での MoE 推論](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全対全、MoE カーネル。
- [TensorRT-LLM 概要](https://nvidia.github.io/TensorRT-LLM/overview.html) — 公式エンジン ドキュメント。
- [NVIDIA — Dynamo の紹介](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — 分離オーケストレーション TensorRT-LLM 上。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — Blackwell 数字を公開するベンチマーク スイート。
