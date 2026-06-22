# Direct Preference Optimization ファミリー

> Rafailov et al.（2023）は RLHF の最適解が優先度データに関して閉形式を持つので明示的報酬モデルをスキップしポリシーを直接最適化できることを示した。そのインサイトはファミリーを生成 — IPO、KTO、SimPO、ORPO、BPO — 各々が DPO の失敗モードを修正。2026年、直接アライメント アルゴリズムは PPO より多くのフロンティア事後学習実行を配送。でも Lesson 2 からの過最適化曲線は依然適用：DAA は Goodhart をエスケープしない、それがかむ場所をシフトするだけ。

**タイプ:** Learn
**言語:** Python（stdlib、6 バリアント優先度損失比較器）
**前提条件:** Phase 18 · 01 (InstructGPT)、Phase 18 · 02 (Reward hacking)、Phase 10 · 08 (DPO basics)
**所要時間:** ~75分

## 学習目標

- RLHF-KL 最適値から DPO 閉形式を導出。
- IPO、KTO、SimPO、ORPO、BPO のそれぞれが DPO で修正する失敗モードを述べる。
- 「暗黙報酬ギャップ」を「優先度強度」から区別し、なぜ IPO の同一映射が重要かを説明。
- ノーズ明示 RM があるが Rafailov et al.（NeurIPS 2024）DAA が過最適化する理由を説明。

## 問題

RLHF 目的（レッスン 1）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

既知最適解を持つ：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

報酬は最適ポリシーと参照の比率で暗黙的に定義：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

これをBradley-Terry 優先度尤度に代わるし、パーティション関数 `Z(x)` は `x` のみに依存するため崩壊。残りはポリシー パラメータのみの損失 — 報酬モデル不要。これが DPO。

皺：導出は最適値が到達可能、優先度データが分布内、参照ポリシーが真のモード アンカーであることを想定。これらのどれも正確に成立しない。ファミリー メンバーは異なる違反想定を修正。

## コンセプト

### DPO（Rafailov et al.、2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

間違える可能性：

- 暗黙報酬ギャップ `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` は無限。小さい優先度は恣意的に大きなギャップを生産。
- 損失は選択と拒否ログ確率を反対方向にドライブ。選択的な絶対ログ確率を落とせる拒否がより速く落ちる限り。これは劣化選択レスポンス現象。
- 分布外優先度（希少希少ペアvs希少希少ペア）は恣意的暗黙報酬を生産。

### IPO（Azar et al.、2024）

Identity Preference Optimization がログシグモイドをアイデンティティ マッピングに優先度確率上で置換。損失は限定ターゲットのスクエア エラーになる：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

マージンは `1/(2 beta)` で限定。優先度強度と暗黙報酬ギャップは比例。爆発なし。

### KTO（Ethayarajh et al.、2024）

Kahneman-Tversky Optimization はペアワイズ構造を全部ドロップ。単一ラベル出力と「望ましい」または「望ましくない」バイナリ シグナル指定で、プロスペクト理論ユーティリティにマップ：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

利得と損失で異なる重み（損失回避）。利益：ペアでないデータを使用できる、遥かにより豊富。

### SimPO（Meng et al.、2024）

Simple Preference Optimization はトレーニング シグナルを生成と合わせて参照ポリシーを完全削除して長さによって正規化ログ尤度：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

マージン `gamma` で安定化。長さ正規化は DPO の長さバイアス失敗モードを利用する動機を削除（より長い `y_w` は構造で大きいログ確率ギャップを与える）。

### ORPO（Hong et al.、2024）

Odds-Ratio Preference Optimization は標準 SFT 負の対数尤度に優先度ターム を追加：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

参照ポリシーなし — SFT タームは正則化子。ベース モデルから配列モデルへ単一ステージでトレーニング。別の SFT チェックポイントなし。

### BPO（ICLR 2026 提案、OpenReview id=b97EwMUWu7）

劣化選択レスポンス問題を識別：DPO はランキング `y_w > y_l` を保持するが `y_w` の絶対ログ確率はドロップできる。BPO は選択レスポンス上のダウン移動にペナルティして単一行修正を追加。数学推敲で Llama-3.1-8B-Instruct 上の DPO 以上 +10.1% 精度をレポート。

### ユニバーサル結果：DAA 依然過最適化

Rafailov et al.「Direct Alignment Algorithms での報酬モデル過最適化のスケーリング則」（NeurIPS 2024）多数データセット全体と KL 予算の DPO、IPO、SLiC でポリシーをトレーニング。金報酬-対-KL 曲線は同じ Gao et al. ピーク-崩壊形を持つ。暗黙報酬クエリはトレーニング中分布外サンプル、KL 正則化はこれを安定化しない。

DAA は Goodhart をエスケープしない。「報酬モデル過最適化」から「参照ポリシー比率過最適化」へ咬く表面をシフト。ユニバーサル修正 — より良いデータ、アンサンブル、早期停止 — 両方に適用。

### 2026年のそれらの間での選択

- 大きなペアワイズ優先度データを持つ場合：保守的ベータで DPO、長さバイアスが明らかな場合 SimPO。
- ペアでないバイナリ フィードバックを持つ場合：KTO。
- ベース モデルから単一ステージ パイプラインをしたい場合：ORPO。
- DPO ログで劣化選択ログ確率を見る場合：BPO。
- 優先度強度は広く変わりDPO は飽和する場合：IPO。

すべての実験室はタスク毎に勝者をピックして電池で 5 つすべてを実行。最適がタスク全体同じである理由はない。数学推敲と安全は同じでない。

## これを使用

`code/main.py` は 6 つの損失（DPO、IPO、KTO、SimPO、ORPO、BPO）をオモチャ優先度データセットで比較。真の優先度強度はペアにより変わる場所。各損失は同じ 500 ペア サンプルでソフトマックス ポリシーに対して最適化。最終的勝率、選択-ログ確率ドリフト、メソッド毎の暗黙報酬スプレッドをプロット。

## 配送

このレッスンは `outputs/skill-preference-loss-selector.md` を生成。データセット統計（ペアワイズvs ペアでない、変数 vs ユニフォーム優先度強度、長さ分布）と対象（単一ステージまたは SFT-次-優先度）を指定で、優先度損失を推奨してそれが保護する失敗モードをレポート。

## 演習

1. `code/main.py` を実行。DPO と BPO の最終選択-ログ確率ドロップをレポート。BPO はより高い選択絶対確率を保持すべき — 確認。

2. すべてのペアが等しい強度を持つため優先度データを修正。6 つのメソッドのどれが最もロバスト？どれが劣化？ここで IPO の利点を説明。

3. 拒否されたレスポンスを平均で選択より 2 倍長くする。他は変更なし、数値で DPO の長さ利用と SimPO の修正を表示。

4. Rafailov et al.（NeurIPS 2024）は DAA が過最適化する主張。単一ポイント バージョンを再現：選択-マイナス-拒否 KL ダイバージェンスをプロットし大きなベータで DPO での過最適化を観察。

5. BPO 論文抄録を読む（OpenReview b97EwMUWu7）。BPO が DPO に追加する 1 行修正を書き落す。`code/main.py` での実装に対して確認。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|--------|
| DPO | 「報酬モデルなし RLHF」 | 閉形 RLHF 最適値から導出する損失、ポリシー パラメータのみ |
| 暗黙報酬 | 「ログ比率」 | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO 暗黙報酬 |
| IPO | 「限定 DPO」 | ログシグモイドを同一で置換、暗黙報酬ギャップ限定 1/(2 beta) で |
| KTO | 「ペアでない DPO」 | 単一ラベルでの市場理論ユーティリティ。損失回避 |
| SimPO | 「参照フリー DPO」 | 長さ正規化ログ尤度 + マージン、参照ポリシーなし |
| ORPO | 「1 ステージ DPO」 | NLL + オッズ比優先度ターム、ベース モデルから 1 回で トレーニング |
| BPO | 「選択保存 DPO」 | DPO プラス選択レスポンス絶対ログ確率減少へのペナルティ |
| 劣化選択 | 「選択がダウン」 | DPO が拒否が速くより速く落ちる限り選択ログ確率を減少 |
| DAA | 「直接アライメント アルゴリズム」 | 明示RM をスキップする任意の優先度損失メソッド |

## 参考文献

- [Rafailov et al. — Direct Preference Optimization（NeurIPS 2023、arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences（AISTATS 2024、arXiv:2310.12036）](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng、Xia、Chen — SimPO（NeurIPS 2024、arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong、Lee、Thorne — ORPO（EMNLP 2024、arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization（ICLR 2026 OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs（NeurIPS 2024、arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
