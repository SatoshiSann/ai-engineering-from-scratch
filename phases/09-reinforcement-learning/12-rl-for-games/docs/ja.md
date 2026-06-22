# RL for Games — AlphaZero、MuZero、LLM推論時代

> 1992：TD-Gammonはバックギャモンでヒューマンチャンピオンを倒した。2016：AlphaGoはLee Sedolを倒した。2017：AlphaZeroはチェス、将棋、囲碁を支配。2024：DeepSeek-R1はPPOをGRPOで置き換える同じレシピ、推論で機能証明。ゲームは このフェーズのすべてのブレークスルーを運転するベンチマーク。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 9 · 05 (DQN)、Phase 9 · 08 (PPO)、Phase 9 · 09 (RLHF)、Phase 9 · 10 (MARL)
**所要時間:** ~120分

## 問題

ゲームはRLが望むすべてを持つ。クリーン報酬（勝/敗）。無限エピソード（自己プレイリセット）。完全シミュレーション（ゲーム*は*シミュレーター）。離散または小さい連続アクション空間。対抗ロバストネスを強制するマルチエージェント構造。

そしてゲームはどうやってすべての主要RLブレークスルーがテストされた。TD-Gammon（バックギャモン、1992）。Atari-DQN (2013)。AlphaGo (2016)。AlphaZero (2017)。OpenAI Five (Dota 2、2019)。AlphaStar (StarCraft II、2019)。MuZero (学習モデル、2019)。AlphaTensor (行列乗算、2022)。AlphaDev (ソートアルゴリズム、2023)。DeepSeek-R1 (数学推論、2025) — ゲームRLテクニックはテキストで機能証明の最新。

このキャップストーン 単一統一レンズを通じて3つのランドマークアーキテクチャ — AlphaZero、MuZero、GRPO を調査：**自己プレイ + 探索 + ポリシー改善**。各々は前を一般化；特にGRPO はアクションとしてのトークンとマッチ信号としての数学検証を持つLLM推論に適用されたAlphaZeroのレシピ。

## コンセプト

![AlphaZero ↔ MuZero ↔ GRPO：同じループ、異なる環境](../assets/rl-games.svg)

**統一ループ。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017)。** Silver et al. ゲーム（チェス、将棋、囲碁）が与えられたら既知のルール：

- ポリシー値ネットワーク：1つのタワー `f_θ(s) → (p, v)`。`p` は法律手上の優先。`v` は期待ゲーム出来。
- モンテカルロ木探索 (MCTS)：各手に、可能な継続の木を拡張。`(p, v)` を優先 + ブートストラップとして使う。UCB (PUCT) でノード選択：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自己プレイ：ゲームをエージェント対エージェント再生。手 `t` で、MCTS訪問配布 `π_t` はポリシー訓練ターゲット。
- 損失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` はゲーム出力 (+1 / 0 / -1)。

ゼロヒューマン知識。ゼロハンドクラフト ヒューリスティック。チェス、将棋、囲碁をマスター後に数十百万自己プレイゲーム後した単一レシピ。

**MuZero (2019)。** Schrittwieser et al. 既知ルール要件を削除。

- 固定環境の代わりに、*潜在動力学モデル* を学習 `(h, g, f)`：
  - `h(s)`：観測をエンコード潜在状態。
  - `g(s_latent, a)`：次潜在状態 + 報酬を予測。
  - `f(s_latent)`：ポリシー優先 + 値を予測。
- MCTS は *学習潜在空間* で実行。同じ探索、同じ訓練ループ。
- Go、チェス、将棋 *と* Atariで機能 — 1つのアルゴリズム、ルール知識なし。

**確率MuZero (2022)。** 確率動力学と機会ノード追加；バックギャモンクラスゲーム拡張。

**Muesli、Gumbel MuZero (2022-2024)。** サンプル効率と決定論的探索での改善。

**GRPO (2024-2025)。** DeepSeek-R1レシピ。同じAlphaZero形状ループ、言語モデル推論に適用：

- 「ゲーム」：数学 / コード / 推論問題に答える。「勝」= 検証者（テストケースパス、数値答えマッチ） リターン 1。
- ポリシー：LLM。アクション：トークン。状態：プロンプト + 応答到元。
- クリティック (PPO スタイル V_φ)。代わりに、各プロンプト、ポリシーから `G` 完成サンプル。各々のために報酬計算。**グループ相対アドバンテージ** `A_i = (r_i - mean_r) / std_r` をREINFORCEスタイル更新用シグナルとして使う。
- 参照ポリシーへのKLペナルティはドリフト防止 (RLHFのような)。
- 完全損失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

報酬モデルなし、クリティックなし、MCSなし。グループ相対ベースラインはすべての3つを置き換え。推論ベンチマークでPPO-RLHFと一致またはそれを超える品質でコンピュートの一部。

**完全なR1レシピ。** DeepSeek-R1 (DeepSeek 2025) は1論文で2つのモデル：

- **R1-Zero。** DeepSeek-V3ベースモデルから開始。SFTなし。2つの報酬コンポーネントを持つGRPO直接適用：*精度報酬*（ルールベース — 最終答えを正確な数字にパースするか、コード単位テストパス）と*フォーマット報酬*（完成は鎖-of-think を `<think>…</think>` タグにラップするか）。数千ステップ上、平均応答長は~100から~10,000トークン成長し、数学ベンチマークスコアが近いo1-previewレベルにクライム。モデルは0からスクラッチで理由を学習。ダウンサイド：それのチェーン思考はしばしば読めない、言語混合、スタイル洗練不足。
- **R1。** R1-Zeroの読み易性問題をクリーンなフォーマットを持つ少数の長CoT例を集める4ステージパイプラインで修正：
  1. **冷スタート SFT。** 読み易長CoT例（数千）を集める。ベースモデルで微調整。読み易い開始点を与える。
  2. **推論志向GRPO。** 精度+フォーマット報酬プラス*言語一貫性* 報酬を持つGRPOを適用してコードスイッチング防止。
  3. **リジェクション サンプリング + SFT ラウンド 2。** RL チェックポイントから ~600K 推論軌跡サンプル、正確な最終答えと読み易い CoT を持つもののみ保つ、~200K 非推論SFT例（文、QA、自己認知）と合わせる。ベースを再度微調整。
  4. **フル スペクトラムGRPO。** 推論（ルールベース報酬）と一般アライメント（有益/害なしプリファレンス報酬）両方を対象とした別のRLラウンド。

結果はAIMEとMATH-500でo1をマッチ、オープンウェイトで、蒸留するのに十分に小さい。同じ論文もリリース 6蒸留された密モデル (Qwen-1.5B からLlama-70B) を R1の推論トレース上のSFTで — RLなし学生で。強いRL教師で蒸留は一貫して、学生スケール でゼロからのRLを打つ。

**推論で PPOではなくGRPOな理由。** DeepSeekMath論文（2024年2月）の3つの理由：(1) 訓練する値ネットワークなし、メモリを半減；(2) グループベースラインはスパース終了軌跡報酬を自然に処理、推論タスク生成；(3) プロンプトごと正規化は不可解なように異なった難しさの問題で比較可能なアドバンテージを作る、単一クリティックは PP ができない。

**探索フリー対探索ベース。** ゲームは枝分かれた：

- *完全情報長期ゲーム* (Go、チェス)：いまだ探索ベース。AlphaZero / MuZeroを支配。
- *LLM推論*：本番でまだMCSなし；完全ロールアウトで GRPO、推論コンピュート Best-of-N。プロセス報酬モデル (PRM) ステップレベル探索が追加されるヒント。

## 構築する

コード `code/main.py` では **GRPOをミニチュア** に実装 — 複数グループのサンプル持つバンディット。アルゴリズムはLLMで同じ；ポリシーと環境のみシンプル。*損失* と*グループ相対アドバンテージ* を教え、2025イノベーション。

### ステップ1：ちっぽけ検証者環境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

本物GRPOで検証者は単位テストを実行または数学等式をチェック。

### ステップ2：ポリシー：プロンプトごと K 答えトークン上のソフトマックス

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

LLMの最終層出力，プロンプト条件と等価。

### ステップ3：グループサンプリングとグループ相対アドバンテージ

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

グループ相対アドバンテージは2024年DeepSeekトリック。クリティック不要。「ベースライン」はグループ平均で、正規化はグループ標準偏差使う。

### ステップ4：REINFORCEベースライン (値フリー) と比較

同じセットアップ、同じコンピュート、プレーンREINFORCE。GRPOはより速く、安定して収束。

### ステップ5：エントロピーとKLを観測

RLHF と同じ診断：参照へのメディアンKL、ポリシーエントロピー、時間上の報酬。一度これらが安定したら、訓練は完了。

## 落とし穴

- **検証者ゲーミングを通した報酬ハッキング。** GRPOはRLHFのリスク受け継ぐ：検証者が間違ったまたは搾取可能なら、LLMはそれを見つけ。堅牢検証者（複数テストケース、形式証明）重要。
- **グループサイズが小さすぎる。** グループベースラインの分散は `1/√G` のようにゆく。`G = 4` 以下で、アドバンテージ信号はノイズ；標準選択は `G = 8` から `64`。
- **長さバイアス。** 異なる長さの LLM 完成は異なるログ確率を持つ。トークン数で正規化、またはシーケンスレベルログ確率使う、または max 長さに切り詰め。
- **純粋自己プレイサイクル。** AlphaZero スタイル訓練は一般サムゲーム上優位ループに立ち往生できる。多様対手プール（リーグプレイ、Lesson 10）で緩和。
- **探索ポリシー不一致。** AlphaZeroは探索出力マネをするようポリシーを訓練。ポリシーネットが探索の配布を表現するには小さすぎたら、訓練は失速。
- **コンピュートフロア。** MuZero / AlphaZero は大規模コンピュート必要。単一アブレーション はしばしば数百GPU時間。ミニデモ存在 (例えば、AlphaZero on Connect Four) 学習。
- **検証者カバレッジ。** バグのある解決策パスのユニットテスト。バグ強化。エッジケース抓むガイドライン設計。

## 使う

2026年のゲームRLランドスケープ、ドメイン：

| ドメイン | 支配的方法 |
|--------|-----------------|
| 2プレイヤーゼロサムボードゲーム（Go、チェス、将棋） | AlphaZero / MuZero / KataGo |
| 不完全情報カードゲーム（ポーカー） | CFR + 深層学習 (DeepStack、Libratus、Pluribus) |
| Atari / ピクセルゲーム | Muesli / MuZero / IMPALA-PPO |
| 大規模マルチプレイヤー戦略（Dota、StarCraft） | PPO + 自己プレイ + リーグ (OpenAI Five、AlphaStar) |
| LLM数学/コード推論 | GRPO (DeepSeek-R1、Qwen-RL、オープン複製) |
| LLMアライメント | DPO / RLHF-PPO (GRPOではなく；検証者は優先、検証可能ではなく) |
| ロボティクス | PPO + DR (ゲームRLではなく、同じポリシー勾配ツール使う) |
| 組み合わせ問題 | AlphaZero変種 (AlphaTensor、AlphaDev) |

*レシピ* — 自己プレイ、探索増強改善、ポリシー蒸留 — テキスト、ピクセル、物理制御を超えてスパン。GRPOはヤング最年少インスタンス；さらに来ている。

## 納入する

`outputs/skill-game-rl-designer.md` として保存：

```markdown
---
name: game-rl-designer
description: 与えられたドメイン用のゲームRLまたは推論RLパイプラインを設計 (AlphaZero / MuZero / GRPO)。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

ターゲット（完全情報ゲーム / 不完全情報 / Atari / LLM推論 / 組み合わせ）が与えられたら、出力：

1. 環境フィット。既知ルール？マルコフ？確率的？マルチエージェント？AlphaZero対MuZero対GRPOを通知。
2. 探索戦略。MCTS (学習優先付きPUCT)、Gumbelサンプリング、Best-of-N、またはなし。
3. 自己プレイプラン。対称自己プレイ / リーグ / オフラインデータ / 検証者生成。
4. ターゲット信号。ゲーム出力 / 検証者報酬 / プリファレンス / 学習モデル。ロバストネスプラン含める。
5. 診断。ベースライン対勝率、ELO曲線、検証者パス率、参照へのKL。

不完全情報ゲーム上でAlphaZeroを拒否 (CFRにルート)。信頼された検証者なしでGRPOを拒否。固定ベースラインシャットセット なしのゲームRLパイプラインを拒否（自己プレイELOは他の方法はキャリブレーション）。
```

## 演習

1. **簡単。** `code/main.py` のGRPOバンディットを実装。2プロンプト × 4答えトークン各で訓練。`G=8` で < 1,000更新で収束。
2. **中程度。** PPO（切り取られた）とバニラREINFORCE プラグイン。同じバンディットで GRPOへのサンプル効率と報酬分散を比較。
3. **難しい。** 長さ2「推論チェーン」に拡張：エージェント2トークンを出力し、検証者がペアに報酬。GRPOがどう2ステップシーケンスの信用配分を処理するか測定。（ヒント：計算グループアドバンテージ *フルシーケンス* ごと、両方トークン位置に伝播。）

## 重要用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| MCTS | "学習ネット付きツリー探索" | モンテカルロ木探索；学習 `(p, v)` 優先付きUCB1/PUCT選択。 |
| AlphaZero | "自己プレイ + MCTS" | MCSの訪問とゲーム出力にマッチするように訓練されたポリシー値ネット。 |
| MuZero | "学習モデルAlphaZero" | 同じループだが学習動力学を通した潜在空間。 |
| GRPO | "クリティックフリーPPO" | グループ相対ポリシー最適化；グループ平均ベースライン + KLを持つREINFORCE。 |
| PUCT | "AlphaZeroのUCB" | `Q + c · p · √N / (1 + N_a)` — 値推定と優先をバランス。 |
| 自己プレイ | "エージェント対過去自己" | ゼロサムに標準；対称訓練信号。 |
| リーグプレイ | "ポピュレーションベース自己プレイ" | 過去 + 現在 + エクスプロイター見本対手。 |
| 検証者報酬 | "検証可能RL" | 報酬は決定論的チェッカー（テスト パス、答えマッチ）から来。 |
| プロセス報酬 | "PRM" | 各推論ステップをスコア、最終答えではなく。 |

## 参考文献

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari、Go、chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath：Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — GRPOを導入しグループ相対ベースライン論文。
- [DeepSeek-AI (2025). DeepSeek-R1：Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — フル4ステージR1レシピ プラスR1-Zeroアブレーション。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + 深層学習スケール。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — それすべてを開始した論文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — カスタム報酬関数を持つGRPOを適用するための本番参照。
- [Qwen Team (2024). Qwen2.5-Math — GRPO複製](https://github.com/QwenLM/Qwen2.5-Math) — 複数スケール時R1レシピのオープン複製。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 自己プレイ、探索、「設計報酬」のテキストブックフレーミング R1がLLMスケール例示。
