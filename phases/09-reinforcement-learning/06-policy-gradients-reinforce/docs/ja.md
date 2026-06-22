# ポリシー勾配 — REINFORCE ゼロから構築

> 価値を推定することをやめて、ポリシーを直接パラメータ化し、期待リターンの勾配を計算して、上り坂に進む。Williams (1992) はこれを 1 つの定理で示した。これが PPO、GRPO、そしてあらゆる LLM RL ループが存在する理由である。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 3 · 03 (バックプロパゲーション)、Phase 9 · 03 (モンテカルロ)、Phase 9 · 04 (TD 学習)
**所要時間:** ~75分

## 問題

Q-learning と DQN は*価値*関数をパラメータ化する。`argmax Q` で行動を選択する。これは離散的な行動と離散的な状態には問題ない。行動が連続である場合（10 次元のトルクで `argmax` はどうする？）、または確率的ポリシーが必要な場合（`argmax` は構成上決定論的）に破綻する。

ポリシー勾配は代わりに*ポリシー*をパラメータ化する。`π_θ(a | s)` はニューラルネットであり、行動の分布を出力する。それをサンプリングして行動する。`θ` に対する期待リターンの勾配を計算する。上り坂に進む。`argmax` がない。ベルマン再帰がない。ただ `J(θ) = E_{π_θ}[G]` に対する勾配上昇。

REINFORCE 定理 (Williams 1992) はこの勾配が計算可能であることを示している: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。エピソードを実行する。リターンを計算する。すべてのステップで `∇ log π_θ(a | s)` で乗算する。平均化する。勾配上昇。終了。

2026 年のすべての LLM-RL アルゴリズム — PPO、DPO、GRPO — は REINFORCE の改良である。これを指の動きで理解することは、このフェーズの残りの部分、および Phase 10 · 07 (RLHF 実装) と Phase 10 · 08 (DPO) の前提条件である。

## コンセプト

![ポリシー勾配: ソフトマックスポリシー、log-π勾配、リターン加重更新](../assets/policy-gradient.svg)

**ポリシー勾配定理。** `θ` でパラメータ化されたあらゆるポリシー `π_θ` に対して:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

ここで `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` はステップ `t` からの割引リターンであり。期待値は `π_θ` からサンプリングされた完全な軌跡 `τ` に対するものである。

**証明は短い。** `J(θ) = Σ_τ P(τ; θ) G(τ)` を期待値の下で微分する。`∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (対数導関数トリック) を使う。`log P(τ; θ) = Σ log π_θ(a_t | s_t)` + θ に依存しない環境項に因数分解する。環境項は消える。2 行の代数が定理を与える。

**分散削減トリック。** Vanilla REINFORCE は殺人的な分散を持つ — リターンはノイズが多く、`∇ log π` はノイズが多く、それらの積は非常にノイズが多い。2 つの標準的な修正:

1. **ベースライン減算。** `G_t` を `G_t - b(s_t)` で置き換える。任意のベースライン `b(s_t)` は `a_t` に依存しない。不偏だる理由は `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型的な選択: `b(s_t) = V̂(s_t)` は批評家によって学習された → アクター・クリティック (レッスン 07)。
2. **報酬から行く。** `Σ_t G_t · ∇ log π_θ(a_t | s_t)` を `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` で置き換える。与えられた行動に対して将来のリターンだけが重要である — 過去の報酬は零平均ノイズに寄与する。

組み合わせると、以下が得られる:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

これは REINFORCE with a baseline — A2C (レッスン 07) と PPO (レッスン 08) の直接的な祖先である。

**ソフトマックスポリシーパラメータ化。** 離散的な行動の場合、標準的な選択:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

ここで `f_θ` は行動ごとにスコアを出力する任意のニューラルネット。勾配はクリーンな形を持つ:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

つまり、取られた行動のスコアから、ポリシー下の期待値を引いたもの。

**連続的な行動に対するガウスポリシー。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` はクローズドフォームを持つ。それが Phase 9 · 07 の SAC が必要なすべてである。

## 構築

### ステップ 1: ソフトマックスポリシーネットワーク

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

表形式環境に対して線形ポリシー (行動ごとに 1 つの重みベクトル) を使う。Atari の場合、CNN に交換してソフトマックスヘッドを維持する。

### ステップ 2: サンプリングとログ確率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### ステップ 3: ログ確率をキャプチャするロールアウト

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### ステップ 4: REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

勾配 `∇ log π(a|s) = e_a - π(·|s)` (a のワンホット マイナス確率) はソフトマックスポリシー勾配の中核である。筋肉記憶に焼き付けろ。

### ステップ 5: ベースライン

最近のエピソードにわたる `G` の移動平均は、十分な分散削減で 4×4 GridWorld を実行するのに十分である。約 500 エピソードで収束する。ベースラインを学習された `V̂(s)` にアップグレードして、アクター・クリティックを取得する。

## 落とし穴

- **勾配爆発。** リターンは巨大になる可能性がある。常に `G` を `~N(0, 1)` に正規化してから `∇ log π` で乗算してください。
- **エントロピー崩壊。** ポリシーはニアデターミニスティックな行動に早期に収束し、探索を停止し、スタックする。修正: `β · H(π(·|s))` のエントロピー ボーナスを目的に追加する。
- **高い分散。** Vanilla REINFORCE は数千のエピソードが必要である。批評家ベースライン (レッスン 07) または TRPO/PPO の信頼領域 (レッスン 08) は標準的な修正である。
- **サンプル非効率性。** オンポリシーは、1 つの更新後にあらゆるトランジションをスローすることを意味する。重要度サンプリングによるオフポリシー補正がデータを戻す (PPO の比率はクリップされた IS 重みである)。
- **非定常勾配。** 100 エピソード前と同じ勾配は古い `π` を使用する。オンポリシーメソッドは、この理由から数回のロールアウトごとに更新される。
- **信用割り当て。** 報酬から行くことなく、過去の報酬はノイズに寄与する。常に報酬から行く。

## 使用

2026 年では、REINFORCE はめったに直接実行されないが、その勾配公式はいたるところにある:

| ユースケース | 派生メソッド |
|----------|---------------|
| 連続制御 | PPO / SAC with ガウスポリシー |
| LLM RLHF | PPO with KL ペナルティ、トークンレベルのポリシーで実行 |
| LLM 推論 (DeepSeek) | GRPO — グループ相対ベースラインを備えた REINFORCE、批評家なし |
| マルチエージェント | 集中批評家 REINFORCE (MADDPG、COMA) |
| 離散行動ロボティクス | A2C、A3C、PPO |
| 好みのみの設定 | DPO — REINFORCE は好み尤度損失として書き直された、サンプリングなし |

2026 年のトレーニング スクリプトで `loss = -advantage * log_prob` を読むと、それはベースラインを備えた REINFORCE である。全体の論文 (DPO、GRPO、RLOO) は、この 1 行の最上位の分散削減トリックである。

## 配信

`outputs/skill-policy-gradient-trainer.md` として保存します:

```markdown
---
name: policy-gradient-trainer
description: 与えられたタスクの REINFORCE / アクター・クリティック / PPO トレーニング設定を生成し、分散問題を診断します。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

環境 (離散 / 連続行動、水平線、報酬統計) が与えられた場合、以下を出力します:

1. ポリシーヘッド。ソフトマックス (離散) またはガウス (連続)、パラメータ カウント付き。
2. ベースライン。なし (vanilla)、移動平均、学習された `V̂(s)`、または A2C 批評家。
3. 分散制御。報酬から行く デフォルトでオン、戻り値の正規化、勾配クリップ値。
4. エントロピー ボーナス。係数 β と減衰スケジュール。
5. バッチ サイズ。更新ごとのエピソード。オンポリシーデータの鮮度契約。

`β = 0` かつ観測されたポリシー エントロピー < 0.1 の 500 ステップを超える水平線で REINFORCE-no-baseline を拒否します。連続行動制御とソフトマックス ヘッドの組み合わせを拒否します。エントロピーが崩壊した場合として `β = 0` と任意の実行にフラグを立てます。
```

## 演習

1. **簡単。** 線形ソフトマックスポリシーで 4×4 GridWorld に REINFORCE を実装します。ベースラインなしで 1,000 エピソード トレーニングします。学習曲線をプロット; 分散 (リターンの std) を測定します。
2. **中程度。** 移動平均ベースラインを追加します。もう一度トレーニングしてください。vanilla 実行へのサンプル効率と分散を比較します。ベースラインは収束までのステップをどの程度削減しますか?
3. **難しい。** エントロピー ボーナス `β · H(π)` を追加します。`β ∈ {0, 0.01, 0.1, 1.0}` をスイープします。最終的なリターンとポリシー エントロピーをプロット します。このタスクでの甘い点はどこですか?

## キーワード

| 用語 | 人々が言う | 実際の意味 |
|------|-----------------|-----------------------|
| ポリシー勾配 | "ポリシーを直接トレーニング" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; 対数導関数トリックから導出。 |
| REINFORCE | "元のPGアルゴリズム" | Williams (1992); ログポリシー勾配で乗算されたモンテカルロリターン。 |
| 対数導関数トリック | "スコア関数推定量" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; 期待値の勾配を扱いやすくします。 |
| ベースライン | "分散削減" | `G` から差し引かれた任意の `b(s)`; `E[b · ∇ log π] = 0` であるため不偏。 |
| 報酬から行く | "将来のリターンだけがカウント" | `G_0` ではなく `G_t^{from t}`; 正確かつ低分散。 |
| エントロピー ボーナス | "探索を促進" | `+β · H(π(·\|s))` 項はポリシーが崩壊するのを防ぎます。 |
| オンポリシー | "あなたが見たばかりのことについてトレーニング" | 勾配期待値は現在のポリシーに関してである — 古いデータを直接再利用することはできません。 |
| アドバンテージ | "平均よりどの程度良いか" | `A(s, a) = G(s, a) - V(s)`; REINFORCE-with-baseline が乗算する符号付き量。 |

## 参考文献

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 元の REINFORCE 論文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 関数近似を備えた現代的なポリシー勾配定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — テキスト本の提示。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — PyTorch コード付きの明確な教育的説明。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 分散削減と、REINFORCE を信頼領域ファミリー (TRPO、PPO) に接続する自然勾配ビュー。
