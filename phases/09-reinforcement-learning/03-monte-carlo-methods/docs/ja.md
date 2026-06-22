# モンテカルロ法 — 完全エピソードから学ぶ

> 動的計画法はモデルが必要です。モンテカルロは何も必要ありません。方策を実行し、リターンを見て、それらを平均化します。RL で最も単純な考え — その後のすべてをアンロックする。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 9 · 01 (MDP)、Phase 9 · 02 (動的計画法)
**所要時間:** 約 75 分

## 問題

動的計画法はエレガントですが、すべての状態とアクションに対して `P(s' | s, a)` をクエリできると仮定します。現実世界のほぼ何もそのように機能しません。ロボットは関節トルク後のカメラピクセルの分布を分析的に計算することはできません。価格設定アルゴリズムはすべての可能な顧客反応に統合できません。LLM はトークン後のすべての可能な継続を列挙できません。

環境から *サンプル* する能力のみを必要とするメソッドが必要です。方策を実行します。軌跡 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T` を取得します。それを使用して値を推定します。それはモンテカルロです。

DP から MC への変更は哲学的に重要です: *既知のモデル + 正確なバックアップ* から *サンプル ロールアウト + 平均リターン* に移動します。分散はジャンプしますが、適用可能性は爆発します。このレッスン後のすべての RL アルゴリズム — TD、Q-learning、REINFORCE、PPO、GRPO — は、ときどきブートストラップで層化した、モンテカルロ推定器です。

## コンセプト

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**コアアイデア、1 行で:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` ここで `G^{(i)}(s)` は方策 `π` の下で `s` への訪問に続いて観測されたリターン。

**最初訪問対すべての訪問 MC。** 状態 `s` を複数回訪問するエピソードが与えられた場合、最初訪問 MC は最初訪問からのリターンのみをカウント。すべての訪問 MC はすべての訪問をカウント。どちらも制限でアンバイアスです。最初訪問はより単純に分析でき (iid サンプル)。すべての訪問はエピソードあたりより多くのデータを使用し、通常は実際に高速で収束します。

**増分平均。** すべてのリターンを保存する代わりに、実行中の平均を更新:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

再構成: `V_new = V_old + α · (target - V_old)` で `α = 1/n`。定数ステップサイズ `α ∈ (0, 1)` で `1/n` を交換し、`π` の変更を追跡する非定常 MC 推定器を取得します。その動きは MC から TD へとすべての最新 RL アルゴリズムへの全ジャンプです。

**探索は今すぐ問題です。** DP はすべての状態に列挙によって接しました。MC は方策が訪問する状態のみが表示されます。`π` が決定的な場合、状態空間の領域全体がサンプルされることはなく、それらの値の推定は永遠にゼロで留まります。3 つの修正、歴史的な順序で:

1. **スタートを探検します。** 各エピソードをランダム (s, a) ペアから開始します。カバレッジを保証します。実際には非現実的 (ロボットを任意の状態に「リセット」することはできません)。
2. **ε-greedy。** 現在の Q に関してグリーディに行動しますが、確率 `ε` でランダムなアクションをピック。すべての状態行動ペアは漸近的にサンプルされます。
3. **オフポリシー MC。** 動作ポリシー `μ` の下でデータを収集し、重要度サンプリング経由でターゲット ポリシー `π` について学びます。高分散ですが、DQN のようなリプレイ バッファ メソッドへのブリッジです。

**モンテカルロ制御。** 評価 → 改善 → 評価、方策反復と同じですが、評価はサンプリングベースです:

1. `π` を実行し、エピソードを取得します。
2. 観測されたリターンから `Q(s, a)` を更新します。
3. `π` を ε-greedy w.r.t. `Q` にします。
4. 繰り返します。

穏やかな条件の下で確率 1 で `Q*` と `π*` に収束 (すべてのペアが無限に訪問、`α` は Robbins-Monro を満たします)。

## ビルド

### ステップ 1: ロールアウト → (s, a, r) のリスト

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

モデルなし、`env.reset()` と `env.step(s, a)` のみ。削除した gym 環境と同じインターフェース。

### ステップ 2: リターンを計算 (逆スイープ)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

1 パス、`O(T)`。後ろの反復 `G_t = r_{t+1} + γ G_{t+1}` は合計を再計算するのを避けます。

### ステップ 3: 最初訪問 MC 評価

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

3 行は機能を実行: 最初訪問の状態をマーク、カウントをインクリメント、実行中の平均を更新。

### ステップ 4: ε-greedy MC 制御 (オンポリシー)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### ステップ 5: DP 金標準と比較

`V^π` の MC 推定は Lesson 02 からの DP 結果と、エピソード → ∞ に同意する必要があります。実際: 4×4 GridWorld で 50,000 エピソード DP 答えの ~0.1 内に取得します。

## 落とし穴

- **無限エピソード。** MC がエピソード *を終了* する必要があります。方策が永遠にループできる場合、`max_steps` を上限し、上限を暗黙的な失敗として扱います。ランダムな方策を持つ GridWorld は定期的にタイムアウトします — それは通常です。正確にカウントするだけ確認してください。
- **分散。** MC は完全なリターンを使用します。長いエピソードでは、分散は巨大です — 終わりで不運な報酬で 1 つ `V(s_0)` 同じ量だけシフト。TD メソッド (Lesson 04) はブートストラップで切ります。
- **状態カバレッジ。** タイのあるフレッシュ Q でグリーディ MC は 1 つのアクションのみを試してみます。探索 (ε-greedy、探検スタート、UCB) する *必要があります*。
- **非定常ポリシー。** `π` が変わる (MC 制御で) 場合、古いリターンは別のポリシーからです。定数α MC はこれを処理します。サンプル平均 MC はしません。
- **オフポリシー重要度サンプリング。** 重み `π(a|s)/μ(a|s)` は軌跡全体に掛けます。分散は地平線で爆発。決定ごとの重み付き IS で上限またはおろしてください TD。

## 使用

2026 年のモンテカルロ法の役割:

| ユースケース | なぜ MC |
|-----------|--------|
| 短地平線ゲーム (ブラックジャック、ポーカー) | エピソードは自然に終了。リターンはクリーン |
| ログに記録されたポリシーのオフライン評価 | 保存された軌跡の平均割引リターン |
| モンテカルロ木検索 (AlphaZero) | MC ロールアウト ツリーの葉からガイド選択 |
| LLM RL 評価 | 与えられたポリシーの完成にわたる平均報酬を計算 |
| PPO でのベースライン推定 | 利点ターゲット `A_t = G_t - V(s_t)` は MC `G_t` を使用 |
| RL を教える | 実際に機能する最も単純なアルゴリズム — ブートストラップを取り除いてコアを見る |

最新の深い RL アルゴリズム (PPO、SAC) は純粋 MC (完全なリターン) と純粋 TD (1 ステップ ブートストラップ) `n` ステップ リターンまたは GAE との間で内挿。両方のエンドポイントは同じ推定器のインスタンスです。

## シップ

`outputs/skill-mc-evaluator.md` として保存:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 演習

1. **簡単。** 4×4 GridWorld での均一ランダム方策の最初訪問 MC 評価を実装します。10,000 エピソード実行します。エピソード カウント対 DP 答えの関数として `V(0,0)` をプロット。
2. **中程度。** `ε ∈ {0.01, 0.1, 0.3}` で ε-greedy MC 制御を実装します。20,000 エピソード後に平均リターンを比較します。曲線はどのように見えます? バイアス分散トレードオフはどこに住んでいますか?
3. **難しい。** *オフポリシー* MC と重要度サンプリング: 均一ランダム ポリシー `μ` の下でデータを収集し、決定的な最適ポリシー `π` の `V^π` を推定します。プレーン IS 対決定ごとの IS 対重い IS を比較。最も低い分散があります?

## キー用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------|--------|
| Monte Carlo | "Random sampling" | 分布からの iid サンプルの平均をとることで期待値を推定 |
| Return `G_t` | "Future reward" | ステップ `t` からエピソード終了まで割引報酬の合計: `Σ_{k≥0} γ^k r_{t+k+1}` |
| First-visit MC | "Count each state once" | エピソードの最初の訪問のみが値の推定に貢献 |
| Every-visit MC | "Use all visits" | すべての訪問が貢献。わずかにバイアスですがより多くのサンプル効率 |
| ε-greedy | "Exploration noise" | 確率 `1-ε` でグリーディ アクションをピック。確率 `ε` でランダム アクション |
| Importance sampling | "Correcting for sampling from the wrong distribution" | `π(a\|s)/μ(a\|s)` 製品でリターンを再重み付けして `V^π` を `μ` データから推定 |
| On-policy | "Learn from my own data" | ターゲット方策 = 動作方策。バニラ MC、PPO、SARSA |
| Off-policy | "Learn from someone else's data" | ターゲット方策 ≠ 動作方策。重要度サンプル MC、Q-learning、DQN |

## 参考文献

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 正規の扱い。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 最初訪問対すべての訪問分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — オフポリシー MC と分散制御。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 最新の低分散 IS 推定器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自己再生が超人的再生に収束する最初の大規模経験的実証。この段階の後半のすべてのレッスンへの概念的先駆者。
