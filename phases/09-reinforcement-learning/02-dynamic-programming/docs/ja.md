# 動的計画法 — 方策反復と値反復

> 動的計画法は RL の不正行為です。遷移と報酬関数はすでに知っています。ベルマン方程式が `V` または `π` 動きを止めるまで反復するだけです。サンプリングベースのメソッドが近づこうとするベンチマークです。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 9 · 01 (MDP)
**所要時間:** 約 75 分

## 問題

既知のモデルを持つ MDP があります。`P(s' | s, a)` と `R(s, a, s')` を任意の状態行動ペアに対してクエリできます。在庫マネージャーは需要分布を知っています。ボード ゲームは決定的な遷移を持っています。GridWorld は Python の 4 行です。*モデル* があります。

モデルフリー RL (Q-learning、PPO、REINFORCE) は、モデルがない場合のために発明されました — 環境からサンプルのみできます。しかし、1 つあるときは、より速くより良いメソッド: 動的計画法があります。ベルマンは 1957 年にそれらを設計しました。彼らはまだ正確さを定義します: 「この MDP に対する最適方策」と言うとき、DP が返す方策を意味します。

2026 年で 3 つの理由で必要です。最初に、RL 研究の各表形式環境 (GridWorld、FrozenLake、CliffWalking) は DP で解決され、金標準方策を生成します。2 番目、正確な値は *デバッグ* サンプリング メソッドを許可します: Q-learning の `V*(s_0)` の推定が DP 答えと 30% 異なる場合、Q-learning にバグがあります。3 番目、最新のオフライン RL と計画メソッド (MCTS、AlphaZero の検索、Phase 9 · 10 のモデルベース RL) はすべて学習モデルまたは与えられたモデルに対するベルマン バックアップを反復します。

## コンセプト

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**2 つのアルゴリズム、どちらもベルマンに対する固定点反復。**

**方策反復。** 方策が変わるまで 2 つのステップを交替させます。

1. *評価:* 方策 `π` が与えられた場合、`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` を繰り返し適用して `V^π` を計算し、収束するまで。
2. *改善:* `V^π` が与えられた場合、`π` を `V^π` w.r.t. でグリーディにします: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

収束は保証されます。なぜなら (a) 各改善ステップ は `π` を同じままにするか、いくつかの状態の `V^π` を厳密に増加させ、(b) 決定的な方策の空間は有限です。通常、大きな状態空間でも約 5 ~ 20 の外側反復で収束します。

**値反復。** 評価と改善を 1 つのスイープに崩壊させます。ベルマン *最適性* 方程式を適用:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε` まで反復します。最後に、グリーディアクションを取得して方策を抽出します。反復あたり厳密に高速です — 内側評価ループなし — しかし通常はより多くの反復が収束に必要です。

**一般化方策反復 (GPI)。** 統一フレーミング。値関数と方策は相互改善ループにロックされています。動く両方をお互いに一貫性へ (非同期値反復、変更方策反復、Q-learning、アクター批評、PPO) 任意のメソッドは GPI のインスタンスです。

**なぜ `γ < 1` が重要。** ベルマン演算子は sup ノルムで `γ` 圧縮: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。圧縮は一意の固定点と幾何学的収束を意味します。`γ < 1` をドロップするとテクノロジーを失い、有限地平線か吸収ターミナル状態が必要です。

## ビルド

### ステップ 1: GridWorld MDP モデルを構築

Lesson 01 から同じ 4×4 GridWorld を使用します。確率的バリアントを追加します: 確率 `0.1` でエージェントが確率的に垂直方向にスリップします。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` は `(s', r, p)` のリストを返します。これは環境全体です。

### ステップ 2: 方策評価

方策 `π(s) = {action: prob}` が与えられた場合、`V` が動きを止めるまでベルマン方程式を反復:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### ステップ 3: 方策改善

`π` を `V` に関するグリーディ方策に置き換えます。`π` が変わらない場合、返す — 最適に到着しています。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### ステップ 4: それらを一緒に縫い合わせる

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 での典型的な収束: 4-6 の外側反復。`V*(0,0) ≈ -6` を出力し、ステップ数を厳密に減らす方策。

### ステップ 5: 値反復 (1 ループ バージョン)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同じ固定点、より少ないコード行。

## 落とし穴

- **ターミナルの処理を忘れる。** 吸収状態にベルマンを適用する場合でも、変わらない「最適アクション」を拾います。`if s == terminal: V[s] = 0` でガード。
- **Sup-norm 対 L2 収束。** `max |V_new - V|` を使用します。平均ではなく。理論的保証は sup ノルムです。
- **インプレース対同期更新。** `V[s]` をインプレース (ガウス-ザイデル) で更新すると、別の `V_new` 辞書 (ヤコビ) よりも速く収束します。本番コードはインプレースを使用します。
- **方策タイ。** 2 つのアクションが等しい Q 値を持つ場合、`argmax` は各反復で異なるタイを破ることがあり、「方策安定」チェックが振動する原因になります。安定したタイ破り (固定順序の最初のアクション) を使用します。
- **状態空間爆発。** DP は 1 スイープあたり `O(|S| · |A|)` です。約 10 ⁷ 状態まで機能します。それ以上、関数近似が必要 (Phase 9 · 05 以降)。

## 使用

2026 年では、DP は正確さベースラインであり計画者の内側ループです:

| ユースケース | メソッド |
|----------|--------|
| 小さな表形式 MDP を正確に解く | 値反復 (より単純) またはアルゴリズム反復 (より少ない外側ステップ) |
| Q-learning / PPO 実装を確認 | おもちゃ環境で DP 最適 V* と比較 |
| モデルベース RL (Phase 9 · 10) | 学習遷移モデルのベルマン バックアップ |
| AlphaZero / MuZero での計画 | モンテカルロ木検索 = 非同期ベルマン バックアップ |
| オフライン RL (CQL、IQL) | 保守的 Q 反復 — OOD アクションの DP ペナルティ |

誰かが「最適値関数」と言うたびに、「DP 固定点」を意味します。ペーパーで `V*` または `Q*` を見るとき、このループの画像を描きます。

## シップ

`outputs/skill-dp-solver.md` として保存:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 演習

1. **簡単。** 4×4 GridWorld で `γ ∈ {0.9, 0.99}` で値反復を実行します。`max |ΔV| < 1e-6` まで何回スイープ? `V*` を 4×4 グリッドとして印刷します。
2. **中程度。** スリップ確率 `0.1` の *確率的* GridWorld でポリシー反復と値反復を比較します。カウント: スイープ、壁時計時間、最終 `V*(0,0)`。どれが反復速く収束します? 壁時計で?
3. **難しい。** 変更方策反復を構築: 評価ステップで、収束までではなく `k` スイープのみを実行します。`k ∈ {1, 2, 5, 10, 50}` について `V*(0,0)` エラー対 `k` をプロットします。曲線は評価/改善トレードオフについて何を教えていますか?

## キー用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------|--------|
| Policy iteration | "DP algorithm" | 方策が変わるまで評価 (`V^π`) と改善 (グリーディ `π` w.r.t. `V^π`) を交替させる |
| Value iteration | "Faster DP" | ベルマン最適性バックアップを 1 つのスイープで適用。幾何学的に `V*` に収束 |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))` は sup ノルムで `γ` 圧縮 |
| Contraction | "Why DP converges" | `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` を持つ任意の演算子 `T` は一意の固定点を持つ |
| GPI | "Everything is DP" | 一般化方策反復: `V` と `π` を相互一貫性へ運転するすべてのメソッド |
| Synchronous update | "Jacobi-style" | スイープ全体で古い `V` を使用。洗浄可能に分析できますがより遅い |
| In-place update | "Gauss-Seidel-style" | `V` を更新するときに使用。実際にはより速く収束 |

## 参考文献

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 方策反復と値反復の正規プレゼンテーション。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 圧縮マッピング議論の厳密な扱い。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 変更方策反復とその収束分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 元の方策反復ペーパー。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — DP から近似 DP / 深い RL へのブリッジで、その後のすべてのレッスンで使用されます。
