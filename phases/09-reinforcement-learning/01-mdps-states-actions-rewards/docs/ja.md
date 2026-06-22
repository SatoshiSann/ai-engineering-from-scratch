# マルコフ決定過程、状態、行動、報酬

> マルコフ決定過程は 5 つのもの: 状態、行動、遷移、報酬、割引。RL のすべて — Q-learning、PPO、DPO、GRPO — はこの形に関して最適化します。一度学んでください。残りの強化学習を無料で読むことができます。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 1 · 06 (確率と分布)、Phase 2 · 01 (ML 分類法)
**所要時間:** 約 45 分

## 問題

チェスボットを書いています。または在庫計画者。また取引エージェント。または推論モデルをトレーニングする PPO ループ。4 つの異なるドメイン、1 つの驚くべき事実: 4 つすべてが同じ数学的オブジェクトに崩壊します。

教師あり学習は `(x, y)` ペアを与え、関数にフィットさせるよう求めます。強化学習はラベルを与えません — ただ状態、取った行動、スカラー報酬のストリームです。その動きはゲームに勝ちましたか? リストック決定はお金を節約しましたか? 取引は利益を上げましたか? LLM が出力したばかりのトークンは、審査人からより高い報酬につながりましたか?

このストリームから学ぶまで、形式化する必要があります。「私が見た」、「私がしたこと」、「次に何が起こったか」、「それがどの程度良かったか」— それぞれが推理できるオブジェクトになる必要があります。その形式化はマルコフ決定過程です。このフェーズのすべての RL アルゴリズム、終わりの RLHF と GRPO ループを含めて、この形に関して最適化します。

## コンセプト

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**5 つのオブジェクト。**

- **状態** `S`。エージェントが決定するのに必要なすべて。GridWorld では、セル。チェスでは、ボード。LLM では、コンテキスト ウィンドウとメモリ。
- **行動** `A`。選択肢。上下左右に移動。移動を実行。トークンを出力。
- **遷移** `P(s' | s, a)`。状態 `s` と行動 `a` が与えられた場合、次の状態に対する分布。チェスでは決定的、在庫では確率的、LLM デコーディングではほぼ決定的。
- **報酬** `R(s, a, s')`。スカラー信号。勝利 = +1、敗北 = -1。収入マイナス コスト。GRPO の対数尤度比項。
- **割引** `γ ∈ [0, 1)`。未来の報酬が現在のどちらかについてカウントします。`γ = 0.99` は約 100 ステップの地平線を買う。`γ = 0.9` は約 10 を買う。

**マルコフ特性** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来は現在の状態のみに依存します。そうでない場合、「状態」表現は不完全です — メソッドの失敗ではなく、状態の失敗。

**方策とリターン。** 方策 `π(a | s)` は状態をアクション分布にマップします。リターン `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` は未来報酬の割引合計です。値 `V^π(s) = E[G_t | s_t = s]` は方策 `π` の下で `s` から開始するリターンの期待値です。Q 値 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` は特定の行動で開始するリターンの期待値です。すべての RL アルゴリズムはこれら 2 つのいずれかを推定してから、それに応じて `π` を改善します。

**ベルマン方程式。** このフェーズが使用する固定点方程式:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

これらは期待リターンを「このステップの報酬」プラス「あなたが着地した場所の割引値」に分割します。再帰的。Phase 9 のすべてのアルゴリズムは、この方程式を収束に反復 (動的計画法)、そこからサンプル (モンテカルロ)、または 1 ステップ ブートストラップ (時間差分) のいずれかです。

## ビルド

### ステップ 1: 小さな決定的 MDP

4×4 GridWorld。エージェントは左上から開始、端末は右下、各ステップの報酬は -1、アクションは `{up, down, left, right}`。`code/main.py` を見てください。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

5 行。環境全体です。決定的な遷移、定数ステップペナルティ、吸収ターミナル状態。

### ステップ 2: 方策をロールアウト

方策は状態からアクション分布への関数です。最も単純: 均一ランダム。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

ランダム方策を 1000 回実行します。平均リターンはこの 4×4 ボードで約 -60 から -80 です。最適なリターンは -6 (斜線パス下右) です。そのギャップを埋めることが Phase 9 のすべてです。

### ステップ 3: ベルマン方程式を使用して `V^π` を正確に計算

小さな MDP の場合、ベルマン方程式は線形システムです。状態を列挙し、期待値を適用し、値が変わるまで反復します。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

これは反復的な方策評価です。Sutton & Barto の最初のアルゴリズムであり、それに続くすべての RL メソッドの理論的基礎です。

### ステップ 4: `γ` は物理的意味を持つハイパーパラメータ

有効地平線は約 `1 / (1 - γ)` です。`γ = 0.9` → 10 ステップ。`γ = 0.99` → 100 ステップ。`γ = 0.999` → 1000 ステップ。

低すぎるとエージェントは近視的に動作します。高すぎるとクレジット割り当てはノイズになります。多くの早期ステップが遠い将来の報酬を共有責任だからです。LLM RLHF は通常 `γ = 1` を使用します。エピソードは短くて有界だからです。制御タスクは `0.95–0.99` を使用します。長地平線戦略ゲームは `0.999` を使用します。

## 落とし穴

- **非マルコフ状態。** 決定するのに最後の 3 つの観測が必要な場合、「状態」は現在の観測だけではありません。修正: フレームをスタック (DQN on Atari は 4 をスタック) または観測の上に再帰状態を使用 (LSTM/GRU)。
- **スパース報酬。** 勝利のみの報酬は大きな状態空間での学習をほぼ不可能にします。報酬を形成する (中間信号) またはイミテーション (Phase 9 · 09) でブートストラップします。
- **報酬ハッキング。** プロキシ報酬の最適化はしばしば病的な動作を生成します。OpenAI のボートレーシング エージェントは、レースを終わらせる代わりに、パワーアップを収集して永遠に円を描きました。常に報酬を目標の成果から定義し、プロキシではありません。
- **割引仕様の誤り。** `γ = 1` 無限地平線タスクはすべての値を無限にします。有限地平線で上限を設定するか、`γ < 1`。
- **報酬スケール。** `{+100, -100}` と `{+1, -1}` の報酬は同じ最適方策を与えますが、勾配の大きさは大きく異なります。PPO/DQN に接続する前に `[-1, 1]` のようなものに正規化します。

## 使用

2026 スタックは、コードに触れる前にすべての RL パイプラインを MDP に減らします:

| 状況 | 状態 | 行動 | 報酬 | γ |
|------|------|------|------|---|
| 制御 (移動、操作) | 関節角度と速度 | 連続トルク | タスク固有形成 | 0.99 |
| ゲーム (チェス、囲碁、ポーカー) | ボードと履歴 | 合法移動 | 勝利=+1 / 敗北=-1 | 1.0 (有限) |
| 在庫/価格設定 | 在庫と需要 | オーダー数量 | 収入 - コスト | 0.95 |
| LLM 用 RLHF | コンテキスト トークン | 次トークン | リターンの終わりの報酬モデル スコア | 1.0 (エピソード 約 200 トークン) |
| 推論用 GRPO | プロンプト + 部分応答 | 次トークン | ベリファイア 0/1 終わり | 1.0 |

5 つのタプルを訓練ループを書く前に書きます。ほとんどの「RL が機能しない」バグ レポートは、紙の上で破られた MDP 定式化に遡ります。

## シップ

`outputs/skill-mdp-modeler.md` として保存:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 演習

1. **簡単。** `code/main.py` で 4×4 GridWorld とランダム方策ロールアウトを実装します。10,000 エピソード実行します。リターンの平均と標準偏差を報告します。最適リターン (-6) と比較します。
2. **中程度。** `γ ∈ {0.5, 0.9, 0.99}` で `policy_evaluation` を実行します。各 4×4 グリッドとして `V` を印刷します。より大きな `γ` でターミナルの近くの状態値がより速く成長するのはなぜですか?
3. **難しい。** GridWorld を確率的にします: 各行動は確率 `p = 0.1` で隣接する方向にスリップします。均一方策を再評価します。`V[start]` は良くなるか悪くなるか? なぜ?

## キー用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------|--------|
| MDP | "Reinforcement learning setup" | マルコフ特性を満たすタプル `(S, A, P, R, γ)` |
| State | "What the agent sees" | 選択した方策クラスの下での将来の動的のための十分統計量 |
| Policy | "Agent's behavior" | 条件付き分布 `π(a \| s)` または決定的なマップ `s → a` |
| Return | "Total reward" | 現在のステップからの割引合計 `Σ γ^t r_t` |
| Value | "How good a state is" | `s` から `π` の下で開始するリターンの期待値 |
| Q-value | "How good an action is" | `s` で開始して最初のアクション `a` で `π` の下でのリターンの期待値 |
| Bellman equation | "Dynamic programming recursion" | 値 / Q の 1 ステップ報酬プラス割引継承者値への固定点分解 |
| Discount `γ` | "Future vs present" | 遠い将来の報酬への幾何学的重み。有効地平線 `~1/(1-γ)` |

## 参考文献

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — テキストブック。Ch. 3 は MDP とベルマン方程式をカバー。Ch. 1 はそれに続くすべてのレッスンの下にある報酬仮説を動機付けます。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — ベルマン方程式の起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 深い RL 角度からの簡潔な MDP プライマー。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP と正確な解法方法に関する操作研究の参照。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — MDP を動的計画法の特殊化として推導する最も洗練された導出。
