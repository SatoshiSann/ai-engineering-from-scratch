# 時間差分 — Q-Learning と SARSA

> モンテカルロはエピソードが終わるまで待ちます。TD は次の値の推定をブートストラップすることで毎ステップ更新します。Q-learning はオフポリシーで楽観的です。SARSA はオンポリシーで慎重です。どちらも 1 行のコードです。どちらもこのフェーズのすべての深い RL メソッドの下にあります。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 9 · 01 (MDP)、Phase 9 · 02 (動的計画法)、Phase 9 · 03 (モンテカルロ)
**所要時間:** 約 75 分

## 問題

モンテカルロは機能しますが 2 つの高い需要があります。終了するエピソードが必要で、最終リターンが入るまで更新するだけです。エピソードが 1,000 ステップの場合、MC は 1,000 ステップ待って何かを更新します。それは高分散、低バイアス、実際に遅い。

動的計画法は反対のプロファイル — ゼロ分散ブートストラップ バックアップ — を持つが、既知のモデルが必要です。

時間差分 (TD) 学習は差を分割します。単一遷移 `(s, a, r, s')` から、1 ステップ ターゲット `r + γ V(s')` を形成し、`V(s)` をそれに向かってナッジします。モデルなし。完全なエピソードなし。近似 `V` を使用することからバイアスしますが、MC より劇的に低い分散とステップ 1 からのオンライン更新。

これは、すべての最新 RL — DQN、A2C、PPO、SAC — が回転する軸です。Phase 9 の残りは、このレッスンで書く 1 ステップ TD 更新上で構築されたレイヤーと関数近似のトリックです。

## コンセプト

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V の TD(0) 更新:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括弧内の量は TD エラー `δ = r + γ V(s') - V(s)`。MC での `G_t - V(s_t)` のオンライン アナログです。収束は `α` ロビンス-モンロ (`Σ α = ∞`、`Σ α² < ∞`) と無限に訪問されたすべての状態を満たす必要があります。

**Q-learning。** 制御のためのオフポリシー TD メソッド:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` は、エージェントが実際に取るアクションに関係なく、`s'` から先にグリーディ方策が従うことを想定しています。そのデカップリングにより Q-learning はエージェントが ε-greedy を介して探索しながら `Q*` を学びます。Mnih ら (2015) これを Atari (Lesson 05) の深い Q-learning に変換しました。

**SARSA。** オンポリシー TD メソッド:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名前はタプル `(s, a, r, s', a')` です。SARSA はエージェント *実際に* グリーディ `argmax` ではなく、次に取るアクション `a'` を使用します。どのような `ε-greedy` `π` が実行されている場合でも `Q^π` に収束し、制限 `ε → 0` で `Q*` になります。

**崖歩き違い。** 古典的な崖歩きタスク (崖から落下 = 報酬 -100) で、Q-learning は崖のエッジに沿った最適なパスを学びますが、探索中に時々ペナルティを取ります。SARSA は崖の 1 ステップ離れた安全なパスを学びます。それは探索ノイズを Q 値に組み込むからです。訓練では、両方ともで最適に到着します `ε → 0`。実際には重要です: 探索が配置に実際に発生している場合、SARSA の動作はより保守的です。

**期待 SARSA。** `Q(s', a')` を `π` の下での期待値に置き換えます:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSA より低分散 (`a'` のサンプルなし)、同じオンポリシー ターゲット。最新の教科書では多くの場合デフォルト。

**n ステップ TD と TD(λ)。** `n` ステップを待ってからブートストラップ するまで TD(0) と MC の間に内挿。`n=1` は TD、`n=∞` は MC。TD(λ) は幾何学的重み `(1-λ)λ^{n-1}` を持つすべての `n` 上で平均化。ほとんどの深い RL は `n` 3 から 20 の間で使用します。

## ビルド

### ステップ 1: ε-greedy ポリシーで SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

8 行。Q-learning からの *唯一* の違いはターゲット行です。

### ステップ 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` はターゲットを動作からデカップルします。その 1 つのシンボルはオンポリシーとオフポリシー間の違いです。

### ステップ 3: 学習曲線

100 エピソード あたり平均リターンを追跡します。Q-learning は単純な決定的 GridWorld で高速に収束します。SARSA は崖歩きでより保守的です。`code/main.py` の 4×4 GridWorld で、両方は `α=0.1, ε=0.1` で約 2,000 エピソード後にほぼ最適です。

### ステップ 4: DP 真実と比較

値反復を実行 (Lesson 02) `Q*` を取得します。`max_{s,a} |Q_learned(s,a) - Q*(s,a)|` をチェック。健全な表形式 TD エージェントは 4×4 GridWorld で 10,000 エピソード後に ~0.5 内に着地します。

## 落とし穴

- **初期 Q 値が重要。** 楽観的初期化 (`Q = 0` 負報酬タスクの場合) 探索を奨励します。悲観的な初期化は貪欲な方策を永遠にトラップできます。
- **α スケジュール。** 定数 `α` は非定常な問題に罰金です。減衰 `α_n = 1/n` は理論に収束を与えますが、実際に遅すぎます — `[0.05, 0.3]` に `α` をピンしレーニング曲線を監視します。
- **ε スケジュール。** 高から開始 (`ε=1.0`)、`ε=0.05` に減衰。「GLIE」 (無限の探索の制限でグリーディ) は収束条件です。
- **Q-learning で最大バイアス。** `max` 演算子は `Q` がノイズの場合、上向きにバイアスされています。過大推定につながります — Hasselt の Double Q-learning (Lesson 05 で DDQN で使用) は 2 つの Q テーブルで修正します。
- **終了しないエピソード。** TD はターミナルなしで学ぶことができますが、ステップをキャップするか、キャップ時のブートストラップを処理する必要があります。標準: キャップを非ターミナルとして扱い、ブートストラップを保つ。
- **状態ハッシング。** 状態がタプル/テンソルの場合、ハッシュ可能キーを使用 (タプル、リストなし。浮動小数点のタプル、生のもの)。

## 使用

2026 年の TD ランドスケープ:

| タスク | メソッド | 理由 |
|------|--------|------|
| 小さな表形式環境 | Q-learning | 直接最適な方策を学びます |
| オンポリシー安全批判的 | SARSA / Expected SARSA | 探索中は保守的です |
| 高次元状態 | DQN (Phase 9 · 05) | リプレイとターゲット ネットを持つニューラル ネット Q 関数 |
| 連続アクション | SAC / TD3 (Phase 9 · 07) | Q ネットワークの TD 更新。方策ネット出力アクション |
| LLM RL (報酬モデルベース) | PPO / GRPO (Phase 9 · 08、12) | GAE を使用した TD スタイル利点を持つ アクター批評 |
| オフライン RL | CQL / IQL (Phase 9 · 08) | 保守的な正則化を持つ Q-learning |

2026 年のペーパーで読む「RL」の 90% は Q-learning または SARSA のいくつかの精密化です。より深く読む前に、表形式の更新を指の中に理解してください。

## シップ

`outputs/skill-td-agent.md` として保存:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 演習

1. **簡単。** 4×4 GridWorld で Q-learning と SARSA を実装します。学習曲線をプロット (100 エピソード あたり平均リターン) 2,000 エピソード。誰がより速く収束?
2. **中程度。** 崖歩き環境を構築 (4×12、最後の行は崖で報酬 -100 とスタートにリセット)。最終的な Q-learning と SARSA ポリシーを比較します。各スクリーンショット パス。どちらが崖に近いですか?
3. **難しい。** Double Q-learning を実装します。ノイズの多い報酬 GridWorld で (ガウス ノイズ σ=5 がステップごとの報酬に追加), show Q-learning は `V*(0,0)` を有意な量で過大推定しながら Double Q-learning はしません。

## キー用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------|--------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`、ブートストラップ残差 |
| TD(0) | "One-step TD" | 次の状態の推定のみを使用してすべての遷移後に更新 |
| Q-learning | "Off-policy RL 101" | 次状態アクションの `max` を持つ TD 更新。動作方策に関係なく `Q*` を学びます |
| SARSA | "On-policy Q-learning" | 実際の次のアクションを使用する TD 更新。現在の ε-greedy π に対して `Q^π` を学びます |
| Expected SARSA | "The low-variance SARSA" | サンプル `a'` を π の下での期待値に置き換えます |
| GLIE | "Correct exploration schedule" | 無限の探索で制限でグリーディ。Q-learning 収束に必要 |
| Bootstrapping | "Using current estimate in the target" | TD を MC から区別するもの。バイアスのソースですが巨大な分散削減 |
| Maximization bias | "Q-learning overestimates" | ノイズ推定上の `max` は上向きにバイアスします。Double Q-learning で固定 |

## 参考文献

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 元のペーパーと収束証明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大バイアスの修正。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA の動機。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — SARSA を造語したペーパー (その後「変更接続主義 Q-learning」と呼ばれる)。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0) を TD(n)、Q-learning から適格トレース、その後 PPO の GAE へのパスに一般化します。
