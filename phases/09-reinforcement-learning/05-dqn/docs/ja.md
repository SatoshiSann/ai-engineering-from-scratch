# ディープ Q ネットワーク (DQN)

> 2013: Mnih は 1 つの Q-learning ネットワークを生ピクセルでトレーニング、7 つの Atari ゲームのすべての古典的 RL エージェントを倒しました。2015: 49 ゲームに拡張、Nature で公開、深い RL の時代を引き起こしました。DQN は Q-learning プラス関数近似を安定させる 3 つのトリックです。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 3 · 03 (バックプロパゲーション)、Phase 9 · 04 (Q-learning、SARSA)
**所要時間:** 約 75 分

## 問題

表形式 Q-learning は各 (状態、アクション) ペアに対して個別の Q 値が必要です。チェスボードには約 10⁴³ の状態があります。Atari フレームは 210×160×3 = 100,800 の機能です。表形式 RL は数千の状態で死ぬ、数十億ではなく。

修正は後ろ向きに明らかです: Q テーブルをニューラル ネットワーク `Q(s, a; θ)` に置き換えます。しかし明らかなのに後向きは数十年かかりました。素朴な関数近似から Q-learning は「致命的な 3 つ」— 関数近似 + ブートストラップ + オフポリシー 学習の下では分岐します。Mnih ら (2013、2015) 3 つのエンジニアリング トリック学習を安定させることを特定:

1. **経験リプレイ** 遷移を相関なくします。
2. **ターゲット ネットワーク** ブートストラップ ターゲットを凍結します。
3. **報酬クリッピング** 勾配の大きさを正規化します。

Atari の DQN は、単一のアーキテクチャで単一のハイパーパラメータ セットを持つ生ピクセルからの数十の制御問題を解く最初の時間でした。それ以来すべての「深い RL」ビルド — DDQN、Rainbow、Dueling、分布、R2D2、Agent57 — この 3 つ トリック ベースの上にスタックされています。

## コンセプト

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目的。** DQN はニューラル Q 関数での 1 ステップ TD 損失を最小化:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 勾配降下で毎ステップ更新されるオンライン ネットワーク。`θ^-` = ターゲット ネットワーク、定期的に `θ` からコピー (約 10,000 ステップごと)。`D` = 過去の遷移のリプレイ バッファ。

**3 つのトリック、重要性の順序で:**

**経験リプレイ。** `~10⁶` の遷移のリング バッファ。各訓練ステップは均一にランダムにミニバッチをサンプル。これは時間的相関を破り (連続フレームはほぼ同一)、ネットワークが希な報酬遷移を多くの回実行して学び、連続した勾配更新を相関なくします。このなし Atari でのオンポリシー TD ニューラル ネットは分岐します。

**ターゲット ネットワーク。** 同じネットワーク `Q(·; θ)` をベルマン方程式の両側で使用すると、ターゲットは毎回更新移動します — 「自分の尾を追う」。修正: 凍結された重みを持つ 2 番目のネットワーク `Q(·; θ^-)` を保つ。毎 `C` ステップ、`θ → θ^-` をコピーします。これは数千の勾配ステップ回の時間の回帰ターゲットを安定させます。ソフト更新 `θ^- ← τ θ + (1-τ) θ^-` (DDPG、SAC で使用) はより滑らかな変種です。

**報酬クリッピング。** Atari 報酬の大きさは 1 から 1000+ に変わります。`{-1, 0, +1}` にクリッピング任意の単一ゲームから勾配を支配するのを停止します。報酬の大きさが重要な場合は間違い。Atari サインのみが重要な場合は罰金。

**Double DQN。** Hasselt (2016) は最大バイアスを修正: オンライン ネットを使用してアクション *を選択します* 、ターゲット ネットを *評価します*。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

ドロップイン交換、一貫してより良い。デフォルトで使用します。

**その他の改善 (Rainbow、2017):** 優先レプレイ (高 TD エラー遷移をより多くサンプル)、決闘アーキテクチャ (別の `V(s)` と利点ヘッド)、ノイズ ネットワーク (学習探索)、n ステップ リターン、分布 Q (C51/QR-DQN)、マルチステップ ブートストラップ。各いくつかのパーセントを追加します。ゲインはほぼ加算です。

## ビルド

ここのコードは stdlib のみ numpy なし — おもちゃの連続 GridWorld で手巻き単一隠れレイヤー MLP を使用します。すべての訓練ステップはマイクロ秒で実行します。アルゴリズムはスケールで Atari DQN と同じです。

### ステップ 1: リプレイ バッファ

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari の ~50,000 容量。5,000 はおもちゃ env に十分です。

### ステップ 2: 小さな Q ネットワーク (手動 MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

フォワード パス: リニアー → ReLU → リニアー。ネット全体です。

### ステップ 3: DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状は Lesson 04 からの Q-learning で 2 つの違いです: (a) テーブルをインデックスする代わりに微分可能 `Q(·; θ)` を通じてバックプロップします、(b) ターゲットは `Q(·; θ^-)` を使用します。

### ステップ 4: 外側ループ

各エピソード、act ε-greedy オンラインで `Q(·; θ)`、バッファに遷移をプッシュ、ミニバッチをサンプル、勾配ステップを実行、定期的に同期 `θ^- ← θ`。パターン:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16 次元 1 ホット状態を持つおもちゃ GridWorld で、エージェントは約 500 エピソードでほぼ最適な方策を学びます。Atari では、200M フレームにスケーリングし、CNN 特徴抽出器を追加します。

## 落とし穴

- **致命的な 3 つ。** 関数近似 + オフポリシー + ブートストラップは分岐できます。DQN はターゲット ネット + リプレイで軽減します。どちらも削除しないでください。
- **探索。** `ε` は減衰する必要があります。通常は最初の訓練の ~10% で 1.0 から 0.01 へ。十分な早期探索がないと Q ネットはローカル盆地に収束します。
- **過大推定。** ノイズ Q を超える `max` は上向きにバイアスします。本番では常に Double DQN を使用します。
- **報酬スケール。** クリップまたはリターンを正規化します。勾配の大きさはレーニング報酬の大きさに比例します。
- **リプレイ バッファ コールドスタート。** バッファが数千の遷移を持つまでトレーニングしないでください。~20 サンプルの早期勾配は過学習します。
- **ターゲット同期頻度。** 頻繁すぎる ≈ ターゲット ネットなし。頻繁すぎない ≈ 古いターゲット。Atari DQN は 10,000 env ステップを使用します。経験則: 訓練地平線の ~1/100 ごとに同期します。
- **観測の前処理。** Atari DQN はスタック フレーム 4 を状態マルコフにします。速度情報を持つすべての env は、フレーム スタッキングまたは再帰状態が必要です。

## 使用

2026 年では、DQN はめったに最新技術ではありませんが、オフポリシー アルゴリズムの参照のままです:

| タスク | 選択のメソッド | なぜ DQN ではなく? |
|------|------------|-----------|
| 離散アクション Atari のような | Rainbow DQN または Muesli | 同じフレームワーク、より多くのトリック |
| 連続制御 | SAC / TD3 (Phase 9 · 07) | DQN はポリシー ネットワークを持たない |
| オンポリシー / 高スループット | PPO (Phase 9 · 08) | リプレイ バッファなし。スケールが簡単 |
| オフライン RL | CQL / IQL / Decision Transformer | 保守的 Q ターゲット、ブートストラップ爆発なし |
| 大規模な離散アクション スペース (推奨者) | アクション埋め込みを持つ DQN、または IMPALA | 罰金。装飾が重要 |
| LLM RL | PPO / GRPO | シーケンス レベル、ステップ レベルなし。別損失 |

レッスンはまだ移動します。リプレイとターゲット ネットワークは SAC、TD3、DDPG、SAC-X、AlphaZero の自己再生バッファ、およびすべてのオフライン RL メソッドに表示されます。報酬クリッピングは PPO の利点正規化として生存します。アーキテクチャはブループリントです。

## シップ

`outputs/skill-dqn-trainer.md` として保存:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 演習

1. **簡単。** `code/main.py` を実行します。エピソード ごとのリターン曲線をプロット。実行平均が -10 を超えるまで何エピソード?
2. **中程度。** ターゲット ネットワークを無効にします (ベルマン ターゲットの両側にオンライン ネットを使用)。訓練の不安定性を測定 — リターンが振動または分岐?
3. **難しい。** Double DQN を追加: オンライン ネットを使用して `argmax a'` をピック、ターゲット ネットを評価します。ノイズの多い報酬 GridWorld で 1,000 エピソード後に Double DQN なし対ありで `Q(s_0, best_a)` 対真 `V*(s_0)` のバイアスを比較。

## キー用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------|--------|
| DQN | "Deep Q-learning" | リプレイ バッファとターゲット ネットワークを持つニューラル Q 関数を持つ Q-learning |
| Experience replay | "Shuffled transitions" | リング バッファは各勾配ステップで均一にサンプル。データを相関なくします |
| Target network | "Frozen bootstrap" | ベルマン ターゲットで使用される Q の定期的コピー。訓練を安定させます |
| Deadly triad | "Why RL diverges" | 関数近似 + ブートストラップ + オフポリシー = 収束保証なし |
| Double DQN | "Fix for maximization bias" | オンライン ネットはアクションを選択、ターゲット ネットはそれを評価 |
| Dueling DQN | "V and A heads" | Q = V + A - mean(A) に分解。同じ出力、より良い勾配フロー |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + 1 つの分布 |
| PER | "Prioritized Replay" | TD エラーの大きさに比例してトランジションをサンプル |

## 参考文献

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 深い RL を始めた 2013 NeurIPS ワークショップ ペーパー。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature ペーパー、49 ゲーム DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 決闘 DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — スタック トリック ペーパー。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 明確な最新解説。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 「致命的な 3 つ」(関数近似 + ブートストラップ + オフポリシー) DQN のターゲット ネットワークとリプレイ バッファの教科書扱いとなる馴化を設計した。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 参照単一ファイル DQN はアブレーション研究で使用。このレッスンのゼロから始めたバージョンと一緒に読む価値あり。
