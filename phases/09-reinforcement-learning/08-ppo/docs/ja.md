# Proximal Policy Optimization (PPO)

> A2C は各ロールアウト後にデータを捨てるが、PPO はポリシー勾配を切り取られた重要度比率でラップしているので、ポリシーが爆発することなく同じデータで10回以上のエポック実行できる。Schulman et al. (2017)。2026年でもまだデフォルトのポリシー勾配アルゴリズムである。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 9 · 06 (REINFORCE)、Phase 9 · 07 (Actor-Critic)
**所要時間:** ~75分

## 問題

A2C (Lesson 07) はオンポリシーである。勾配 `E_{π_θ}[A · ∇ log π_θ]` は、現在の `π_θ` からサンプリングされたデータを必要とする。1回更新すると、`π_θ` が変わり、使ったデータはオフポリシーになる。それを再利用すると、勾配は偏ってしまう。

ロールアウトは高コストである。Atariでは、8つの環境×128ステップ = 1024遷移のロールアウトが数十秒の環境時間がかかる。1つの勾配ステップの後にそれを捨てるのは無駄である。

Trust Region Policy Optimization (TRPO, Schulman 2015) が最初の修正案だった。古いポリシーと新しいポリシー間のKLダイバージェンスが `δ` 以下に留まるように各更新を制約する。理論的にはきれいだが、更新ごとに共役勾配求解が必要である。2026年で誰もTRPOを実行していない。

PPO (Schulman et al. 2017) は、ハードな信頼領域制約を単純な切り取り目的関数に置き換える。1行のコード追加。ロールアウトあたり10エポック。共役勾配なし。十分な理論的保証。9年後の現在でも、MuJoCoからRLHFまで、すべてのデフォルトポリシー勾配アルゴリズムである。

## コンセプト

![PPO切り取り代替目的関数：1 ± εでの比率切り取り](../assets/ppo.svg)

**重要度比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

これはデータを収集したポリシーと比較した新しいポリシーの尤度比である。`r_t = 1` は変化がないことを意味する。`r_t = 2` は新しいポリシーが `a_t` を選ぶ確率が古いポリシーの2倍であることを意味する。

**切り取られた代替目的。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

2つの項：

- もしアドバンテージ `A_t > 0` で比率が `1 + ε` を超えて成長しようとしたら、クリップが勾配を平坦にする — 良いアクションを古い確率の上に `+ε` 以上押し上げない。
- もしアドバンテージ `A_t < 0` で比率が `1 - ε` を超えて成長しようとしたら（つまり悪いアクションをクリップ削減と比較してより可能性を高くする）、クリップが勾配を制限する — 悪いアクションを `-ε` 以下に押し下げない。

`min` は他の方向を処理する。もし比率が *有利な* 方向に移動していたら、勾配を得られる（あなたを傷つけるであろう側のクリップなし）。

典型的には `ε = 0.2`。目的を `r_t` の関数としてプロット：「良い側」に平坦な屋根を持ち「悪い側」に平坦な床を持つ区分線形関数。

**完全なPPO損失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2Cと同じアクター・クリティック構造。3つの係数、通常 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**訓練ループ。**

1. `N` 個の並列環境で `T` ステップずつ、合計 `N × T` 遷移を収集する。
2. アドバンテージ (GAE) を計算し、定数として凍結する。
3. `π_{θ_old}` を現在の `π_θ` のスナップショットとして凍結する。
4. `K` エポックについて、各ミニバッチの `(s, a, A, V_target, log π_old(a|s))` について：
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))` を計算する。
   - `L^{CLIP}` + 値損失 + エントロピーを適用する。
   - 勾配ステップ。
5. ロールアウトを破棄する。ステップ1に戻る。

`K = 10` とミニバッチサイズ 64 が標準的なハイパーパラメータセットである。PPOはロバスト：正確な数値は±50%以内ではほとんど重要でない。

**KL-ペナルティ変種。** オリジナルペーパーは適応型KLペナルティを使う代替案を提案した：`L = L^{PG} - β · KL(π_θ || π_old)` で、`β` は観測されたKLに基づいて調整される。クリッピング版が支配的になった；KL変種はRLHF（参照ポリシーへのKLが常に必要な別の制約）で生き残っている。

## 構築する

### ステップ1：ロールアウト時に `log π_old(a | s)` をキャプチャする

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

スナップショットはロールアウト時に一度取られる。それは更新エポック中に変わらない。

### ステップ2：GAEアドバンテージを計算する (Lesson 07)

A2Cと同じ。バッチ全体で正規化する。

### ステップ3：切り取り代替更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

「切り取られた → ゼロ勾配」パターンはPPOの心臓である。新しいポリシーが有利な方向にすでに遠く漂ってしまったら、更新は停止する。

### ステップ4：値とエントロピー

クリティックターゲットに標準MSEを追加し、アクターにエントロピーボーナスを追加する。A2Cと同じ。

### ステップ5：診断

すべての更新で監視する3つのこと：

- **平均KL** `E[log π_old - log π_θ]`。`[0, 0.02]` に留まるべき。`0.1` を超えて爆発したら、`K_EPOCHS` または `LR` を削減する。
- **クリップ分数** — 比率が `[1-ε, 1+ε]` の外に位置するサンプルの割合。`~0.1-0.3` であるべき。`~0` なら、クリップは決して発動しない → `LR` または `K_EPOCHS` を引き上げる。`~0.5+` なら、ロールアウトをオーバーフィッティングしている → それらを下げる。
- **説明される分散** `1 - Var(V_target - V_pred) / Var(V_target)`。クリティック品質メトリック。クリティックが学習するにつれて1に向かって上昇すべき。

## 落とし穴

- **クリップ係数が誤調整されている。** `ε = 0.2` が事実上の標準。`0.1` に行くと更新が臆病になり過ぎ；`0.3+` は不安定性を招く。
- **エポックが多すぎる。** `K > 20` は定期的に不安定化する。なぜなら、ポリシーは `π_old` から遠く漂うから。エポックを制限する、特に大きなネットワークでは。
- **報酬正規化がない。** 大きな報酬スケールはクリップ範囲を食い潰す。報酬（実行標準偏差）をアドバンテージ計算前に正規化する。
- **アドバンテージ正規化を忘れる。** バッチごとのゼロ平均/単位標準偏差正規化が標準。それをスキップするとほとんどのベンチマークでPPOを破壊する。
- **学習率が減衰していない。** PPOはゼロへの線形LR減衰から利益を得る。定数LRはしばしばより悪い。
- **重要度比率数学エラー。** 数値安定性のために、常に `exp(log_new - log_old)` を使う；`new / old` ではなく。
- **勾配符号が間違っている。** 代替を最大化する = `-L^{CLIP}` を *最小化* する。フリップした符号はもっとも一般的なPPOバグである。

## 使う

PPOは2026のデフォルトRLアルゴリズムで、多くの領域を超えている：

| ユースケース | PPO変種 |
|----------|-------------|
| MuJoCo / ロボティクス制御 | ガウシアンポリシー、GAE(0.95)とのPPO |
| Atari / 離散ゲーム | カテゴリカルポリシー、128ステップロールアウトをローリングとのPPO |
| LLM用RLHF | 参照モデルへのKLペナルティ、応答終了時のRMからの報酬とのPPO |
| 大規模ゲームエージェント | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 推論LLM | GRPO (Lesson 12) — クリティックなしのPPO変種 |
| プリファレンス専用データ | DPO — PPO+KLを閉形式で一緒に潰す、オンラインサンプリングなし |

PPO *損失形状* — 切り取り代替 + 値 + エントロピー — はDPO、GRPO、ほぼすべてのRLHFパイプラインの足場である。

## 納入する

`outputs/skill-ppo-trainer.md` として保存する：

```markdown
---
name: ppo-trainer
description: 与えられた環境に対してPPO訓練コンフィグと診断プランを生成する。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

環境と訓練予算が与えられたら、出力：

1. ロールアウトサイズ。`N` 環境 × `T` ステップ。
2. 更新スケジュール。`K` エポック、ミニバッチサイズ、LRスケジュール。
3. 代替パラメータ。`ε` (クリップ)、`c_v`、`c_e`、アドバンテージ正規化オン。
4. アドバンテージ。明示的な `γ` と `λ` を持つGAE(`λ`)。
5. 診断プラン。KL、クリップ分数、説明分散しきい値とアラート。

`K > 30` または `ε > 0.3` を拒否する（危険な信頼領域）。アドバンテージ正規化またはKL/クリップ監視なしのPPO実行を拒否する。0.4以上で持続するクリップ分数をドリフトとしてフラグ。
```

## 演習

1. **簡単。** 4×4 GridWorldで `ε=0.2, K=4` を使ってPPOを実行。マッチした環境ステップでA2C（ロールアウトごと1エポック）へのサンプル効率を比較。
2. **中程度。** `K ∈ {1, 4, 10, 30}` をスイープ。環境ステップと平均KLを対リターングラフで追跡。この課題でKLが爆発する `K` は？
3. **難しい。** 切り取り代替を適応型KLペナルティに置き換える（`KL > 2·target` なら `β` を2倍に、`KL < target/2` なら半分に）。最終リターン、安定性、クリップフリーネスを比較。

## 重要用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 重要度比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；データを収集したポリシーからの偏差。 |
| 切り取られた代替 | "PPOの主要なトリック" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；有利側のクリップを超えた平坦勾配。 |
| 信頼領域 | "TRPO / PPO意図" | 単調改善を保証するために、各更新のKLを制限。 |
| KLペナルティ | "ソフト信頼領域" | 代替PPO：`L - β · KL(π_θ \|\| π_old)`。適応型 `β`。 |
| クリップ分数 | "クリップがいつ発動するか" | 診断 — 0.1-0.3であるべき；外側は誤調整を意味する。 |
| マルチエポック訓練 | "データ再利用" | 各ロールアウトで K エポック；分散コストをサンプル効率とトレード。 |
| オンポリシーっぽい | "ほぼオンポリシー" | PPOは名目上オンポリシーだが K>1 エポックは若干オフポリシーデータを安全に使う。 |
| PPO-KL | "もう一つのPPO" | KLペナルティ変種；RLHFで参照へのKLがすでに制約である場所で使用。 |

## 参考文献

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 論文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO、PPOの前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — すべてのPPOハイパーパラメータをアブレーション。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO-in-RLHFレシピ。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PyTorchでのきれいな最新の説明。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 多くの論文で使用される参照シングルファイルPPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 言語モデルのPPOの本番レシピ；Lesson 09 (RLHF) と一緒に読む。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — 「37コードレベル最適化」論文；PPOのどのトリックが負荷ベアリングで、どれが民俗学。
