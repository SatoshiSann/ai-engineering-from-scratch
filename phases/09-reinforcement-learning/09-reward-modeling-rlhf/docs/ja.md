# 報酬モデリング & RLHF

> 人間は「良い助手応答」の報酬関数を書くことができないが、2つの応答を比較してより良い方を選ぶことはできる。その比較に報酬モデルをフィットさせて、その報酬に対して言語モデルをRLする。Christiano 2017。InstructGPT 2022。GPT-3をChatGPTに変えたレシピ。2026年ではほぼDPOに置き換えられているが、心のモデルは留まる。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 05 (Sentiment)、Phase 9 · 08 (PPO)
**所要時間:** ~45分

## 問題

次トークン予測目的で言語モデルを訓練した。文法的な英語を書く。また嘘をつき、延々と話し、拒否を拒否する。これ以上の事前訓練では修正できない — ウェブテキストが問題であり、解決ではない。

あなたは「応答Aは指示Xに対して応答Bより良い」と言うスカラー報酬が欲しい。その報酬関数を手で書くことは不可能。「有益さ」はトークン上の閉形式表現ではない。しかし、人間は2つの出力を比較して優先順位を付けることができる。それは規模で安く収集できる。

RLHF (Christiano et al. 2017; Ouyang et al. 2022) は優先順位を報酬モデルに変換し、その報酬に対してLMをPPOで最適化する。3つのステップ：SFT → RM → PPO。これはChatGPT、Claude、Gemini、そして2023-2025のその他すべてのアライメント済みLLMを出荷したレシピ。

2026年ではPPOステップはほぼDPO (Phase 10 · 08) で置き換えられている。なぜならそれはより安くほぼアライメント訓練に同じくらい良いから。しかし、*報酬モデル* ピースはすべてのBest-of-Nサンプラー、すべてのRL-from-verifiable-rewards パイプライン、そしてプロセス報酬モデルを使うすべての推論モデルの下にある。RLHFを理解すれば、あなたはアライメント全体のスタックを理解する。

## コンセプト

![3段階RLHF：SFT、ペアワイズ優先度のRM訓練、KLペナルティ付きPPO](../assets/rlhf.svg)

**ステージ1：教師あり微調整（SFT）。** 事前訓練されたベースモデルから開始。目的の振る舞いのヒューマンライティング例（指示に従う応答、有益な返信など）で微調整。結果：*良い行動に偏った* モデル `π_SFT` だが、依然として無制限のアクション空間がある。

**ステージ2：報酬モデル訓練。**

- 「y_+ は y_- より好まれている」とヒューマンラベル付けされたプロンプト `x` に対する応答ペア `(y_+, y_-)` を収集。
- 報酬モデル `R_φ(x, y)` を訓練して `y_+` にはより高いスコアを割り当てる。
- 損失：**Bradley-Terry ペアワイズロジスティック**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σはシグモイド。報酬の差は優先度のログオッズを意味する。BTは1952年以来標準（Bradley-Terry）であり、最新RLHFで支配的な選択肢。

- `R_φ` は通常SFTモデルから初期化され、その上にスカラーヘッドがある。同じトランスフォーマーバックボーン；単一の線形層が報酬を出力。

**ステージ3：KLペナルティ付きのRMに対するPPO。**

- 訓練可能なポリシー `π_θ` を `π_SFT` から初期化。凍結された *参照* `π_ref = π_SFT` を保つ。
- 応答 `y` の終わりでの報酬：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KLペナルティは `π_θ` が `π_SFT` から任意に漂うのを防ぐ — これは *正則化子*、ハードな信頼領域ではない。`β` は通常 `0.01`-`0.05`。
- このレポートでPPO (Lesson 08) を実行。アドバンテージはトークンレベルの軌跡で計算されるが、RMはフル応答のみスコア。

**なぜKL？** なしで、PPOは喜んで報酬ハッキング戦略を見つける — RMは配布内完成でのみ訓練された。配布外応答はヒューマンライティングより高くスコアするかもしれない。KLは `π_θ` をRMが訓練された多様体の近くに保つ。これはRLHFで最も重要な一つのノブ。

**2026状態：**

- **DPO** (Rafailov 2023)：閉形式代数がステージ2+3を優先度データ上の単一教師あり損失に潰す。RM、PPOなし。アライメントベンチマークで同じ品質で計算の一部。Phase 10 · 08で対象。
- **GRPO** (DeepSeek 2024-2025)：クリティック代わりのグループ相対ベースラインとのPPO、ヒューマン訓練RMの代わりに検証者（コード実行 / 数学答えマッチ）からの報酬。推論モデルで支配的。Phase 9 · 12で対象。
- **プロセス報酬モデル (PRMs)：** 部分解（各推論ステップ）をスコア；RLHFと推論用GRPO変種で使用。
- **Constitutional AI / RLAIF：** アライメント済みLLMを使用して人間の代わりに優先度を生成。優先度予算を拡大。

## 構築する

このレッスンは小さな総合「プロンプト」と「応答」を文字列として表現する。RMはbag-of-tokensの表現上の線形スコアラー。本物のLLMではない — パイプラインの *形状* が重要で、スケールではない。`code/main.py` を参照。

### ステップ1：総合優先度データ

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

本物のRLHFではこれはヒューマンラベラーで置き換えられる。形状 — `(prompt, preferred_response, rejected_response)` — は同一。

### ステップ2：Bradley-Terry報酬モデル

線形スコア：`R(x, y) = w · bag(y)`。BT ペアワイズログロスを最小化するように訓練：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

数百更新後、`w` は良い単語トークンに正の重みを、悪いに負の重みを割り当てる。

### ステップ3：RMの上のPPOのようなポリシー

私たちのおもちゃポリシーは語彙から単一トークンを生成。RMの下でトークンをスコア、`log π_θ(token | prompt)` を計算、KL-to-reference ペナルティを追加、クリップPPO代替を適用。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### ステップ4：KLを監視する

すべての更新でメディアン `KL(π_θ || π_ref)` を追跡。`~5-10` を超えて忍び寄ったら、ポリシーは `π_SFT` から遠く漂ってしまった — `β` が上昇しているか報酬ハッキングが開始している。これは本物のRLHFで最高の診断。

### ステップ5：TRLとの本番レシピ

おもちゃパイプラインを理解したら、これは同じループが本物のライブラリユーザーが書く通り。Hugging Faceの [TRL](https://huggingface.co/docs/trl) は参照実装 — ステージ2用 `RewardTrainer` とステージ3用 `PPOTrainer` (組み込みKL-to-reference付き)。

```python
# ステージ2：ペアワイズ優先度からの報酬モデル
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# データセット行：{"prompt", "chosen", "rejected"} — Bradley-Terry形式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# ステージ3：KLペナルティ付きRMに対するPPO（SFT参照へ）
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

ライブラリがあなたのためにする3つのこと。`adap_kl_ctrl=True` は適応型β スケジュール実装：観測されたKLが `target_kl` を超えたら、β は倍になる；半分以下なら、β は半分に。参照モデルは慣例で凍結される — 誤ってパラメータを `policy` と共有しないこと。そして値ヘッドはポリシーと同じバックボーンに住む (`AutoModelForCausalLMWithValueHead` はスカラーMLP head を付ける)、なぜTRLが `policy/kl` と `value/loss` を別々に報告。

## 落とし穴

- **オーバー最適化 / 報酬ハッキング。** RMは不完全；`π_θ` は高くスコアするが悪い対抗的完成を見つける。症状：報酬は無限に上昇しながらヒューマン評価スコアはプラトーか低下。修正：早期停止、`β` 引き上げ、RM訓練データ拡大。
- **長さハッキング。** 有益応答で訓練されたRMはしばしば暗黙的に長さに報酬。ポリシーはパディング応答を学ぶ。修復：長さ正規化報酬、または長さ認識RMとのRLAIF。
- **小さすぎるRM。** RMはポリシーと少なくとも同じくらい大きい必要。小さいRMはポリシーの出力を忠実にスコアできない。
- **KL調整。** 低すぎるβ → ドリフトと報酬ハッキング。高すぎるβ → ポリシーはほぼ変わらない。標準的なトリックは、ステップごと固定KLをターゲットとする *適応型* β。
- **優先度データノイズ。** ヒューマンラベルの ~30% はノイズか曖昧。同意フィルタリングデータで、またはBTで温度を使ってRM訓練をキャリブレーション。
- **オフポリシー問題。** PPOデータは最初のエポック後やや オフポリシー。Lesson 08のようにクリップ分数を監視。

## 使う

2026年のRLHF はレイヤー状：

| レイヤー | ターゲット | 方法 |
|-------|--------|--------|
| 指示に従う、有益さ、危害なさ | アライメント | DPO (Phase 10 · 08) は RLHF-PPO より好まれる。 |
| 推論正確性（数学、コード） | 能力 | 検証者報酬を持つGRPO (Phase 9 · 12)。 |
| 長期マルチステップタスク | 代理的 | ステップ上のプロセス報酬モデルを持つPPO / GRPO。 |
| 安全性 / 拒否行動 | 安全性 | 別々の安全性RM、またはConstitutional AIを持つRLHF-PPO。 |
| 推論でBest-of-N | 高速アライメント | デコード時にRMを使う；ポリシー訓練は不要。 |
| 報酬蒸留 | 推論計算 | 凍結LMの上の小さい「報酬ヘッド」を訓練。 |

RLHF は 2022-2024の *その* 方法だった。2026年で、本番アライメント パイプラインはDPO-first、PPO-only for the RM-intensive または safety-critical ステップ。

## 納入する

`outputs/skill-rlhf-architect.md` として保存：

```markdown
---
name: rlhf-architect
description: 言語モデル用のRLHF / DPO / GRPOアライメント パイプラインを設計、RM、KL、データ戦略を含める。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

ベースLM、目標行動（アライメント / 推論 / 拒否 / エージェント）、および優先度または検証者予算が与えられたら、出力：

1. ステージ。SFT？RM？DPO？GRPO？正当化付き。
2. 優先度または検証者ソース。ヒューマン、AI フィードバック、ルールベース、ユニットテストパス、または報酬蒸留。
3. KL戦略。固定β、適応型β、またはDPO (暗黙的KL)。
4. 診断。平均KL、報酬安定性、オーバー最適化ガード（ホールドアウト ヒューマン評価）。
5. 安全性ゲート。レッドチーム設定、拒否率、有益さRMから別々の安全性RM。

KL監視なしでRLHF-PPOを出荷することを拒否。ターゲットポリシーより小さいRMを使うことを拒否。長さのみ報酬を拒否。盲目ヒューマン評価セットを保っていないパイプラインをオーバー最適化保護の欠落とフラグ。
```

## 演習

1. **簡単。** `code/main.py` のBradley-Terry報酬モデルを500総合優先度ペアで訓練。ホールドアウト100ペアでペアワイズ精度を測定。90% を超えるべき。
2. **中程度。** `β ∈ {0.0, 0.1, 1.0}` を持つおもちゃPPO-RLHFループを実行。それぞれに対して、RM スコアと更新上のKL-to-reference をプロット。どの実行が報酬ハッキング？
3. **難しい。** 同じ優先度データ上のDPO (閉形式優先度-尤度損失) を実装して、計算使用とRLHF-PPOパイプラインで達成された最終RM スコアを比較。

## 重要用語

| 用語 | 人が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| RLHF | "アライメントRL" | 3段階SFT + RM + PPOパイプライン (Christiano 2017, Ouyang 2022)。 |
| 報酬モデル (RM) | "スコアリングネット" | Bradley-Terryのペアワイズプリファレンスにフィットした学習スカラー関数。 |
| Bradley-Terry | "ペアワイズロジスティック損失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；標準RM目的。 |
| KLペナルティ | "参照の近くに留まる" | 報酬内 `β · KL(π_θ \|\| π_ref)`；報酬ハッキング防止正則化子。 |
| 報酬ハッキング | "グッドハート法則" | ポリシーはRM欠陥を悪用；症状：報酬アップ、ヒューマン評価フラット。 |
| RLAIF | "AIラベル付き優先度" | ラベルが人間の代わりに別のLMから来るRLHF。 |
| PRM | "プロセス報酬モデル" | 部分推論ステップをスコア；推論パイプラインのRLHFとGRPO変種で使用。 |
| Constitutional AI | "Anthropicの方法" | AI生成優先度は明示的ルールで導かれる。 |

## 参考文献

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — RLHFを開始した論文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPTの背後のレシピ。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 要約のための以前のRLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026でのポストRLHFデフォルト。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIFと自己批判ループ。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH論文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 本番 `RewardTrainer` と `PPOTrainer`。適応型KLと値ヘッドの詳細については訓練者ソースを読む。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) Lambert、Castricato、von Werra、Havrilla — ダイアグラム付きの3段階パイプラインの規範的なウォークスルー。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — ライブラリ；`examples/` はLlama、Mistral、QwenのエンドツーエンドRLHFスクリプト。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 報酬仮説ビュー；報酬ハッキングについて考える必須前提条件。
