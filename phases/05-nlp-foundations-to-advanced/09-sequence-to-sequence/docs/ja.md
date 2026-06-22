# Sequence-to-Sequence モデル

> 翻訳器のふりをする 2 つの RNN。それらがぶつかるボトルネックこそ、アテンションが存在する理由だ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 08 (CNNs + RNNs for Text)、Phase 3 · 11 (PyTorch Intro)
**所要時間:** 約75分

## 問題

分類は可変長の系列を単一のラベルへ写す。翻訳は可変長の系列を別の可変長の系列へ写す。入力と出力は異なる語彙、おそらく異なる言語に属し、長さが揃う保証はない。

seq2seq アーキテクチャ (Sutskever、Vinyals、Le、2014) は、意図的にシンプルなレシピでこれを解いた。2 つの RNN。一方はソース文を読んで固定サイズのコンテキストベクトルを生成する。もう一方はそのベクトルを読んでターゲット文をトークンごとに生成する。レッスン 08 で書いたのと同じコードを、異なる形で接着しただけだ。

これは 2 つの理由で研究する価値がある。第一に、コンテキストベクトルのボトルネックは NLP において教育的に最も有用な失敗だ。それはアテンションとトランスフォーマーが得意とするすべてを動機づける。第二に、訓練レシピ (teacher forcing、scheduled sampling、推論時のビームサーチ) は、LLM を含むあらゆる現代の生成システムに今なお適用される。

## コンセプト

**エンコーダ。** ソース文を読む RNN。その最終隠れ状態が**コンテキストベクトル**だ。入力全体の固定サイズの要約だ。ソース以外は何も失わない、はずだ。

**デコーダ。** コンテキストベクトルから初期化される別の RNN。各ステップで前に生成したトークンを入力として受け取り、ターゲット語彙上の分布を生成する。サンプリングまたは argmax で次のトークンを選ぶ。それを戻して入力する。`<EOS>` トークンが生成されるか最大長に達するまで繰り返す。

**訓練:** 各デコーダステップでのクロスエントロピー損失を系列にわたって合計する。両ネットワークを通じた標準的な BPTT (通時的バックプロパゲーション)。

**Teacher forcing。** 訓練中、ステップ `t` でのデコーダの入力は、デコーダ自身の前回の予測ではなく、位置 `t-1` の*正解*トークンだ。これは訓練を安定化させる。これなしでは初期の誤りが連鎖し、モデルは決して学習しない。推論時はモデル自身の予測を使わざるを得ないので、常に訓練/推論の分布ギャップがある。そのギャップは**露出バイアス (exposure bias)** と呼ばれる。

**ボトルネック。** エンコーダがソースについて学んだすべてが、その一つのコンテキストベクトルに押し込められなければならない。長い文は詳細を失う。まれな語はぼやける。語順の入れ替え (chat noir 対 black cat) は計算ではなく記憶で扱わなければならない。

アテンション (レッスン 10) は、デコーダが最後の状態だけでなく*すべての*エンコーダ隠れ状態を見られるようにすることでこれを修正する。それがすべての売り文句だ。

## ビルド

### ステップ 1: エンコーダ

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` は形状 `[batch, seq_len, hidden_dim]` を持つ。入力位置ごとに一つの隠れ状態だ。`hidden` は形状 `[1, batch, hidden_dim]` を持つ。最終ステップだ。レッスン 08 は「分類のために outputs にわたってプーリングせよ」と言った。ここではコンテキストベクトルとして最後の隠れ状態を保持し、ステップごとの outputs は無視する。

### ステップ 2: デコーダ

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

デコーダは一度に一ステップずつ呼び出される。入力: 単一トークンのバッチと現在の隠れ状態。出力: 次のトークンの語彙ロジットと更新された隠れ状態。

### ステップ 3: teacher forcing つきの訓練ループ

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

名前を付ける価値のあるつまみが 2 つ。`ignore_index=0` はパディングトークンでの損失をスキップする。`teacher_forcing_ratio` は各ステップで真のトークン対モデルの予測を使う確率だ。1.0 (完全な teacher forcing) から始めて、露出バイアスのギャップを閉じるために訓練を通じて約 0.5 までアニーリングする。

### ステップ 4: 推論ループ (貪欲)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

貪欲デコードは各ステップで最も確率の高いトークンを選ぶ。それは脱線しうる。いったんトークンにコミットすると、それを取り消せない。**ビームサーチ**は上位 `k` 個の部分系列を生かしておき、最後に最高スコアの完全な系列を選ぶ。ビーム幅 3〜5 が標準だ。

### ステップ 5: ボトルネックの実証

おもちゃのコピータスクでモデルを訓練する。ソース `[a, b, c, d, e]`、ターゲット `[a, b, c, d, e]`。系列長を増やす。精度を観察する。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

単一の GRU 隠れ状態は 40 トークンの入力を無損失で記憶できない。情報はすべてのエンコーダステップに存在するが、デコーダは最後の状態しか見ない。アテンションがこれを直接修正する。

## 使う

PyTorch には `nn.Transformer` と `nn.LSTM` ベースの seq2seq テンプレートがある。Hugging Face の `transformers` ライブラリは、数十億トークンで訓練された完全なエンコーダ・デコーダモデル (BART、T5、mBART、NLLB) を同梱している。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

現代のエンコーダ・デコーダは RNN をトランスフォーマーに置き換えた。高レベルの形 (エンコーダ、デコーダ、トークンごとの生成) は 2014 年の seq2seq 論文と同一だ。各ブロック内部のメカニズムが異なる。

### それでも RNN ベースの seq2seq に手を伸ばすとき

新規プロジェクトではほぼない。具体的な例外:

- 入力を一度に一トークンずつ、有界なメモリで消費するストリーミング翻訳。
- トランスフォーマーのメモリコストが許容できないオンデバイステキスト生成。
- 教育。エンコーダ・デコーダのボトルネックを理解することは、なぜトランスフォーマーが勝ったのかを理解する最速の道だ。

### 露出バイアスとその緩和策

- **Scheduled sampling。** 訓練中に teacher forcing の比率をアニーリングし、モデルが自身の誤りから回復することを学習させる。
- **Minimum risk training。** トークンレベルのクロスエントロピーではなく、文レベルの BLEU スコアで訓練する。実際に欲しいものに近い。
- **強化学習によるファインチューニング。** 系列生成器をメトリクスで報酬づけする。現代の LLM の RLHF で使われる。

3 つすべてが今なおトランスフォーマーベースの生成に適用される。

## 出荷

`outputs/prompt-seq2seq-design.md` として保存する。

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 演習

1. **易しい。** おもちゃのコピータスクを実装せよ。ターゲットがソースと等しい入出力ペアで GRU seq2seq を訓練せよ。長さ 5、10、20 で精度を測定せよ。ボトルネックを再現せよ。
2. **中級。** ビーム幅 3 のビームサーチデコードを追加せよ。小さな対訳コーパスで貪欲に対する BLEU を測定せよ。ビームサーチが勝つ箇所 (通常は最後のトークン) と差がない箇所を文書化せよ。
3. **難しい。** 1 万ペアの言い換えデータセットで `facebook/bart-base` をファインチューニングせよ。ホールドアウト入力で、ファインチューニング済みモデルの beam-4 出力をベースモデルと比較せよ。BLEU を報告し、10 個の定性的な例を選べ。

## 重要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| エンコーダ | 入力 RNN | ソースを読む。ステップごとの隠れ状態と最終コンテキストベクトルを生成する。 |
| デコーダ | 出力 RNN | コンテキストベクトルから初期化。ターゲットトークンを一度に一つずつ生成する。 |
| コンテキストベクトル | 要約 | 最終エンコーダ隠れ状態。固定サイズ。アテンションが解決するボトルネック。 |
| Teacher forcing | 真のトークンを使う | 訓練時に正解の前トークンを与える。学習を安定化させる。 |
| 露出バイアス | 訓練/テストギャップ | 真のトークンで訓練されたモデルは、自身の誤りから回復する練習をしていない。 |
| ビームサーチ | より良いデコード | 貪欲にコミットする代わりに、各ステップで上位 k 個の部分系列を生かしておく。 |

## 参考文献

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 元の seq2seq 論文。4 ページ。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — GRU とエンコーダ・デコーダの枠組みを導入した。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — アテンション論文。このレッスンの直後に読むこと。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 構築可能な seq2seq + アテンションのコード。
