# テキストのための CNN と RNN

> 畳み込みは n-gram を学習する。再帰は記憶する。両方ともアテンションに取って代わられた。両方とも制約のあるハードウェアでは今なお重要だ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 3 · 11 (PyTorch Intro)、Phase 5 · 03 (Word Embeddings)、Phase 4 · 02 (Convolutions from Scratch)
**所要時間:** 約75分

## 問題

TF-IDF と Word2Vec は語順を無視した平坦なベクトルを生成した。それらの上に構築された分類器は `dog bites man` を `man bites dog` と区別できなかった。語順は時にシグナルを担う。

トランスフォーマー登場以前、その隙間を埋めたアーキテクチャの系統が 2 つある。

**テキスト用畳み込みネット (TextCNN)。** 単語埋め込みの系列に対して 1D 畳み込みを適用する。幅 3 のフィルタは学習可能なトライグラム検出器だ。3 つの語にまたがってスコアを出力する。異なる幅 (2、3、4、5) を積み重ねてマルチスケールパターンを検出する。最大プーリングで固定サイズの表現にする。平坦で、並列で、高速だ。

**再帰ネット (RNN、LSTM、GRU)。** トークンを一度に一つずつ処理し、情報を前方に運ぶ隠れ状態を維持する。逐次的で、記憶を持ち、入力長が柔軟だ。2014 年から 2017 年まで系列モデリングを支配し、その後アテンションが起きた。

このレッスンでは両方を構築し、その後アテンションを動機づけた失敗に名前を付ける。

## コンセプト

**TextCNN** (Kim、2014)。トークンが埋め込まれる。幅 `k` の 1D 畳み込みが埋め込みの連続する `k`-gram にわたってフィルタをスライドさせ、特徴マップを生成する。そのマップ上のグローバル最大プーリングが最も強い活性を選ぶ。複数のフィルタ幅からの最大プーリング出力を連結する。分類ヘッドに渡す。

なぜ機能するか。フィルタは学習可能な n-gram だ。最大プーリングは位置不変なので、"not good" はレビューの先頭でも途中でも同じ特徴を発火させる。各 100 フィルタの 3 つのフィルタ幅で、300 個の学習された n-gram 検出器が得られる。訓練は並列で、逐次依存はない。

**RNN。** 各時間ステップ `t` で、隠れ状態は `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` を時間にわたって共有する。時刻 `T` の隠れ状態はプレフィックス全体の要約だ。分類では `h_1 ... h_T` にわたってプーリングする (最大、平均、または最後)。

素の RNN は勾配消失に苦しむ。**LSTM** は、何を忘れ、何を保存し、何を出力するかを決めるゲートを追加し、長い系列を通じて勾配を安定化させる。**GRU** は LSTM を 2 つのゲートに簡略化する。より少ないパラメータで同等の性能を発揮する。

**双方向 RNN** は一方の RNN を前向きに、もう一方を後ろ向きに走らせ、隠れ状態を連結する。すべてのトークンの表現が左右両方の文脈を見る。タグ付けタスクに不可欠だ。

## ビルド

### ステップ 1: PyTorch での TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` は `[batch, seq_len, embed_dim]` を `[batch, embed_dim, seq_len]` に整形する。`nn.Conv1d` は中央の軸をチャネルとして扱うからだ。プーリング出力は入力長によらず固定サイズだ。

### ステップ 2: LSTM 分類器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

最後の状態のプーリングではなく、系列にわたる最大プーリングを使う。分類では、最大プーリングが通常最後の隠れ状態を取るより優れている。長い系列の末尾の情報が最後の状態を支配しがちだからだ。

### ステップ 3: 勾配消失のデモ (直観)

ゲートのない素の RNN は長距離依存を学習できない。おもちゃのタスクを考えよう。トークン `A` が系列のどこかに現れたかを予測する。`A` が位置 1 にあって系列が 100 トークン長なら、損失からの勾配は再帰重みの 99 回の掛け算を逆向きに流れねばならない。重みが 1 未満なら勾配は消失する。1 より大きければ爆発する。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM は、加算的な相互作用だけでネットワークを通る**セル状態 (cell state)** でこれを修正する (忘却ゲートが乗算的にスケールするが、勾配は依然「ハイウェイ」に沿って流れる)。GRU はより少ないパラメータで同様のことをする。両方とも 100 ステップ以上の系列を通じて安定した訓練を提供する。

### ステップ 4: なぜこれでもまだ不十分だったか

LSTM でも 3 つの問題が残った。

1. **逐次的ボトルネック。** 長さ 1000 の系列で RNN を訓練するには 1000 回の直列の順伝播/逆伝播ステップが必要だ。時間方向に並列化できない。
2. **エンコーダ・デコーダ構成での固定サイズコンテキストベクトル。** デコーダはエンコーダの最終隠れ状態だけを見る。これは入力全体にわたって圧縮されている。長い入力は詳細を失う。レッスン 09 がこれを直接扱う。
3. **遠距離依存の精度の天井。** LSTM は素の RNN を上回るが、200 ステップ以上にわたって特定の情報を伝播させるのに依然として苦労する。

アテンションは 3 つすべてを解決した。トランスフォーマーは再帰を完全に捨てた。レッスン 10 が転換点だ。

## 使う

PyTorch の `nn.LSTM`、`nn.GRU`、`nn.Conv1d` は本番対応だ。訓練コードは標準的だ。

Hugging Face は入力層として差し込める事前学習済み埋め込みを同梱している。

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

制約に合うときに使うチェックリスト。

- **エッジ / オンデバイス推論。** GloVe 埋め込みつきの TextCNN はトランスフォーマーの 10〜100 倍小さい。デプロイ対象が携帯電話なら、これがそのスタックだ。
- **ストリーミング / オンライン分類。** RNN はトークンを一度に一つずつ処理する。トランスフォーマーは完全な系列を必要とする。リアルタイムで入ってくるテキストには、LSTM が今なお勝つ。
- **ベースライン用の小型モデル。** 新しいタスクでの高速な反復。CPU 上で 5 分で TextCNN を訓練できる。
- **限られたデータでの系列ラベリング。** BiLSTM-CRF (レッスン 06) は 1k〜10k のラベル付き文に対して今なお本番品質の NER アーキテクチャだ。

それ以外はすべてトランスフォーマーへ。

## 出荷

`outputs/prompt-text-encoder-picker.md` として保存する。

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 演習

1. **易しい。** 3 クラスのおもちゃデータセット (データは自分で作る) で TextCNN を訓練せよ。フィルタ幅 (2、3、4) が単一幅 (3) を平均 F1 で上回ることを検証せよ。
2. **中級。** LSTM 分類器に対して最大プーリング、平均プーリング、最後の状態プーリングを実装せよ。小さなデータセットで比較し、どのプーリングが勝つかを文書化し、その理由を仮説立てよ。
3. **難しい。** BiLSTM-CRF NER タガーを構築せよ (レッスン 06 とこれを組み合わせる)。CoNLL-2003 で訓練せよ。レッスン 06 の CRF 単独ベースラインおよび BERT のファインチューニングと比較せよ。訓練時間、メモリ、F1 を報告せよ。

## 重要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| TextCNN | テキスト用 CNN | 単語埋め込みに対する 1D 畳み込みのスタックとグローバル最大プーリング。Kim (2014)。 |
| RNN | 再帰ネット | 各時間ステップで隠れ状態を更新: `h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | ゲート付き RNN | 入力 / 忘却 / 出力ゲート + セル状態を追加。長い系列を通じて安定して訓練する。 |
| GRU | より単純な LSTM | 3 つではなく 2 つのゲート。同等の精度、より少ないパラメータ。 |
| 双方向 | 両方向 | 前向き + 後ろ向き RNN を連結。すべてのトークンが文脈の両側を見る。 |
| 勾配消失 | 訓練シグナルが死ぬ | 素の RNN での 1 未満の重みの繰り返し乗算により、初期ステップの勾配が事実上ゼロになる。 |

## 参考文献

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 論文。8 ページ。読みやすい。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 論文。意外なほど明快。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTM を誰にとっても理解しやすくした図解。
