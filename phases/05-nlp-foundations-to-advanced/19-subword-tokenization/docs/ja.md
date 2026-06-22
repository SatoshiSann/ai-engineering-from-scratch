# サブワードトークン化 — BPE、WordPiece、Unigram、SentencePiece

> 単語トークナイザーは未見の単語で詰まる。文字トークナイザーは系列長を爆発させる。サブワードトークナイザーは両者の中間を取る。現代のあらゆる LLM はそのいずれかの上に出荷される。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 5 · 01 (テキスト処理)、Phase 5 · 04 (GloVe / FastText / サブワード)
**所要時間:** 約60分

## 問題

あなたの語彙は50,000語を持つ。ユーザーが「untokenizable」と入力する。あなたのトークナイザーは `[UNK]` を返す。モデルはもうその単語についての信号を持たない。さらに悪いことに、あなたのコーパスの90パーセンタイルの文書には40の稀な単語があり、これは文書あたり40ビットの情報が失われることを意味する。

サブワードトークン化はこれを解決する。よくある単語は単一のトークンのままである。稀な単語は意味のある断片へ分解される。`untokenizable` → `un`、`token`、`izable`。任意の文字列は究極的にはバイトの系列であるため、訓練データはすべてをカバーする。

2026年のあらゆるフロンティア LLM は、3つのアルゴリズム (BPE、Unigram、WordPiece) のいずれかの上に出荷され、3つのライブラリ (tiktoken、SentencePiece、HF Tokenizers) のいずれかに包まれている。1つを選ばずに言語モデルを出荷することはできない。

## コンセプト

![BPE 対 Unigram 対 WordPiece、文字ごと](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding)。** 文字レベルの語彙から始める。隣接するすべてのペアを数える。最も頻繁なペアを新しいトークンへマージする。目標の語彙サイズに達するまで繰り返す。支配的なアルゴリズム: GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**バイトレベル BPE。** 同じアルゴリズムだが、Unicode 文字の代わりに生のバイト (256個の基本トークン) を対象とする。`[UNK]` トークンがゼロであることを保証する。どんなバイト系列もエンコードできる。GPT-2 は50,257トークンを使う (256バイト + 50,000マージ + 1個の特殊トークン)。

**Unigram。** 巨大な語彙から始める。各トークンにユニグラム確率を割り当てる。コーパスの対数尤度の増加が最も小さいトークンを反復的に枝刈りする。推論時に確率的: トークン化をサンプリングできる (サブワード正則化によるデータ拡張に有用)。T5、mBART、ALBERT、XLNet、Gemma で使われる。

**WordPiece。** 生の頻度ではなく、訓練コーパスの尤度を最大化するペアをマージする。BERT、DistilBERT、ELECTRA で使われる。

**SentencePiece 対 tiktoken。** SentencePiece は、生の Unicode テキストに対して直接語彙 (BPE または Unigram) を*訓練する*ライブラリで、空白を `▁` としてエンコードする。tiktoken は OpenAI の、事前構築された語彙に対する高速な*エンコーダ*である。訓練はしない。

経験則:

- **新しい語彙を訓練する場合:** SentencePiece (多言語、事前トークン化なし) または HF Tokenizers。
- **GPT 語彙に対する高速な推論:** tiktoken (cl100k_base、o200k_base)。
- **両方:** HF Tokenizers — 1つのライブラリで訓練 + サービング。

## 作ってみよう

### ステップ1: ゼロから BPE

`code/main.py` を参照。そのループ:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

アルゴリズムがエンコードする3つの事実。`</w>` は単語末を示すので、「low」(接尾辞) と「lower」(接頭辞) は区別されたままになる。頻度の重み付けにより、高頻度のペアが早く勝つ。マージリストは順序付けられている。推論は訓練時の順序でマージを適用する。

### ステップ2: 学習したマージでエンコードする

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

ナイーブな O(n·|merges|)。本番の実装 (tiktoken、HF Tokenizers) は、優先度キューを用いたマージランクの参照を使い、ほぼ線形時間で動く。

### ステップ3: 実践での SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注目すべき点: 事前トークン化は不要、空白は `▁` としてエンコードされる、`character_coverage` は稀な文字をどれだけ積極的に保持するか、それとも `<unk>` にマッピングするかを制御する。

### ステップ4: OpenAI 互換の語彙のための tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

エンコードのみ。高速 (Rust バックエンド)。バイト数のカウント、コスト見積もり、コンテキストウィンドウの予算管理のための GPT-4/5 トークン化と完全に一致する。

## 2026年も今なお出荷される落とし穴

- **トークナイザーのドリフト。** 語彙 A で訓練し、語彙 B に対してデプロイする。トークン ID が異なる。モデルはゴミを出力する。CI で `tokenizer.json` のハッシュをチェックすること。
- **空白の曖昧さ。** BPE では「hello」と「 hello」は異なるトークンを生む。常に `add_special_tokens` と `add_prefix_space` を明示的に指定すること。
- **多言語の訓練不足。** 英語に偏ったコーパスは、非ラテン文字を5〜10倍多くのトークンに分割する語彙を生む。同じプロンプトが、GPT-3.5 では日本語/アラビア語で5〜10倍のコストになる。o200k_base はこれを部分的に修正した。
- **絵文字の分割。** 1つの絵文字が5トークンを取ることがある。コンテキストの予算管理時には絵文字の扱いをチェックポイントすること。

## 使ってみよう

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| 単言語モデルをゼロから訓練する | HF Tokenizers (BPE) |
| 多言語モデルを訓練する | SentencePiece (Unigram、`character_coverage=0.9995`) |
| OpenAI 互換の API をサービングする | tiktoken (GPT-4+ には `o200k_base`) |
| ドメイン固有の語彙 (コード、数学、タンパク質) | ドメインコーパスでカスタム BPE を訓練し、ベース語彙とマージする |
| エッジ推論、小さなモデル | Unigram (小さな語彙がよりよく機能する) |

語彙サイズは定数ではなくスケーリングの判断である。大まかな経験則: 10億未満のパラメータには32k、1〜100億には50〜100k、多言語/フロンティアには200k以上。

## 仕上げよう

`outputs/skill-bpe-vs-wordpiece.md` として保存する:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 演習

1. **易しい。** `code/main.py` の小さなコーパスで500マージの BPE を訓練する。3つのホールドアウト単語をエンコードする。ちょうど1トークンになったものと、2トークン以上になったものはいくつか。
2. **普通。** 100文の英語 Wikipedia の文に対して、`cl100k_base`、`o200k_base`、そして vocab=32k で訓練した SentencePiece BPE のトークン数を比較する。それぞれの圧縮率を報告する。
3. **難しい。** 同じコーパスを BPE、Unigram、WordPiece で訓練する。小さな感情分類器でそれぞれを使ったときの下流タスクの精度を測定する。その選択は F1 で1ポイントを超えて結果を動かすか。

## 重要な用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 目標の語彙サイズに達するまで、最も頻繁な文字ペアを貪欲にマージする。 |
| バイトレベル BPE | 未知トークンが決して出ない | 生の256バイトに対する BPE。GPT-2 / Llama がこれを使う。 |
| Unigram | 確率的なトークナイザー | 対数尤度を使って大きな候補集合から枝刈りする。T5、Gemma が使う。 |
| SentencePiece | 空白を扱うもの | 生のテキストに対して BPE/Unigram を訓練するライブラリ。空白は `▁` としてエンコードされる。 |
| tiktoken | 高速なもの | 事前構築された語彙のための OpenAI の Rust バックエンドの BPE エンコーダ。訓練なし。 |
| マージリスト | 魔法の数字 | 順序付けられた `(a, b) → ab` マージのリスト。推論は順序どおりに適用する。 |
| 文字カバレッジ | どれだけ稀なら稀すぎるか? | トークナイザーがカバーしなければならない訓練コーパス中の文字の割合。約0.9995が典型的。 |

## 参考文献

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE の論文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram の論文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — ライブラリ。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 簡潔なリファレンス。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — クックブック + エンコーディングリスト。
