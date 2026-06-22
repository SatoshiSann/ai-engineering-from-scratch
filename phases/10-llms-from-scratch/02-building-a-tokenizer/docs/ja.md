# ゼロからトークナイザーを構築する

> レッスン 01 はおもちゃを与えた。このレッスンは武器を与える。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 10、レッスン 01 (トークナイザー: BPE、WordPiece、SentencePiece)
**所要時間:** ~90分

## 学習目標

- Unicode、ホワイトスペース正規化、および特殊トークンを処理する本番グレードの BPE トークナイザーを構築する
- バイトレベルのフォールバックを実装して、未知のトークンなしで (絵文字、CJK、コードを含む) 任意の入力をエンコードできるようにする
- テキストをワード境界で分割する事前トークナイザー正規表現パターンを追加して、BPE マージが適用される前に
- カスタム トークナイザーをコーパスでトレーニングし、多言語テキストで tiktoken に対する圧縮率を評価する

## 問題

レッスン 01 の BPE トークナイザーは英語テキストで機能します。今、日本語をそこに投げて。または絵文字。またはタブとスペースが混在した Python コード。

破綻します。

BPE が間違っているわけではありません — 実装が不完全であるためです。本番トークナイザーは、任意のエンコーディングで生のバイトを処理し、分割前に Unicode を正規化し、マージされないことのない特殊トークンを管理し、事前トークナイザーをサブワード分割とチェーンし、15 兆のトークンを処理するトレーニング パイプラインをボトルネックにしないほど高速でこれすべてを行います。

GPT-2 のトークナイザーは 50,257 トークンを持つ。Llama 3 は 128,256 を持つ。GPT-4 はおよそ 100,000 を持つ。これらはおもちゃの数字ではありません。それらの語彙の背後にあるマージテーブルは、数百ギガバイトのテキストで訓練されました。周囲の機械 — 正規化、事前トークナイザー、特殊トークン注入、チャット テンプレート フォーマット — が、"hello world" を処理するトークナイザーをインターネット全体を処理するものと区別するものです。

その機械を構築するつもりです。

## コンセプト

### フルパイプライン

本番トークナイザーは 1 つのアルゴリズムではありません。異なる問題を解く 5 つのステージのパイプラインです。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

各ステージには特定の仕事があります:

| ステージ | 何をするか | なぜ重要か |
|-------|-------------|----------------|
| 正規化 | NFKC Unicode、小文字オプション、アクセント削除オプション | "fi" リガチャ (U+FB01) は "fi" (2 文字) になります。これなしで、同じ単語は異なるトークンを取得します。 |
| 事前トークナイザー | BPE の前にテキストをチャンクに分割 | BPE がワード境界を越えてマージするのを防ぎます。"the cat" は決して "e c" トークンを生成することはありませんでした。 |
| BPE マージ | 学習されたマージ規則をバイト シーケンスに適用 | コアの圧縮。生のバイトをサブワード トークンに変えます。 |
| 特殊トークン | [BOS]、[EOS]、[PAD]、チャット テンプレート マーカーを注入 | これらのトークンは固定 ID を持つ。BPE マージに参加することはありません。モデルは構造に必要です。 |
| ID マッピング | トークン文字列を整数 ID に変換 | モデルは文字列ではなく整数を見ます。 |

### バイトレベル BPE

レッスン 01 のトークナイザーは UTF-8 バイトで動作します。それが正しい呼び出しでした。しかし、私たちはスキップしたこと重要なことがあります: これらのバイトが有効な UTF-8 でない場合はどうなりますか?

バイトレベル BPE はすべての可能なバイト値 (0-255) を有効なトークンとして扱うことによってこれを解決します。ベース語彙は正確に 256 エントリです。任意のファイル — テキスト、バイナリ、破損した — 未知のトークンを生成することなくトークン化できます。

GPT-2 がトリック を追加: 各バイトを印刷可能な Unicode 文字にマップして、語彙が人間が読める状態に保つ。バイト 0x20 (スペース) は彼らのマッピングで文字 "G" になります。これは純粋に化粧品です。アルゴリズムは気にしません。

本当の力: バイトレベル BPE は地球上のすべての言語を処理します。中国語の文字は 3 UTF-8 バイトです。日本語は 3-4 バイトかもしれません。アラビア語、デーバナーガリー、絵文字 — すべてはバイト シーケンスです。BPE アルゴリズムは、これらのバイト シーケンスのパターンを、英語 ASCII バイトのパターンを見つけるのと同じ方法で見つけます。

### 事前トークナイザー

BPE がテキストに触れる前に、チャンクに分割する必要があります。これにより、マージ アルゴリズムがワード境界を越えるトークンを作成するのを防ぎます。

GPT-2 は正規表現パターンを使用してテキストを分割します:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

このパターンは縮約形 ("don't" は "don" + "'t" になる)、オプションの先頭スペース付きの単語、数字、句読点、ホワイトスペースで分割。先頭スペースは単語に添付されたままです — "the cat" は ["the", " ", "cat"] ではなく [" the", " cat"] になります。

Llama は SentencePiece を使用し、正規表現を完全にスキップします。生のバイト ストリームを 1 つの長いシーケンスとして扱い、BPE アルゴリズムに境界を把握させます。これはシンプルですが、クロスワード トークンを作成する BPE により多くの自由を与えます。

選択は重要です。GPT-2 の正規表現は、トークナイザーが 1 つの単語の最後の "the" と次の単語の開始の "the" をマージすべきであることを学習するのを防ぎます。SentencePiece はそれを許可し、時々より効率的な圧縮を生成しますが、解釈が難しいトークンです。

### 特殊トークン

あらゆる本番トークナイザーは構造マーカーの token ID を予約します:

| トークン | 目的 | 使用済み |
|-------|---------|---------|
| `[BOS]` / `<s>` | シーケンスの開始 | Llama 3、GPT |
| `[EOS]` / `</s>` | シーケンスの終了 | すべてのモデル |
| `[PAD]` | バッチ調整のためのパディング | BERT、T5 |
| `[UNK]` | 未知のトークン (バイトレベル BPE はこれを排除) | BERT、WordPiece |
| `<\|im_start\|>` | チャット メッセージ境界開始 | ChatGPT、Qwen |
| `<\|im_end\|>` | チャット メッセージ境界終了 | ChatGPT、Qwen |
| `<\|user\|>` | ユーザー ターン マーカー | Llama 3 |
| `<\|assistant\|>` | アシスタント ターン マーカー | Llama 3 |

特殊トークンは BPE で分割されません。マージ アルゴリズムが実行される前に正確にマッチし、固定 ID に置き換えられ、周囲のテキストは正常にトークン化されます。

### チャット テンプレート

ほとんどの人がここで混乱し、ほとんどの実装が破綻するところです。

チャット モデルにメッセージを送ると、API はメッセージのリストを受け入れます:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

モデルは JSON を見ません。それはフラット トークン シーケンスを見ます。チャット テンプレートは、特殊トークンを使用してメッセージをそのフラット シーケンスに変換します。すべてのモデルがこれを異なる方法で行います:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

テンプレートを取得して、モデルはガベージを生成します。それは 1 つの正確なフォーマットで訓練されました。任意の逸脱 — 欠けた改行、スワップされたトークン、追加スペース — が入力をトレーニング分布の外に置きます。

### スピード

Python は本番トークン化には遅すぎます。

tiktoken (OpenAI) は Rust で記述され、Python バインディングを備えています。HuggingFace トークナイザーも Rust です。SentencePiece は C++ です。これらは純粋な Python より 10-100 倍の高速化を実現します。

視点の場合: Llama 3 事前トレーニング用 15 兆トークンを 1 秒あたり 100 万トークン (高速 Python) でトークン化するのに 174 日かかります。100 秒あたり 100 万トークン (Rust) では 1.7 日かかります。

アルゴリズムを理解するため Python で構築しています。本番では、コンパイル済み実装を使用し、Python ラッパーのみに触れます。

## 構築

### ステップ 1: バイトレベル エンコーディング

基礎。任意の文字列をバイト シーケンスに変換し、各バイトを表示用の印刷可能な文字にマップし、プロセスを逆にします。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

多言語テキストでテストして、バイト カウントを確認:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" は 5 バイト。"你好" は 6 バイト (文字ごと 3)。火の絵文字は 4 バイト。バイトレベル トークナイザーは言語のことを気にしません。バイトはバイトです。

### ステップ 2: 正規表現による事前トークナイザー

GPT-2 正規表現パターンを使用してテキストをチャンクに分割。各チャンクは BPE で独立してトークン化されます。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` モジュールは Unicode プロパティエスケープをサポートしています (`\p{L}` 文字、`\p{N}` 数字)。標準ライブラリ `re` モジュールは そうではない、ASCII 文字クラスにフォール バック。本番多言語トークナイザーの場合、`regex` をインストール。

それを試します:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

先頭スペースは単語に付いたまま。縮約形はアポストロフィで分割。句読点は独自のチャンクになります。BPE はこれらの境界を越えてトークンをマージしない。

### ステップ 3: バイト シーケンスの BPE

レッスン 01 のコアアルゴリズム、しかし今は事前トークン化されたチャンクで独立して動作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### ステップ 4: 特殊トークン処理

特殊トークンは正確なマッチングと固定 ID が必要です。彼らは BPE を完全にバイパス。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### ステップ 5: フル トークナイザー クラス

すべてをチェーン: 正規化、特殊トークンで分割、事前トークナイザー、BPE マージ、ID にマップ。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### ステップ 6: 多言語テスト

本当のテスト。英語、中国語、絵文字、コードをそこに投げ。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中国語の文字は文字ごと 3 バイトを生成。絵文字は 4 バイトを生成。これらのどちらもトークナイザーをクラッシュさせません。これらのどちらも未知のトークンを生成しません。それがバイトレベル BPE の力です。

## 使用

### 本物のトークナイザーを比較

Llama 3、GPT-4、Mistral から実際のトークナイザーをロード。各は同じ多言語段落をどのように処理するかを確認。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

同じテキストで異なるトークン数が表示。128K 語彙を持つ Llama 3 は一般的なパターンをより積極的にマージします。100K を持つ GPT-4 は真ん中に座ります。32K を持つ Mistral はより多くのトークンを生成しますが、小さな埋め込みレイヤーを持ります。

トレードオフはいつも同じです: より大きな語彙はより短いシーケンスを意味しますが、より多くのパラメータです。

## 配信

このレッスンは本番トークナイザーを構築およびデバッグするためのプロンプトを生成します。`outputs/prompt-tokenizer-builder.md` を参照してください。

## 演習

1. **簡単:** トークン ID の生のバイトを表示する `get_token_bytes(id)` メソッドを追加します。それを使用して最も一般的にマージされたトークンが実際に何を表すかを検査。
2. **中程度:** ホワイトスペースと数字で分割するが先頭スペースを保つ Llama スタイルの事前トークナイザーを実装します。同じコーパスで GPT-2 正規表現アプローチと比較して語彙を比較。
3. **難しい:** `{"role": ..., "content": ...}` メッセージのリストを取得し、Llama 3 チャット フォーマットの正しいトークン シーケンスを生成するチャット テンプレート メソッドを追加。HuggingFace 実装に対してテスト。

## キーワード

| 用語 | 人々が言う | 実際の意味 |
|------|----------------|----------------------|
| バイトレベル BPE | "バイトで動作するトークナイザー" | 256 バイト値のベース語彙を持つ BPE — 未知のトークンなしで任意の入力を処理 |
| 事前トークナイザー | "BPE の前に分割" | ワード境界を越えたマージを防ぐ正規表現またはルールベースの分割 |
| NFKC 正規化 | "Unicode クリーンアップ" | 正規分解に続く互換性の組成 — "fi" リガチャは "fi" になります、フルウィド "A" は "A" になります |
| チャット テンプレート | "メッセージがトークンになる方法" | ロール/コンテンツメッセージのリストをフラット トークン シーケンスに変換するための正確なフォーマット — モデル固有で訓練フォーマットと一致する必要 |
| 特殊トークン | "コントロール トークン" | BPE をバイパスする予約済み token ID — [BOS]、[EOS]、[PAD]、チャット マーカー — マージの前に正確に一致 |
| 受胎能力 | "単語ごとのトークン" | 出力トークンから入力単語への比率 — GPT-4 で英語の場合は 1.3、韓国語の場合は 2-3、高いは浪費されたコンテキスト |
| tiktoken | "OpenAI トークナイザー" | Rust BPE 実装 Python バインディング付き — 純粋な Python より 10-100 倍高速 |
| マージ テーブル | "語彙" | トレーニング中に学習されたバイトペア マージの順序付きリスト — これはトークナイザーの学習知識です |

## 参考文献

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4 で使用される Rust BPE 実装
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- BPE、WordPiece、Unigram をサポートする Rust トークナイザー ライブラリ
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K 語彙とトークナイザー訓練の詳細
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 言語ニュートラル なトークン化
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 元のバイト対 Unicode マッピング
