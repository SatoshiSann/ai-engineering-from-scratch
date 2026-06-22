# 多言語 NLP

> 1つのモデル、100以上の言語、そしてそのほとんどに対して訓練データはゼロ。クロスリンガル転移は2020年代の実用的な奇跡である。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 5 · 04 (GloVe、FastText、サブワード)、Phase 5 · 11 (機械翻訳)
**所要時間:** 約45分

## 問題

英語には数十億のラベル付き例がある。ウルドゥー語には数千ある。マイティリー語にはほとんどない。グローバルな利用者に応える実用的な NLP システムは、タスク固有の訓練データが存在しない言語のロングテールでも機能しなければならない。

多言語モデルは、1つのモデルを多くの言語で同時に訓練することでこれを解決する。共有された表現により、モデルは高リソース言語で学んだスキルを低リソース言語へ転移できる。モデルを英語の感情分析でファインチューニングすれば、そのままウルドゥー語に対して驚くほど良い感情予測を生成する。それがゼロショットのクロスリンガル転移であり、NLP が世界へ出荷される仕方を作り変えてきた。

このレッスンでは、トレードオフ、定番のモデル、そして多言語作業に不慣れなチームを引っかける1つの判断、すなわち転移のためのソース言語の選択について名指しする。

## コンセプト

![共有された多言語埋め込み空間を介したクロスリンガル転移](../assets/multilingual.svg)

**共有された語彙。** 多言語モデルは、すべての対象言語のテキストで訓練された SentencePiece または WordPiece のトークナイザーを使う。語彙は共有される。同じサブワード単位が、関連する言語間で同じ形態素を表す。英語とイタリア語の `anti-` は同じトークンになる。

**共有された表現。** 多くの言語にわたるマスク言語モデリングで事前学習されたトランスフォーマーは、異なる言語の意味的に類似した文が類似した隠れ状態を生むことを学ぶ。mBERT、XLM-R、NLLB はいずれもこれを示す。英語の「cat」の埋め込みは、フランス語の「chat」やスペイン語の「gato」の近くにクラスタを形成し、文全体の埋め込みも同様である。

**ゼロショット転移。** モデルを1つの言語 (通常は英語) のラベル付きデータでファインチューニングする。推論時には、モデルがサポートする他のどの言語でも実行する。対象言語のラベルは不要である。結果は類型的に関連する言語では強く、遠い言語では弱い。

**少数ショットのファインチューニング。** 対象言語で100〜500のラベル付き例を加える。精度は分類タスクで英語ベースラインの95〜98%まで跳ね上がる。これは多言語 NLP において最も費用対効果の高い唯一のレバーである。

## モデル

| モデル | 年 | カバレッジ | 備考 |
|-------|------|----------|-------|
| mBERT | 2018 | 104言語 | Wikipedia で訓練。最初の実用的な多言語 LM。低リソースでは弱い。 |
| XLM-R | 2019 | 100言語 | CommonCrawl (Wikipedia よりはるかに大きい) で訓練。クロスリンガルのベースラインを定める。Base 270M、Large 550M。 |
| XLM-V | 2023 | 100言語 | 100万トークンの語彙 (250k に対して) を持つ XLM-R。低リソースで優れる。 |
| mT5 | 2020 | 101言語 | 多言語生成のための T5 アーキテクチャ。 |
| NLLB-200 | 2022 | 200言語 | Meta の翻訳モデル。55の低リソース言語を含む。 |
| BLOOM | 2022 | 46言語 + 13プログラミング | 多言語で訓練されたオープンな176B LLM。 |
| Aya-23 | 2024 | 23言語 | Cohere の多言語 LLM。アラビア語、ヒンディー語、スワヒリ語で強い。 |

ユースケースで選ぶ。分類は穏当なデフォルトとして XLM-R-base でうまくいく。生成タスクは、翻訳かオープン生成かによって mT5 か NLLB を求める。LLM スタイルの作業は、Aya-23 や、明示的な多言語プロンプティングを用いた Claude と組み合わせる。

## ソース言語の判断 (2026年の研究)

ほとんどのチームはファインチューニングのソースとしてデフォルトで英語を使う。最近の研究 (2026年) は、これがしばしば間違いであることを示している。

言語の類似性は、生のコーパスサイズよりも転移品質をよく予測する。スラブ語系の対象には、ドイツ語やロシア語がしばしば英語を上回る。インド語系の対象には、ヒンディー語がしばしば英語を上回る。**qWALS** 類似度指標 (2026年、World Atlas of Language Structures の特徴に基づく) がこれを定量化する。**LANGRANK** (Lin et al., ACL 2019) は別の、より早期の手法で、言語的類似性、コーパスサイズ、系統的近縁性の組み合わせから候補のソース言語をランク付けする。

実践的なルール: 対象言語に類型的に近い高リソースの親戚がいるなら、まずそれでファインチューニングを試し、それから英語ファインチューニングと比較する。

## 作ってみよう

### ステップ1: ゼロショットのクロスリンガル分類

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

1つのモデル、3つの言語、同じ API。NLI データで訓練された XLM-R は、含意 (entailment) のトリックを介して分類によく転移する。

### ステップ2: 多言語の埋め込み空間

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻訳は埋め込み空間で近くに着地する。異なる英語の文はさらに遠くに着地する。これが、クロスリンガルの検索、クラスタリング、類似度を機能させるものである。

### ステップ3: 少数ショットのファインチューニング戦略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

100〜500の対象言語の例には、`num_train_epochs=5` と `learning_rate=2e-5` が安全なデフォルトである。学習率が高すぎると多言語のアライメントが崩壊し、英語のみのモデルになってしまう。

## 実際に機能する評価

- **ホールドアウトセットでの言語ごとの精度。** 集約しない。集約はロングテールを隠す。
- **単言語ベースラインとの比較。** 十分なデータのある言語では、ゼロから訓練された単言語モデルが多言語モデルを上回ることがある。テストすること。
- **エンティティレベルのテスト。** 対象言語の固有表現。多言語モデルは、ラテン文字から遠い文字に対して弱いトークン化を持つことが多い。
- **クロスリンガルの一貫性。** 2つの言語での同じ意味は、同じ予測を生むべきである。そのギャップを測定する。

## 使ってみよう

2026年のスタック:

| タスク | 推奨 |
|-----|-------------|
| 分類、100言語 | XLM-R-base (約270M) のファインチューニング |
| ゼロショットのテキスト分類 | `joeddav/xlm-roberta-large-xnli` |
| 多言語の文埋め込み | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻訳、200言語 | `facebook/nllb-200-distilled-600M` (レッスン11を参照) |
| 生成型の多言語 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低リソース言語の NLP | XLM-V、または関連する高リソース言語でのドメイン固有のファインチューニング |

性能が重要なら、対象言語でのファインチューニングのために常に予算を確保すること。ゼロショットは出発点であって、最終的な答えではない。

### トークン化税 (低リソース言語で何が問題になるか)

多言語モデルは、すべての言語にわたって1つのトークナイザーを共有する。その語彙は、英語、フランス語、スペイン語、中国語、ドイツ語が支配的なコーパスで訓練されている。支配的なセットの外側にあるどの言語についても、3つの税が静かに積み重なる:

- **稔性 (fertility) 税。** 低リソース言語のテキストは、英語よりも単語あたりはるかに多くのトークンにトークン化される。ヒンディー語の文は、等価な英語の文の3〜5倍のトークンを必要としうる。その3〜5倍が、コンテキストウィンドウ、訓練効率、レイテンシを食い尽くす。
- **異形回復税。** あらゆるタイプミス、ダイアクリティカルマークの異形、Unicode 正規化の不一致、大文字小文字の差異が、埋め込み空間でコールドスタートの無関係な系列になる。モデルは、ネイティブスピーカーが自明とみなす正書法の対応を学べない。
- **容量あふれ税。** 税1と税2は、コンテキスト位置、層の深さ、埋め込み次元を消費する。実際の推論に残るものは、同じモデルから高リソース言語が得るものより体系的に小さい。

実践的な症状は次のとおり。あなたのモデルはヒンディー語で正常に訓練され、損失曲線は正しく見え、評価パープレキシティは妥当に見え、それでも本番の出力が微妙に間違っている。文の途中で形態論が崩壊する。稀な活用形は回復不能なままである。**壊れたトークナイザーから、データのスケールアップで抜け出すことはできない。**

緩和策: 対象言語に対して良いカバレッジを持つトークナイザーを選ぶ (XLM-V の100万トークン語彙は直接的な解決策である)。訓練の前に、ホールドアウトの対象テキストでトークン化の稔性を検証する。本当にロングテールの文字に対しては、バイトレベルのフォールバック (SentencePiece の `byte_fallback=True`、GPT-2 スタイルのバイトレベル BPE) を使い、決して OOV にならないようにする。

## 仕上げよう

`outputs/skill-multilingual-picker.md` として保存する:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 演習

1. **易しい。** 英語、フランス語、ヒンディー語、アラビア語にわたって、言語ごとに10文でゼロショット分類パイプラインを実行する。それぞれの精度を報告する。フランス語は強く、ヒンディー語はそこそこ、アラビア語はばらつくのが見られるはずである。
2. **普通。** `paraphrase-multilingual-MiniLM-L12-v2` を使って、小さな混合言語コーパスに対するクロスリンガルの検索器を作る。英語でクエリし、任意の言語の文書を取得する。recall@5 を測定する。
3. **難しい。** ヒンディー語の分類タスクについて、英語ソースとヒンディー語ソースのファインチューニングを比較する。両方の体制で、少数ショットのファインチューニングに500の対象言語の例を使う。どちらのソースがより良いヒンディー語の精度を生み、どれだけの差があるかを報告する。これは LANGRANK のテーゼをミニチュアで示したものである。

## 重要な用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| 多言語モデル | 1つのモデル、多くの言語 | 言語間で共有された語彙とパラメータ。 |
| クロスリンガル転移 | ある言語で訓練し、別の言語で実行する | ソースでファインチューニングし、対象言語のラベルなしで対象を評価する。 |
| ゼロショット | 対象言語のラベルなし | 対象言語でのファインチューニングなしの転移。 |
| 少数ショット | 小さな対象ラベル | ファインチューニングに使う100〜500の対象言語の例。 |
| mBERT | 最初の多言語 LM | Wikipedia で事前学習された104言語の BERT。 |
| XLM-R | 標準的なクロスリンガルベースライン | CommonCrawl で事前学習された100言語の RoBERTa。 |
| NLLB | Meta の200言語 MT | No Language Left Behind。55の低リソース言語を含む。 |

## 参考文献

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R の論文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — クロスリンガル転移の研究の流れを始めた分析論文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 の論文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya、Cohere の多言語 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK のソース言語論文。
