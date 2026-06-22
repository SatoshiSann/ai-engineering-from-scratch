# エンティティリンキングと曖昧性解消

> NER は「Paris」を見つけた。エンティティリンキングはこう決める: フランスのパリ？ パリス・ヒルトン？ テキサス州パリス？ パリス（トロイの王子）？ リンキングなしでは、知識グラフは曖昧なままになる。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 06（NER）、Phase 5 · 24（共参照解決）
**所要時間:** 約60分

## 問題

ある文にこうある: 「Jordan beat the press（ジョーダンがプレスを破った）。」 NER は「Jordan」を PERSON とタグ付けする。よろしい。だが*どの* Jordan か？

- マイケル・ジョーダン（バスケットボール）？
- マイケル・B・ジョーダン（俳優）？
- マイケル・I・ジョーダン（バークレーの ML 教授 — そう、この混同は ML 論文で実際に起こる）？
- ヨルダン（国）？
- Jordan（ヘブライ語の名前）？

エンティティリンキング（EL）は、各言及を知識ベースの一意のエントリに解決する: Wikidata、Wikipedia、DBpedia、あるいはドメイン KB。2つのサブタスクがある。

1. **候補生成（Candidate generation）。** 「Jordan」が与えられたとき、どの KB エントリが妥当か？
2. **曖昧性解消（Disambiguation）。** コンテキストが与えられたとき、どの候補が正しいか？

どちらのステップも学習可能だ。どちらもベンチマークがある。組み合わせたパイプラインは10年間安定している — 変化するのは曖昧性解消器の品質だ。

## コンセプト

![エンティティリンキングのパイプライン: 言及 → 候補 → 曖昧性解消されたエンティティ](../assets/entity-linking.svg)

**候補生成。** 言及の表層形（「Jordan」）が与えられたら、エイリアスインデックスで候補を引く。Wikipedia のエイリアス辞書はほとんどの固有表現をカバーする: 「JFK」 → ジョン・F・ケネディ、ジャクリーン・ケネディ、JFK 空港、JFK（映画）。典型的なインデックスは言及ごとに10〜30件の候補を返す。

**曖昧性解消: 3つのアプローチ。**

1. **事前確率 + コンテキスト（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。よく機能し、高速で、訓練不要。
2. **埋め込みベース（ESS / REL / Blink）。** 言及 + コンテキストを符号化する。各候補の説明文を符号化する。最大コサインを選ぶ。2020〜2024年のデフォルト。
3. **生成型（GENRE, 2021; LLM ベース, 2023年〜）。** エンティティの正規名をトークンごとにデコードする。有効なエンティティ名のトライ木に制約されるため、出力は有効な KB id であることが保証される。

**エンドツーエンド対パイプライン。** 現代のモデル（ELQ、BLINK、ExtEnD、GENRE）は NER + 候補生成 + 曖昧性解消を1パスで実行する。コンポーネントを差し替えられるため、本番ではパイプライン型のシステムが依然として主流だ。

### 2つの測定指標

- **言及再現率（候補生成）。** 正しい KB エントリが候補リストに現れる正解言及の割合。パイプライン全体の下限。
- **曖昧性解消の精度／F1。** 正しい候補が与えられたとき、top-1 が正しい頻度。

常に両方を報告すること。候補再現率80%で曖昧性解消が99%のシステムは、80%のパイプラインだ。

## 作ってみよう

### ステップ1: Wikipedia のリダイレクトからエイリアスインデックスを構築する

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia のエイリアスデータ: 約1,800万件の (エイリアス, エンティティ) ペア。Wikidata のダンプからダウンロードする。転置インデックスとして保存する。

### ステップ2: コンテキストベースの曖昧性解消

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard オーバーラップはおもちゃだ。埋め込み上のコサイン類似度で置き換える（トランスフォーマー版は `code/main.py` のステップ2を参照）。

### ステップ3: 埋め込みベース（BLINK スタイル）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

インデックス時にすべての KB エンティティを一度埋め込む。クエリ時に言及 + コンテキストを一度埋め込み、候補プールに対してドット積を取り、最大を選ぶ。

### ステップ4: 生成型エンティティリンキング（コンセプト）

GENRE はエンティティの Wikipedia タイトルを文字ごとにデコードする。制約付きデコーディング（レッスン20参照）により、有効なタイトルのみが出力されることを保証する。KB を裏付けとするトライ木と緊密に統合される。現代の後継は REL-GEN と、構造化出力を備えた LLM プロンプトによる EL だ。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

ホワイトリスト（Outlines の `choice`）と組み合わせれば、これは2026年に出荷できる最もシンプルな EL パイプラインだ。

### ステップ5: AIDA-CoNLL で評価する

AIDA-CoNLL は標準の EL ベンチマークだ: 1,393件のロイター記事、34k件の言及、Wikipedia エンティティ。KB 内精度（`P@1`）と KB 外の NIL 検出率を報告する。

## 落とし穴

- **NIL の扱い。** 一部の言及は KB に存在しない（新興エンティティ、無名の人物）。システムは間違ったエンティティを推測する代わりに NIL を予測しなければならない。別途測定する。
- **言及境界の誤り。** 上流の NER が部分スパンを取りこぼす（「Bank of America」が単に「Bank」とタグ付けされる）。EL の再現率が下がる。
- **人気バイアス。** 訓練されたシステムは頻出エンティティを過剰に予測する。ML 論文上の「Michael I. Jordan」への言及が、バスケットボールの Jordan にリンクされることがよくある。
- **言語横断 EL。** 中国語テキストの言及を英語 Wikipedia のエンティティにマッピングする。多言語エンコーダか翻訳ステップが必要。
- **KB の陳腐化。** 新しい企業、出来事、人物は昨年の Wikipedia ダンプに存在しない。本番パイプラインには更新ループが必要だ。

## 使ってみよう

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| 汎用の英語 + Wikipedia | BLINK または REL |
| 言語横断、KB = Wikipedia | mGENRE |
| LLM 向き、1日あたり少数の言及 | 候補リスト + 制約付き JSON で Claude/GPT-4 にプロンプト |
| ドメイン固有 KB（医療、法律） | KB 対応の検索を備えたカスタム BERT + ドメインの AIDA 風データでファインチューニング |
| 極めて低レイテンシ | 完全一致の事前確率のみ（Milne-Witten ベースライン） |
| 研究の SOTA | GENRE / ExtEnD / 生成型 LLM-EL |

2026年に出荷される本番パターン: NER → 共参照 → 各言及に EL → クラスタをクラスタごとに1つの正規エンティティへ集約。出力: ドキュメント内のエンティティごとに1つの KB id（言及ごとに1つではない）。

## 出荷しよう

`outputs/skill-entity-linker.md` として保存する。

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 演習

1. **易。** `code/main.py` の事前確率+コンテキスト曖昧性解消器を、10件の曖昧な言及（Paris、Jordan、Apple）に対して実装する。正しいエンティティを手作業でラベル付けする。精度を測定する。
2. **中。** 50件の曖昧な言及を文トランスフォーマーで符号化する。各候補の説明文を埋め込む。埋め込みベースの曖昧性解消を Jaccard コンテキストオーバーラップと比較する。
3. **難。** 1kエンティティのドメイン KB（例: 自社の従業員 + 製品）を構築する。NER + EL をエンドツーエンドで実装する。100件のホールドアウト文に対して適合率と再現率を測定する。

## 重要用語

| 用語 | 一般に言われること | 実際の意味 |
|------|-----------------|-----------------------|
| エンティティリンキング（EL） | Wikipedia へのリンク | 言及を一意の KB エントリにマッピングする。 |
| 候補生成（Candidate generation） | 誰でありうるか？ | 言及に対する妥当な KB エントリの候補リストを返す。 |
| 曖昧性解消（Disambiguation） | 正しいものを選ぶ | コンテキストを用いて候補をスコア付けし、勝者を選ぶ。 |
| エイリアスインデックス（Alias index） | 引き表 | 表層形 → 候補エンティティへのマッピング。 |
| NIL | KB に存在しない | どの KB エントリも一致しないという明示的な予測。 |
| KB | 知識ベース | Wikidata、Wikipedia、DBpedia、またはドメイン KB。 |
| AIDA-CoNLL | ベンチマーク | 正解エンティティリンクが付いた1,393件のロイター記事。 |

## 参考文献

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基礎となる事前確率+コンテキストアプローチ。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 埋め込みベースの主力。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 制約付きデコーディングによる生成型 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — ベンチマーク論文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — オープンな本番スタック。
