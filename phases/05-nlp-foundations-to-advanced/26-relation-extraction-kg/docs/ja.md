# 関係抽出と知識グラフ構築

> NER はエンティティを見つけた。エンティティリンキングはそれらを固定した。関係抽出はそれらの間のエッジを見つける。知識グラフはノード、エッジ、そしてそれらの来歴（プロヴェナンス）の総和である。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 06（NER）、Phase 5 · 25（エンティティリンキング）
**所要時間:** 約60分

## 問題

アナリストがこう読む: 「ティム・クックは2011年に Apple の CEO になった。」4つの事実がある。

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

関係抽出（RE）は自由テキストを構造化されたトリプル `(subject, relation, object)` に変換する。コーパス全体で集約すれば知識グラフになる。集約してクエリできれば、RAG、分析、コンプライアンス監査のための推論基盤になる。

2026年の問題: LLM は熱心に関係を抽出する。熱心すぎる。元のテキストが裏付けないトリプルを幻覚する。来歴がなければ、本物のトリプルともっともらしい虚構を見分けられない。2026年の答えは、AEVS スタイルのアンカー・アンド・ベリファイ・パイプラインだ。

## コンセプト

![テキスト → トリプル → 知識グラフ](../assets/relation-extraction.svg)

**トリプル形式。** `(subject_entity, relation_type, object_entity)`。関係は閉じたオントロジー（Wikidata プロパティ、FIBO、UMLS）から来るか、開いた集合（OpenIE スタイル、何でもあり）から来る。

**3つの抽出アプローチ。**

1. **ルール／パターンベース。** Hearst パターン: 「X such as Y」 → `(Y, isA, X)`。加えて手作りの正規表現。脆く、精密で、説明可能。
2. **教師あり分類器。** 文中の2つのエンティティ言及が与えられたとき、固定の集合から関係を予測する。TACRED、ACE、KBP で訓練。2015〜2022年の標準。
3. **生成型 LLM。** モデルにトリプルを出力するようプロンプトする。すぐに機能する。来歴が必要で、さもなければもっともらしく見えるゴミを幻覚する。

**AEVS（Anchor-Extraction-Verification-Supplement、2026）。** 現在の幻覚軽減フレームワーク:

- **Anchor（アンカー）。** すべてのエンティティスパンと関係フレーズスパンを正確な位置とともに特定する。
- **Extract（抽出）。** アンカースパンにリンクされたトリプルを生成する。
- **Verify（検証）。** 各トリプル要素を元テキストに照合し、裏付けのないものを棄却する。
- **Supplement（補完）。** カバレッジパスでアンカー付きスパンが取りこぼされないことを保証する。

幻覚は急減する。より多くの計算を要するが、監査可能だ。

**開放対閉鎖のトレードオフ。**

- **閉じたオントロジー。** 固定のプロパティリスト（例: Wikidata の11,000以上のプロパティ）。予測可能。クエリ可能。創作しにくい。
- **オープン IE。** あらゆる動詞句が関係になる。高再現率。低精度。クエリしにくく雑然とする。

本番の KG はたいてい両者を混ぜる: 発見にはオープン IE を使い、メインのグラフにマージする前に関係を閉じたオントロジー上に正規化する。

## 作ってみよう

### ステップ1: パターンベースの抽出

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完全なおもちゃの抽出器は `code/main.py` を参照。Hearst パターンはデバッグ可能なため、今もドメイン固有のパイプラインで出荷されている。

### ステップ2: 教師あり関係分類

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL は seq2seq の関係抽出器だ: テキストを入れるとトリプルが出て、すでに Wikidata のプロパティ id になっている。遠隔監督データでファインチューニングされている。標準のオープンウェイトベースライン。

### ステップ3: アンカリングを伴う LLM プロンプトの抽出

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

返されたすべてのスパンを元テキストに対して検証する。`text[start:end] != triple_entity` のものはすべて棄却する。これは AEVS の「検証（verify）」ステップを最小形で実装したものだ。

### ステップ4: 閉じたオントロジー上に正規化する

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

正規化はしばしばエンジニアリング作業の60〜80%を占める。そのための予算を見込んでおくこと。

### ステップ5: 小さなグラフを構築してクエリする

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

これはあらゆる RAG-over-KG システムの原子だ。RDF トリプルストア（Blazegraph、Virtuoso）、プロパティグラフ（Neo4j）、またはベクトル拡張グラフストアでスケールさせる。

## 落とし穴

- **RE の前に共参照。** 「He founded Apple」 — RE は「he」が誰かを知る必要がある。先に共参照を実行する（レッスン24）。
- **エンティティの正規化。** 「Apple Inc」と「Apple」は同じノードに解決されなければならない。先にエンティティリンキングを行う（レッスン25）。
- **幻覚したトリプル。** LLM はテキストが裏付けないトリプルを出力する。スパン検証を強制すること。
- **関係正規化のドリフト。** オープン IE の関係は一貫性がない（「was born in」「came from」「is a native of」）。正規 id にまとめなければグラフはクエリ不能になる。
- **時間的な誤り。** 「Tim Cook is CEO of Apple」 — 今は真だが、2005年には偽だった。多くの関係は時間的に区切られている。修飾子（Wikidata の `P580` 開始時刻、`P582` 終了時刻）を使う。
- **ドメインのミスマッチ。** REBEL は Wikipedia で訓練されている。法律、医療、科学のテキストはしばしばドメインでファインチューニングした RE モデルを必要とする。

## 使ってみよう

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| 高速な本番、汎用ドメイン | REBEL または LlamaPred + Wikidata 正規化 |
| ドメイン固有（バイオメディカル、法律） | SciREX スタイルのドメインファインチューニング + カスタムオントロジー |
| LLM プロンプト、監査済み出力 | AEVS パイプライン: アンカー → 抽出 → 検証 → 補完 |
| 大量のニュース IE | パターンベース + 教師ありのハイブリッド |
| ゼロから KG を構築 | オープン IE + 手動の正規化パス |
| 時間的 KG | 修飾子（開始/終了時刻、時点）付きで抽出 |

統合パターン: NER → 共参照 → エンティティリンキング → 関係抽出 → オントロジーマッピング → グラフロード。各段階が品質ゲートになりうる。

## 出荷しよう

`outputs/skill-re-designer.md` として保存する。

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 演習

1. **易。** `code/main.py` のパターン抽出器を、5つのニュース記事の文に対して実行する。精度を手作業で確認する。
2. **中。** 同じ文に REBEL（または小さな LLM）を使う。トリプルを比較する。どちらの抽出器が高精度か？ 高再現率か？
3. **難。** AEVS パイプラインを構築する: LLM で抽出し、元テキストに対してスパンを検証する。50件の Wikipedia 風の文に対して、検証ステップの前後で幻覚率を測定する。

## 重要用語

| 用語 | 一般に言われること | 実際の意味 |
|------|-----------------|-----------------------|
| トリプル（Triple） | 主語-関係-目的語 | KG の原子単位である `(s, r, o)` タプル。 |
| オープン IE | 何でも抽出 | 開語彙の関係フレーズ。高再現率、低精度。 |
| 閉じたオントロジー | 固定スキーマ | 関係タイプの有界な集合（Wikidata、UMLS、FIBO）。 |
| 正規化（Canonicalization） | すべて正規化 | 表層名／関係を正規 id にマッピングする。 |
| AEVS | 接地された抽出 | Anchor-Extraction-Verification-Supplement パイプライン（2026）。 |
| 来歴（Provenance） | 出所リンク | すべてのトリプルが出所への doc id + char-span を持つ。 |
| 遠隔監督（Distant supervision） | 安価なラベル | 既存の KG とテキストを整合させて訓練データを作る。 |

## 参考文献

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 遠隔監督の論文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE の主力。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 結合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026年の幻覚軽減設計。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 定番のグラフクエリ。
