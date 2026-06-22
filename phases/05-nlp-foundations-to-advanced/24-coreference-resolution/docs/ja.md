# 共参照解決（Coreference Resolution）

> 「彼女は彼に電話した。彼は出なかった。その医師は昼食中だった。」二人の人物への3つの参照、そして誰の名前も挙げられていない。共参照解決は誰が誰なのかを突き止める。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 5 · 06（NER）、Phase 5 · 07（POS と構文解析）
**所要時間:** 約60分

## 問題

300語の記事から Apple Inc. へのすべての言及を抽出する。記事が「Apple」と言っている場合は簡単だ。難しいのは「その会社」「彼ら」「クパチーノのテクノロジー大手」「ジョブズの会社」と言っている場合だ。これらの言及を同じエンティティに解決しなければ、NER パイプラインは言及の60〜80%を取りこぼす。

共参照解決は、同じ実世界のエンティティを指すすべての表現を1つのクラスタにリンクする。これは表層レベルの NLP（NER、構文解析）と下流の意味処理（IE、QA、要約、KG）をつなぐ接着剤である。

2026年に重要な理由:

- 要約: 「その CEO は発表した……」対「ティム・クックは発表した……」 — 要約は CEO を名指しすべきだ。
- 質問応答: 「彼女は誰に電話したか？」には「彼女」を解決する必要がある。
- 情報抽出: 「PER1 が Apple を創業した」と「ジョブズが Apple を創業した」を別エントリとして持つ知識グラフは誤っている。
- マルチドキュメント IE: 同じ出来事に関する複数記事にまたがる言及をマージすることは、ドキュメント横断の共参照である。

## コンセプト

![共参照クラスタリング: 言及 → エンティティ](../assets/coref.svg)

**タスク。** 入力: ドキュメント。出力: 各クラスタが1つのエンティティを指す、言及（スパン）のクラスタリング。

**言及の種類。**

- **固有表現。** 「ティム・クック」
- **名詞句的（Nominal）。** 「その CEO」「その会社」
- **代名詞的（Pronominal）。** 「彼」「彼女」「彼ら」「それ」
- **同格（Appositive）。** 「Apple の CEO、ティム・クック」

**アーキテクチャ。**

1. **ルールベース（Hobbs, 1978）。** 文法規則を用いた構文木ベースの代名詞解決。良いベースライン。代名詞では驚くほど打ち負かしにくい。
2. **言及ペア分類器（Mention-pair classifier）。** 言及のあらゆるペア (m_i, m_j) について共参照するかどうかを予測する。推移閉包でクラスタ化する。2016年以前の標準。
3. **言及ランキング（Mention-ranking）。** 各言及について、候補となる先行詞（「先行詞なし」を含む）をランク付けする。最上位を選ぶ。
4. **スパンベースのエンドツーエンド（Lee et al., 2017）。** トランスフォーマーエンコーダ。長さの上限まですべての候補スパンを列挙する。言及スコアを予測する。各スパンについて先行詞確率を予測する。貪欲にクラスタ化する。現代のデフォルト。
5. **生成型（2024年〜）。** LLM にプロンプトを与える: 「このテキスト中のすべての代名詞とその先行詞を列挙せよ。」簡単なケースではうまく機能するが、長いドキュメントや稀な指示対象には苦戦する。

**評価指標。** 5つの標準指標（MUC、B³、CEAF、BLANC、LEA）がある。単一の指標ではクラスタリング品質を捉えきれないからだ。最初の3つの平均を CoNLL F1 として報告する。2026年の CoNLL-2012 における最先端: 約83 F1。

**既知の難しいケース。**

- 数ページ前に導入されたエンティティを指す定記述。
- 橋渡し照応（bridging anaphora）（「車輪」 → 以前に言及された車）。
- 中国語や日本語などの言語におけるゼロ照応。
- 後方照応（cataphora、指示対象より前の代名詞）: 「**彼女**が入ってきたとき、メアリーは微笑んだ。」

## 作ってみよう

### ステップ1: 事前学習済みのニューラル共参照（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

より長いドキュメントでは、次のような結果が得られる。
- クラスタ1: [Apple, The company, they]
- クラスタ2: [new products]

### ステップ2: ルールベースの代名詞解決器（教育用）

stdlib のみの実装は `code/main.py` を参照。

1. 言及を抽出する: 固有表現（大文字始まりのスパン）、代名詞（辞書引き）、定記述（「the X」）。
2. 各代名詞について、直前のK個の言及を見て、以下でスコア付けする:
   - 性／数の一致（ヒューリスティック）
   - 近接性（近いものが勝つ）
   - 構文的役割（主語が優先される）
3. 最高スコアの先行詞をリンクする。

ニューラルモデルには太刀打ちできない。しかし、探索空間と、エンドツーエンドモデルが下さねばならない判断を示してくれる。

### ステップ3: 共参照に LLM を使う

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意すべき失敗モードが2つある。第一に、LLM は過剰にマージする（別々の二人を指す「彼」と「彼女」を一緒にしてしまう）。第二に、LLM は長いドキュメントで言及を黙って取りこぼす。常にスパンオフセットのチェックで検証すること。

### ステップ4: 評価

標準の conll-2012 スクリプトは MUC、B³、CEAF-φ4 を計算し、その平均を報告する。社内評価では、注釈付きテストセット上のスパンレベルの適合率・再現率から始め、その後で言及リンク F1 を追加するとよい。

## 落とし穴

- **シングルトンの爆発。** 一部のシステムはすべての言及を独立したクラスタとして報告する。B³ は寛容だ。MUC はこれを罰する。常に3つの指標すべてを確認すること。
- **長コンテキストでの代名詞。** 2,000トークンを超えるドキュメントでは性能が約15 F1 低下する。慎重にチャンキングすること。
- **性別の仮定。** ハードコードされた性別ルールは、ノンバイナリの指示対象、組織、動物では破綻する。学習済みモデルか中立的なスコアリングを使うこと。
- **長いドキュメントでの LLM のドリフト。** 単一の API 呼び出しで50段落以上にまたがる言及を確実にクラスタ化することはできない。スライディングウィンドウ + マージを使うこと。

## 使ってみよう

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| 英語、単一ドキュメント | `en_coreference_web_trf`（spaCy-experimental）または AllenNLP ニューラル共参照 |
| 多言語 | OntoNotes または Multilingual CoNLL で訓練した SpanBERT / XLM-R |
| ドキュメント横断のイベント共参照 | 専用のエンドツーエンドモデル（2025〜26年の SOTA） |
| 手早い LLM ベースライン | 構造化出力の共参照プロンプトを用いた GPT-4o / Claude |
| 本番の対話システム | ルールベースのフォールバック + ニューラル主系統 + 重要スロットの手動レビュー |

2026年に出荷される統合パターン: まず NER を実行し、共参照を実行し、共参照クラスタを NER エンティティにマージする。下流タスクは、言及ごとに1エンティティではなく、クラスタごとに1エンティティを見る。

## 出荷しよう

`outputs/skill-coref-picker.md` として保存する。

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 演習

1. **易。** `code/main.py` のルールベース解決器を、手作りの5段落に対して実行する。正解に対する言及リンク精度を測定する。
2. **中。** 事前学習済みのニューラル共参照モデルをニュース記事に使う。クラスタを自分の手作業の注釈と比較する。どこで失敗したか？
3. **難。** 共参照で強化した NER パイプラインを構築する: まず NER、次に共参照クラスタでマージする。100記事上で NER 単独に対するエンティティ網羅率の改善を測定する。

## 重要用語

| 用語 | 一般に言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 言及（Mention） | 参照 | エンティティ（名前、代名詞、名詞句）を指すテキストのスパン。 |
| 先行詞（Antecedent） | 「それ」が指すもの | 後の言及が共参照する、より前の言及。 |
| クラスタ（Cluster） | エンティティの言及群 | すべて同じ実世界のエンティティを指す言及の集合。 |
| 照応（Anaphora） | 後方参照 | 後の言及が前を指す（「彼」 → 「ジョン」）。 |
| 後方照応（Cataphora） | 前方参照 | 前の言及が後を指す（「彼が到着したとき、ジョンは……」）。 |
| 橋渡し（Bridging） | 暗黙の参照 | 「車を買った。車輪が悪かった。」（あの車の車輪。） |
| CoNLL F1 | リーダーボードの数値 | MUC、B³、CEAF-φ4 の F1 スコアの平均。 |

## 参考文献

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 定番の教科書の章。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — スパンベースのエンドツーエンド。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 共参照を改善する事前学習。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — ベンチマーク。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — ルールベースの古典。
