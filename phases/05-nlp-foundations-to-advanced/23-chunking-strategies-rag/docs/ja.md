# RAG のためのチャンキング戦略

> チャンキングの設定は、埋め込みモデルの選択と同じくらい検索品質に影響する（Vectara NAACL 2025）。チャンキングを誤れば、どれだけリランキングしても救えない。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 14（情報検索）、Phase 5 · 22（埋め込みモデル）
**所要時間:** 約60分

## 問題

50ページの契約書を RAG システムに投入する。ユーザーが「解約条項は何ですか？」と尋ねる。検索器は表紙を返す。なぜか？ モデルは512トークンのチャンクで訓練されており、解約条項は20ページ目にあって、ページ区切りで分断され、クエリに紐づくローカルなキーワードが近くに存在しないからだ。

解決策は「もっと良い埋め込みモデルを買う」ことではない。解決策はチャンキングだ。どのくらいの大きさにするか？ オーバーラップは？ どこで分割するか？ 周辺コンテキストは付けるか？

2026年2月のベンチマークは意外な結果を示している。

- Vectara の2026年の研究: 再帰的512トークンチャンキングがセマンティックチャンキングを精度69% → 54%で上回った。
- Natural Questions における SPLADE + Mistral-8B: オーバーラップは測定可能な利益をまったくもたらさなかった。
- コンテキストの崖: コンテキストが約2,500トークンを超えたあたりで応答品質が急落する。

「当たり前」とされる答え（セマンティックチャンキング、20%オーバーラップ、1000トークン）は、しばしば間違っている。このレッスンでは6つの戦略に対する直感を養い、どんなときにどれを使うべきかを示す。

## コンセプト

![6つのチャンキング戦略を1つのパッセージ上で可視化](../assets/chunking.svg)

**固定チャンキング（Fixed chunking）。** N文字またはNトークンごとに分割する。最もシンプルなベースライン。文の途中で切れる。圧縮率は良いが、一貫性は悪い。

**再帰（Recursive）。** LangChain の `RecursiveCharacterTextSplitter`。まず `\n\n` で分割を試み、次に `\n`、次に `.`、最後にスペースで試す。きれいにフォールバックする。2026年のデフォルト。

**セマンティック（Semantic）。** 各文を埋め込む。隣接する文同士のコサイン類似度を計算する。類似度が閾値を下回るところで分割する。トピックの一貫性を保つ。低速で、ときに40トークンの極小フラグメントを生成し、検索の妨げになる。

**文（Sentence）。** 文の境界で分割する。1チャンクにつき1文、またはN文のウィンドウ。約5kトークンまではセマンティックチャンキングに匹敵する性能を、わずかなコストで実現する。

**親ドキュメント（Parent-document）。** 検索用に小さな子チャンクを保存し、コンテキスト用に大きな親チャンク*も*保存する。子で検索し、親を返す。優雅に劣化する。悪い子チャンクでも、それなりの親が返る。

**遅延チャンキング（Late chunking、2024）。** まずドキュメント全体をトークンレベルで埋め込み、その後トークン埋め込みをチャンク埋め込みにプーリングする。チャンク間のコンテキストを保持する。長コンテキスト埋め込みモデル（BGE-M3、Jina v3）で機能する。計算コストは高い。

**コンテキスト検索（Contextual retrieval、Anthropic、2024）。** 各チャンクの先頭に、ドキュメント内での位置を LLM が生成した要約を付加する（「このチャンクは解約条項のセクション3.2である……」）。Anthropic 自身のベンチマークで35〜50%の検索改善。インデックス作成のコストは高い。

### あらゆるデフォルトに勝つルール

チャンクサイズをクエリの種類に合わせる。

| クエリの種類 | チャンクサイズ |
|------------|-----------|
| ファクトイド（「CEO の名前は？」） | 256〜512トークン |
| 分析的／マルチホップ | 512〜1024トークン |
| セクション全体の理解 | 1024〜2048トークン |

NVIDIA の2026年ベンチマーク。チャンクは、答えとローカルなコンテキストを含むのに十分大きく、かつ検索器の top-K がコンテキストのノイズではなく答えに集中できる程度に小さくあるべきだ。

## 作ってみよう

### ステップ1: 固定チャンキングと再帰チャンキング

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### ステップ2: セマンティックチャンキング

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

`threshold` はドメインに合わせて調整する。高すぎると断片化し、低すぎると1つの巨大なチャンクになる。

### ステップ3: 親ドキュメント

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

重要な洞察: 親を重複排除すること。複数の子が同じ親に対応しうるため、すべてを返すとコンテキストが無駄になる。

### ステップ4: コンテキスト検索（Anthropic パターン）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

コンテキスト化したチャンクをインデックスする。クエリ時には、追加された周辺シグナルから検索が恩恵を受ける。

### ステップ5: 評価する

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

常にベンチマークを取ること。あなたのコーパスにとっての「最良」戦略は、どのブログ記事とも一致しないかもしれない。

## 落とし穴

- **チャンキングをファクトイドクエリのみで評価する。** マルチホップクエリではまったく異なる勝者が明らかになる。クエリ種別で層別化した評価セットを使うこと。
- **最小サイズなしのセマンティックチャンキング。** 検索を妨げる40トークンのフラグメントを生成する。常に `min_tokens` を強制すること。
- **カーゴカルト的なオーバーラップ。** 2026年の研究では、オーバーラップはしばしばまったく利益をもたらさず、インデックスコストを2倍にすることがわかっている。仮定せず、測定すること。
- **最小／最大の強制なし。** 5トークンのチャンクも5000トークンのチャンクも、どちらも検索を壊す。範囲を制限すること。
- **ドキュメント横断のチャンキング。** チャンクが2つのドキュメントにまたがることを決して許してはならない。常にドキュメントごとにチャンキングしてからマージすること。

## 使ってみよう

2026年のスタック:

| 状況 | 戦略 |
|-----------|----------|
| 初回構築、未知のコーパス | 再帰、512トークン、オーバーラップなし |
| ファクトイド QA | 再帰、256〜512トークン |
| 分析的／マルチホップ | 再帰、512〜1024トークン + 親ドキュメント |
| 相互参照が多い（契約書、論文） | 遅延チャンキングまたはコンテキスト検索 |
| 会話／対話コーパス | ターンレベルのチャンク + 話者メタデータ |
| 短い発話（ツイート、レビュー） | 1ドキュメント = 1チャンク |

再帰512から始めること。50件のクエリ評価セットで recall@5 を測定する。そこから調整する。

## 出荷しよう

`outputs/skill-chunker.md` として保存する。

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 演習

1. **易。** 20ページのドキュメント1つを fixed(512, 0)、recursive(512, 0)、recursive(512, 100) でチャンキングする。チャンク数と境界品質を比較する。
2. **中。** 5つのドキュメントにわたる30件のクエリ評価セットを構築する。再帰、セマンティック、親ドキュメントの recall@5 を測定する。どれが勝つか？ ブログ記事と一致するか？
3. **難。** コンテキスト検索を実装する。ベースラインの再帰に対する MRR の改善を測定する。インデックスコスト（LLM 呼び出し）対精度向上を報告する。

## 重要用語

| 用語 | 一般に言われること | 実際の意味 |
|------|-----------------|-----------------------|
| チャンク（Chunk） | ドキュメントの一部 | 埋め込み・インデックス・検索されるサブドキュメント単位。 |
| オーバーラップ（Overlap） | 安全マージン | 隣接チャンク間で共有されるNトークン。2026年のベンチマークではしばしば無用。 |
| セマンティックチャンキング | 賢いチャンキング | 隣接文の埋め込み類似度が下がるところで分割する。 |
| 親ドキュメント（Parent-document） | 2層検索 | 小さな子を検索し、大きな親を返す。 |
| 遅延チャンキング（Late chunking） | 埋め込み後のチャンキング | ドキュメント全体をトークンレベルで埋め込み、チャンクベクトルにプーリングする。 |
| コンテキスト検索（Contextual retrieval） | Anthropic の手法 | インデックス前に各チャンクへ LLM 生成の要約を前置する。 |
| コンテキストの崖（Context cliff） | 2500トークンの壁 | RAG においてコンテキスト約2.5kトークン付近で観測される品質低下（2026年1月）。 |

## 参考文献

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 本番環境でのデフォルト。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — チャンキングは埋め込みの選択と同じくらい重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 遅延チャンキングの論文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM 生成のコンテキスト接頭辞で35〜50%の検索改善。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — クエリ種別ごとのチャンクサイズ。
