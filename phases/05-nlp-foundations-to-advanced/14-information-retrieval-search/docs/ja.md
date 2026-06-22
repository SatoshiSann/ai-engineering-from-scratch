# 情報検索と検索

> BM25 は精密だが脆い。密なキャストは広いネット但しキーワードを見落とす。ハイブリッドは2026デフォルト。他は全てチューニング。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 02（BoW + TF-IDF）、Phase 5 · 04（GloVe、FastText、Subword）
**所要時間:** 約75分

## 問題

ユーザーは「誰かがお金を得るために嘘をついたら何が起こるか」を入力し、実際その法令をカバーする「第420条IPC」を見つけることを期待。キーワード検索は完全にそれを見落とす（共有語彙なし）。意味検索はそれを見落とします埋め込みが法的テキストで訓練されていなかった場合。本物の検索は両方を処理する必要。

IRは全ての RAG システムの下のパイプライン、全て検索バー、全て docs サイトのあいまいい検索。2026 本番で動作するアーキテクチャは単一の方法ではない。それは相互補完方法のチェーン、各がその前のもの失敗をキャッチ。

このレッスンはそれぞれのピースをビルドして、各ピースがどの失敗をキャッチするかを名付ける。

## コンセプト

![ハイブリッド検索：BM25 + 密 + RRF + クロス符号化 rerank](../assets/retrieval.svg)

4つのレイヤー。あなたが必要なのをピック。

1. **スパース検索（BM25）。** 高速、正確なマッチで精密、意味で悪い。逆引きインデックス実行。百万文書上のサブ10ミリ秒/クエリ。法令参照、製品コード、エラーメッセージ、命名エンティティを手に入れるのは正しい。
2. **密な検索。** クエリとドキュメントをベクトルにエンコード。最近傍検索。paraphrases とセマンティック類似度をキャプチャ。１文字の正確キーワード マッチを見落とし異なるもの。FAISS またはベクトルDB で 50-200ms/クエリ。
3. **融合。** スパースと密なからランク付けされたリストをマージ。相互 Rank 融合（RRF）は生スコア（異なるスケールで住む）を無視して単にランク位置を使用するため簡単デフォルト。重み付け融合は あなたのドメイン 1 つのシグナルが支配するときをオプション。
4. **クロス符号化 rerank。** 融合から最上位30をとる。クロス符号化実行（クエリ+ドキュメント、各ペアをスコア）。トップ5を保つ。クロス符号化はペアあたり二項符号より遅いが遠くはるかに正確。あなたは最上位30のみで実行することにより償却。

3方向検索（BM25 + 密 + SPLADE のような学習スパース）は 2026 ベンチマークで2方向を上回るが学習スパース インデックスのためのインフラが必要。多数のチーム、2方向プラスクロス符号化 rerank は甘いスポット。

## ビルド

### ステップ1：BM25 をゼロから

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

知る価値のある2つのパラメータ。`k1=1.5` はタームー頻度飽和をコントロール；高い意味はターム反復に加えた重み。`b=0.75` はドキュメント長正規化をコントロール；0は長さを無視、1は完全に正規化。デフォルトはロバートソンのお薦め元論文と滅多にチューニング必要。

### ステップ 2：ビエンコーダで密な検索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 正規化埋め込みのでドット積はコサイン等しい。`all-MiniLM-L6-v2` は 384次元、高速、ほどんどの英語検索のため十分に強い。多言語の仕事のため、`paraphrase-multilingual-MiniLM-L12-v2` を使用。最上位精度で、`bge-large-en-v1.5` または `e5-large-v2`。

### ステップ 3：相互ランク融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 定数は元の RRF 論文から。より高い `k` ランク差の寄与をフラット；より低い `k` はトップランク支配。60 は発表されたデフォルトで滅多にチューニング必要。

### ステップ 4：ハイブリッド検索 + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

3ステージ構成。BM25 は字句マッチを見つけ。密は意味マッチを見つけ。RRF は2つのランキングをスコア較正が必要なくマージ。クロス符号化はクエリドキュメント ペアを一緒に使用してスコアし、二項符号を見落とした微細な関連性をキャプチャ。トップ5を保つ。

### ステップ 5：評価

| メトリック | 意味 |
|--------|---------|
| リコール@k | どの質問の正しいドキュメントが存在する、トップ `k` にそれの頻度？|
| MRR（平均相互ランク） | 最初に関連ドキュメントのランクの 1/ランク の平均。|
| nDCG@k | 関連グラデーション説明、バイナリではない。|

RAG 特に、取得 **リコール@k** 最も重要な数。あなたの読むは取得セットにある権利 通路がない場合答えることができない。

デバッグティップ：失敗するクエリのため、スパースと密なランキングを diff。1つが権利ドキュメントを見つけ他ではない場合、あなたは語彙ミスマッチ（フィックス：欠落した半分を追加）または意味曖昧（フィックス：より良い埋め込みまたは reranker）を持つ。

## 使う

2026 スタック：

| スケール | スタック |
|-------|-------|
| 1k-100k ドキュメント | メモリ内BM25 + `all-MiniLM-L6-v2` 埋め込み + RRF。別々DB なし。|
| 100k-10M ドキュメント | 密むけFAISS または pgvector + BM25 むけ Elasticsearch / OpenSearch。並行実行。|
| 10M+ ドキュメント | ハイブリッドサポート持つ Qdrant / Weaviate / Vespa / Milvus。トップ30の上で クロス符号化 rerank。|
| 最高品質フロンティア | 3方向（BM25 + 密 + SPLADE）+ ColBERT 後期相互作用 reranking |

何をピックしても、評価の予算。読む正確性の前に取得リコール ベンチマーク。読むは取得が見落とした訂正することができない。

### 2026 本番 RAG からの苦労して得た教訓

- **80% RAG 失敗はモデル摂動でなく、摂取とチャンキングに痕跡。** チーム週をLLM スワップ+プロンプトチューニングに浪費する一方、取得は静かに各3番目のクエリで間違ったコンテキストを返す。チャンキング最初に修正。
- **チャンキング戦略はサイズより重要。** 固定サイズ分割テーブル、コード、ネストされたヘッダーを壊す。文認識はデフォルト；セマンティックまたはLLMベース チャンキングは技術的ドキュメント+製品マニュアルで負担される。
- **親ドキュメントパターン。** 小さい「子」チャンクで精密のため取得。複数の子が同じ親セクションから表示される場合、親ブロックで交換してコンテキスト保持。これは一貫して答え品質を上げるなしで再訓練。
- **k_rerank=3 は通常最適。** その過去各チャンク追加はトークンコスト+世代遅延を追加するなしで答え品質を上げ。k=8 はまだk=3 より良い あなたのため、reranker は低性能。
- **HyDE / クエリ拡張。** クエリから仮説答えを生成、埋め込み、取得。短い質問と長いドキュメント間の表記ギャップをブリッジ。訓練が必要なしの無料精密リフト。
- **コンテキスト予算下 8K トークン。** 一貫したヒット その制限でなはreranker 閾値が緩すぎること意味。
- **全てをバージョン。** プロンプト、チャンキングルール、埋め込みモデル、reranker。あらゆるドリフトは静かに答え品質を壊す。CI ゲートで忠実性、コンテキスト精密、答え不可能な率で回帰はユーザーが見る前にブロック。
- **3方向検索（BM25 + 密 + SPLADE のような学習スパース）2方向を上回る** 2026 ベンチマークで、特にクエリ固有名+セマンティクスを混合するもの。インフラは支援 SPLADE インデックスのときにそれを出荷。

適切な取得設計は 2026 業界測定に従って 70-90% で幻想を削減。ほどんどの RAG 性能利益はより良い取得からくる、モデル微調整ではない。

## 出荷

`outputs/skill-retrieval-picker.md` として保存：

```markdown
---
name: retrieval-picker
description: 与えられたコーパスとクエリパターンのため取得スタックをピック。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

要件が与えられたら（コーパスサイズ、クエリパターン、遅延予算、品質バー、インフラ制約）、出力：

1. スタック。BM25 のみ、密のみ、ハイブリッド（BM25 + 密 + RRF）、ハイブリッド + クロス符号化 rerank、または3方向（BM25 + 密 + 学習スパース）。
2. 密なエンコーダ。具体的モデルを名付け。言語（複）、ドメイン、コンテキスト長にマッチ。
3. Reranker。使用された場合、具体的クロス符号化モデルを名付け。rerank がトップ30に 30-100ms 遅延を追加するとしてフラグ。
4. 評価プラン。リコール@10 は主取得メトリック。複数答えのための MRR。ベースラインまず、それに対してインクリメント改善を測定。

名前エンティティ、エラーコード、または製品SKU を持つコーパスのため密のみを推奨するを拒否、ユーザーは密が正確マッチを処理する証拠を持つことを除き。高賭け取得（法律、医学）をスキップ reranking をしないでください、最終的なトップ5はユーザー答えを決定するところ。
```

## 演習

1. **簡単。** 500 ドキュメントコーパスで上記の `hybrid_search` を実装。20 クエリをテスト。BM25 のみ、密のみ、ハイブリッド間でリコール 5 を比較。
2. **中程度。** MRR 計算を追加。各テスト既知の正しいドキュメント持つクエリのため、BM25、密、ハイブリッド ランキング内の正しい ドキュメントのランクを見つけ。各の MRR を報告。
3. **難しい。** MultipleNegativesRankingLoss を使用してあなたのドメインで密エンコーダをファインチューン（Sentence Transformers）。500 クエリドキュメント ペアから訓練セットをビルド。前後ファインチューンのリコールを比較。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| BM25 | キーワード検索 | Okapi BM25。用語頻度、IDF、長さ別にドキュメントをスコア。|
| 密な検索 | ベクトル検索 | クエリ+ドキュメントをベクトルにエンコード、最近傍を見つけ。|
| ビエンコーダ | 埋め込みモデル | クエリとドキュメントを独立にエンコード。クエリ時に高速。|
| クロス符号化 | Reranker モデル | 一緒にクエリ+ドキュメントをエンコード。遅いが正確。|
| RRF | ランク融合 | 2つのランキング合計でマージ `1/(k + rank)`。|
| リコール@k | 取得メトリック | 関連 ドキュメントはトップ `k` にある質問の割合。|

## 参考文献

- [Robertson と Zaragoza (2009)。確率的関連性フレームワーク：BM25 以降](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 決定的な BM25 処理。
- [Karpukhin et al. (2020)。オープン領域 QA のための密な通路取得](https://arxiv.org/abs/2004.04906) — DPR、標準的なビエンコーダ。
- [Formal et al. (2021)。SPLADE：スパース字句と拡張モデル](https://arxiv.org/abs/2107.05720) — 学習スパース取得。密とギャップを閉じる。
- [Cormack、Clarke、Büttcher (2009)。相互ランク融合は Condorcet と個々ランク学習方法を上回る](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 論文。
- [Khattab と Zaharia (2020)。ColBERT：効率的で有効な通路検索](https://arxiv.org/abs/2004.12832) — 後期相互作用取得。
