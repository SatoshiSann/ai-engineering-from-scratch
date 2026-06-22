# トピックモデリング — LDA と BERTopic

> LDA: 文書はトピックの混合であり、トピックは単語上の分布である。BERTopic: 文書は埋め込み空間上でクラスタを形成し、クラスタがトピックとなる。目標は同じだが、分解の仕方が異なる。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 5 · 02 (BoW + TF-IDF)、Phase 5 · 03 (Word2Vec)
**所要時間:** 約45分

## 問題

あなたは10,000件のカスタマーサポートチケット、50,000件のニュース記事、あるいは200,000件のツイートを抱えている。それらを読まずに、このコレクションが何についてのものなのかを知る必要がある。ラベル付きのカテゴリは持っていない。そもそもカテゴリがいくつ存在するのかすら分からない。

トピックモデリングは、それを教師なしで答える。コーパスを与えれば、少数の一貫したトピックの集合と、各文書についてそれらのトピック上の分布が返ってくる。

2つのアルゴリズムのファミリーが主流である。LDA (2003) は各文書を潜在トピックの混合として扱い、各トピックを単語上の分布として扱う。推論はベイズ的である。混合メンバーシップのトピック割り当てや、説明可能な単語レベルの確率分布が必要な本番環境では、今なお使われている。

BERTopic (2020) は BERT で文書をエンコードし、UMAP で次元削減を行い、HDBSCAN でクラスタリングし、クラスベースの TF-IDF を介してトピック単語を抽出する。短いテキスト、ソーシャルメディア、そして単語の重なりよりも意味的類似性が重要となる場面で優れている。1つの文書には1つのトピックが割り当てられるため、長文コンテンツでは制約となる。

このレッスンでは両者についての直観を養い、与えられたコーパスに対してどちらを選ぶべきかを示す。

## コンセプト

![LDA 混合モデルと BERTopic クラスタリング](../assets/topic-modeling.svg)

**LDA の生成ストーリー。** 各トピックは単語上の分布である。各文書はトピックの混合である。文書内の単語を生成するには、まず文書の混合からトピックをサンプリングし、次にそのトピックの分布から単語をサンプリングする。推論はこれを逆向きに行う。観測された単語が与えられたとき、文書ごとのトピック分布とトピックごとの単語分布を推論する。崩壊ギブスサンプリングまたは変分ベイズがその計算を担う。

LDA の主要な出力:

- `doc_topic`: 行列 `(n_docs, n_topics)`、各行の和は1 (文書のトピック混合)。
- `topic_word`: 行列 `(n_topics, vocab_size)`、各行の和は1 (トピックの単語分布)。

**BERTopic のパイプライン。**

1. 各文書を sentence transformer (例: `all-MiniLM-L6-v2`) でエンコードする。384次元のベクトル。
2. UMAP で次元を約5次元まで削減する。BERT 埋め込みはクラスタリングには次元が高すぎる。
3. HDBSCAN でクラスタリングする。密度ベースで、可変サイズのクラスタと「外れ値」ラベルを生成する。
4. 各クラスタについて、そのクラスタの文書全体に対してクラスベースの TF-IDF を計算し、上位の単語を抽出する。

出力は文書ごとに1つのトピック (加えて -1 の外れ値ラベル) である。オプションで、HDBSCAN の確率ベクトルを介したソフトメンバーシップも得られる。

## 作ってみよう

### ステップ1: scikit-learn による LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注目すべき点: ストップワードを除去し、min_df と max_df で稀な語とどこにでも出現する語をフィルタリングしている。また LDA は生のカウントを期待するため、TfidfVectorizer ではなく CountVectorizer を使っている。

### ステップ2: BERTopic (本番)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` というフィルタは、BERTopic の外れ値バケット (HDBSCAN がクラスタリングできなかった文書) を取り除く。`min_topic_size` は HDBSCAN の最小クラスタサイズを制御する。BERTopic ライブラリのデフォルトは10である。この例では、このレッスンの規模に合わせて明示的に15に設定している。10,000件を超えるコーパスでは、50または100に増やすこと。

### ステップ3: 評価

どちらの手法もトピック単語を出力する。問題は、それらの単語が一貫しているかどうかである。

- **トピックコヒーレンス (c_v)。** スライディングウィンドウのコンテキストにわたる上位単語ペアの NPMI (正規化相互情報量) を組み合わせ、そのスコアをトピックベクトルに集約し、それらのベクトルをコサイン類似度で比較する。高いほど良い。`gensim.models.CoherenceModel` を `coherence="c_v"` で使う。
- **トピックの多様性。** すべてのトピックの上位単語にわたるユニークな単語の割合。高いほど良い (トピックが重複していない)。
- **定性的な検査。** 各トピックの上位単語を読む。それらは実在するものを表しているか。人間の判断は依然として最後の防衛線である。

## どちらをいつ選ぶか

| 状況 | 選択 |
|-----------|------|
| 短いテキスト (ツイート、レビュー、見出し) | BERTopic |
| トピックの混合を含む長い文書 | LDA |
| GPU なし / 計算資源が限られる | LDA または NMF |
| 文書レベルのマルチトピック分布が必要 | LDA |
| トピックラベル付けのための LLM 統合 | BERTopic (直接サポート) |
| リソース制約のあるエッジデプロイ | LDA |
| 最大の意味的コヒーレンス | BERTopic |

最大の実務的な考慮事項は文書の長さである。BERT 埋め込みは切り詰められる。LDA のカウントはどんな長さでも機能する。埋め込みモデルのコンテキストより長い文書の場合は、チャンク化 + 集約を行うか、LDA を使う。

## 使ってみよう

2026年のスタック:

- **BERTopic。** 短いテキストや、意味が重要なあらゆる場面のデフォルト。
- **`gensim.models.LdaModel`。** 本番向けの古典的 LDA。成熟しており、実戦で鍛えられている。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 実験用の手軽な LDA。
- **NMF。** 非負値行列因子分解。LDA の高速な代替で、短いテキストでは同等の品質。
- **Top2Vec。** BERTopic と似た設計。コミュニティは小さいが、一部のベンチマークでは良好。
- **FASTopic。** より新しく、非常に大きなコーパスでは BERTopic より高速。
- **LLM ベースのラベル付け。** 任意のクラスタリングを実行し、その後モデルにプロンプトを与えて各クラスタに名前を付けさせる。

## 仕上げよう

`outputs/skill-topic-picker.md` として保存する:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 演習

1. **易しい。** 20 Newsgroups データセットに対して5トピックの LDA をフィットする。トピックごとの上位10単語を出力する。各トピックに手作業でラベルを付ける。アルゴリズムは実際のカテゴリを見つけられたか。
2. **普通。** 同じ 20 Newsgroups のサブセットに対して BERTopic をフィットする。見つかったトピックの数、上位単語、定性的なコヒーレンスを LDA と比較する。どちらがより明確に実際のカテゴリを浮かび上がらせるか。
3. **難しい。** あなたのコーパスに対して LDA と BERTopic の両方について c_v コヒーレンスを計算する。それぞれを5、10、20、50トピックで実行する。コヒーレンスとトピック数の関係をプロットする。どちらの手法がトピック数にわたってより安定しているか報告する。

## 重要な用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| トピック | コーパスが扱っているもの | 単語上の確率分布 (LDA) または類似文書のクラスタ (BERTopic)。 |
| 混合メンバーシップ | 文書は複数のトピック | LDA は各文書に全トピック上の分布を割り当てる。 |
| UMAP | 次元削減 | 局所構造を保存する多様体学習。BERTopic で使われる。 |
| HDBSCAN | 密度クラスタリング | 可変サイズのクラスタを見つける。外れ値には「ノイズ」ラベル (-1) を生成する。 |
| c_v コヒーレンス | トピック品質指標 | スライディングウィンドウ内における上位トピック単語の平均相互情報量。 |

## 参考文献

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA の論文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic の論文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — c_v とその仲間を導入した論文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 本番向けのリファレンス。優れた例が豊富。
