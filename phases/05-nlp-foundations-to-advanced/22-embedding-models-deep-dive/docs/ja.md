# 埋め込みモデル — 2026年の深掘り

> Word2Vecは1単語あたり1つのベクトルを与えた。モダン埋め込みモデルはパッセージあたり1つのベクトル、クロスリンガル、スパース、デンス、マルチベクトルビュー、インデックスに合わせてサイズを与える。間違いを選ぶと、RAGは間違ったものを取得する。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 5 · 03 (Word2Vec)、Phase 5 · 14 (情報検索)
**所要時間:** 約60分

## 問題

RAGシステムが40%の時間、間違ったパッセージを取得する。犯人はめったにベクトルデータベースまたはプロンプトではない。埋め込みモデルだ。

2026年で埋め込みを選択することは、5つの軸全体で選択することを意味する:

1. **デンス対スパース対マルチベクトル。** パッセージあたり1つのベクトル、またはトークンあたり1つ、またはスパース重み付き単語の袋。
2. **言語カバレッジ。** 単一言語英語モデルは英語のみのタスクでまだ勝つ。コーパスが混合されているときは多言語モデルが勝つ。
3. **コンテキスト長。** 512トークン対8,192対32,768 — そして実際の有効容量は多くの場合、広告されたマックスの60-70%だ。
4. **次元バジェット。** 3,072フロート完全精度 = ベクトルあたり12 KB。100Mベクトルでは、ストレージは月額$1,300だ。Matryoshka截断は4×これを削減する。
5. **オープン対ホスト。** オープンウェイトはスタックとデータを制御することを意味する。ホストは常に最新への管理をトレードオフを意味する。

このレッスンは、最後の四半期が人気があるもの基づかず、証拠に基づいて選択できるようにトレードオフを名付ける。

## コンセプト

![デンス、スパース、マルチベクトル埋め込み](../assets/embedding-modes.svg)

**デンス埋め込み。** パッセージあたり1つのベクトル(通常384-3,072次元)。コサイン類似度は意味的近接度でパッセージをランク付けする。OpenAI `text-embedding-3-large`、BGE-M3デンスモード、Voyage-3。デフォルト選択。

**スパース埋め込み。** SPLADEスタイル。トランスフォーマーはすべてのボキャブラリートークンの重みを予測し、ほとんどをゼロアウトする。結果は|vocab|のサイズのスパースベクトル。字句マッチング(BM25のような)を取得するが、学習された用語の重みを使用。キーワード多くのクエリで強い。

**マルチベクトル(後期相互作用)。** ColBERTv2、Jina-ColBERT。トークンあたり1つのベクトル。MaxSimでスコアリング: 各クエリトークンについて、最も類似したドキュメントトークンを見つけ、スコアを合計する。保存とスコアリングにより高くつくが、長いクエリと領域固有コーパスで勝つ。

**BGE-M3: 3つすべて一度に。** 単一モデルは同時にデンス、スパース、マルチベクトル表現を出力する。それぞれを独立してクエリできる; スコアは重み付けされた合計で融合する。1つのチェックポイントから柔軟性が必要なときの2026年デフォルト。

**Matryoshka表現学習。** 訓練されるため、ベクトルの最初のN次元は有用なスタンドアロン埋め込みを形成する。1,536次元ベクトルを256次元に截断して、精度で約1%を支払うが、ストレージ節約は6×。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+でサポート。

### MTEBリーダーボードは部分的なストーリーを伝える

Massive Text Embedding Benchmark — 起動時(2022)に8つのタスクタイプ全体で56のタスク、MTEB v2では100以上のタスクに拡張。2026年初め、Gemini Embedding 2は検索(67.71 MTEB-R)でトップ。Cohere embed-v4はゼネラル(65.2 MTEB)を導く。BGE-M3はオープンウェイト多言語(63.0)を導く。リーダーボードは必要だが十分ではない — 常にドメインでベンチマークする。

### 3層パターン

| ユースケース | パターン |
|----------|---------|
| 高速第1パス | デンスバイエンコーダ(BGE-M3、text-3-small) |
| リコール向上 | スパース(SPLADE、BGE-M3スパース)+RRF融合 |
| トップ50での精度 | マルチベクトル(ColBERTv2)またはクロスエンコーダreranker |

ほとんどの本番スタックは3つすべてを使用する。

## ビルドしてみよう

### ステップ1: ベースライン — Sentence-BERTでデンス埋め込み

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`はドット積をコサイン類似度と同等にする。常に設定する。

### ステップ2: Matryoshka截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断後に正規化しなおす。Nomic v1.5、OpenAI text-3、Voyage-4は最初の数段階でこれが無損失になるように訓練されている。非Matryoshkaモデル(オリジナルSentence-BERT)は截断時に急激に低下する。

### ステップ3: BGE-M3多機能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

3つのインデックス、1つの推論呼び出し。スコア融合:

```python
dense_score = ... # dense_vecsの上のコサイン
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

ドメインでの重みをチューニングする。

### ステップ4: カスタムタスクでのMTEB評価

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

候補モデルを*代表的な*サブセットで実行する。リーダーボードランクだけを信頼しない — ドメインが重要だ。

### ステップ5: 手巻きコサインをスクラッチから

`code/main.py`を参照。平均化ハッシングトリック埋め込み(stdlibのみ)。トランスフォーマー埋め込みと競争力がない、しかし形を示す: トークン化 → ベクトル → 正規化 → ドット積。

## ピットフォール

- **クエリとドキュメントに同じモデルを使用。** いくつかのモデル(Voyage、Jina-ColBERT)は非対称エンコーディングを使用 — クエリとドキュメントは異なるパスを通す。常にモデルカードをチェック。
- **プレフィックスが無い。** `bge-*`モデルは`"Represent this sentence for searching relevant passages: "`をクエリの前に付ける必要がある。忘れると3-5ポイントのリコール低下。
- **Matryoshkaの過度な削減。** 1,536 → 256は通常安全だ。1,536 → 64はそうではない。評価セットで検証する。
- **コンテキスト截断。** ほとんどのモデルは最大長を超える入力を黙って截断する。長いドキュメントは分割が必要だ(レッスン23参照)。
- **レイテンシーテール無視。** MTEBスコアはp99レイテンシーを隠す。600Mモデルは335Mモデルを2ポイント上回るかもしれないが、クエリあたり3倍のコスト。

## 使用方法

2026年スタック:

| 状況 | 選択 |
|-----------|------|
| 英語のみ、高速、API | `text-embedding-3-large`または`voyage-3-large` |
| オープンウェイト、英語 | `BAAI/bge-large-en-v1.5` |
| オープンウェイト、多言語 | `BAAI/bge-m3`または`Qwen3-Embedding-8B` |
| 長いコンテキスト(32k+) | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| CPU専用デプロイメント | Nomic Embed v2(137Mパラム、MoE) |
| ストレージ制約 | Matryoshka截断+int8量子化 |
| キーワード多くのクエリ | SPLADEスパースを追加、RRFでデンスと融合 |

2026パターン: BGE-M3またはtext-3-largeで開始、MTEBでドメインで評価、領域固有モデルが3ポイント以上勝つ場合は交換。

## 出荷しよう

`outputs/skill-embedding-picker.md`として保存:

```markdown
---
name: embedding-picker
description: 与えられたコーパスとデプロイメント用の埋め込みモデル、次元、検索モードを選択する。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

コーパス(サイズ、言語、ドメイン、平均長)、デプロイメントターゲット(クラウド/エッジ/オンプレ)、レイテンシーバジェット、ストレージバジェットを指定して、以下を出力:

1. モデル。名前が付けられたチェックポイントまたはAPI。1文の理由。
2. 次元。フル/Matryoshka截断/int8量子化。ストレージバジェットに結び付けられた理由。
3. モード。デンス/スパース/マルチベクトル/ハイブリッド。理由。
4. クエリプレフィックス/モデルカードで必要な場合のテンプレート。
5. 評価計画。ドメイン関連のMTEBタスク+ドメイン評価でnDCG@10をホールドアウト。

ドメイン検証なしでMatryoshkaを<64次元に截断する推奨を拒否しなさい。10k未満のパッセージのコーパスのColBERTv2を拒否しなさい(オーバーヘッドは正当化されない)。512トークンウィンドウを持つモデルにルーティングされた長ドキュメントコーパス(>8kトークン)にフラグを立てなさい。
```

## 演習

1. **簡単。** `bge-small-en-v1.5`で100文をフル次元(384)で、次にMatryoshka 128でエンコード。10クエリでMRR低下を測定。
2. **中程度。** ドメインから500パッセージでBGE-M3デンス、スパース、colbertを比較。リコール@10でどれが勝つ?RRF融合が最良の単一モードを上回るか?
3. **難しい。** トップ2領域タスク全体で3つの候補モデルでMTEBを実行。MTEBスコア、100クエリバッチでのp99レイテンシー、$/1Mクエリを報告。Pareto最適なものを選択。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| デンス埋め込み | ベクトル | テキストあたり1つの固定サイズベクトル。ランク付けのためのコサイン類似度。 |
| スパース埋め込み | 学習されたBM25 | ボキャブラリートークンあたり1つの重み; ほとんどゼロ; 端から端まで訓練。 |
| マルチベクトル | ColBERTスタイル | トークンあたり1つのベクトル; MaxSimスコアリング; より大きなインデックス、より良いリコール。 |
| Matryoshka | ロシア人形トリック | 最初のN次元は単独で有効なより小さい埋め込み。 |
| MTEB | ベンチマーク | Massive Text Embedding Benchmark — 起動時56タスク、v2では100+。 |
| BEIR | 検索ベンチマーク | 18のゼロショット検索タスク; 多くの場合、クロスドメインの堅牢性で引用。 |
| 非対称エンコーディング | クエリ ≠ ドキュメントパス | モデルはクエリとドキュメントに異なる投影を使用。 |

## 参考文献

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — バイエンコーダペーパー。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — リーダーボードペーパー。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 統一された3モードモデル。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 次元ラダー訓練目標。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 本番での後期相互作用。
- [MTEBリーダーボードHugging Faceで](https://huggingface.co/spaces/mteb/leaderboard) — ライブランキング。
