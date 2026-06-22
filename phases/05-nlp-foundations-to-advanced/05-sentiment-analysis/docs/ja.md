# 感情分析

> 標準的なNLPタスク。古典的なテキスト分類について知る必要があることのほとんどがここに出現。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 02（BoW + TF-IDF）、Phase 2 · 14（ナイーブベイズ）
**所要時間:** 約75分

## 問題

「The food was not great.」ポジティブか負か？

感情は単純に聞こえる。レビュアーは何かを好きか嫌いか言った。文にラベル付け。それが標準的なNLPタスクになった理由は、全ての簡単に見える場合は難しい場合を隠す。否定がフリップ意味。皮肉はそれを反転。「Not bad at all」は2つの否定符号化単語にもかかわらずポジティブ。絵文字は周囲のテキストより多くの信号を運ぶ。ドメイン語彙は重要（音楽レビューの「tight」対ファッションレビューの「tight」）。

感情は古典的なNLPの働く実験室。すべてのナイーブベースラインが特定の失敗モードを持つことを理由を理解した場合、なぜすべてのより豊かなモデルが発明されたか理由を理解。このレッスンはナイーブベイズベースラインをゼロから構築し、ロジスティック回帰を追加し、本番感情を順守グレード問題にする陥阱に名付ける。

## コンセプト

古典的な感情は2ステップレシピ。

1. **表現。** テキストを機能ベクトルに変える。BoW、TF-IDF、またはn-グラム。
2. **分類。** 訓練されたサンプルの上にリニアモデル（ナイーブベイズ、ロジスティック回帰、SVM）をフィット。

ナイーブベイズは機能するダムモデル。全機能はラベル与えられ独立を仮定。`P(word | positive)` と `P(word | negative)` を数から推定。推論で、確率を乗じ。「naive」独立仮定は大きく間違っているし、結果は驚くほど強い。理由：スパーステキスト機能と中程度データで、分類器はどの方向へおよび多くあるかより各単語が傾斜する方向について気づかう。

ロジスティック回帰は独立仮説を修正。それは各機能のための重み学ぶ、否定的な重み含む。`not good` バイグラム機能は否定的な重み取得。ナイーブベイズはそれがめったに訓練ラベル付けされなかったバイグラムそれのためのことを行うことができない。

## ビルド

### ステップ1：本物のミニデータセット

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

意図的に小さい。実際の仕事はテキスト数万の例（IMDb、SST-2、Yelp polarity）を使用。数学は同一。

### ステップ2：多項ナイーブベイズをゼロから

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法スムージング（alpha=1.0）はラプラス平滑化。なしで、クラスで見たことない単語はゼロ確率と対数爆発。`alpha=0.01` は実装で一般的。`alpha=1.0` は教えるデフォルト。

### ステップ3：ロジスティック回帰をゼロから

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2正則化はここで重要。テキスト機能はスパース；L2なしでモデルは訓練サンプルを暗記。`0.01` で開始しチューン。

### ステップ4：否定を処理（失敗モード）

「not good」と「not bad」を考える。BoWクラシファイアは `{not, good}` と `{not, bad}` を見て、どちらが訓練で表示されたが多いかから学ぶ。バイグラムクラシファイアは `not_good` と `not_bad` を見て、それらを異なる機能として学ぶ。それは通常十分。

バイグラムないときに働く粗いフィックス：**否定スコーピング**。トークンに否定単語次の前缀トークン `NOT_` 次のポイントまで。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

今 `good` と `NOT_good` は異なる機能。分類器はそれらを反対に重み付けできる。前処理の3行、感情ベンチマークの測定可能な精度ジャンプ。

### ステップ5：重要なメトリック評価

精度単独はクラス不均衡の場合誤解。実感情コーパスは通常70-80%ポジティブか70-80%ネガティブ；定数多数クラシファイアは80%精度取得しおよび無価値。報告以下のすべて：

- **あたりクラス精度とリコール。** あたり1ペアクラス。マクロ平均それら単一の数取得クラス均衡を尊重。
- **マクロF1（不均衡データの主メトリック）。** あたりクラスF1スコアの平均、等しく重み付け。クラスが不均衡のときのクラス代わりに精度使用。
- **重み付けF1（代替）。** 同じマクロが重み付けクラス頻度。不均衡自体が事業意味を持つときのクラス近くマクロF1を報告。
- **混同マトリックス。** 生数。常に任意スカラーメトリック信頼前に検査；それはモデルが混同するどのペアクラスの明示。
- **あたりクラスエラーサンプル。** あたりクラス5つ引き出し誤り予測。読む。スカラーメトリックは実エラー読みに置き換わり。

激しく不均衡データのため（> 95-5比）、報告 **AUROC** と **AUPRC** 精度の代わりに。AUPRCはマイノリティクラスに対しより敏感、通常あなたが気づかう（スパム、詐欺、希な感情）。

**避けるべき一般的バグ。** マクロF1代わりにマイクロF1不均衡データで報告は多数クラスが支配的であるため高く見える数与える。マクロF1はマイノリティクラス性能を見ることを強制。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使う

scikit-learnは6行でそれを行い、正確に。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

3つ気付くこと。`stop_words=None` 否定を保持。`ngram_range=(1, 2)` バイグラムを追加、`not_good` は機能になるので。`sublinear_tf=True` は反復した単語をダンプ。これら3つのフラグはSST-2で75%-精度ベースラインと85%-精度ベースラインの差。

### トランスフォーマーに手を伸ばすべき場合

- 皮肉検出。古典的なモデルはここで失敗。時期。
- ドキュメント中盤シフト感情で長いレビュー。
- 側面基づ感情。「Camera was great but battery was terrible.」側面への感情を属性する必要。トランスフォーマーまたは構造化出力モデル。
- 非英語、低リソース言語。多言語BERTは無料でゼロショット基準をくれます。

あなたが上記のいずれも必要とした場合、フェーズ7にスキップ（トランスフォーマー深いダイブ）。それ以外、ナイーブベイズまたはロジスティック回帰TF-IDFプラスバイグラムプラス否定処理で2026本番ベースライン。

### 再現可能性陥阱（再び）

感情モデルを再訓練することは日常。再評価はしない。精度数値論文は特定スプリット、特定前処理、特定トークナイザー。あなたが同一パイプラインを使用せずにベースラインに新しいモデルを比較した場合、あなたは誤解デルタ取得。常に論文の数より、あなたのパイプラインの上でベースラインを再生成。

## 出荷

`outputs/prompt-sentiment-baseline.md` として保存：

```markdown
---
name: sentiment-baseline
description: 新しいデータセットのため感情分析ベースラインを設計。
phase: 5
lesson: 05
---

データセット記述が与えられたら（ドメイン、言語、サイズ、ラベル細粒度、遅延予算）、出力：

1. 機能抽出レシピ。トークナイザー、n-グラム範囲、ストップワードポリシー指定（通常保持）、否定処理（スコーププレフィックスまたはバイグラム）。
2. クラシファイア。ベースラインのためナイーブベイズ、本番のためロジスティック回帰、トランスフォーマー場合のみドメイン必要なら皮肉 / 側面 / 交差言語。
3. 評価プラン。精度、リコール、F1、混同マトリックス、および（スカラー単独ではなく）あたりクラスエラーサンプルを報告。
4. 展開後監視する1つの失敗モード。ドメインドリフトと皮肉は最上位2。

感情タスクのためストップワード削除を推奨拒否。クラスが不均衡のときの場合の単独メトリックとして精度報告を拒否（例えば、90%ポジティブ）。単語リッチ言語を言葉レベルTF-IDF超えるFastTextまたはトランスフォーマー埋め込みとしてフラグ。
```

## 演習

1. **簡単。** `apply_negation` を前処理ステップとしてscikit-learnパイプラインに追加し、小さな感情データセット上でF1デルタを測定。
2. **中程度。** クラス重み付けロジスティック回帰を実装（scikit-learnに `class_weight="balanced"` をパス、または勾配自分派生）。合成90-10クラス不均衡効果を測定。
3. **難しい。** 感情モデルの残差上のセカンド クラシファイアを訓練することで皮肉検出器を構築。実験セットアップを文書。あなたの精度がチャンス下にあることを警告（2クラス皮肉でチャンスレベルは~50%、最初試みはほとんどそこに着地）。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| ポーラリティ | ポジティブまたはネガティブ | バイナリラベル；時々ニュートラルまたは細粒度（5スター）に拡張。|
| 側面基づ感情 | あたり側面ポーラリティ | テキストで述べられた特定のエンティティまたは属性への感情を属性。|
| 否定スコーピング | 反転近いトークン | `NOT_` 「not」の後のトークンに前缀、ポイントまで。|
| ラプラス平滑化 | カウントに1を追加 | ナイーブベイズのゼロ確率機能を防止。|
| L2正則化 | 重み縮小 | `lambda * sum(w^2)` を損失に追加。スパーステキスト機能のためのエッセンシャル。|

## 参考文献

- [Pang and Lee (2008)。意見採掘と感情分析](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基礎的なサーベイ。長い、最初の4セクションは古典を全てカバー。
- [Wang and Manning (2012)。ベースラインとバイグラム：シンプル、良い感情とトピック分類](https://aclanthology.org/P12-2018/) — 短いテキストでbaigram + ナイーブベイズが打つのは難しい論文。
- [scikit-learnテキスト機能抽出ドキュメント](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer`、すべてのノブのための参考チューンします。
