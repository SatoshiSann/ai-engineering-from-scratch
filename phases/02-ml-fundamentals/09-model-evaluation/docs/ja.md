# モデル評価

> モデルの価値は、それをどう測定するかにかかっている。

**タイプ:** 構築
**言語:** Python
**前提条件:** Phase 1（確率と分布、MLのための統計）、Phase 2 レッスン1〜8
**所要時間:** 約90分

## 学習目標

- K分割交差検証と層化K分割交差検証をスクラッチで実装し、不均衡データにおいて層化が重要な理由を説明できる
- 適合率、再現率、F1、AUC-ROC、および回帰指標（MSE、RMSE、MAE、決定係数）をスクラッチで計算できる
- 学習曲線を解釈して、モデルが高バイアスか高分散かを診断できる
- データリーク、誤った指標の選択、テストセットの汚染など、よくある評価ミスを特定できる

## 問題

モデルを訓練した。データ上で95%の精度が出た。これは良いモデルだろうか？

場合による。データの95%が一つのクラスに属している場合、常にそのクラスを予測するモデルも95%の精度を達成できるが、それはまったく役に立たない。訓練に使ったデータで評価した場合、モデルが単に答えを暗記しただけなので、その95%という数字は無意味だ。データセットに時系列の要素があり、分割前にランダムにシャッフルした場合、モデルが過去を予測するために将来のデータを使っている可能性がある。

モデル評価は、ほとんどのMLプロジェクトが失敗する場所だ。誤った指標は悪いモデルを良く見せる。誤った分割はモデルにカンニングを許す。誤った比較では、より悪いモデルを選んでしまう。評価を正しく行うことは任意ではない。本番環境で機能するモデルと、実際のデータを見た瞬間に失敗するモデルの違いがここにある。

## コンセプト

### 訓練・検証・テスト

```mermaid
flowchart LR
    A[全データセット] --> B[訓練セット 60-70%]
    A --> C[検証セット 15-20%]
    A --> D[テストセット 15-20%]
    B --> E[モデルを適合]
    E --> C
    C --> F[ハイパーパラメータを調整]
    F --> E
    F --> G[最終モデル]
    G --> D
    D --> H[性能を報告]
```

3つの分割、3つの目的：

- **訓練セット**: モデルがこのデータから学習する。訓練中にこれらの例を見る。
- **検証セット**: ハイパーパラメータの調整やモデル選択に使う。モデルはこのデータで訓練しないが、あなたの判断はこれに影響される。
- **テストセット**: 最終的な性能を報告するために、最後に一度だけ使う。テスト性能を見てモデルを変更しに戻った場合、それはもはやテストセットではなく、第二の検証セットになってしまう。

テストセットは、報告された性能が真に未見のデータに対する性能を反映していることを保証するためのホールドアウト保証だ。

### K分割交差検証

小さいデータセットでは、単一の訓練/検証分割はデータを無駄にし、ノイズの多い推定を与える。K分割交差検証は、訓練と検証の両方にすべてのデータを使う：

```mermaid
flowchart TB
    subgraph Fold1["分割 1"]
        direction LR
        V1["検証"] --- T1a["訓練"] --- T1b["訓練"] --- T1c["訓練"] --- T1d["訓練"]
    end
    subgraph Fold2["分割 2"]
        direction LR
        T2a["訓練"] --- V2["検証"] --- T2b["訓練"] --- T2c["訓練"] --- T2d["訓練"]
    end
    subgraph Fold3["分割 3"]
        direction LR
        T3a["訓練"] --- T3b["訓練"] --- V3["検証"] --- T3c["訓練"] --- T3d["訓練"]
    end
    subgraph Fold4["分割 4"]
        direction LR
        T4a["訓練"] --- T4b["訓練"] --- T4c["訓練"] --- V4["検証"] --- T4d["訓練"]
    end
    subgraph Fold5["分割 5"]
        direction LR
        T5a["訓練"] --- T5b["訓練"] --- T5c["訓練"] --- T5d["訓練"] --- V5["検証"]
    end
    Fold1 --> R["スコアの平均"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. データをK個の等サイズの分割に分ける
2. 各分割に対して、K-1個の分割で訓練し、残りの分割で検証する
3. K個の検証スコアの平均を取る

K=5またはK=10が標準的な選択だ。すべてのデータポイントが検証に正確に一度使われる。平均スコアは単一の分割よりも安定した推定値だ。

**層化K分割**: 各分割でクラス分布を保持する。データセットがクラスAが70%、クラスBが30%の場合、各分割はほぼ同じ比率になる。ランダムな分割ではすべての少数クラスサンプルが一つの分割に入ってしまう可能性がある不均衡データセットで重要だ。

### 分類指標

**混同行列**: 基盤となるもの。二値分類の場合：

|  | 予測陽性 | 予測陰性 |
|--|---|---|
| 実際に陽性 | 真陽性 (TP) | 偽陰性 (FN) |
| 実際に陰性 | 偽陽性 (FP) | 真陰性 (TN) |

この行列から、他のすべての指標が導かれる：

- **精度 (Accuracy)** = (TP + TN) / (TP + TN + FP + FN)。正しい予測の割合。クラスが不均衡な場合は誤解を招く。
- **適合率 (Precision)** = TP / (TP + FP)。陽性と予測されたものの中で、実際に陽性だったものはどれくらいか？偽陽性が高コストな場合に使う（例：スパムフィルターが本物のメールをスパムとしてマークする）。
- **再現率 (Recall)**（感度）= TP / (TP + FN)。実際の陽性のうち、どれくらい捉えられたか？偽陰性が高コストな場合に使う（例：がんのスクリーニングで腫瘍を見逃す）。
- **F1スコア** = 2 * 適合率 * 再現率 / (適合率 + 再現率)。適合率と再現率の調和平均。どちらも明確に優位でない場合のバランス指標。
- **AUC-ROC**: ROC曲線（受信者操作特性曲線）の下面積。さまざまな分類閾値で真陽性率と偽陽性率をプロットする。AUC=0.5はランダムな推測、AUC=1.0は完全な分離を意味する。閾値に依存しない：選択した閾値に関係なく、モデルが陽性を陰性よりも高くランク付けする能力を測定する。

### 回帰指標

- **MSE**（平均二乗誤差）= mean((y_true - y_pred)^2)。大きな誤差を二乗で罰する。外れ値に敏感。
- **RMSE**（二乗平均平方根誤差）= sqrt(MSE)。目標変数と同じ単位。MSEより解釈しやすい。
- **MAE**（平均絶対誤差）= mean(|y_true - y_pred|)。すべての誤差を線形に扱う。MSEより外れ値に頑健。
- **決定係数（R二乗）** = 1 - SS_res / SS_tot。ここで SS_res = sum((y_true - y_pred)^2)、SS_tot = sum((y_true - y_mean)^2)。モデルによって説明される分散の割合。R^2=1.0は完璧。R^2=0.0はモデルが常に平均を予測することと同じレベル。R^2はモデルが平均より悪い場合に負になることもある。

### 学習曲線

訓練スコアと検証スコアを訓練セットサイズの関数としてプロットする：

- **高バイアス（過小適合）**: 両方の曲線が低いスコアに収束する。より多くのデータを加えても助けにならない。より複雑なモデルが必要だ。
- **高分散（過学習）**: 訓練スコアは高いが、検証スコアははるかに低い。それらの間のギャップが大きい。より多くのデータを加えることで改善するはずだ。

### 検証曲線

訓練スコアと検証スコアをハイパーパラメータの関数としてプロットする：

- 低複雑度では：両方のスコアが低い（過小適合）
- 適切な複雑度では：両方のスコアが高く近い
- 高複雑度では：訓練スコアは高いままだが検証スコアが下がる（過学習）

最適なハイパーパラメータ値は、検証スコアがピークになるところだ。

### よくある評価ミス

**データリーク**: テストセットからの情報が訓練に漏れる。例：分割前に全データセットでスケーラーを適合する、時系列予測に将来のデータを含める、目標から派生した特徴量を使う。常に先に分割し、その後に前処理する。

**クラス不均衡**: 取引の99%は正当で、1%が不正だ。常に「正当」と予測するモデルは99%の精度を得るが、まったく役に立たない。代わりに適合率、再現率、F1、またはAUC-ROCを使う。

**誤った指標**: 再現率を最適化すべき（医療診断）ときに精度を最適化する、あるいはデータに大きな外れ値がある（代わりにMAEを使う）ときにRMSEを最適化する。

**層化分割を使わない**: 不均衡データでは、ランダムな分割によって検証分割に少数クラスのサンプルがほとんど入らず、不安定な推定値が得られることがある。

**頻繁なテスト**: テスト性能を確認して調整するたびに、テストセットに過学習する。テストセットは一回しか使えない。

## 構築

### ステップ1: 訓練/検証/テスト分割

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### ステップ2: K分割および層化K分割交差検証

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### ステップ3: 混同行列と分類指標

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### ステップ4: 回帰指標

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### ステップ5: 学習曲線

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### ステップ6: テスト用の簡単な分類器とデモ全体

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== 訓練/検証/テスト分割 ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")
    print(f"  訓練クラス分布: {sum(y_train)}/{len(y_train)} 陽性")
    print(f"  検証クラス分布: {sum(y_val)}/{len(y_val)} 陽性")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== 分類指標 ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  混同行列: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  精度:      {accuracy(y_test, y_pred):.4f}")
    print(f"  適合率:    {precision(y_test, y_pred):.4f}")
    print(f"  再現率:    {recall(y_test, y_pred):.4f}")
    print(f"  F1スコア:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K分割交差検証 (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  分割スコア: {[round(s, 4) for s in cv_scores]}")
    print(f"  平均: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== 層化K分割交差検証 (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  分割スコア: {[round(s, 4) for s in strat_scores]}")
    print(f"  平均: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== 不均衡データ: なぜ精度は嘘をつくか ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  クラス分布: {positives} 陽性, {len(y_imb) - positives} 陰性 ({positives/len(y_imb)*100:.1f}% 陽性)")

    always_negative = [0] * len(y_imb)
    print(f"  常に陰性のベースライン:")
    print(f"    精度:      {accuracy(y_imb, always_negative):.4f}")
    print(f"    適合率:    {precision(y_imb, always_negative):.4f}")
    print(f"    再現率:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1スコア:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  不均衡データで訓練されたモデル:")
    print(f"    精度:      {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    適合率:    {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    再現率:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1スコア:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== 回帰指標 ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R二乗:     {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  平均ベースライン:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R二乗:     {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== 学習曲線 ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'サイズ':>6} {'訓練':>8} {'検証':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== 統計的モデル比較 ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  モデルA (100エポック) 平均: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  モデルB (500エポック) 平均: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  平均差: {mean_diff:.4f}")
    print(f"  対応t統計量: {t_stat:.4f}")
    print(f"  (|t| > 2.78 で df=4 のとき p<0.05 で有意)")
```

## 活用

scikit-learnでは、評価はワークフローに組み込まれている：

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

スクラッチ実装は、交差検証が実際に何をしているか（マジックなし、ただのforループとインデックス追跡）、各指標がどのように計算されるか（TP/FP/TN/FNを数えるだけ）、なぜ層化が重要か（各分割でクラス比率を保持する）を正確に示す。ライブラリ版は並列性、より多くのスコアリングオプション、パイプラインとの統合を追加する。

## Ship It

このレッスンが生成するもの：
- `outputs/skill-evaluation.md` - 分類と回帰モデルの評価戦略をカバーするスキル

## 演習

1. 適合率-再現率曲線を実装する：異なる閾値での適合率と再現率をプロットする。PR曲線の下面積（平均適合率）を計算する。不均衡データセットでPR曲線とROC曲線を比較し、それぞれがより有益な場合を説明する。
2. ネストされた交差検証ループを構築する：外側のループはモデル性能を評価し、内側のループはハイパーパラメータを調整する。検証データを評価に漏らさずに、2つのモデルを公平に比較するために使う。
3. モデル比較のための置換検定を実装する：ラベルをシャッフルし、再訓練し、性能を測定する。100回繰り返してヌル分布を構築する。この分布に対する観測モデル性能のp値を計算する。

## 用語集

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| 過学習 | 「訓練データを暗記している」 | モデルが訓練データのノイズを捉えており、訓練では良い性能だが未見のデータでは悪い性能を示す |
| 交差検証 | 「異なるサブセットでテストする」 | どのデータ部分を検証に使うかを体系的に入れ替え、すべての入れ替えで結果を平均する |
| 適合率 | 「陽性予測のうち正しいものがどれくらいか」 | TP / (TP + FP): 陽性予測のうち実際に陽性のものの割合 |
| 再現率 | 「実際の陽性をどれくらい見つけられたか」 | TP / (TP + FN): 正しく識別された実際の陽性の割合 |
| AUC-ROC | 「モデルがクラスをどれくらい分離できるか」 | すべての閾値での真陽性率と偽陽性率の曲線の下面積。0.5（ランダム）から1.0（完璧） |
| 決定係数 | 「どれくらいの分散が説明されているか」 | 1 - (残差の二乗和 / 全二乗和): モデルが捉えた目標分散の割合 |
| データリーク | 「モデルがカンニングした」 | 予測時に利用できない情報を訓練中に使用し、楽観的な評価につながる |
| 学習曲線 | 「データが増えると性能がどう変わるか」 | 訓練セットサイズに対する訓練スコアと検証スコアのプロット。過小適合または過学習を明らかにする |
| 層化分割 | 「クラス比率をバランスよく保つ」 | 各サブセットが全データセットと同じ割合の各クラスを持つようにデータを分割する |

## 参考文献

- [scikit-learn モデル選択ガイド](https://scikit-learn.org/stable/model_selection.html) - 交差検証、指標、ハイパーパラメータ調整の包括的なリファレンス
- [精度を超えて：適合率と再現率（Google ML集中講座）](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - インタラクティブな例を用いた明確な説明
- [交差検証手続きの調査（Arlot & Celisse, 2010）](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - 異なるCV戦略がいつ、なぜ機能するかの厳密な考察
