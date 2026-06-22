# 教師なし学習

> ラベルも、教師もいない。アルゴリズムが自分自身でデータの構造を発見する。

**タイプ:** ビルド
**言語:** Python
**前提条件:** フェーズ1（[レッスン14 ノルムと距離](file:///Users/satoshimochizuki/Documents/github/ai-engineering-from-scratch/phases/01-math-foundations/14-norms-and-distances/)、[レッスン06 確率と分布](file:///Users/satoshimochizuki/Documents/github/ai-engineering-from-scratch/phases/01-math-foundations/06-probability-and-distributions/)）、フェーズ2 レッスン01〜06
**時間:** 約90分

## 学習目標

- K-Means、DBSCAN、および混合ガウスモデル（GMM: Gaussian Mixture Model）をゼロから実装し、それらのクラスタリングの挙動を比較する
- シルエットスコア（Silhouette score）とエルボー法（Elbow method）を用いてクラスタの品質を評価し、最適なクラスタ数 K を選択する
- DBSCANがどのような状況で K-Means を上回るのかを説明し、非球状のクラスタや外れ値に対応できるアルゴリズムを特定する
- クラスタリング手法を用いた異常検知（Anomaly detection）パイプラインを構築し、正常なパターンから逸脱した点を検出する

## 問題の背景

これまでのすべての機械学習のレッスンでは、「入力があり、正解の出力がある」というラベル付きデータを前提としてきた。しかし現実世界において、データへのラベル付け（アノテーション）は非常にコストがかかる。病院には数百万もの患者の記録があるが、各レコードにどの疾患カテゴリに属するかのタグを手作業で付けた人はいない。ECサイトには数百万ものユーザーセッションがあるが、顧客セグメントを手作業で分類した人はいない。セキュリティチームには膨大なネットワークログがあるが、すべての異常値にフラグを立ててくれた人などいない。

教師なし学習（Unsupervised learning）は、何を探索すべきかを指示されることなく、データ内のパターンを見つけ出す。類似したデータポイントをグループ化し、隠れた構造を発見し、異常値を浮かび上がらせる。教師あり学習が「解答付きの教科書」から学習することであるならば、教師なし学習は「生のデータをじっと眺め、パターンが自然に浮かび上がるのを待つ」ようなものである。

課題：ラベルがないため、何が「正解」で何が「不正解」かを直接測定することができない。そのため、アルゴリズムが発見した構造が有意義であるかどうかを評価するには、異なるアプローチ（評価ツール）が必要になる。

## 概念

### クラスタリング：類似したものをグループ化する

クラスタリングは、各データポイントをグループ（クラスタ）に割り当てる。同じグループ内のポイント同士は互いに類似し、異なるグループのポイントとは類似しないようにグループ分けを行う。ここで常に問われるのは、「類似している」とは何を意味するのか、ということである。

```mermaid
flowchart LR
    A[生データ] --> B{手法の選択}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[階層的クラスタリング]
    B --> F[混合ガウスモデル (GMM)]
    C --> G[平坦で球状のクラスタ]
    D --> H[任意の形状、ノイズ検出]
    E --> I[入れ子状クラスタのツリー構造]
    F --> J[ソフトな割り当て、楕円状のクラスタ]
```

### K-Means：クラスタリングの主力手法

K-Meansは、データを正確に $K$ 個のクラスタに分割する。各クラスタには重心（セントロイド: Centroid、クラスタの重心）があり、すべてのデータポイントは最も近い重心のクラスタに属する。

ロイドのアルゴリズム（Lloyd's algorithm）：

1. 初期重心として、データからランダムに $K$ 個の点を選択する
2. 各データポイントを最も近い重心に割り当てる
3. 各重心を、その重心に割り当てられたポイントの平均値として再計算する
4. 割り当てが変化しなくなるまで、ステップ 2 と 3 を繰り返す

目的関数（慣性: Inertia）は、各点からその割り当てられた重心までの距離の2乗和を測定する。K-Meansはこの値を最小化するが、見つけられるのは局所最適解（ローカルミニマ）のみである。初期重心の選び方によって異なる結果が得られることがある。

### クラスタ数 K の選択

主に使われる2つの手法：

**エルボー法（Elbow method）：** $K = 1, 2, 3, \dots, n$ に対して K-Means を実行する。$K$ の値に対する慣性（Inertia）をプロットする。クラスタ数を増やしても慣性があまり減少しなくなる「エルボー（肘）」と呼ばれる屈曲点を探す。

**シルエットスコア（Silhouette score）：** 各データポイントについて、自分が属するクラスタ内の他の点との類似度 $a$ と、最も近い他のクラスタの点との類似度 $b$ を測定する。シルエット係数は $(b - a) / \max(a, b)$ で計算され、範囲は -1（誤ったクラスタに分類されている）から +1（適切にクラスタリングされている）である。すべての点のシルエット係数の平均をとってグローバルな評価スコアとする。

### DBSCAN：密度ベースのクラスタリング

K-Meansはクラスタが球状であると想定し、事前にクラスタ数 $K$ を決定する必要がある。DBSCAN（Density-Based Spatial Clustering of Applications with Noise）はどちらの想定も必要としない。データが密集している「高密度な領域」をクラスタとし、データの薄い「低密度な領域」によってそれらが隔てられていると捉える。

2つのパラメータ：
- **eps**：近傍とみなす半径
- **min_samples**：高密度な領域を形成するために必要な最小データポイント数

3つのポイントの種類：
- **コア点（Core point）**：半径 `eps` 以内に少なくとも `min_samples` 個のポイントを持つ点
- **ボーダー点（境界点: Border point）**：コア点の `eps` 以内にあるが、自身はコア点ではない点
- **ノイズ点（Noise point）**：コア点でもボーダー点でもない点。これらは外れ値（Outliers）である。

DBSCANは、互いに `eps` 以内にあるコア点同士を接続して同じクラスタにする。ボーダー点は、近くのコア点のクラスタに結合する。ノイズ点はどのクラスタにも属さない。

強み：任意の形状のクラスタを発見できる、クラスタ数を自動で決定できる、外れ値を識別できる。弱み：クラスタごとに密度が大きく異なるデータの処理が苦手である。

### 階層的クラスタリング

入れ子（Nested）状のクラスタからなるツリー（デンドログラム: Dendrogram、系統図）を構築する。

凝集型（ボトムアップ方式）の手順：
1. 各データポイントをそれぞれ単一のクラスタとして開始する
2. 最も近い2つのクラスタを結合する
3. クラスタが1つになるまでステップ2を繰り返す
4. 得られたデンドログラムを望むレベル（高さ）で切断することで、任意のクラスタ数 $K$ を得る

クラスタ間の「近さ」は以下の方法で測定される：
- **単一連結法（Single linkage）**：2つのクラスタの点同士のうち、最も近い2点間の距離
- **完全連結法（Complete linkage）**：最も遠い2点間の距離
- **平均連結法（Average linkage）**：すべてのペアの距離の平均
- **ウォード法（Ward's method）**：結合した結果、クラスタ内の総分散の増加量が最小になるような組み合わせを選択する

### 混合ガウスモデル（GMM）

K-Meansは「ハードな割り当て（各データポイントは正確に1つのクラスタに属する）」を行う。これに対してGMM（Gaussian Mixture Model）は「ソフトな割り当て（各データポイントが各クラスタに属する確率を算出する）」を行う。

GMMは、データがそれぞれ異なる平均と共分散を持つ $K$ 個のガウス分布（正規分布）の混合から生成されたと仮定する。期待値最大化（EM: Expectation-Maximization）アルゴリズムは、以下のステップを交互に繰り返す：

- **Eステップ（Expectation）**：各データポイントが各ガウス成分に属する確率（負担率）を計算する
- **Mステップ（Maximization）**：データの尤度（Likelihood）を最大化するように、各ガウス成分の平均、共分散、および混合重みを更新する

GMMは、（K-Meansのように球状だけでなく）楕円状のクラスタをモデル化でき、重なり合うクラスタも自然に処理できる。

### 手法の使い分け

| 手法 | 最適な状況 | 避けるべき状況 |
|---|---|---|
| K-Means | 大規模データセット、球状クラスタ、クラスタ数 K が既知である場合 | 不規則な形状、外れ値が存在する場合 |
| DBSCAN | クラスタ数 K が未知、任意の形状、外れ値検出が必要な場合 | 領域ごとに密度が異なる場合、超高次元データ |
| 階層的 | 小規模データセット、デンドログラムが必要、K が未知である場合 | 大規模データセット（メモリ計算量が $O(n^2)$ となるため） |
| GMM | クラスタ同士が重なり合っている場合、ソフトな確率的割り当てが必要な場合 | 極端に大規模なデータ、特徴量が多すぎる場合 |

### クラスタリングによる異常検知

クラスタリングは、異常検知（Anomaly detection）に自然に応用できる：
- **K-Means**：どの重心からも極端に遠いデータポイントを異常値とする
- **DBSCAN**：ノイズ点として分類されたものを、定義通り異常値とする
- **GMM**：すべてのガウス分布において発生確率（尤度）が極めて低いポイントを異常値とする

## 実装してみよう

### ステップ 1：K-Means のスクラッチ実装

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### ステップ 2：エルボー法とシルエットスコア

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### ステップ 3：DBSCAN のスクラッチ実装

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### ステップ 4：混合ガウスモデル（EM アルゴリズム）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### ステップ 5：テストデータの生成と実行

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

完全な実装、ヘルパーメソッド、およびデモについては [clustering.py](file:///Users/satoshimochizuki/Documents/github/ai-engineering-from-scratch/phases/02-ml-fundamentals/07-unsupervised-learning/code/clustering.py) を参照。

## 使ってみよう

scikit-learn を用いれば、同じアルゴリズムを数行で記述できる：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

このスクラッチ実装によって、これらのライブラリが内部で何を計算しているのかが正確に理解できる。K-Means は割り当てと再計算を繰り返す。DBSCAN は高密度のシードからクラスタを拡張する。GMM は期待値の算出とパラメータの最大化を繰り返す。ライブラリによる実装では、数値的な安定性やスマートな初期化（K-Means++）、GPUアクセラレーションなどが追加されているが、そのコアとなるロジックは同一である。

## 演習問題

1. K-Means++ 初期化を実装せよ。重心を完全にランダムに選ぶ代わりに、最初の重心をランダムに選び、その後の重心は「既存の最も近い重心からの距離の2乗」に比例する確率で選択する。ランダム初期化と収束速度を比較せよ。
2. 階層的凝集クラスタリングをコードに追加せよ。ウォード法による連結を実装し、デンドログラム（マージ順序を表す入れ子のリスト）を生成せよ。異なる高さで木を切断し、K-Means の結果と比較せよ。
3. シンプルな異常検知パイプラインを構築せよ：同じデータに対して DBSCAN と GMM を実行し、両方の手法で外れ値（DBSCAN におけるノイズ点、GMM における低確率値）と判定されたデータポイントにフラグを立てよ。一致度を測定し、結果が食い違ったケースについて考察せよ。

## 主要用語

| 用語 | よくある説明 | 実際の意味 |
|---|---|---|
| クラスタリング (Clustering) | 「似たもの同士をグループ化する」 | データをいくつかの部分集合に分割すること。特定の距離尺度を用いて、グループ内の類似度がグループ間の類似度を上回るように分割する |
| 重心 (Centroid) | 「クラスタの中心」 | クラスタに割り当てられたすべてのデータポイントの平均値。K-Means においてクラスタを代表する点として用いられる |
| 慣性 (Inertia) | 「クラスタの密集度」 | 各データポイントからその割り当てられた重心までの距離の2乗和。値が小さいほどクラスタが密集していることを示す |
| シルエットスコア (Silhouette score) | 「クラスタの分離の良さ」 | 各データポイントについて $(b - a) / \max(a, b)$ で計算される。ここで $a$ はクラスタ内平均距離、$b$ は最も近い他クラスタとの平均距離 |
| コア点 (Core point) | 「密度の高い領域内の点」 | DBSCAN において、自身から半径 `eps` 以内に少なくとも `min_samples` 個の近傍点を持つデータポイント |
| EMアルゴリズム (EM algorithm) | 「ソフトな K-Means」 | 期待値最大化（Expectation-Maximization）。所属確率を算出する Eステップと、確率分布のパラメータを更新する Mステップを交互に繰り返す |
| デンドログラム (Dendrogram) | 「クラスタのツリー」 | 階層的クラスタリングにおいて、クラスタが結合（または分割）されていった順序と距離を示すツリー状の図 |
| 異常値 (Anomaly) | 「外れ値」 | 期待されるパターンに従わないデータポイント。DBSCAN におけるノイズ点や、GMM における低確率点として検出される |

## 推薦図書・論文

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng氏によるクラスタリングとEMアルゴリズムの講義ノート
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - すべてのクラスタリングアルゴリズムを視覚的例とともに実用的に解説したガイド
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 密度ベースのクラスタリングを導入した原著論文
