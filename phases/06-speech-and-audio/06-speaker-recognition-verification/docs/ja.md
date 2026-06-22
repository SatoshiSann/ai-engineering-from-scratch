# 話者認識と話者照合

> ASRは「彼らは何と言ったか?」を問う。話者認識は「誰が言ったか?」を問う。数学は同じに見える — 埋め込みプラスコサイン — が、あらゆる本番の判断は単一のEER数値にかかっている。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 02 (Spectrograms & Mel)、Phase 5 · 22 (Embedding Models)
**所要時間:** 約45分

## 問題

ユーザーがパスフレーズを言う。あなたは知りたい。これは本人が主張する通りの人物か(*照合*、1:1)、それとも登録バンクの最初の人物か(*識別*、1:N)? あるいはどちらでもない — これは未知の話者か(*オープンセット*)?

2018年以前: GMM-UBM + i-vectors。妥当なEERだが、チャネルシフト(電話対ラップトップ)や感情に対して脆弱。2018〜2022年: x-vectors(角度マージンで訓練されたTDNNバックボーン)。2022年以降: ECAPA-TDNNとWavLM-large埋め込み。2026年までに、この分野は3つのモデルと1つの指標に支配されている。

その指標が**EER** — Equal Error Rate(等価エラー率)だ。False Accept Rate = False Reject Rateとなるように決定しきい値を設定する。その交差点がEERだ。あらゆる論文、あらゆるリーダーボード、あらゆる調達案件で使われる。

## コンセプト

![埋め込み + コサイン + EERによる登録 + 照合パイプライン](../assets/speaker-verification.svg)

**パイプライン。** 登録: ターゲット話者の5〜30秒を記録し、固定次元の埋め込み(ECAPA-TDNNで192次元、WavLM-largeで256次元)を計算する。照合: テスト発話の埋め込みを取得し、コサイン類似度を計算し、しきい値と比較する。

**ECAPA-TDNN (2020年、2026年でも依然として支配的)。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。squeeze-excitation付きの1D convブロック、マルチヘッドアテンションプーリング、続いて192次元への線形層。VoxCeleb 1+2(2,700話者、110万発話)上でAdditive Angular Margin損失(AAM-softmax)で訓練。

**WavLM-SV (2022年以降)。** 事前学習済みのWavLM-large SSLバックボーンをAAM損失でファインチューニングする。より高品質だが遅い — 300+ MB対15 MB。

**x-vector (ベースライン)。** TDNN + 統計プーリング。古典的。CPU / エッジでは今も有用。

**AAM-softmax。** 角度空間に追加マージン`m`を加えた標準的なsoftmax: 正しいクラスに対して`cos(θ + m)`。クラス間の角度分離を強制する。典型的には`m=0.2`、スケール`s=30`。

### スコアリング

- 登録埋め込みとテスト埋め込みの間の**コサイン**。しきい値ベースの決定。
- **PLDA (Probabilistic LDA)。** 埋め込みを、同一話者対異話者が閉形式の尤度比を持つ潜在空間に射影する。コサインの上に追加して、EERを10〜20%削減する。2020年以前は標準。現在はクローズドセット構成でのみ使われる。
- **スコア正規化。** `S-norm`または`AS-norm`: 各スコアを偽者の平均と標準偏差のコホートに対して正規化する。クロスドメイン評価に不可欠。

### 知っておくべき数値(2026年)

| モデル | VoxCeleb1-O EER | パラメータ | スループット (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (古典的) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 セグメンテーション + 埋め込み | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 話者ダイアライゼーション

マルチ話者クリップにおける「いつ誰が話したか」。パイプライン: VAD → セグメント化 → 各セグメントを埋め込み → クラスタリング(凝集型またはスペクトル) → 境界を平滑化。現代のスタック: `pyannote.audio` 3.1。話者セグメンテーション + 埋め込み + クラスタリングを1回の呼び出しの裏にまとめている。AMIにおける2026年のSOTA DERは約15%(2022年の23%から低下)。

## 作ってみる

### ステップ1: MFCC統計からのトイ埋め込み

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

SOTAには遥かに及ばない — 教育目的のみ。`code/main.py`はこれを合成話者データ上の概念実証として使う。

### ステップ2: コサイン類似度 + しきい値

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### ステップ3: 類似度ペアからのEER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

(eer, threshold_at_eer)を返す。両方を報告すること。

### ステップ4: SpeechBrainによる本番

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### ステップ5: pyannoteによる話者ダイアライゼーション

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| クローズドセット1:1照合、エッジ | ECAPA-TDNN + コサインしきい値 |
| オープンセット照合、クラウド | WavLM-SV + AS-norm |
| 話者ダイアライゼーション(会議、ポッドキャスト) | `pyannote/speaker-diarization-3.1` |
| アンチスプーフィング(リプレイ / ディープフェイク検出) | AASISTまたはRawNet2 |
| 小型組み込み(KWS + 登録) | Titanet-Small (NeMo) |

## 落とし穴

- **チャネルの不一致。** VoxCeleb(ウェブ動画)で訓練されたモデル ≠ 電話通話オーディオ。常にターゲットチャネルで評価すること。
- **短い発話。** EERはテストオーディオが3秒を下回ると急激に劣化する。
- **ノイズ付きの登録。** 1つのノイジーな登録がアンカーを汚染する。3つ以上のクリーンなサンプルを使い、平均すること。
- **条件をまたいだ固定しきい値。** 常にターゲットドメインからのホールドアウト開発セットでしきい値を調整すること。
- **非正規化埋め込み上のコサイン。** まずL2正規化すること。さもないとマグニチュードが支配する。

## 出荷する

`outputs/skill-speaker-verifier.md`として保存する。モデル、登録プロトコル、しきい値調整計画、不正対策を選択する。

## 演習

1. **易しい。** `code/main.py`を実行する。合成「話者」(異なるトーンプロファイル)を構築し、登録し、100ペアの試行リストでEERを計算する。
2. **中級。** 30個のVoxCeleb1発話(5話者 × 各6)でSpeechBrain ECAPAを使え。コサイン対PLDAでEERを計算せよ。
3. **難しい。** `pyannote.audio`で完全な登録 → 話者ダイアライゼーション → 照合パイプラインを構築せよ。AMI開発セットでDERを評価せよ。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| EER | 看板指標 | False Accept = False Rejectとなるしきい値。 |
| Verification | 1:1 | 「これはAliceか?」 |
| Identification | 1:N | 「誰が話しているか?」 |
| Open-set | 未知の可能性あり | テストセットに未登録話者が含まれうる。 |
| Enrollment | 登録 | 話者の参照埋め込みを計算すること。 |
| AAM-softmax | 損失 | 加法的角度マージン付きsoftmax。クラスタ分離を強制する。 |
| PLDA | 古典的スコアリング | Probabilistic LDA。埋め込みの上の尤度比スコアリング。 |
| DER | 話者ダイアライゼーション指標 | Diarization Error Rate — ミス + 誤警報 + 混同。 |

## 参考文献

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 古典的な深層埋め込みの論文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020〜2026年の支配的アーキテクチャ。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SVと話者ダイアライゼーションのためのSSLバックボーン。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 本番の話者ダイアライゼーション + 埋め込みスタック。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — モデル間の現在のEER順位。
