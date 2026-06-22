# テキスト要約

> 抽出システムはドキュメントが何を言ったか教えます。抽出的システムはドキュメントが何を意味したか教えます。異なるタスク、異なる陥阱。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 02（BoW + TF-IDF）、Phase 5 · 11（機械翻訳）
**所要時間:** 約75分

## 問題

2,000単語のニュース記事はあなたのフィードに着地。あなたは120単語がそれをキャプチャするが必要。あなたはどちらかできます記事から3つの最も重要な文を選ぶ（抽出）またはあなた自身の言葉で内容を書き直す（抽象的）。どちらでも要約という。それらは完全に異なる問題。

抽出要約はランキング問題。全て文をスコア、トップ `k` を戻す。出力は常に文法的である理由それは文字通り持ち上げられた。リスク記事全体に分散されたコンテンツを見落とすこと。

抽象的要約は世代問題。トランスフォーマーは入力で条件付けられた新しいテキストを生成。出力は流暢で圧縮的ですが事実を幻想するかもしれません。リスクは自信に満ちた捏造。

このレッスンは両方をビルド、そのいずれかが所有する失敗モード持つ。

## コンセプト

![抽出TextRank 対 抽象的トランスフォーマー](../assets/summarization.svg)

**抽出的。** 記事をグラフとして扱う各ノードは文と境は類似度のどこ。PageRank（またはそのようなもの）実行全体に接続される方法によって得点するグラフを超えて文。最高得点文は要約。標準的な実装は **TextRank** （Mihalcea と Tarau、2004）。

**抽象的。** トランスフォーマーエンコーダデコーダ（BART、T5、Pegasus）をドキュメント要約ペアで微調整。推論で、モデルはドキュメントを読み交差注意経由token-by-token 要約を生成。特に Pegasus は要約なし多くをファインチューニング優れてさせるギャップ文事前訓練目的を使用。

**ROUGE** による評価（Gisting評価の想起志向の代用）。ROUGE-1 と ROUGE-2 はユニグラムとバイグラムオーバーラップをスコア。ROUGE-L は最長一般シーケンスをスコア。高い良いですが 40 ROUGE-L は「良い」と 50 は「例外的」全ての論文は3つすべてを報告。`rouge-score` パッケージを使用。

## ビルド

### ステップ1：TextRank（抽出的）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

名前が値する2つのこと。類似度機能はログ正規化単語オーバーラップ、元 TextRank バリアント。TF-IDF ベクトルのコサイン。ダンピング係数 0.85 と反復カウントは PageRank デフォルト。

### ステップ2：BART で抽象的

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN は CNN/DailyMail コーパスで微調整。外のボックスでニュース スタイルの要約を生成。他のドメイン（科学論文、対話、法律）のために、対応する Pegasus チェックポイント を使用またはあなたのターゲット データで微調整。

### ステップ3：ROUGE 評価

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

常にステミングを使用。なしで、「running」と「run」は異なる単語として数え ROUGE は undercounts。

### 超える ROUGE（2026 要約評価）

ROUGE は 20 年要約で支配的メトリックであり、2026 で単独で不充分。NLG 論文の大規模メタ分析は示した：

- **BERTScore** （文脈埋め込み類似度）は 2023 を通じて地面を得、今の要約論文でほとんど ROUGE と共に報告。
- **BARTScore** は評価を世代として扱う：要約ソースを与えられた事前訓練 BART 指定する可能性によってスコア。
- **MoverScore** （文脈埋め込み上の地球の移動距離）は 2025 要約ベンチマークで最上位スポット達した理由それは意味オーバーラップをキャプチャより良く ROUGE。
- **FactCC** と **QA ベース忠実性** は 2021-2023 一般的、今よく **G-Eval** で置き換え（GPT-4 プロンプトチェーン一貫性、一貫性、流暢性、関連性スコア推論で考える）。
- **G-Eval** と同様 LLM 判定アプローチはマッチ人間の判断 ~80% ルーブリックがよく設計されて場合。

本番推奨：ROUGE-L は遺産比較のため、BERTScore は意味オーバーラップのため、G-Eval は一貫性と真実性のため報告。50-100 人間標識された要約に対してキャリブレーション。

### ステップ4：真実性問題

抽象的要約は幻想に陥りやすい。抽出要約ははるかに低い幻想リスクを運ぶ理由は出力はソースから文字通り持ち上げられたですが、それらは依然誤解できる場合出荷文は脱視化、古い、または順序外引用。これは本番システムが依然順守隣接コンテンツのための抽出方法を好む単一最大理由。

幻想タイプ：

- **エンティティ交換。** ソースは「John Smith」と言う。要約は「John Brown」と言う。
- **数漂流。** ソースは「25,000」と言う。要約は「25 百万」と言う。
- **ポーラリティフリップ。** ソースは「拒否した提供」と言う。要約は「受け入れた提供」と言う。
- **事実の発明。** ソースはCEOを述べない。要約はCEOが承認したと言う。

動作評価アプローチ：

- **FactCC。** 二項分類器はソース文と要約文間の含意で訓練。真実/非真実を予測。
- **QA ベース真実性。** ソースで答えが回答の質問をする QA モデルに求める。要約が別の答えをサポートした場合、フラグ。
- **エンティティレベル F1。** ソースと要約のため指定された名前エンティティを比較。エンティティは要約のみ存在は懸念される。

ユーザー向けどこで真実性は重要（ニュース、医学、法律、財政）のため、抽出はより安全なデフォルト。抽象的なループで真実性チェックが必要。

## 使う

2026 スタック：

| ケース使用 | 推奨 |
|---------|-------------|
| ニュース、3-5 文要約、英語 | `facebook/bart-large-cnn` |
| 科学論文 | `google/pegasus-pubmed` またはチューン T5 |
| マルチドキュメント、長形式 | 任意 LLM 32k+ 文脈、プロンプト持つ |
| 対話要約 | `philschmid/bart-large-cnn-samsum` |
| 抽出、低幻想リスク構造により | TextRank または `sumy` の LSA / LexRank |

LLM は 2026 長文脈は通常特化モデルを計算は制約でないときに勝つ。トレードオフは本番コストと再現可能性；特化モデルはより一貫した出力を与える。

## 出荷

`outputs/skill-summary-picker.md` として保存：

```markdown
---
name: summary-picker
description: 抽出または抽象的なら、指定ライブラリ、真実性チェックをピック。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

タスク（ドキュメントタイプ、準拠要件、長さ、計算予算）が与えられたら、出力：

1. アプローチ。抽出または抽象的。1文で理由を説明。
2. 開始モデル / ライブラリ。それを名付け。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed`、またはLLMプロンプト。
3. 評価プラン。ROUGE-1、ROUGE-2、ROUGE-L（ステミング持つrouge-scoreを使用）。プラス真実性チェック抽象的の場合。
4. プローブするため1つの失敗モード。エンティティ交換は最も一般的な抽象的ニュース要約；ソースエンティティは要約に表示されない場合にフラグサンプル。

医学、法律、財政、または規制されたコンテンツのための抽象的要約を拒否真実性ゲートなし。モデルの文脈ウィンドウ超える入力をフラグとして分割チャンク地図削減要約が必要（単なるトランケーションではなく）。
```

## 演習

1. **簡単。** TextRank を 5 つのニュース記事で実行。トップ3文を参照要約に比較。ROUGE-L を測定。CNN/DailyMail スタイル記事で 30-45 ROUGE-L 見るべき。
2. **中程度。** エンティティレベル真実性を実装：指定ソースと要約からエンティティを抽出（spaCy）、ソースエンティティ要約でのリコールと精度のサマリーエンティティ対ソース。高精度と低リコールは安全ですが簡潔；低精度は幻想エンティティが意味。
3. **難しい。** 50 CNN/DailyMail 記事で BART-large-CNN を LLM（Claude または GPT-4）に対して比較。ROUGE-L、真実性（エンティティF1で）、ことのさら報告。每れぞれどこで勝つか文書。

## キー用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 抽出的 | 文をピック | ソースから文字通りに文を戻す。決に幻想せず。|
| 抽象的 | 書き直し | ソースに条件付けられた新しいテキストを生成。できる幻想。|
| ROUGE | 要約メトリック | N-グラム / LCS ゼロシステム出力と参照間のオーバーラップ。|
| TextRank | グラフベース抽出的 | 文類似度グラフ上の PageRank。|
| 真実性 | それは正しいか | 要約クレームはソースでサポートされているかどうか。|
| 幻想 | 構成されたコンテンツ | 要約のコンテンツするソースはサポートしません。|

## 参考文献

- [Mihalcea and Tarau (2004)。TextRank：テキストに秩序を取る](https://aclanthology.org/W04-3252/) — 抽出標準論文。
- [Lewis et al. (2019)。BART：抽象化シーケンスツーシーケンス事前訓練](https://arxiv.org/abs/1910.13461) — BART 論文。
- [Zhang et al. (2019)。PEGASUS：抽出ギャップ文で事前訓練](https://arxiv.org/abs/1912.08777) — Pegasus と ギャップ文目的。
- [Lin (2004)。ROUGE：自動要約評価のためパッケージ](https://aclanthology.org/W04-1013/) — ROUGE 論文。
- [Maynez et al. (2020)。抽象的要約の忠実性と真実性に](https://arxiv.org/abs/2005.00661) — 真実性風景論文。
