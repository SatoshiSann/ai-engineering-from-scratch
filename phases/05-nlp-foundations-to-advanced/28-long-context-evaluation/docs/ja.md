# 長コンテキスト評価 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Proは10Mトークンのコンテキストを広告する。1Mトークンで、8-needle MRCRは26.3%に低下する。広告 ≠ 使用可能。長コンテキスト評価は、出荷しているモデルの実際の容量を示す。

**タイプ:** 学習
**言語:** Python
**前提条件:** Phase 5 · 13 (質問応答)、Phase 5 · 23 (分割戦略)
**所要時間:** 約60分

## 問題

200ページの契約書がある。モデルは1Mトークンのコンテキストを主張する。契約を貼り付けて、「終了条項は何か?」と尋ねる。モデルが答える — しかし表紙ページから答える、なぜなら終了条項は120kトークン深く、モデルが実際に注視する場所過ぎて座っているため。

これは2026年のコンテキスト容量ギャップだ。仕様シートは1Mまたは10Mを言う。現実は実際の使用可能な60-70%、そして「使用可能」はタスクに依存する。

- **検索(干し草の中の単一針):** 最先端モデルで広告されたマックスまでほぼ完璧。
- **マルチホップ/集約:** ほとんどのモデルで〜128k過ぎて急激に低下。
- **分散されたファクトに対する推論:** 最初に失敗するタスク。

長コンテキスト評価はこれらの軸を測定する。このレッスンはベンチマーク、それぞれが実際に測定するもの、ドメイン用にカスタム針テストを構築する方法を名付ける。

## コンセプト

![NIAHベースライン、RULERマルチタスク、LongBench全体的](../assets/long-context-eval.svg)

**干し草の中の針(NIAH、2023)。** 長いコンテキストに制御された深さで事実(「魔法の言葉はパイナップルだ」)を配置する。モデルに検索するよう求める。深さ×長さを掃引。オリジナルの長コンテキストベンチマーク。最先端モデルは今その飽和; それは必要だが十分ではないベースラインだ。

**RULER (Nvidia、2024)。** 4つのカテゴリ全体で13のタスクタイプ: 検索(単一/マルチキー/マルチ値)、マルチホップトレーシング(変数トレーシング)、集約(共通単語周波数)、QA。構成可能なコンテキスト長(4kから128k+)。NIAHで飽和するがマルチホップで失敗するモデルを明かす。2024年のリリースでは、32k+コンテキストを主張する17モデルのうち、32kで品質を維持したのは半分だった。

**LongBench v2 (2024)。** 503の多肢選択問題、8k-2Mワードのコンテキスト、6つのタスクカテゴリ: 単一ドキュメントQA、マルチドキュメントQA、長いコンテキスト内学習、長いダイアログ、コードリポジトリ、長構造化データ。本当の長コンテキスト挙動の本番ベンチマーク。

**MRCR (マルチラウンド共参照解析)。** マルチターン共参照规模。8-針、24-針、100-針バリアント。注視が低下する前にモデルが何個のファクトをジャグリングできるかを露光。

**NoLiMa。** 「非字句針。」針とクエリは文字通りオーバーラップを共有しない; 検索は1つのステップの意味論的推論が必要。NIAHより難しい。

**HELMET。** 多くのドキュメントを連結し、任意の1つから質問を尋ねる。選択的注視をテストする。

**BABILong。** 無関係な干し草にbAbI推論チェーンを埋め込む。干し草への推論をテストする、単なる検索ではない。

### 実際に報告するもの

- **広告されたコンテキストウィンドウ。** 仕様シート番号。
- **有効検索長。** NIAHはいくつかのしきい値(例えば、90%)で合格。
- **有効推論長。** マルチホップまたは集約はそのしきい値で合格。
- **劣化曲線。** コンテキスト長対精度、タスクタイプごとにプロット。

仕様シートの2つの番号: 検索有効と推論有効。通常、推論有効は広告ウィンドウの25-50%だ。

## ビルドしてみよう

### ステップ1: ドメイン用カスタムNIAH

`code/main.py`を参照。スケルトン:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # フィラーを干し草本体を満たすのに十分長くなるまで繰り返す。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

`depth_ratio` ∈ {0、0.25、0.5、0.75、1.0} × `total_tokens` ∈ {1k、4k、16k、64k}を掃引。ヒートマップをプロット。それは対象モデルのNIAHカードだ。

### ステップ2: マルチ針バリアント

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

「3つの魔法の言葉は何か?」のような質問は、3つすべてを取得する必要がある。単一針の成功は、マルチ針の成功を予測しない。

### ステップ3: マルチホップ変数トレーシング (RULERスタイル)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答えは3つの割り当てをチェーンする必要がある。最先端モデルは128kで、ここで50-70%精度に低下することが多い。

### ステップ4: スタックでのLongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

カテゴリごとの精度を報告する。集計スコアは大きなタスクレベルの違いを隠す。

## ピットフォール

- **NIAHのみの評価。** 1Mトークンで NIAHに合格することは、マルチホップについて何も言わない。常にRULERまたはカスタムマルチホップテストを実行する。
- **一様な深さサンプリング。** 多くの実装は depth=0.5 だけをテストする。depth=0、0.25、0.5、0.75、1.0をテストする — 「真ん中で失われている」効果は本物だ。
- **フィラーとの字句オーバーラップ。** 針がフィラーとキーワードを共有すれば、検索は些細になる。NoLiMaスタイルの重複なし針を使用する。
- **レイテンシーを無視。** 1Mトークンプロンプトは30-120秒かかる。精度と一緒にtime-to-first-tokenを測定。
- **ベンダー自己報告番号。** OpenAI、Google、Anthropicはすべて自身のスコアを公開。常にユースケースで独立して再実行。

## 使用方法

2026年スタック:

| 状況 | ベンチマーク |
|-----------|-----------|
| 高速サニティチェック | 3つの深さ×3の長さでカスタムNIAH |
| 本番モデル選択 | 対象長でのRULER (13タスク) |
| 現実のQA品質 | LongBench v2単一ドキュメントQAサブセット |
| マルチホップ推論 | BABILongまたはカスタム変数トレーシング |
| 会話/ダイアログ | 対象長での MRCR 8-針 |
| モデルアップグレード回帰 | 固定社内NIAH +RULERハーネス、すべての新しいモデルで実行 |

本番用のプラムルール: 意図された長さでNIAH + 1つの推論タスクまでコンテキストウィンドウを信頼しない。

## 出荷しよう

`outputs/skill-long-context-eval.md`として保存:

```markdown
---
name: long-context-eval
description: 与えられたモデルとユースケースのための長コンテキスト評価バッテリーを設計する。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

対象モデル、対象コンテキスト長、ユースケースを指定して、以下を出力:

1. テスト。NIAH深さ×長さグリッド; RULERマルチホップ; カスタムドメインタスク。
2. サンプリング。各長さで深さ0、0.25、0.5、0.75、1.0。
3. メトリクス。検索合格率; 推論合格率; time-to-first-token; cost-per-query。
4. カットオフ。有効検索長(90%合格)と有効推論長(70%合格)。両方を報告。
5. 回帰。固定ハーネス、すべてのモデルアップグレードで再実行、デルタを表面化。

モデルカードだけからコンテキストウィンドウを信頼するのを拒否しなさい。任意のマルチホップワークロード用のNIAHのみ評価を拒否しなさい。ベンダー自己報告の長コンテキストスコアを独立した証拠として拒否しなさい。
```

## 演習

1. **簡単。** 3つの深さ(0.25、0.5、0.75) × 3の長さ(1k、4k、16k)でNIAHを構築。任意のモデルで実行。3×3ヒートマップとして合格率をプロット。
2. **中程度。** 3-針バリアントを追加。各長さで3すべての検索を測定。同じ長さでの単一針合格率と比較。
3. **難しい。** 変数トレーシングタスク(X1 → X2 → X3、3ホップ付き)を64kのフィラーに埋め込むを構築。3つの最先端モデル全体で精度を測定。モデルあたりの有効推論長を報告。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| NIAH | 干し草の中の針 | ファクトをフィラーに植え、モデルに検索するよう求める。 |
| RULER | ステロイドのNIAH | 検索/マルチホップ/集約/QA全体で13タスクタイプ。 |
| 有効コンテキスト | 実際の容量 | 精度がしきい値を超えて保つ長さ。 |
| 真ん中で失われている | 深さバイアス | モデルは長い入力の真ん中の内容の下注視。 |
| マルチ針 | 一度に多くのファクト | 複数の植え; 注視ジャグリング、単なる検索ではない。 |
| MRCR | マルチラウンド共参照 | 8、24、または100-針共参照; 注視飽和を露光。 |
| NoLiMa | 非字句針 | 針とクエリが文字通りトークンを共有しない; 推論が必要。 |

## 参考文献

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — オリジナルNIAHリポ。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — マルチタスクベンチマーク。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 本当の長コンテキスト評価。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — より難しい針。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 推論-イン干し草。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度バイアスペーパー。
