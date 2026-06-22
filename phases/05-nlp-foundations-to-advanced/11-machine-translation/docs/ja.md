# 機械翻訳

> 翻訳は、30 年間 NLP 研究の資金を支え、今も支え続けているタスクだ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 5 · 10 (Attention Mechanism)、Phase 5 · 04 (GloVe, FastText, Subword)
**所要時間:** 約75分

## 問題

モデルはある言語の文を読み、別の言語の文を生成する。長さは変わる。語順は変わる。一部のソース語は複数のターゲット語に対応し、その逆もある。慣用句は一対一の対応を拒む。フランス語で "I miss you" は "tu me manques" — 文字通りには「あなたが私に欠けている」だ。語レベルのアラインメントはそれを生き延びない。

機械翻訳は、NLP にエンコーダ・デコーダ、アテンション、トランスフォーマー、そして最終的には LLM パラダイム全体を発明させたタスクだ。翻訳品質が測定可能で、人間と機械のギャップが頑固だったからこそ、すべての前進が訪れた。

このレッスンは歴史の講義を飛ばし、2026 年の実用パイプラインを教える。事前学習済みの多言語エンコーダ・デコーダ (NLLB-200 または mBART)、サブワードトークン化、ビームサーチ、BLEU と chrF による評価、そして今なお検出されずに本番へ出荷される少数の失敗モード。

## コンセプト

![MT パイプライン: トークン化 → エンコード → アテンションつきデコード → デトークン化](../assets/mt-pipeline.svg)

現代の MT は、対訳テキストで訓練されたトランスフォーマーのエンコーダ・デコーダだ。エンコーダはソースをその言語のトークン化で読む。デコーダはクロスアテンション (レッスン 10) を通じてエンコーダの出力を使い、ターゲットを一度に一サブワードずつ生成する。デコードは貪欲デコードの罠を避けるためにビームサーチを使う。出力はデトークン化され、デトゥルーケース化され、参照に対してスコア付けされる。

3 つの運用上の選択が現実世界の MT 品質を左右する。

- **トークナイザー。** 混合言語コーパスで訓練された SentencePiece BPE。言語間で共有された語彙が、NLLB でのゼロショットペアを可能にするものだ。
- **モデルサイズ。** NLLB-200 distilled 600M はラップトップに収まる。NLLB-200 3.3B は公表された本番のデフォルトだ。54.5B が研究の上限だ。
- **デコード。** 一般コンテンツにはビーム幅 4〜5。短すぎる出力を避けるための長さペナルティ。用語の一貫性が必要なときは制約付きデコード。

## ビルド

### ステップ 1: 事前学習済み MT の呼び出し

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

ここで重要なことが 3 つ。`src_lang` はどのスクリプトとセグメンテーションを適用するかをトークナイザーに伝える。`forced_bos_token_id` はどの言語を生成するかをデコーダに伝える。両方とも NLLB 固有のトリックだ。mBART と M2M-100 は独自の慣例を使い、互換性はない。

### ステップ 2: BLEU と chrF

BLEU は出力と参照の間の n-gram 重複を測定する。4 つの参照 n-gram サイズ (1〜4)、適合率の幾何平均、短すぎる出力への簡潔性ペナルティ。スコアは [0, 100] にある。広く使われる。解釈にいらだつ。BLEU 30 は「使える」、40 は「良い」、50 は「例外的」、1 BLEU 未満の差はノイズだ。

chrF は文字レベルの F スコアを測定する。BLEU が一致を過小カウントしがちな形態的に豊かな言語に対してより敏感だ。しばしば BLEU と並べて報告される。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

常に `sacrebleu` を使うこと。トークン化を正規化するので、論文間でスコアが比較可能になる。自前の BLEU 計算を作ることが、誤解を招くベンチマークが生まれる原因だ。

### 3 層評価階層 (2026)

現代の MT 評価は 3 つの相補的なメトリクスファミリーを使う。少なくとも 2 つを伴って出荷せよ。

- **ヒューリスティック** (BLEU、chrF)。高速、参照ベース、解釈可能、言い換えに鈍感。レガシー比較と回帰検出に使う。
- **学習済み** (COMET、BLEURT、BERTScore)。人間の判断で訓練されたニューラルモデル。翻訳とソースおよび参照の意味的類似度を比較する。COMET は 2023 年以降 MT 研究との関連が最も高く、品質が重要な場合の 2026 年本番デフォルトだ。
- **LLM-as-judge** (参照不要)。大規模モデルに、流暢さ、妥当性、トーン、文化的適切さで翻訳をスコア付けするよう促す。GPT-4-as-judge は、ルーブリックがよく設計されていれば人間との一致を約 80% の確率で達成する。参照が存在しない自由形式コンテンツに使う。

実用的な 2026 年スタック: BLEU と chrF には `sacrebleu`、COMET には `unbabel-comet`、最終的な人間向けシグナルにはプロンプトされた LLM。本番データで信頼する前に、すべてのメトリクスを 50〜100 個の人間ラベル付き例に対して較正せよ。

参照不要メトリクス (COMET-QE、BLEURT-QE、LLM-as-judge) は参照なしに翻訳を評価できる。これは参照訳が存在しないロングテール言語ペアにとって重要だ。

### ステップ 3: 本番で壊れるもの

上の実用パイプラインは 80% の確率で流暢に翻訳し、残りの 20% で静かに失敗する。名前を付けた失敗モード:

- **ハルシネーション。** モデルがソースになかった内容を捏造する。馴染みのないドメイン語彙でよくある。症状: 出力は流暢だがソースが述べなかった事実を主張する。緩和策: ドメイン用語での制約付きデコード、規制対象コンテンツでの人間レビュー、入力より大幅に長い出力の監視。
- **ターゲット外生成。** モデルが間違った言語に翻訳する。NLLB はまれな言語ペアでこれに驚くほど陥りやすい。緩和策: `forced_bos_token_id` を検証し、常に出力に対する言語 ID モデルチェックでデコードする。
- **用語ドリフト。** "Sign up" がドキュメント 1 では "s'inscrire"、ドキュメント 2 では "créer un compte" になる。UI テキストやユーザー向け文字列では、生の品質より一貫性が重要だ。緩和策: 用語集制約付きデコードまたはポスト編集辞書。
- **敬語の不一致。** フランス語の "tu" 対 "vous"、日本語の丁寧度レベル。モデルは訓練でより一般的だった形式を選ぶ。顧客向けコンテンツではこれは通常間違いだ。緩和策: モデルが対応していれば敬語トークンつきのプロンプトプレフィックス、または丁寧体のみのコーパスで小さなモデルをファインチューニング。
- **短い入力での長さの爆発。** 非常に短い入力文は、約 5 ソーストークン未満で長さペナルティが崖から落ちるため、しばしば過度に長い翻訳を生成する。緩和策: ソース長に比例したハードな最大長キャップ。

### ステップ 4: ドメインへのファインチューニング

事前学習済みモデルはジェネラリストだ。法律、医療、ゲーム対話の翻訳は、ドメイン対訳データでのファインチューニングから測定可能に恩恵を受ける。レシピは風変わりではない。

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

数千個の高品質な対訳例は、数十万個のノイズの多いウェブスクレイプされたものに勝る。訓練データの品質が、最大の単一の本番レバーだ。

## 使う

MT の 2026 年本番スタック:

| ユースケース | 推奨される出発点 |
|---------|---------------------------|
| 任意から任意、200 言語 | `facebook/nllb-200-distilled-600M` (ラップトップ) または `nllb-200-3.3B` (本番) |
| 英語中心、高品質、50 言語 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短い実行、安価な推論、英語-フランス語/ドイツ語/スペイン語 | Helsinki-NLP / Marian モデル |
| レイテンシ重視のブラウザ側 | ONNX 量子化された Marian (約 50 MB) |
| 最高品質、支払う意志あり | 翻訳プロンプトつきの GPT-4 / Claude / Gemini |

LLM は 2026 年現在、いくつかの言語ペアで、特に慣用的なコンテンツと長い文脈で、専用 MT モデルを上回るようになった。トレードオフはトークンあたりのコストとレイテンシだ。文脈長、文体の一貫性、またはプロンプトを通じたドメイン適応がスループットより重要なときは LLM を選ぶ。

## 出荷

`outputs/skill-mt-evaluator.md` として保存する。

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 演習

1. **易しい。** `nllb-200-distilled-600M` を使って、5 文の英語段落をフランス語に、そして英語に戻して翻訳せよ。往復が元の文にどれだけ近いかを測定せよ。語選択のドリフトを伴う意味の保存が見えるはずだ。
2. **中級。** `fasttext lid.176` または `langdetect` を使って、翻訳出力に対する言語 ID チェックを実装せよ。MT 呼び出しに統合し、ターゲット外生成が返される前に捕まえよ。
3. **難しい。** 自分で選んだ 5,000 ペアのドメインコーパスで `nllb-200-distilled-600M` をファインチューニングせよ。ファインチューニング前後でホールドアウトセット上の BLEU を測定せよ。どの種類の文が改善し、どれが悪化したかを報告せよ。

## 重要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| BLEU | 翻訳スコア | 簡潔性ペナルティつきの n-gram 適合率。[0, 100]。 |
| chrF | 文字 F スコア | 文字レベルの F スコア。形態的に豊かな言語に対してより敏感。 |
| NMT | ニューラル MT | 対訳テキストで訓練されたトランスフォーマーのエンコーダ・デコーダ。2017 年以降のデフォルト。 |
| NLLB | No Language Left Behind | Meta の 200 言語 MT モデルファミリー。 |
| 制約付きデコード | 制御された出力 | 特定のトークンや n-gram を出力に現れさせる / 現れさせないよう強制する。 |
| ハルシネーション | 捏造された内容 | ソースによって裏付けられないモデル出力。 |

## 参考文献

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 論文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — なぜ `sacrebleu` が BLEU を報告する唯一の正しい方法なのか。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 論文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 実用的なファインチューニングのウォークスルー。
