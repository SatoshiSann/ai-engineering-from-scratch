# BERT — マスク言語モデリング

> GPT は次の単語を予測。BERT は欠落単語を予測。1文の違い — そして埋め込み形のすべて半十年。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 05 （完全トランスフォーマー）, Phase 5 · 02 （テキスト表現）
**所要時間:** 約45分

## 問題

2018年、あらゆるNLPタスク — 感情、NER、QA、含意 — 独自のデータ上で独自のモデルから訓練した。「English を理解する」事前訓練チェックポイントがあらなかった あなたがファインチューン可能だった。ELMo（2018）は双方向LSTMで文脈的埋め込みを事前訓練可能なことを示した; それはいくぶん助けたが一般化しなかった。

BERT（Devlin et al. 2018）は尋ねた: インターネット上のあらゆる文で訓練したトランスフォーマー エンコーダを取り、両側のコンテキストから欠落単語を予測するように強制した場合？その後、あなたはダウンストリームタスク上の1つの頭をファインチューン。パラメータ効率は啓示だった。

結果: 18か月以内に BERT とその変種（RoBERTa、ALBERT、ELECTRA）すべてのNLPリーダーボードを支配した存在したもの。2020年までに、あらゆる検索エンジン、コンテンツモデレーションパイプライン、意味的検索システムが、地球上には BERT を持つ。

2026年エンコーダのみモデルは依然として分類、検索、構造化抽出向けの右ツール — デコーダより5–10倍高速トークンあたり実行、埋め込みは近代的検索スタックのすべてのバックボーン。ModernBERT（2024年12月）はアーキテクチャを Flash Attention + RoPE + GeGLU で 8K コンテキストに押した。

## コンセプト

![マスク言語モデリング: トークンを選択、マスク、元を予測](../assets/bert-mlm.svg)

### 訓練信号

文を取る: `the quick brown fox jumps over the lazy dog`。

トークンの15%をランダムにマスク:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

マスク位置で元のトークンを予測するようにモデルを訓練。エンコーダが双方向性だから、位置1の `[MASK]` を予測するは `brown fox jumps` を位置2+で使用できる。GPT ができない。

### BERT マスク ルール

予測向けに選択された15%のトークン:

- 80% は `[MASK]` で置き換えられ。
- 10% はランダムトークンで置き換えられ。
- 10% は変更なく残される。

なぜ常に `[MASK]` ではない？ `[MASK]` は推論時に決してあらわれない。モデルに期待させること `[MASK]` を マスク位置の100%で、事前訓練とファインチューン間の配布シフトを作成するだろう。ランダム10% + 変更なし10% はモデルを正直に保つ。

### 次文予測（NSP） — そしてなぜそれが落とされた

元のBERTはNSP にも訓練した: 2つの文AとBを与えて、Bが A に続くかを予測。RoBERTa（2019）はそれをアブレートし、NSP がでなく傷つけたことを示した。最新エンコーダはそれをスキップ。

### 2026年で何が変わった: ModernBERT

2024年 ModernBERT ペーパーは 2026年 プリミティブでブロックを再構築:

| コンポーネント | 元の BERT （2018） | ModernBERT （2024） |
|-----------|----------------------|-------------------|
| 位置的 | 学習絶対 | RoPE |
| 活性化 | GELU | GeGLU |
| 正規化 | LayerNorm | Pre-norm RMSNorm |
| アテンション | 完全密 | 交互ローカル（128）+ グローバル |
| コンテキスト長 | 512 | 8192 |
| トークナイザー | WordPiece | BPE |

そして 2018年スタック と異なり、それは Flash-Attention-native。推論は シーケンス長 8K で DeBERTa-v3 より 2–3倍高速、より好い GLUE スコア。

### エンコーダがまだ 2026年でピックするユースケース

| タスク | なぜエンコーダはデコーダを打つ |
|------|---------------------------|
| 検索 / 意味的検索埋め込み | 双方向コンテキスト = あたりトークン埋め込み品質 |
| 分類（感情、意図、毒性） | 1つのフォワードパス; 生成オーバーヘッドなし |
| NER / トークンラベリング | あたり位置出力、ネイティブ双方向 |
| ゼロショット含意（NLI） | エンコーダの上の分類器頭 |
| RAG のためのリランカー | クロスエンコーダスコアリング、LLMリランカーより10倍高速 |

## ビルド

### ステップ1: マスクロジック

`code/main.py` を見よ。関数 `create_mlm_batch` はトークンIDのリスト、ボキャブ サイズ、マスク確率を取る。入力ID（マスク適用）とラベル（マスク位置のみ、他は -100 — PyTorch の ignore index 慣例）を返す。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 元を保つ
    return input_ids, labels
```

### ステップ2: 小さいコーパス上 MLM 予測実行

ボキャブ 20 語、200 文で 2 層エンコーダ + MLM 頭で訓練。勾配なし — フォワードパス健全性チェック。完全訓練には PyTorch が必要。

### ステップ3: マスク型を比較

3方ルールが `[MASK]` なしでモデルを使用可能に保つ方法を示す。マスクされていない文と マスク文上で予測。両方は妥当なトークン配布を生産すべき 設定で両方パターンをモデルが見たから。

### ステップ4: ファインチューン頭

MLM 頭をおもちゃ感情データセット上の分類頭で置き換え。ヘッドのみ訓練; エンコーダは凍結。これはあらゆる BERT 応用が従うパターン。

## 使用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**埋め込みモデルはファインチューン BERT。** `sentence-transformers` `all-MiniLM-L6-v2` のようなモデルは対比損失で訓練 BERT。エンコーダは同じ。損失は変わった。

**クロスエンコーダリランカーもまたファインチューン BERT。** ペア分類 `[CLS] query [SEP] doc [SEP]`。ある双方向アテンション query と doc の間は 正確に クロスエンコーダに品質縁を与える。

**BERT をピックしてはいけないときは 2026年。** あらゆる生成的。エンコーダはトークンを自動回帰的に生成するための健全な方法を持たない。また: 小さいデコーダが品質と柔軟性をより一致させることができる何か 1B パラメータ未満（Phi-3-Mini, Qwen2-1.5B）。

## 展開

`outputs/skill-bert-finetuner.md` を見よ。スキルは BERT ファインチューン（バックボーン選択、頭仕様、データ、評価、停止）を範囲化 新分類または抽出タスク向け。

## 演習

1. **簡単。** `code/main.py` を実行し、10,000トークン上のマスク分布を印刷。~15% が選択される、そのものの ~80% が `[MASK]` になる確認。
2. **中程度。** 全単語マスクを実装: 単語がサブワードに字句化される場合、すべてのサブワード一緒にマスクまたはなし。500文コーパス上で MLM 精度を測定。
3. **難しい。** 小さい（2層、d=64）BERT を 公開データセットから 10,000 文で訓練。SST-2 感情向けの `[CLS]` トークンをファインチューン。マッチパラメータでのデコーダのみベースラインに対して比較 — どち勝つ？

## 主要用語

| 用語 | 人々は何を言うか | 実際には何を意味するか |
|------|-----------------|-----------------------|
| MLM | 「マスク言語モデリング」 | 訓練信号: ランダムに 15% トークン `[MASK]` で置き換え、元を予測。 |
| 双方向 | 「両方を見る」 | エンコーダ注目は因果マスクを持たない — あらゆる位置はあらゆる他位置を見る。 |
| `[CLS]` | 「プーラー トークン」 | あらゆるシーケンスに前置されたトークン; 最終埋め込みは文レベル表現として使用。 |
| `[SEP]` | 「セグメント分離器」 | ペアシーケンス分離（例 query/doc, 文 A/B）。 |
| NSP | 「次文予測」 | BERT の2番目の事前訓練タスク; RoBERTa で無駄だと示される、2019年後落とされた。 |
| ファインチューン | 「タスクに適応」 | エンコーダをほぼ凍結させておく; ダウンストリームタスク向け頭で小さいヘッドを訓練。 |
| クロスエンコーダ | 「リランカー」 | BERT が両方クエリと doc を入力として取る、関連スコアを出力。 |
| ModernBERT | 「2024年更新」 | RoPE、RMSNorm、GeGLU、交互ローカル/グローバルアテンション、8K コンテキストでエンコーダ再構築。 |

## 参考文献

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 元論文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — BERT を正しく訓練する方法; NSP を殺す。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — マッチコンピュートで MLM で置き換えられたトークン検出勝つ。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT ペーパー。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 規範的エンコーダ参照。
