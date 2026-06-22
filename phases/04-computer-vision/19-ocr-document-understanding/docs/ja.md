# OCRとドキュメント理解

> OCRは三段階のパイプラインだ — テキストボックスを検出し、文字を認識し、レイアウトを整える。現代のすべてのOCRシステムはこれらの段階を並び替えるか統合する。

**タイプ:** 学習 + 活用
**言語:** Python
**前提条件:** Phase 4 レッスン 06 (検出)、Phase 7 レッスン 02 (セルフアテンション)
**所要時間:** 約45分

## 学習目標

- 古典的なOCRパイプライン（検出 -> 認識 -> レイアウト）と現代のエンドツーエンドの代替手法（Donut、Qwen-VL-OCR）をたどることができる
- シーケンス間OCR学習のためのCTC（Connectionist Temporal Classification）損失関数を実装できる
- 学習なしに本番ドキュメント解析にPaddleOCRまたはEasyOCRを使用できる
- OCR、レイアウト解析、ドキュメント理解を区別し、タスクに応じて適切なツールを選択できる

## 問題

テキストが詰まった画像はいたるところにある：レシート、請求書、身分証明書、スキャンされた本、フォーム、ホワイトボード、看板、スクリーンショット。それらから構造化データを抽出すること — 単なる文字だけでなく「これが合計金額だ」 — は、応用ビジョンの中でも最も価値の高い問題の一つだ。

この分野は三つのスキル層に分かれる：

1. **OCR本体**：ピクセルからテキストへ変換する。
2. **レイアウト解析**：OCR出力を領域（タイトル、本文、表、ヘッダー）にグループ化する。
3. **ドキュメント理解**：レイアウトから構造化フィールド（「invoice_total = $42.50」）を抽出する。

各層に古典的なアプローチと現代的なアプローチがあり、「画像からテキストが欲しい」と「このレシートから合計金額が必要だ」の間の差は、ほとんどのチームが気づくよりも大きい。

## コンセプト

### 古典的なパイプライン

```mermaid
flowchart LR
    IMG["画像"] --> DET["テキスト検出<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["単語/行の<br/>バウンディングボックス"]
    BOX --> CROP["各領域を切り抜く"]
    CROP --> REC["認識<br/>(CRNN + CTC)"]
    REC --> TXT["テキスト文字列"]
    TXT --> LAY["レイアウト<br/>順序付け"]
    LAY --> OUT["読み順テキスト"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **テキスト検出**は行ごとまたは単語ごとの四辺形を生成する。
- **認識**は各領域を固定高さに切り抜き、CNN + BiLSTM + CTCで文字シーケンスを生成する。
- **レイアウト**は読み順を再構築する（ラテン語は上から下、左から右；アラビア語、日本語は異なる）。

### CTCを一段落で

OCR認識は固定長の特徴マップから可変長のシーケンスを生成する。CTC（Graves et al.、2006）は文字レベルのアライメントなしにこれを学習できる。モデルは各タイムステップで（語彙 + ブランク）上の分布を出力し；CTC損失は繰り返しをマージしブランクを除去した後にターゲットテキストに還元されるすべてのアライメントを周辺化する。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTCは2015年にCRNNが機能した理由であり、2026年でもほとんどの本番OCRモデルの学習に使われている。

### 現代のエンドツーエンドモデル

- **Donut**（Kim et al.、2022）— ViTエンコーダ + テキストデコーダ；画像を読み込みJSONを直接出力する。テキスト検出器もレイアウトモジュールも不要。
- **TrOCR** — ViT + トランスフォーマーデコーダによる行レベルOCR。
- **Qwen-VL-OCR / InternVL** — OCRタスク用にファインチューニングされた完全な視覚言語モデル；2026年の複雑なドキュメントで最高精度。
- **PaddleOCR** — 成熟した本番パッケージの古典的DB + CRNNパイプライン；オープンソースの主力。

エンドツーエンドモデルはより多くのデータと計算が必要だが、多段階パイプラインのエラー蓄積を回避できる。

### レイアウト解析

構造化されたドキュメントには、各領域にラベル（タイトル、段落、図、表、脚注）を付けるレイアウト検出器（LayoutLMv3、DocLayNet）を実行する。読み順はその後「レイアウト順に領域を反復し、連結する」になる。

フォームには**キー値抽出**モデル（視覚的にリッチなドキュメントにはDonut、プレーンスキャンにはLayoutLMv3）を使用する。これらは画像 + 検出テキスト + 位置を入力として、構造化されたキー値ペアを予測する。

### 評価指標

- **文字誤り率（CER）** — レーベンシュタイン距離 / 参照の長さ。低いほど良い。本番目標：クリーンなスキャンで2%未満。
- **単語誤り率（WER）** — 同じく単語レベル。
- **構造化フィールドのF1** — キー値タスク用；`{invoice_total: 42.50}`が正しく現れるかを測定する。
- **JSONの編集距離** — エンドツーエンドのドキュメント解析用；Donutの論文は正規化木編集距離を導入した。

## 構築

### ステップ1：CTC損失関数 + 貪欲デコーダ

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`は利用可能な場合に効率的なCuDNN実装を使用する。貪欲デコーダはビームサーチより単純で、通常CERは1%以内に収まる。

### ステップ2：小さなCRNN認識器

行OCR用の最小限のCNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高さの入力（CNNが高さを1にマックスプーリングする）。幅がCTCの時間次元になる。

### ステップ3：合成OCR

エンドツーエンドのスモークテスト用に白地に黒の数字列を生成する。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

実際のOCRデータセットはフォント、ノイズ、回転、ブラー、色を追加する。上記のパイプラインは同一だ。

### ステップ4：学習スケッチ

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

この単純な合成データで200ステップで損失は約3から約0.2に低下するはずだ。

## 活用

三つの本番パス：

- **PaddleOCR** — 成熟、高速、多言語。一行での使用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — Pythonネイティブ、多言語、PyTorchバックボーン。
- **Tesseract** — 古典的；モデルが苦手な古いスキャンドキュメントに今でも有用。

エンドツーエンドのドキュメント解析にはDonutまたはVLMを使用する：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

レシート、請求書、繰り返し構造を持つフォームにはDonutをファインチューニングする。任意のドキュメントや推論付きOCRには、Qwen-VL-OCRのようなVLMが現在のデフォルトだ。

## 成果物

このレッスンで生成されるもの：

- `outputs/prompt-ocr-stack-picker.md` — ドキュメントの種類、言語、構造が与えられたとき、Tesseract / PaddleOCR / Donut / VLM-OCRを選択するプロンプト。
- `outputs/skill-ctc-decoder.md` — 長さ正規化を含む貪欲なビームサーチCTCデコーダをゼロから書くスキル。

## 演習

1. **(易)** 5桁のランダムな数値文字列でTinyCRNNを500ステップ学習させる。ホールドアウトセットでCERを報告する。
2. **(中)** 貪欲なデコードをビームサーチ（beam_width=5）に置き換える。CERの差を報告する。どの入力でビームサーチが勝つか？
3. **(難)** PaddleOCRを20枚のレシートに使用し、ラインアイテムを抽出し、{item_name, price}ペアの手動ラベル付き正解と比べてF1を計算する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| OCR | 「ピクセルからテキスト」 | 画像領域を文字シーケンスに変換すること |
| CTC | 「アライメントフリーの損失関数」 | タイムステップごとのラベルなしにシーケンスモデルを学習させる損失関数；アライメントを周辺化する |
| CRNN | 「古典的OCRモデル」 | 畳み込み特徴抽出器 + BiLSTM + CTC；本番でも使われる2015年のベースライン |
| Donut | 「エンドツーエンドOCR」 | ViTエンコーダ + テキストデコーダ；画像から直接JSONを出力する |
| レイアウト解析 | 「領域を見つける」 | ドキュメント内のタイトル/表/図/段落領域を検出してラベル付けする |
| 読み順 | 「テキストシーケンス」 | 認識された領域を文に並べる順序；ラテン語は単純、混合レイアウトは非単純 |
| CER / WER | 「誤り率」 | 文字または単語の粒度でのレーベンシュタイン距離 / 参照の長さ |
| VLM-OCR | 「読むLLM」 | OCRタスク用に学習またはプロンプトされた視覚言語モデル；複雑なドキュメントの現在のSOTA |

## 参考文献

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — CNN+RNN+CTCの元のアーキテクチャ
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 元のCTC論文；アルゴリズムのアイデアが凝縮されている
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) — OCRフリーのドキュメント理解トランスフォーマー
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — オープンソースの本番OCRスタック
