# 音声言語モデル — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026年の音声言語モデルは、音声 + 環境音 + 音楽について推論する。Qwen2.5-Omni-7B は MMAU-Pro で GPT-4o Audio に匹敵する。Audio Flamingo Next は LongAudioBench で Gemini 2.5 Pro を上回る。オープンとクローズドのギャップは本質的に埋まった — ただしマルチオーディオタスクを除く。そこでは誰もがほぼランダムレベルだ。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 6 · 04 (ASR)、Phase 12 · 03 (Vision-Language Models)、Phase 7 · 10 (Audio Transformers)
**所要時間:** 約45分

## 問題

5秒間の音声があるとする。犬が吠え、誰かが「止まれ！」と叫び、その後静寂が訪れる。有用な質問は複数の軸にまたがる。

- **文字起こし。** 「何が言われたか？」 — ASR の領域。
- **意味的推論。** 「この人物は危険な状況にあるか？」 — 吠え声 + 叫び声 + 静寂を統合的に理解する必要がある。
- **音楽推論。** 「メロディを演奏している楽器は何か？」
- **長尺音声検索。** 「この90分の講義のどこで講師が勾配降下法を説明したか？」

これらすべてを単一のプロンプトで答える単一のモデルが**音声言語モデル**（LALM / ALM）だ。純粋な ASR とは別物だ。LALM は単なる文字起こしではなく、自由形式の自然言語の回答を生成する。

## コンセプト

![音声言語モデル: 音声エンコーダ + プロジェクタ + LLM デコーダ](../assets/alm-architecture.svg)

### 3コンポーネントのテンプレート

2026年のすべての LALM は同じ骨格を持つ。

1. **音声エンコーダ。** Whisper エンコーダ · BEATs · CLAP · WavLM · またはモデルごとのカスタムエンコーダ。
2. **プロジェクタ。** 音声エンコーダの特徴を LLM のトークン埋め込み空間に橋渡しする線形層または MLP。
3. **LLM。** Llama / Qwen / Gemma ベースのデコーダ。テキスト + 音声トークンが交互に並んだものを受け取り、テキストを生成する。

訓練:

- **ステージ1。** エンコーダ + LLM を凍結し、ASR / キャプショニングデータでプロジェクタのみを訓練する。
- **ステージ2。** 指示に従う音声タスク（QA、推論、音楽理解）でフル / LoRA ファインチューニングを行う。
- **ステージ3（任意）。** 音声入力 / 音声出力のために音声デコーダを追加する。Qwen2.5-Omni と AF3-Chat がこれを行っている。

### 2026年のモデルマップ

| モデル | バックボーン | 音声エンコーダ | 出力モダリティ | アクセス |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | カスタム + Whisper | テキスト + 音声 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | カスタム | テキスト + 音声 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | テキスト | NVIDIA 非商用 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | テキスト | NVIDIA 非商用 |
| SALMONN | Vicuna | Whisper + BEATs | テキスト | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | テキスト | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | テキスト | Apache-2.0 |
| Gemini 2.5 Flash/Pro (クローズド) | Gemini | 独自 | テキスト + 音声 | API |
| GPT-4o Audio (クローズド) | GPT-4o | 独自 | テキスト + 音声 | API |

### ベンチマークの現実確認（2026年）

**MMAU-Pro。** 音声 / 環境音 / 音楽 / 混合をカバーする1800の QA ペア。マルチオーディオサブセットを含む。

| モデル | 総合 | 音声 | 環境音 | 音楽 | マルチオーディオ |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | 約60% | 73.4% | 51.9% | 64.9% | 約22% |
| Gemini 2.5 Flash | 約57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | 約20% |
| Audio Flamingo 3 | 約54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench で SOTA | — | — | — | — |

**マルチオーディオの列はすべての人にとって痛烈だ。** 4択の選択問題でのランダムな当たり率 = 25%。ほとんどのモデルがその近辺のスコアにとどまる。LALM はいまだに2つのクリップを比較するのに苦労している。

### 2026年に LALM が役立つ場面

- **コールセンター録音のコンプライアンス監査。** 「エージェントは必須の開示事項を述べたか？」
- **アクセシビリティ。** 聴覚障害のあるユーザーに音響イベントを説明する（単なる文字起こしではない）。
- **コンテンツモデレーション。** 暴力的な言葉 + 脅迫的なトーン + 背景の文脈を検出する。
- **ポッドキャスト / 会議のチャプター分け。** 単なる話者交代ではなく、意味的な要約。
- **音楽カタログ分析。** 「B セクションで転調するすべてのトラックを見つける。」

### まだ役立たない場面

- きめ細かな音楽理論（コードレベル以下）。
- 長い会話にわたる話者帰属を伴う推論（10分を超えると劣化する）。
- マルチオーディオ比較（22-26% はランダムをかろうじて上回る程度）。
- リアルタイムストリーミング推論（ほとんどはオフラインのバッチ推論）。

## 作ってみる

### ステップ1: Qwen2.5-Omni に問い合わせる

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### ステップ2: プロジェクタのパターン

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

これだけだ。プロジェクタは通常1〜3層の線形層だ。ASR ペア（音声 → 文字起こし）でこれを訓練するのがステージ1のプリテキストタスクだ。

### ステップ3: MMAU / LongAudioBench でのベンチマーク

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

カテゴリ別（音声 / 環境音 / 音楽 / マルチオーディオ）に分けて報告すること。集計値はモデルがどこで失敗するかを隠してしまう。

## 使ってみる

| タスク | 2026年の選択 |
|------|-----------|
| 自由形式の音声 QA（オープン） | Qwen2.5-Omni-7B |
| 長尺音声でのオープン最良 | Audio Flamingo Next |
| クローズドの最良 | Gemini 2.5 Pro |
| 音声入力 / 音声出力エージェント | Qwen2.5-Omni または GPT-4o Audio |
| 音楽推論 | Audio Flamingo 3 または 2（音楽特化の AF-CLAP） |
| コールセンター監査 | API 経由の Gemini 2.5 Pro、ポリシー文書に対する RAG を併用 |

## 落とし穴

- **マルチオーディオへの過信。** タスクに「どのクリップに X が含まれるか」が必要な場合、ランダム水準の性能は現実だ。
- **長尺音声の劣化。** 10分を超えると、ほとんどのモデルの話者帰属が崩れる。まず話者分離（レッスン6）を行い、その後に要約する。
- **静寂での幻覚。** Whisper エンコーダを使う LALM が継承する、Whisper 由来の同じ問題。VAD でゲートする。
- **ベンチマークのチェリーピッキング。** ベンダーのブログ投稿はベストケースのカテゴリを強調する。MMAU-Pro のマルチオーディオサブセットを自分で実行すること。

## 出荷する

`outputs/skill-alm-picker.md` として保存する。与えられた音声理解タスクに対して、LALM + ベンチマークサブセット + 出力モダリティ（テキスト vs 音声）を選択する。

## 演習

1. **易。** `code/main.py` を実行し、おもちゃのプロジェクタパターンと、（音声埋め込み、テキストトークン） → 出力トークンというフェイク LALM ルーティングを確認する。
2. **中。** Qwen2.5-Omni-7B を100件の MMAU-Pro 音声項目で採点する。論文で報告されている数値と比較する。
3. **難。** 最小限の音声キャプショニングのベースラインを構築する: BEATs エンコーダ + 2層プロジェクタ + 凍結した Llama-3.2-1B。AudioCaps でプロジェクタのみをファインチューニングする。Clotho-AQA で SALMONN と比較する。

## 重要用語

| 用語 | 世間の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| LALM | 音声版 ChatGPT | 音声エンコーダ + プロジェクタ + LLM デコーダ。 |
| プロジェクタ | アダプタ | 音声特徴を LLM 埋め込み空間にマッピングする小さな MLP。 |
| MMAU | 例のベンチマーク | 音声、環境音、音楽にわたる1万の音声 QA ペア。 |
| MMAU-Pro | より難しい MMAU | 1800のマルチオーディオ / 推論重視の質問。 |
| LongAudioBench | 長尺評価 | 意味的クエリを伴う数分のクリップ。 |
| 音声入力 / 音声出力 | スピーチネイティブ | モデルが音声を取り込み、テキストを経由せずに音声を出力する。 |

## 参考文献

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — リファレンスアーキテクチャ。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 音声入力・音声出力。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — オープンな長尺音声のリーダー。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — デュアルエンコーダのパイオニア。
- [MMAU-Pro リーダーボード](https://mmaubenchmark.github.io/) — 2026年のライブランキング。
