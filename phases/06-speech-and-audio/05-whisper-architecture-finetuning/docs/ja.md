# Whisper — アーキテクチャとファインチューニング

> Whisperは30秒窓のtransformerエンコーダ-デコーダで、68万時間の多言語弱教師ありオーディオ-テキストペアで訓練されている。1つのアーキテクチャ、複数のタスク、99言語にわたって堅牢。2026年のリファレンスASRだ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 04 (ASR)、Phase 5 · 10 (Attention)、Phase 7 · 05 (Full Transformer)
**所要時間:** 約75分

## 問題

2022年9月にOpenAIによってリリースされたWhisperは、コモディティとして出荷された最初のASRモデルだった。オーディオを貼り付け、テキストを得る、99言語、ノイズに堅牢、ラップトップで動く。2024年までにOpenAIはLarge-v3とTurboのバリアントを出荷した。2026年までに、Whisperはポッドキャストの文字起こしからボイスアシスタント、YouTubeの字幕まで、あらゆるもののデフォルトベースラインとなっている。

しかしWhisperは、永遠にブラックボックスとして扱えるパイプラインではない。ドメインシフトがそれを殺す — 技術用語、話者のアクセント、固有名詞、短いクリップ、無音。あなたは次のことを知る必要がある:

1. 内部で実際に何であるか。
2. チャンク化された、ストリーミングの、または長尺のオーディオを正しく与える方法。
3. いつファインチューニングし、どのように行うか。

## コンセプト

![Whisperエンコーダ-デコーダ、タスク、チャンク化推論、ファインチューニング](../assets/whisper.svg)

**アーキテクチャ。** 標準的なtransformerエンコーダ-デコーダ。

- 入力: 30秒の対数メルスペクトログラム、80メル、10 msホップ → 3000フレーム。より短いクリップはゼロパディングされ、より長いクリップはチャンク化される。
- エンコーダ: conv-ダウンサンプル(ストライド2) + `N`個のtransformerブロック。Large-v3の場合: 32層、1280次元、20ヘッド。
- デコーダ: 因果的自己アテンション + エンコーダ出力へのクロスアテンションを持つ`N`個のtransformerブロック。エンコーダと同じサイズ。
- 出力: 51,865トークン語彙にわたるBPEトークン。

Large-v3は15.5億パラメータを持つ。Turboは(32層から)4層デコーダを使い、レイテンシを8倍削減し、WERへの影響は1%未満だ。

**プロンプト形式。** Whisperはデコーダプロンプト内の特殊トークンによって操舵されるマルチタスクモデルだ:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 言語タグ。翻訳対文字起こしの動作を強制する。
- `<|transcribe|>`または`<|translate|>` — 任意の言語入力から英語出力を翻訳するか、逐語的に行う。
- `<|notimestamps|>` — 単語レベルのタイムスタンプをスキップする(高速)。

プロンプトこそが、1つのモデルに多くのタスクを行わせるものだ。`<|en|>`を`<|fr|>`に変えれば、フランス語を文字起こしする。

**30秒窓。** すべてが30秒に固定されている。より長いクリップにはチャンク化が必要で、より短いクリップはパディングされる。窓はネイティブにストリームされない — これがWhisperX、Whisper-Streaming、faster-whisperが存在する理由だ。

**対数メル正規化。** `(log_mel - mean) / std`、ここで統計値はWhisper自身の訓練コーパスから来る。`librosa.feature.melspectrogram`ではなく、Whisperの前処理(`whisper.audio.log_mel_spectrogram`)を*必ず*使わなければならない。

### 2026年のバリアント

| バリアント | パラメータ | レイテンシ (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× リアルタイム | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | ストリーミング | 2.0% |

### ファインチューニング

2026年の正典的ワークフロー:

1. 整列された文字起こし付きのターゲットドメインオーディオを10〜100時間集める。
2. `generate_with_loss`コールバック付きで`transformers.Seq2SeqTrainer`を実行する。
3. パラメータ効率的: アテンション層の`q_proj`、`k_proj`、`v_proj`上のLoRAは、WERコスト0.3未満でGPUメモリを4倍削減する。
4. 10時間未満の場合はエンコーダを凍結する。デコーダのみを調整する。
5. Whisper自身のトークナイザーとプロンプト形式を使う。決してトークナイザーを入れ替えないこと。

コミュニティの結果: 20時間の医療口述でMediumをファインチューニングすると、医療語彙でのWERが12%から4.5%に下がる。4時間のアイスランド語でTurboをファインチューニングすると、WERが18%から6%に下がる。

## 作ってみる

### ステップ1: Whisperをそのまま実行する

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

常にオーバーライドすべき主要なデフォルト: `temperature=0.0`(サンプリングはデフォルトで0.0 → 0.2 → 0.4 … のフォールバックチェーン)、`condition_on_previous_text=False`(連鎖する幻覚問題を防ぐ)、`no_speech_threshold=0.6`(無音検出)。

### ステップ2: チャンク化された長尺

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperXは(1) Silero VADゲーティング、(2) wav2vec 2.0経由の単語レベルアラインメント、(3) `pyannote.audio`経由の話者ダイアライゼーションを追加する。本番文字起こしのための2026年の働き者だ。

### ステップ3: LoRAでファインチューニングする

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

その後、標準的なTrainerループ。1000ステップごとにチェックポイント。ホールドアウトでWERにより評価する。

### ステップ4: 各層が何を学習するかを調べる

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

ヒートマップで可視化する — デコーダステップがエンコーダフレームをスキャンするにつれて、対角的なアラインメントが見えるだろう。その対角こそがWhisperの単語タイムスタンプの概念だ。

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| 一般的な英語、オフライン | `whisperx`経由のLarge-v3-turbo |
| モバイル / エッジ | 量子化(int8)したWhisper-TinyまたはMoonshine |
| 多言語長尺 | `whisperx`経由のLarge-v3 + 話者ダイアライゼーション |
| 低リソース言語 | LoRAでMediumまたはTurboをファインチューニング |
| ストリーミング (2秒レイテンシ) | Whisper-StreamingまたはParakeet-TDT |
| 単語レベルタイムスタンプ | WhisperX (wav2vec 2.0経由の強制アラインメント) |

`faster-whisper`(CTranslate2バックエンド)は2026年で最速のCPU+GPU推論ランタイムだ — 同一の出力でバニラより4倍高速。

## 2026年でもまだ出荷される落とし穴

- **無音上での幻覚テキスト。** キャプションで訓練されたWhisperは「Thanks for watching!」、「Subscribe!」、歌詞を含む。呼び出す前に常にVADでゲートすること。
- **`condition_on_previous_text`の連鎖。** 1つの幻覚が後続の窓を汚染する。チャンクをまたいだ流暢さが必要でない限り`False`に設定すること。
- **短いクリップのパディング。** 30秒にパディングされた2秒のクリップは、末尾の無音で幻覚を起こすことがある。`pad=False`を使うかVADでゲートすること。
- **間違ったメル統計値。** Whisperのものの代わりにlibrosaのメルを使うと、ほぼランダムな出力が生じる。`whisper.audio.log_mel_spectrogram`を使うこと。

## 出荷する

`outputs/skill-whisper-tuner.md`として保存する。与えられたドメインに対してWhisperのファインチューニングまたは推論パイプラインを設計する。

## 演習

1. **易しい。** `code/main.py`を実行する。Whisperスタイルのプロンプトをトークン化し、デコードされた形状の予算を計算し、10分のクリップのチャンクスケジュールを出力する。
2. **中級。** `faster-whisper`をインストールし、10分のポッドキャストを文字起こしし、人間の文字起こしに対してWERを比較せよ。`language="auto"`対強制された`language="en"`を試せ。
3. **難しい。** HF `datasets`を使って、Whisperが苦戦する言語(例: ウルドゥー語)を選び、2時間で2エポックLoRAでMediumをファインチューニングし、WERの差分を報告せよ。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 30秒窓 | Whisperの限界 | ハードな入力上限。より長いオーディオをチャンク化する。 |
| SOT | Start-of-transcript | `<\|startoftranscript\|>`がデコーダプロンプトを開始する。 |
| Timestampsトークン | 時間的アラインメント | 0.02秒オフセットごとが51k語彙内の特殊トークン。 |
| Turbo | 高速バリアント | 4デコーダ層、8倍高速、WER劣化1%未満。 |
| WhisperX | 長尺ラッパー | VAD + Whisper + wav2vecアラインメント + 話者ダイアライゼーション。 |
| LoRAファインチューニング | 効率的な調整 | アテンションに低ランクアダプタを追加。パラメータの約0.3%を訓練。 |
| 幻覚 | サイレント障害 | Whisperがノイズ/無音から流暢な英語を生成する。 |

## 参考文献

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 元のアーキテクチャと訓練レシピ。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4層デコーダ、8倍高速化。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 長尺、単語整列、話者ダイアライゼーション。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2バック、4倍高速。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 正典的なLoRA / フルFTのウォークスルー。
