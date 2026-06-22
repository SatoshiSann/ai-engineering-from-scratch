# 音声認識 (ASR) — CTC、RNN-T、アテンション

> 音声認識は、各タイムステップでのオーディオ分類を、英語と無音を知るシーケンスモデルで貼り合わせたものだ。CTC、RNN-T、アテンションがそれを行う3つの方法だ。1つを選び、なぜそうなのかを理解せよ。

**タイプ:** Build
**言語:** Python
**前提条件:** Phase 6 · 02 (Spectrograms & Mel)、Phase 5 · 08 (CNNs & RNNs for Text)、Phase 5 · 10 (Attention)
**所要時間:** 約45分

## 問題

10秒の16 kHzクリップがある。あなたは文字列が欲しい。「turn on the kitchen lights」。課題は構造的だ。オーディオフレームは文字と一対一で整列しない。「okay」という単語は200 msかかるかもしれないし、1200 msかかるかもしれない。無音が発話を区切る。一部の音素は他より長い。出力トークンの数は事前にわからない。

3つの定式化がこれを解決する:

1. **CTC (Connectionist Temporal Classification)。** 特殊な*blank*を含むフレームごとのトークン確率を出力する。デコード時に繰り返しとブランクを畳み込む。非自己回帰的で高速。wav2vec 2.0、MMSが使用。
2. **RNN-T (Recurrent Neural Network Transducer)。** ジョイントネットワークが、エンコーダフレームと以前のトークンが与えられたときに次のトークンを予測する。ストリーム可能。GoogleのオンデバイスASR、NVIDIA Parakeetが使用。
3. **アテンションエンコーダ-デコーダ。** エンコーダがオーディオを隠れ状態に圧縮し、デコーダがクロスアテンションして自己回帰的にトークンを生成する。Whisper、SeamlessM4Tが使用。

2026年、LibriSpeech test-cleanにおけるSOTA WERは1.4% (Parakeet-TDT-1.1B、NVIDIA)と1.58% (Whisper-Large-v3-turbo)だ。違いはわずかだが、デプロイメントの違いは巨大だ。

## コンセプト

![3つのASR定式化: CTC, RNN-T, アテンションエンコーダ-デコーダ](../assets/asr-formulations.svg)

**CTCの直観。** エンコーダに`V+1`トークン(V文字 + ブランク)にわたる`T`個のフレームレベル分布を出力させる。長さ`U < T`のターゲット文字列`y`に対して、`y`に畳み込まれる任意のフレームアラインメントがカウントされる。CTC損失はそのようなすべてのアラインメントにわたって合計する。推論: フレームごとのargmax、繰り返しの畳み込み、ブランクの除去。

利点: 非自己回帰的、ストリーム可能、ルックアヘッドゼロ。欠点: *条件付き独立性の仮定* — 各フレームの予測は他のフレームから独立しているため、内部言語モデルがない。ビームサーチまたはシャローフュージョン経由の外部LMで修正する。

**RNN-Tの直観。** トークン履歴を埋め込む*predictor*ネットワークと、predictorの状態とエンコーダフレームを`V+1`にわたる結合分布(`+1`はnull / 出力なし)に結合する*joiner*を追加する。CTCが無視した条件付き依存関係を明示的にモデル化する。各ステップが過去のフレームと過去のトークンのみに条件付けされるため、ストリーム可能だ。

利点: ストリーム可能 + 内部LM。欠点: 訓練がより複雑でメモリ消費が大きい(3D損失格子)。RNN-T損失カーネルはそれ自体でライブラリのカテゴリ全体を成す。

**アテンションエンコーダ-デコーダ。** 対数メルフレーム上のエンコーダ(6〜32 transformer層)。デコーダ(6〜32 transformer層)がエンコーダ出力にクロスアテンションして自己回帰的にトークンを生成する。アラインメント制約なし — アテンションはオーディオのどこでも見ることができる。アテンションを制限しない限りストリーム不可(チャンク化されたWhisper-Streaming、2024)。

利点: オフラインASRで最高品質、標準的なseq2seqツールで訓練が容易。欠点: 自己回帰的なレイテンシは出力長に比例する。エンジニアリングなしではストリームできない。

### WER: 唯一の数値

**Word Error Rate** = `(S + D + I) / N`、ここでS=置換、D=削除、I=挿入、N=参照単語数。単語レベルでのLevenshtein編集距離に一致する。低いほど良い。20%を超えるWERは一般的に使い物にならない。5%未満は読み上げ音声に対して人間と同等だ。標準ベンチマークでの2026年の数値:

| モデル | LibriSpeech test-clean | LibriSpeech test-other | サイズ |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 11億パラメータ |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

これらすべてはエンコーダ-デコーダまたはRNN-Tベースだ。純粋なCTCシステム(wav2vec 2.0)はtest-cleanで1.8〜2.1%あたりに位置する。

## 作ってみる

### ステップ1: 貪欲なCTCデコード

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

2つのルール: 連続する繰り返しを畳み込み、ブランクを除去する。例: `a a _ _ a b b _ c` → `a a b c`。

### ステップ2: ビームサーチCTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

本番ではLMフュージョン付きのプレフィックスツリービームサーチを使う。これは概念的な骨組みだ。

### ステップ3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### ステップ4: Whisperに対する推論

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026年で最強の汎用ASRのワンライナー。24 GB GPU上で約20倍リアルタイムで動く。

### ステップ5: ParakeetまたはWav2vec 2.0によるストリーミング

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

ストリーミングASRにはチャンク化されたエンコーダアテンションと持ち越し状態が必要だ。それをサポートするライブラリを使うこと(Parakeet用にNeMo、`chunk_length_s`付きの`transformers`パイプライン)。

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|-----------|------|
| 英語、オフライン、最大品質 | Whisper-large-v3-turbo |
| 多言語、堅牢 | SeamlessM4T v2 |
| ストリーミング、低レイテンシ | Parakeet-TDT-1.1BまたはRiva |
| エッジ、モバイル、レイテンシ <500 ms | 量子化したWhisper-TinyまたはMoonshine (2024) |
| 長尺 | VADベースのチャンク化付きWhisper (WhisperX) |
| ドメイン特化 (医療、法律) | wav2vec 2.0をファインチューニング + ドメインLMフュージョン |

## 2026年でもまだ出荷される落とし穴

- **VADなし。** 無音上でWhisperを実行すると幻覚が生じる(「Thanks for watching!」)。常にVADでゲートすること。
- **文字対単語対サブワードWER。** 正規化(小文字化、句読点の除去)*の後*に単語レベルのWERを報告すること。
- **言語IDのドリフト。** Whisperの自動LIDはノイジーなクリップを日本語やウェールズ語に誤ルーティングする。わかっているときは`language="en"`を強制すること。
- **チャンク化なしの長尺クリップ。** Whisperには30秒の窓がある。それより長いものには`chunk_length_s=30, stride=5`を使うこと。

## 出荷する

`outputs/skill-asr-picker.md`として保存する。与えられたデプロイメントターゲットに対して、モデル、デコード戦略、チャンク化、LMフュージョンを選択する。

## 演習

1. **易しい。** `code/main.py`を実行する。手作りのCTC出力を貪欲にデコードし、参照に対してWERを計算する。
2. **中級。** ステップ2のプレフィックスツリービームサーチを適切に実装せよ(ブランクのマージルールを考慮すること)。10例の合成データセットで貪欲法と比較せよ。
3. **難しい。** [LibriSpeech test-clean](https://www.openslr.org/12)で`whisper-large-v3-turbo`を使え。最初の100発話でWERを計算せよ。公表された数値と比較せよ。

## 主要用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| CTC | ブランクトークン損失 | すべてのフレーム-トークンアラインメントにわたる周辺化。非AR。 |
| RNN-T | ストリーミング損失 | CTC + 次トークンpredictor。語順を扱う。 |
| Attention enc-dec | Whisperスタイル | エンコーダ + クロスアテンションするデコーダ。最高のオフライン品質。 |
| WER | あなたが報告する数値 | 単語レベルでの`(S+D+I)/N`。 |
| Blank | その空虚さ | 「このフレームでは出力なし」を知らせるCTCの特殊トークン。 |
| LM fusion | 外部言語モデル | ビームサーチ中に加重したLM対数確率を加える。 |
| VAD | 無音ゲート | 音声活動検出器。非音声をトリミングする。 |

## 参考文献

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTCの論文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-Tの論文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022年の正典的論文。v3-turbo拡張は2024年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 Open ASRリーダーボードのリーダー。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25以上のモデルにわたるライブベンチマーク。
