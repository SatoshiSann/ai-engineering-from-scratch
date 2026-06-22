# GPU セットアップとクラウド

> 学習目的ならCPUで十分。本格的なトレーニングにはGPUが必要です。

**種別:** 実装
**言語:** Python
**前提条件:** フェーズ0、レッスン01
**所要時間:** 約45分

## 学習目標

- `nvidia-smi` とPyTorchのCUDA APIを使ってローカルGPUの使用可否を確認する
- Google ColabでT4 GPUを設定し、無料のクラウド環境で実験する
- CPUとGPUで行列乗算のベンチマークを取り、速度向上を計測する
- fp16の目安ルールを使って、VRAMに収まる最大モデルサイズを見積もる

## 課題

フェーズ1〜3のほとんどのレッスンはCPUで問題なく動作します。しかし、CNN・トランスフォーマー・LLMのトレーニングを始めると（フェーズ4以降）、GPU高速化が必要になります。CPUで8時間かかるトレーニングが、GPUなら10分で完了します。

選択肢は3つあります：ローカルGPU、クラウドGPU、またはGoogle Colab（無料）。

## 概念

```
選択肢：

1. ローカルNVIDIA GPU
   コスト: $0（すでに所有している場合）
   セットアップ: CUDA + cuDNN のインストール
   最適用途: 日常的な使用、大規模データセット

2. Google Colab（無料枠）
   コスト: $0
   セットアップ: 不要
   最適用途: 手軽な実験、自宅にGPUがない場合

3. クラウドGPU（Lambda、RunPod、Vast.ai）
   コスト: $0.20〜2.00/時間
   セットアップ: SSH + インストール
   最適用途: 本格的なトレーニング、大規模モデル
```

## 実装

### オプション1: ローカルNVIDIA GPU

GPUがあるか確認します：

```bash
nvidia-smi
```

CUDA付きでPyTorchをインストールします：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### オプション2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com) にアクセスする
2. ランタイム > ランタイムのタイプを変更 > T4 GPU を選択する
3. `!nvidia-smi` を実行して確認する

このコースのノートブックをColabに直接アップロードして使用できます。

### オプション3: クラウドGPU

Lambda Labs、RunPod、またはVast.aiの場合：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPUがない場合でも大丈夫

ほとんどのレッスンはCPUで動作します。GPUが必要なレッスンにはその旨が記載され、Colabリンクが含まれます。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 実装: GPU vs CPU ベンチマーク

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 演習

1. 上記のベンチマークを実行し、CPUとGPUの処理時間を比較する
2. GPUがない場合は、Google Colabで実行して比較する
3. GPUメモリの容量を確認し、収まる最大モデルサイズを見積もる（目安：fp16では1パラメータあたり2バイト）

## 重要用語

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| CUDA | 「GPUプログラミング」 | NVIDIAの並列コンピューティングプラットフォーム。GPU上でコードを実行できる |
| VRAM | 「GPUメモリ」 | GPU上のビデオRAM。システムRAMとは別でモデルサイズを制限する |
| fp16 | 「半精度」 | 16ビット浮動小数点。fp32の半分のメモリで最小限の精度低下にとどまる |
| Tensor Core | 「高速行列演算ハードウェア」 | 行列乗算専用のGPUコア。通常コアより4〜8倍高速 |
