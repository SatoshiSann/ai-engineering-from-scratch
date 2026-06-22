# エッジ推論 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> コア エッジ制約はメモリ帯域幅であり、計算ではありません。モバイル DRAM は 50-90 GB/s で保持；データセンター HBM3 は 2-3 TB/s をクリア — 30-50 倍ギャップ。デコードはメモリバウンドなので、ギャップは決定的です。2026 年の風景は 4 つの方法に分割します。Apple M4/A18 Neural Engine は統合メモリで最大 38 TOPS に達します（CPU↔NPU コピー なし）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon は 45 TOPS に当たります。WebGPU + WebLLM は M3 Max で Llama 3.1 8B（Q4）を 約 41 tok/s で実行します（ほぼ 70-80%ネイティブ）；17.6k GitHub スター、OpenAI 互換 API、約 70-75% モバイルカバレッジ。NVIDIA Jetson Orin Nano Super（8GB）は Llama 3.2 3B / Phi-3 に収まります；AGX Orin は vLLM を介して gpt-oss-20b を約 40 tok/s で実行します；Jetson T4000（JetPack 7.1）は 2 倍 AGX Orin。TensorRT Edge-LLM は EAGLE-3、NVFP4、チャンク プリフィル をサポート — 2026 CES で Bosch、ThunderSoft、MediaTek によって示されました。

**タイプ:** Learn
**言語:** Python (stdlib、おもちゃの帯域幅バウンド デコード シミュレータ)
**前提条件:** Phase 17 · 04（vLLM Serving Internals）、Phase 17 · 09（本番量子化）
**所要時間:** 約 60 分

## 学習目標

- モバイル LLM 推論がメモリ帯域幅バウンドであり、計算は二次的である理由を説明してください。
- 4 つのエッジターゲット（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）を列挙し、各々をユースケースにマッチさせてください。
- 2026 年 WebGPU カバレッジギャップ（Firefox Android が追いつく）と Safari iOS 26 ランディングを名前で述べてください。
- ターゲットごとに量子化フォーマットを選択してください（ANE は Core ML INT4 + FP16、Hexagon は QNN INT8/INT4、WebGPU は Q4、Jetson Thor は NVFP4）。

## 問題

顧客は、オンデバイス チャットボットを望んでいます：音声ファースト、デフォルトで プライベート、オフラインで動作。MacBook Pro M3 Max では、Llama 3.1 8B Q4 は 約 55 tok/s で実行 — 良いです。iPhone 16 Pro では、同じモデルは 3 tok/s で実行 — 悪い。Snapdragon 8 Gen 3 を持つミッドレンジ Android では、7 tok/s。Chrome Android v121+ 経由で WebGPU を使用したブラウザでは、デバイスに応じて 4-8 tok/s。

スループット分散はポーティング問題ではありません。これは帯域幅ギャップ倍 量子化フォーマット 倍 NPU がユーザースペースからアクセス可能かどうかです。2026 年のエッジ推論は 4 つの異なる問題で 4 つの異なるソリューション です。

## コンセプト

### 帯域幅が本物の上限

デコードはすべての重みセットをトークンごとに読みます。Q4 の 1 つの 7B モデルは 3.5 GB です。3.5 GB を 50 GB/s で読むのに 70 ミリ秒がかかります — 理論上の上限は 約 14 tok/s。高エンド モバイル DRAM（90 GB/s）で上限は 約 25 tok/s に移動します。この数値以下では、計算は助けになりません。

データセンター HBM3 は 3 TB/s で同じ 3.5 GB を 1.2 ミリ秒でクリア — 上限は 830 tok/s です。同じモデル、同じ重み。異なるメモリサブシステム。

### Apple Neural Engine（M4 / A18）

- 最大 38 TOPS。統合メモリ（CPU と ANE は同じプール を共有） — コピー オーバーヘッド なし。
- Core ML + `.mlmodel` コンパイル済みモデル、または PyTorch を通じた Metal Performance Shaders（MPS）経由でのアクセス。
- Llama.cpp Metal バックエンドは MPS を使用し、ANE を直接使用しません；ネイティブ ANE は Core ML 変換を必要とします。
- 2026 年の iOS アプリ向けベストプラクティカル パス：INT4 重み + FP16 アクティベーション を使用した Core ML。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最大 45 TOPS。SoC の CPU と GPU と統合されていますが、別個のメモリドメイン。
- QNN（Qualcomm Neural Network）SDK および AI Hub は PyTorch/ONNX からの変換を提供。
- チャット テンプレート、Llama 3.2、Phi-3 は AI Hub でファースト クラス アーティファクトとして出荷。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。ソフトウェアは Apple/Qualcomm の後ろにあります；OpenVINO は改善していますが、ニッチです。
- Windows ARM コパイロット アプリに最適；ローカル ファースト用の AMD/Intel デスクトップでネイティブ。

### WebGPU + WebLLM

- WebGPU 計算シェーダーを使用したブラウザでモデルを実行；インストール なし。
- M3 Max で Llama 3.1 8B Q4 を 約 41 tok/s で — ほぼ 70-80%ネイティブ（同じバックエンド経由）。
- WebLLM で 17.6k GitHub スター；OpenAI 互換 JS API；Apache 2.0。
- 2026 年カバレッジ：Chrome Android v121+、Safari iOS 26 GA、Firefox Android はまだ追いつく中。全体的に 約 70-75% モバイル カバレッジ。

### NVIDIA Jetson ファミリー

- Orin Nano Super（8GB）：Llama 3.2 3B、Phi-3 に収まり、良い tok/s。
- AGX Orin：vLLM を介して gpt-oss-20b を 約 40 tok/s で実行。
- Thor / T4000（JetPack 7.1）：2 倍 AGX Orin パフォーマンス、EAGLE-3 および NVFP4 サポート。
- TensorRT Edge-LLM（2026）は EAGLE-3 投機的デコード、NVFP4 重み、チャンク プリフィル をサポート — エッジに移植された データセンター 最適化。

### ターゲットごとの量子化選択

| ターゲット | フォーマット | 注記 |
|--------|--------|-------|
| Apple ANE | INT4 重み + FP16 アクティベーション | Core ML 変換パス |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub コンバーター |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | `mlc_llm convert_weight` + コンパイル済み `.wasm` を使用；GGUF はサポートされていません |
| Jetson Orin Nano | Q4 GGUF または TRT-LLM INT4 | メモリバウンド |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM パス |

### エッジでの長コンテキスト罠

Llama 3.1 の 128K コンテキストはデータセンター機能です。8 GB RAM を持つ電話では、4 GB モデル + 32K トークン用 2 GB KV キャッシュ + OS オーバーヘッド = OOM。エッジ デプロイメントは積極的な KV 量子化（Q4 KV）が受け入れられない限り、4K-8K でコンテキストを保つ。

### 音声はキラーアプリ

音声エージェントはレイテンシ感応的（最初のトークン < 500 ミリ秒）。ローカル推論はネットワーク レイテンシを完全に排除。Whisper Turbo バリアント（エッジで実行）を使用した音声テキスト変換と組み合わせ、エッジ推論は本番品質の音声ループになります。

### 覚えるべき数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 での 約 41 tok/s。
- AGX Orin：vLLM を介して gpt-oss-20b での 約 40 tok/s。
- データセンター エッジ帯域幅ギャップ：30-50 倍。
- WebGPU モバイル カバレッジ：約 70-75%（Firefox Android 遅れ）。

## 使用方法

`code/main.py` はエッジ ターゲット全体での理論的デコード スループット上限を帯域幅バウンド 数学から計算します。観測されたベンチマークと比較し、帯域幅がボトルネックである場所、計算ではない場所を強調します。

## 出荷方法

このレッスンは `outputs/skill-edge-target-picker.md` を生成します。プラットフォーム（iOS/Android/ブラウザ/Jetson）、モデル、レイテンシ/メモリ予算が与えられた場合、量子化フォーマットと変換パイプラインを選択します。

## 演習

1. `code/main.py` を実行してください。Snapdragon 8 Gen 3（約 77 GB/s 帯域幅）での Q4 の 7B モデルのデコード上限を計算してください。観測された 6-8 tok/s と比較 — ランタイムは効率的ですか？
2. Android の WebGPU は Chrome v121+ を必要とします。古いブラウザ向けのフォールバックを設計 — 同じ OpenAI 互換 API 経由でサーバーサイド。
3. iOS アプリは 4K コンテキスト ストリーミングが必要です。どのモデル/フォーマット組み合わせが iPhone 16 で 4 GB アクティブ メモリ以下のままを可能にしますか？
4. Jetson AGX Orin は gpt-oss-20b を 40 tok/s で実行。Jetson Nano は 3B にのみ収まります。製品が両方をターゲットした場合、推論スタックをどう統一しますか？
5. 「2026 年に WebLLM は本番環境対応か」を議論してください。カバレッジ、パフォーマンス、Firefox Android ギャップを引用してください。

## 主要用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|----------------|------------------------|
| ANE | "Apple ニューラル エンジン" | M シリーズと A シリーズのオンデバイス NPU；統合メモリ |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；アクセス用 QNN SDK |
| WebGPU | "ブラウザ GPU" | W3C 標準ブラウザ GPU API；Chrome/Safari 2026 |
| WebLLM | "ブラウザ LLM ランタイム" | MLC-LLM プロジェクト；Apache 2.0；OpenAI 互換 JS |
| Jetson | "NVIDIA エッジ" | Orin Nano / AGX / Thor / T4000 ファミリー |
| TRT Edge-LLM | "エッジ TensorRT" | 2026 年エッジ TensorRT-LLM ポート；EAGLE-3 + NVFP4 |
| 統合メモリ | "共有プール" | CPU と NPU は同じ RAM を見る；コピー オーバーヘッド なし |
| バンドバウンド | "メモリ制限" | デコードはウェイト読み込みバイト/秒でゲート |
| Core ML | "Apple 変換" | ANE ネイティブ モデル用 Apple フレームワーク |
| QNN | "Qualcomm スタック" | Qualcomm Neural Network SDK |

## 参考文献

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 風景とベンチマーク。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年エッジポート案内。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — デザインとベンチマーク。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE ネイティブ変換。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 用事前変換済みモデル。
