# Mixture of Experts (MoE)

> 密な70Bトランスフォーマーはすべてのトークンに対してすべてのパラメータを活性化します。671B MoEはトークンごとに37Bのみを活性化し、すべてのベンチマークでそれを上回ります。スパーシティは10年間の最も重要なスケーリングアイデアです。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 7 · 05 (完全なトランスフォーマー), Phase 7 · 07 (GPT)
**所要時間:** 約45分

## 問題

密なトランスフォーマーの推論時FLOPはそのパラメータ数に等しい(前向きパスで2倍)。密なモデルをスケールアップし、すべてのトークンが全額を支払います。2024年までに、フロンティアは計算壁に当たっていました：より良くなるために、トークンごとに指数関数的により多くのFLOPが必要でした。

Mixture of Experts はこのリンクを破ります。各FFNを`E`個の独立した専門家 + `k`個の専門家を選択するルータで置換します。総パラメータ = `E × FFN_size`。トークンごとのアクティブパラメータ = `k × FFN_size`。典型的な2026年設定：`E=256`、`k=8`。ストレージは`E`でスケール、計算は`k`でスケール。

2026年のフロンティアはほぼ完全にMoE：DeepSeek-V3 (671B合計 / 37Bアクティブ)、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。Artificial Analysisの独立したリーダーボード上で、トップ10オープンソースモデルはすべてMoEです。

## コンセプト

![MoEレイヤー：ルータがトークンごとにEのkを選択](../assets/moe.svg)

### FFNスワップ

密なトランスフォーマーブロック：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoEブロック：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # トークンごとにEのkを選択
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

すべての専門家は独立したFFN(通常SwiGLU)。ルータは単一の線形レイヤー。各トークンは独自の`k`専門家を選択し、それらの出力のゲーティングされた混合を取得。

### 負荷バランシング問題

ルータが90%のトークンを専門家3に通し、他の専門家は飢える。3つの修正が試されました：

1. **補助負荷バランシング損失**(Switch Transformer、Mixtral)。専門家使用の分散に比例するペナルティを追加。機能しますが、ハイパーパラメータと2番目の勾配信号を追加。
2. **専門家キャパシティ + トークンドロップ**(初期Switch)。各専門家は最大`C × N/E`トークンを処理；オーバーフロートークンはレイヤーをスキップ。品質に害を与える。
3. **補助損失なし均衡**(DeepSeek-V3)。学習済みの専門家ごとのバイアスを追加して、ルータのトップk選択をシフト。バイアスは学習損失の外で更新。メインの目的に対するペナルティなし。2024年の大きなロック解除。

DeepSeek-V3のアプローチ：各学習ステップの後、すべての専門家について、その使用がターゲット以上/以下であるかをチェック。バイアスを`±γ`で推す。選択は`scores + bias`を使用。ゲーティングに使用される専門家確率は未変更の生の`scores`。ルーティングから表現を分離。

### 共有専門家

DeepSeek-V2/V3は専門家を*共有*と*ルートされた*に分割。すべてのトークンはすべての共有専門家を通す。ルートされた専門家はトップkを通じて選択。共有専門家は一般的な知識を取得；ルートされた専門家は専門化。V3は1つの共有専門家と256のルートされたうち上位8を実行。

### 細粒度専門家

古典的なMoE (GShart、Switch)：各専門家は完全なFFNと同じくらい広い。`E`は小さい(8-64)、`k`は小さい(1-2)。

現代的な細粒度MoE (DeepSeek-V3、Qwen-MoE)：各専門家はより狭い(1/8 FFNサイズ)。`E`は大きい(256+)、`k`はより大きい(8+)。同じ総パラメータですが、組み合わせはより高速にスケール。`C(256, 8) = 400兆`可能な「専門家」/トークン。品質が上がり、遅延が平坦。

### コストプロファイル

トークンあたり、レイヤーあたり：

| 構成 | トークンごとのアクティブパラメータ | 総パラメータ |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (密) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3はLlama 3 70B (密)をほぼすべてのベンチマークで上回りますが、トークンごとに**より少ないアクティブFLOP**を実行。より多くのパラメータ=より多くの知識。より多くのアクティブFLOP=トークンごとにより多くの計算。MoEはそれらを分離。

### キャッチ：メモリ

すべての専門家がどちらかのGPUに存在するかに関わらず。671Bモデルはfp16ウェイトに~1.3 TBのVRAMが必要。フロンティアMoE展開は専門家並列性が必要 — 専門家をGPU間で分割、トークンをネットワーク間でルート。遅延は行列乗算ではなく、すべてツーオール通信で支配。

## ビルド

`code/main.py`を見てください。純粋なstdlibでコンパクトなMoEレイヤー：

- `n_experts=8` SwiGLU-ishな専門家(イラスト用に各1線形)
- top-k=2ルーティング
- ソフトマックス正規化ゲーティング重み
- 専門家ごとのバイアス経由の補助損失なし均衡

### ステップ1：ルータ

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 選択された専門家の元のスコアをソフトマックス化
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

バイアスはゲート重みではなく選択に影響。それはDeepSeek-V3トリック — バイアスはモデルの予測をステアリングせずに負荷不均衡を修正。

### ステップ2：100トークンをルータ経由で実行

どの専門家が何回火るかを追跡。バイアスなしで、使用は歪みます。バイアス更新ループ(過剰使用専門家には`-γ`、未使用専門家には`+γ`)の場合、使用は数回のイテレーション上で一様分布に収束。

### ステップ3：パラメータ数比較

MoE設定の「密な同等物」を印刷。DeepSeek-V3形：256ルート + 1共有、8アクティブ、d_model=7168。総パラメータ数は目を見張るばかり。アクティブ数は密なLlama 3 70Bの7番目。

## 使用

HuggingFaceロード：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026年本番推論：vLLMはMoEルーティングをネイティブにサポート。SGLangは最速の専門家並列パス。両方自動的にトップk選択と専門家並列性を処理。

**MoEをいつ選択するか：**
- より低い推論コストでトークンごとにフロンティア品質が必要。
- VRAM/専門家並列インフラストラクチャを持つ。
- ワークロードがコンテキスト重い(長いドキュメント)ではなくトークン重い(チャット、コード)。

**MoEをいつ選択しないか：**
- エッジ展開 — アクティブなFLOPのフル保管を支払う。
- レイテンシー臨界単一ユーザー提供 — 専門家ルーティングはオーバーヘッドを追加。
- 小さいモデル(<7B) — MoE品質優位は計算閾値(~6Bアクティブパラメータ)以上で表示。

## シップ

`outputs/skill-moe-configurator.md`を見てください。スキルはパラメータ予算、訓練トークン、展開ターゲットが与えられた新しいMoE用のE、k、共有専門家レイアウトを選択。

## 演習

1. **簡単。** `code/main.py`を実行。補助損失なしのバイアス更新が50回のイテレーション上で専門家使用を均等化する方法を見る。
2. **中程度。** 学習されたルータをハッシュベースのルータで置換(決定的、学習なし)。品質とバランスを比較。学習されたルータがなぜ優れているのか？
3. **難しい。** GRPO形「ロールアウトマッチングルーティング」を実装(DeepSeek-V3.2トリック)：推論中にどの専門家が火るかをログ、勾配計算中に同じルーティングを強制。小さいポリシー勾配設定での効果を測定。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 専門家 | 「多くの中の1つのFFN」 | 独立したフィードフォワードネットワーク；FFN計算の疎なスライスに専念するパラメータ。 |
| ルータ | 「ゲート」 | 各トークンを各専門家に対してスコア付けする小さい線形レイヤー；トップk選択。 |
| トップkルーティング | 「トークンごとkアクティブ専門家」 | 各トークンのFFN計算は正確にk個の専門家を経由し、ゲートで重み付け。 |
| 補助損失 | 「負荷バランスペナルティ」 | 歪んだ専門家使用にペナルティを与える追加損失項。 |
| 補助損失なし | 「DeepSeek-V3トリック」 | ルータの選択のみの専門家ごとのバイアス経由の均衡；追加勾配なし。 |
| 共有専門家 | 「常にオン」 | すべてのトークンが通す追加専門家；一般的な知識をキャプチャ。 |
| 専門家並列性 | 「専門家でシャード」 | 異なる専門家を異なるGPU間に分散；トークンをネットワーク間でルート。 |
| スパーシティ | 「アクティブパラメータ < 総パラメータ」 | 比率`k × expert_size / (E × expert_size)`；DeepSeek-V3では37/671 ≈ 5.5%。 |

## 参考文献

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — アイデア。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch、古典的なMoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 補助損失なしMoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — バイアスベース均衡論文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 細粒度 + 共有専門家スプリット。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 元の共有専門家論文。
