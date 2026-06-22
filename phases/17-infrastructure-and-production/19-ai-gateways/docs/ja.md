# AI ゲートウェイ — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> ゲートウェイはアプリとモデルプロバイダーの間に座る。コア機能はプロバイダールーティング、フォールバック、再試行、レート制限、秘密参照、可観測性、ガードレール。2026 年市場分割：**LiteLLM** は MIT OSS で 100+ プロバイダー、OpenAI 互換、しかし約 2000 RPS で崩壊（8 GB メモリ、発表ベンチマークでのカスケード障害）。Python に最適、<500 RPS、dev/プロトタイピング。**Portkey** はコントロールプレーン配置（ガードレール、PII 削除、ジェイルブレイク検出、監査証跡）、2026 年 3 月に Apache 2.0 オープンソース化、20-40 ms レイテンシ オーバーヘッド、$49/mo プロダクションティア。**Kong AI Gateway** は Kong Gateway ビルド — Kong の同じ 12 CPU でのベンチマーク：Portkey より 228% 高速、LiteLLM より 859% 高速。$100/model/月 価格設定（Plus ティア max 5）。Kong 上にいるなら enterprise 適合。**Bifrost**（Maxim AI） — 設定可能なバックオフで自動再試行、OpenAI 429 での Anthropic へのフォールバック。**Cloudflare / Vercel AI ゲートウェイ** — マネージド、ゼロオペレーション、基本的な再試行。データレジデンシーが self-host 決定を駆動。Portkey と Kong はオープンソース + 任意のマネージド でミッドレンジ。

**タイプ:** Learn
**言語:** Python (stdlib, toy gateway-routing simulator)
**前提条件:** Phase 17 · 01 (Managed LLM Platforms)、Phase 17 · 16 (Model Routing)
**所要時間:** 約 60 分

## 学習目標

- 6 つのコア ゲートウェイ機能を列挙（ルーティング、フォールバック、再試行、レート制限、秘密、可観測性、ガードレール）。
- 4 つの 2026 ゲートウェイ（LiteLLM、Portkey、Kong AI、Bifrost）をスケール上限とユースケースにマップ。
- Kong ベンチマーク（Portkey より 228%、LiteLLM より 859%）を引用して >500 RPS で重要な理由を説明。
- データレジデンシーと ops 予算を指定して self-hosted vs マネージドを選択。

## 問題

プロダクトは OpenAI、Anthropic、self-hosted Llama を呼び出す。各プロバイダーは異なる SDK、エラーモデル、レート制限、auth スキーム。フェイルオーバー（OpenAI 429s なら Anthropic try）、シングル資格情報ストア、統一可観測性、テナントあたりレート制限が欲しい。

アプリレイヤーでこれを再発明すると、すべてのサービスをすべてのプロバイダーに結合。ゲートウェイレイヤーはそれをシングル プロセス、シングル API（通常 OpenAI 互換）に統合してプロバイダーにファンアウト。

## コンセプト

### 6 つのコア機能

1. **プロバイダー ルーティング** — OpenAI、Anthropic、Gemini、self-hosted、等。シングル API の背後。
2. **フォールバック** — 429、5xx、品質障害で他を試す。
3. **再試行** — 指数バックオフ、境界試行。
4. **レート制限** — テナントあたり、key あたり、モデルあたり。
5. **秘密参照** — ランタイムでのファイルから資格情報を取得（アプリに決してない）。
6. **可観測性** — OTel + GenAI 属性（Phase 17 · 13）+ コスト帰属。
7. **ガードレール** — PII 削除、ジェイルブレイク検出、許可テーマフィルター。

### LiteLLM — MIT OSS、Python

- 100+ プロバイダー、OpenAI 互換、ルーター設定、フォールバック、基本的な可観測性。
- Kong のベンチマークで約 2000 RPS で崩壊。8 GB メモリフットプリント、持続荷重下でカスケード障害。
- ベストフィット：Python アプリ、<500 RPS、dev/staging ゲートウェイ、実験的ルーティング。
- コスト：OSS $0。クラウド無料ティア存在。

### Portkey — コントロール プレーン配置

- Apache 2.0 OSS 2026 年 3 月時点。ガードレール、PII 削除、ジェイルブレイク検出、監査証跡。
- 20-40 ms リクエストあたりレイテンシ オーバーヘッド。
- $49/mo プロダクション ティア、retention + SLA。
- ベストフィット：ガードレール + 可観測性を束ねて必要な規制産業。

### Kong AI Gateway — スケール プレイ

- Kong Gateway ビルド（成熟 API ゲートウェイプロダクト、lua+OpenResty）。
- Kong の 12 CPU 同等でのベンチマーク：Portkey より 228% 高速、LiteLLM より 859% 高速。
- 価格設定：$100/model/月、Plus ティア max 5。
- ベストフィット：既に Kong 上。>1000 RPS。ライセンス希望。

### Bifrost（Maxim AI）

- 設定可能なバックオフで自動再試行。
- OpenAI 429 での Anthropic へのフォールバックは canonical レシピ。
- 新進気鋭。商用。

### Cloudflare AI Gateway / Vercel AI Gateway

- マネージド、ゼロオペレーション。基本的な再試行と可観測性。
- ベストフィット：Cloudflare/Vercel で edge-serving JavaScript アプリ。
- Kong/Portkey と比較してガードレール、レート制限で限定。

### Self-hosted vs マネージド

データレジデンシーは forcing 関数。Healthcare と finance は self-host デフォルト（LiteLLM または Portkey OSS または Kong）。コンシューマプロダクト はマネージド デフォルト（Cloudflare AI Gateway）またはミッド-ティア（Portkey マネージド）。ハイブリッド：規制テナントには self-hosted、その他マネージド。

### レイテンシ バジェット

- LiteLLM：典型的なオーバーヘッド 5-15 ms。
- Portkey：20-40 ms オーバーヘッド。
- Kong：3-8 ms オーバーヘッド。
- Cloudflare/Vercel：1-3 ms オーバーヘッド（edge アドバンテージ）。

ゲートウェイ レイテンシは直接 TTFT に追加。TTFT P99 < 100 ms SLA に対して Kong または Cloudflare。P99 < 500 ms なら任意。

### レート制限セマンティクスが重要

シンプル token-bucket は中程度スケールまで動作。マルチテナント には sliding-window + バースト allowance + テナントあたりティア分け が必要。LiteLLM は token-bucket を配送。Kong は sliding-window を配送。Portkey はティア分けを配送。

### ゲートウェイ + 可観測性 + ルーティング は複合

Phase 17 · 13（可観測性）+ 16（モデルルーティング）+ 19（ゲートウェイ）は production の同じレイヤー。すべて 3 つをカバーするシングル ツール を選ぶか、注意深く wire：最も 2026 デプロイメント は Helicone（可観測性）または Portkey（ガードレール）と Kong（スケール）を組み合わせる。

### 記憶すべき数字

- LiteLLM：約 2000 RPS で崩壊、8 GB メモリ。
- Portkey：20-40 ms オーバーヘッド。Apache 2.0 2026 年 3 月以降。
- Kong：Portkey より 228% 高速、LiteLLM より 859% 高速。
- Kong 価格設定：$100/model/月、Plus ティア max 5。
- Cloudflare/Vercel：edge で 1-3 ms オーバーヘッド。

## 使用方法

`code/main.py` は 3 プロバイダーで 429/5xx 注入での ゲートウェイ ルーティング、フォールバックをシミュレート。レイテンシ、再試行率、フォールバック ヒット率を報告。

## 配送

このレッスンは `outputs/skill-gateway-picker.md` を作成。スケール、ops posture、compliance、レイテンシ バジェットを指定してゲートウェイを選択。

## 演習

1. `code/main.py` を実行。OpenAI→Anthropic→self-hosted フェイルバック設定。5% プロバイダーエラー率での期待ヒット率は？
2. SLA が TTFT P99 < 200 ms、300 ms ベースライン。どのゲートウェイがバジェット内に留まるか？
3. Healthcare カスタマーが self-hosted + PII 削除 + 監査を要求。Portkey OSS または Kong？
4. LiteLLM vs Kong 比較：何 RPS 上限でチームが移行すべきか？
5. マルチテナント SaaS のレート制限ポリシー設計：free tier、trial tier、paid tier。Token-bucket または sliding-window？

## キーターム

| ターム | 人が言うこと | 実際の意味 |
|--------|----------------|---------------------|
| Gateway | "API ブローカー" | アプリとプロバイダーの間に座るプロセス |
| LiteLLM | "MIT one" | Python OSS、100+ プロバイダー、2K RPS で崩壊 |
| Portkey | "ガードレール ゲートウェイ" | コントロール プレーン + 可観測性、Apache 2.0 |
| Kong AI Gateway | "スケール one" | Kong Gateway ビルド、ベンチマーク リーダー |
| Bifrost | "Maxim's gateway" | 再試行 + Anthropic フォールバック レシピ |
| Cloudflare AI Gateway | "edge マネージド" | Edge-deployed マネージド ゲートウェイ、ゼロオペレーション |
| PII redaction | "データ scrub" | Regex + NER mask、モデルに送信前 |
| Jailbreak detection | "プロンプト インジェクション ガード" | ユーザー入力上の分類器 |
| Audit trail | "規制ログ" | 不変な記録、すべての LLM 呼び出しの |
| Token-bucket | "シンプル レート制限" | Refill-based レート リミッター |
| Sliding-window | "正確 レート制限" | 時間枠 レート リミッター。より良い公正 |

## 参考文献

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
