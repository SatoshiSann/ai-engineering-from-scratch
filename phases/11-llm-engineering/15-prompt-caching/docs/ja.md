# プロンプトキャッシングとコンテキストキャッシング

> システムプロンプトは4,000トークン。RAGコンテキストは20,000トークン。各リクエストで両方を送信。そして両方に対して支払い — 毎回。プロンプトキャッシングはプロバイダが再利用でそのプレフィックスを暖かく保ち、通常レートの10%を請求することを許可。正しく使用されると、推論コストの50-90%とトークンのファースト遅延の40-85%を削減。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 11 · 01 (Prompt Engineering)、Phase 11 · 05 (Context Engineering)、Phase 11 · 11 (Caching and Cost)
**所要時間:** 約60分

## 問題

コーディングエージェントが会話のすべてのターンで同じ15,000トークンシステムプロンプトをClaudeに送信。20ターン(入力トークン$3/Mで$0.90の入力コストのみ — ユーザーの実際のメッセージの前。1日10,000会話でかけ、請求は$9,000/日 — 決して変わらないテキストのため。

プロンプトを縮小できない — 品質を傷つけます。送信を避けられない — モデルは毎回必要です。唯一の動きはプレフィックスのプロバイダが既に見た可能性で、完全な価格を停止すること。

そのムーブはプロンプトキャッシング。Anthropicはそれを2024年8月に出荷(2025年1時間拡張TTLバリアント付き)、OpenAIが後でそれを自動化、Googleがgemini 1.5と共に明示的コンテキストキャッシングを出荷、3つすべてがフロンティアモデルで一級フィーチャーとして提供。

## コンセプト

**メカニック。** リクエストのプレフィックスが最近のリクエストのものと合致するとき、プロバイダは以前の実行からKV-キャッシュを提供する替わりにトークンを再エンコード。最初に書き込みで小さいプレミアム、各回後で大きい読み取りディスカウントを支払う。

**2026年の3プロバイダフレーバー。**

| プロバイダ | APIスタイル | ヒットディスカウント | 書き込みプレミアム | デフォルトTTL | 最小キャッシュ可能 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | コンテンツブロック上の明示的`cache_control`マーカー | 入力で90%オフ | 25%サーチャージ | 5分(1時間拡張可能) | 1,024トークン(Sonnet/Opus)、2,048(Haiku) |
| OpenAI | 自動プレフィックス検出 | 入力で50%オフ | なし | 最大1時間(ベストエフォート) | 1,024トークン |
| Google(Gemini) | 明示的`CachedContent` API | ストレージ請求；読み取りで~25% | ストレージ手数料トークン·時間あたり | ユーザー設定(デフォルト1時間) | 4,096(Flash) / 32,768(Pro) |

**不変。** 3つはすべてプレフィックスのみをキャッシュ。リクエスト間でトークンが異なるなら、最初の異なるトークン後のすべてはミス。*安定*部をトップに、*変数*部をボトムに。

### キャッシュ友好レイアウト

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

順序を違う — システムプロンプト上ユーザーメッセージを置く、少数ショット間で動的検索をインターリーブ — キャッシュは決してヒット。

### ブレークイーブン計算

Anthropicの25%書き込みプレミアムはキャッシュブロックが利益をまとめるのに最少2倍を読む必要を意味。1書き込み + 1読み取りはリクエストあたり平均0.675xコスト(32%節約)；1書き込み + 10読み取りは平均0.205x(80%節約)。ルールオブサムb：TTL内少なくとも3回を再利用するすべてをキャッシュ。

## ビルド

### ステップ1：Anthropicプロンプトキャッシング明示マーカー付き

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control`マーカーはAnthropicに5分の間ブロックを保存するよう指示。そのウィンドウ内再利用がヒット；再利用後有効期限が切れて再び書き込み。

**レスポンス使用フィールド：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

両方フィールドをCIで確認 — `cache_read_input_tokens`がリクエスト全体のゼロに留まるなら、キャッシュキーはドリフトしている。

### ステップ2：1時間拡張TTL

長実行バッチジョブの場合、5分デフォルトはジョブ間に有効期限切れ。`ttl`を設定：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1時間TTLはベースラインの上25%代わりに50%の書き込みプレミアムを費用 しかし5回以上プレフィックスを再利用するすべてのバッチで素早く返却。

### ステップ3：OpenAI自動キャッシング

OpenAIは設定のものを与え。1,024トークン以上の最近のリクエストと合致する任意プレフィックスは50%割引を自動的に取得。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同じキャッシュ友好レイアウトルールが適用。2つはOpenAIのキャッシュをAnthropicのキャッシュではなく殺す：`user`フィールドを変える(キャッシュキー成分として使用)とツールをリオーダー。

### ステップ4：Gemini明示コンテキストキャッシング

Geminiはキャッシュを一級オブジェクトとして扱います：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Geminiはキャッシュが住んでいる限りトークン·時間あたりストレージを費用、読み取りを通常入力レートの~25%で。これは複数セッション上数日の同じ大きいプロンプト再利用するとき、正しい形です。

### ステップ5：本番環境でのヒット率を測定

`code/main.py`の3プロバイダ会計事業をシミュレートしたものは、書き込み/読み取り/ミスカウント追跡、1000リクエスト当たりのブレンドコストを計算。ターゲットヒット率にデプロイをゲート — ほとんどの本番Anthropicセットアップは暖機後の読み取り分数で>80%を見るべきです。

## 2026年にまだ出荷される落とし穴

- **トップの動的タイムスタンプ。** `"Current time: 2026-04-22 15:30:02"`をシステムプロンプトのトップ。すべてのリクエストはミス。キャッシュブレークポイント下のタイムスタンプを移動。
- **ツールリオーダー。** ツールを安定な順で連列化 — デプロイ間の辞書リシャフルはすべてのヒットを破ります。
- **フリーテキスト準複製。** 「You are helpful.」対「You are a helpful assistant.」 — 1バイト差 = 完全ミス。
- **小さすぎるブロック。** Anthropicは1,024トークンのフロアを強制(Haiku 2,048)。より小さいブロック。
- **ブラインドコストダッシュボード。** キャッシュ対非キャッシュに「入力トークン」を分割。さもなくば、トラフィック低下にはキャッシュ勝利のように見えます。

## 使用方法

2026年キャッシングスタック：

| 状況 | 選択 |
|-----------|------|
| 安定10k+ システムプロンプト、多数ターン付きエージェント | Anthropic `cache_control`5分TTL |
| 30+分プレフィックスを再利用するバッチジョブ | Anthropic 1時間TTL付き |
| GPT-5のサーバーレスエンドポイント、カスタムインフラなし | OpenAI自動(プレフィックスをシンプルに安定、長くする) |
| 複数日複製大きいコード/ドキュメントコーパス | Gemini明示`CachedContent` |
| クロスプロバイダフォールバック | プロバイダ全体で同じキャッシュプレフィックスレイアウトを保つ、ヒットはいずれでも機能 |

セマンティックキャッシング(Phase 11 · 11)とコンボ：プロンプトキャッシングは*トークン同一*再利用、セマンティックキャッシングは*意味同一*再利用を処理。

## 出荷

`outputs/skill-prompt-caching-planner.md`を保存：

```markdown
---
name: prompt-caching-planner
description: キャッシュ友好プロンプトレイアウト設計、正しいプロバイダキャッシングモード選択。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

プロンプト(system + tools + few-shot + retrieval + history + user)と使用プロファイル(時間あたりリクエスト、TTL必要、プロバイダ)を与え、出力：

1. レイアウト。再オーダー済みセクションで単一キャッシュブレークポイント；安定と変動セクション説明。
2. プロバイダモード。Anthropic cache_control、OpenAI自動、またはGemini CachedContent。TTLと再利用パターンから正当化。
3. ブレークイーブン。TTL内読み取りあたりの予想書き込み；キャッシュなし対ネット コストと数学。
4. 検証計画。2番目の同じリクエスト上cache_read_input_tokens > 0のCI断言；キャッシュ対非キャッシュトークン分割ダッシュボード。
5. 失敗モード。このセットアップでキャッシュがミスする3つの最も可能性のある理由(動的タイムスタンプ、ツールリオーダー、準複製テキスト)、各1つ防ぐ方法。

キャッシュ友好ブレークポイント上動的フィールドを置くキャッシュ計画を出荷拒否。2倍書き込みプレミアムを返却しない再利用カウント無し1時間TTLを有効化拒否。
```

## 演習

1. **簡単。** Claude対5,000トークンシステムプロンプトでの10ターン会話を取る。キャッシュなしで実行、次に`cache_control`付き。それぞれの入力トークン請求を報告。
2. **中程度。** プロンプトテンプレートと要求ログを与える、期待ヒット率をコンピュート、ドル節約(Anthropic 5m、Anthropic 1h、OpenAI自動、Gemini明示)プロバイダあたり。
3. **難しい。** レイアウト最適化を構築：プロンプトと`stable=True/False`としてマークされたフィールドリストを与え、単一キャッシュブレークポイントを情報を失わない最大キャッシュ友好位置に置くようプロンプトを書き換え。実際のAnthropicエンドポイント上検証。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| Prompt caching | "長いプロンプトを安くする" | マッチングプレフィックスの再利用プロバイダサイドKV-キャッシュ；反復入力トークンで50-90%割引。 |
| `cache_control` | "Anthropicマーカー" | コンテンツブロック属性「ここまですべてキャッシュ可能」；`{"type": "ephemeral"}`。 |
| Cache write | "プレミアムを支払う" | キャッシュを登録するのに最初のリクエスト；Anthropic ~1.25x入力レート、OpenAI無料で請求。 |
| Cache read | "割引" | 後続リクエスト マッチングプレフィックス；10%(Anthropic)、50%(OpenAI)、~25%(Gemini)で請求。 |
| TTL | "どのくらいそれは住んでいるのか" | キャッシュ暖かい秒；Anthropic 5mデフォルト(拡張可能1h)、OpenAI ベストエフォート最大1h、Geminiユーザー設定。 |
| Extended TTL | "1時間Anthropicキャッシュ" | `{"type": "ephemeral", "ttl": "1h"}`；2倍書き込みプレミアムが バッチ再利用で価値。 |
| Prefix match | "マイキャッシュなぜミス" | キャッシュは開始から ブレークポイントまでのすべてのトークンが バイト同一のときのみヒット。 |
| Context caching(Gemini) | "明示的なもの" | Google命名、ストレージ請求キャッシュオブジェクト；複数日大きいコーパスの再利用用最良。 |

## 参考文献

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、ブレークイーブンテーブル。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自動プレフィックスマッチング。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API とストレージ価格。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 遅延番号を持つ本来起動投稿。
- Phase 11 · 05 (Context Engineering) — キャッシュ着地できるようプロンプトをスライスする場所。
- Phase 11 · 11 (Caching and Cost) — ユーザーメッセージでセマンティックキャッシュを使用プロンプトキャッシング。
- [Pope et al., "Efficiently Scaling Transformer Inference"(2022)](https://arxiv.org/abs/2211.05102) — KV-キャッシュメモリモデルプロンプトキャッシング露出。キャッシュプレフィックスが再計算より読み直し~10倍安い理由を説明。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills"(2023)](https://arxiv.org/abs/2308.16369) — プレフィルはプロンプトキャッシングショートカット。TT FT はなぜキャッシュヒット上劇的に低下するか、TPOTは影響されない理由。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"(2023)](https://arxiv.org/abs/2211.17192) — プロンプトキャッシング側推測デコーディング、Flash Attention、MQA/GQAレバー推論コスト曲線曲げ；完全画像について3つを読む。
