# 構造化出力と制約付きデコーディング

> LLMにJSONを求めると、ほとんどの場合JSONが返ってくる。本番環境では「ほとんど」が問題だ。制約付きデコーディングは、サンプリング前にロジットを編集することで「ほとんど」を「常に」に変える。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 17 (チャットボット)、Phase 5 · 19 (部分語トークン化)
**所要時間:** 約60分

## 問題

分類器がLLMにプロンプトを与える: 「{positive, negative, neutral}の1つを返してください。」モデルが返すのは「センチメントはpositiveです — このレビューは圧倒的に好意的です。顧客が明らかに述べているため...」パーサーはクラッシュする。分類器のF1スコアは0.0だ。

自由形式の生成は契約ではない。提案だ。本番システムには契約が必要だ。

2026年には3つのレイヤーが存在する。

1. **プロンプティング。** 丁寧に尋ねる。「JSONオブジェクトだけを返してください。」最先端モデルで約80%機能し、より小さいモデルではそれ以下だ。
2. **ネイティブ構造化出力API。** OpenAIの`response_format`、Anthropicツール使用、Gemini JSONモード。サポートされているスキーマで信頼性がある。ベンダーロックイン。
3. **制約付きデコーディング。** 各生成ステップでロジットを変更するため、モデルは無効なトークンを発行*できない*。構造上100%有効だ。任意のローカルモデルで機能する。

このレッスンは3つすべてに対する直感を構築し、どの場合にどれに到達するかを名付ける。

## コンセプト

![制約付きデコーディングが各ステップで無効なトークンをマスク](../assets/constrained-decoding.svg)

**制約付きデコーディングのしくみ。** 各生成ステップで、LLMはフルボキャブラリー(〜100kトークン)にわたるロジットベクトルを生成する。*ロジットプロセッサー*がモデルとサンプラーの間に位置する。現在の位置でターゲットグラマー内で有効なトークンを計算する — JSONスキーマ、正規表現、文脈自由文法 — および無効なトークンのロジットを負の無限大に設定する。残りのロジットの上でのソフトマックスは、有効な継続だけに確率質量を配置する。

2026年の実装:

- **Outlines。** JSONスキーマまたは正規表現を有限状態機械にコンパイルする。すべてのトークンはO(1)の有効な次のトークンのルックアップを得る。FSMベースなため、再帰的スキーマはフラット化が必要だ。
- **XGrammar / llguidance。** 文脈自由文法エンジン。再帰的JSONスキーマを処理する。デコーディングオーバーヘッドはほぼゼロ。OpenAIは彼らの2025年構造化出力実装でllguidanceにクレジットを与えた。
- **vLLM誘導デコーディング。** Outlines、XGrammar、またはlm-format-enforcerバックエンド経由の組み込み`guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor。** 任意のLLMの上のPydanticベースのラッパー。検証失敗時に再試行する。クロスプロバイダーだが、ロジットを変更しない — 再試行と構造化出力対応プロンプトに依存する。

### 直感に反する結果

制約付きデコーディングは、制約なし生成*より*高速であることが多い。2つの理由がある。まず、次トークンの検索空間を縮小する。次に、巧妙な実装は強制トークンの生成を完全にスキップする(足場のような`{"name": "` — すべてのバイトが決定されている)。

### 代償を払うピットフォール

フィールド順序は重要だ。`answer`を`reasoning`の前に配置すると、モデルは思考する前に答えにコミットする。JSONは有効だ。答えは間違っている。検証で何も捕捉されない。

```json
// 悪い例
{"answer": "yes", "reasoning": "because ..."}

// 良い例
{"reasoning": "... therefore ...", "answer": "yes"}
```

スキーマフィールド順序はロジック、フォーマットではない。

## ビルドしてみよう

### ステップ1: ゼロからの正規表現制約付き生成

`code/main.py`のスタンドアロンFSM実装を参照。30行のコアアイデア:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSMは現在までにグラマーのどの部分を満たしたかを追跡する。`valid_tokens(state, tokenizer)`は、FSMを受理パスから離さずに進める事ができるボキャブラリートークンを計算する。

### ステップ2: JSONスキーマのOutlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

検証エラーはゼロだ。決して起きない。FSMは無効な出力を到達不可能にする。

### ステップ3: プロバイダー不知論のPydanticのためのInstructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

異なるメカニズム。Instructorはロジットに触れない。スキーマをプロンプトにフォーマットし、出力を解析し、検証失敗時に再試行する(デフォルト3回)。任意のプロバイダーで機能する。再試行は遅延とコストを追加する。クロスプロバイダーのポータビリティが売上だ。

### ステップ4: ネイティブベンダーAPI

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

サーバー側の制約付きデコーディング。サポートされているスキーマでOutlinesとの信頼性パリティ。ローカルモデル管理はない。ベンダーにロックインされる。

## ピットフォール

- **再帰的スキーマ。** Outlinesは再帰を固定深度にフラット化する。ツリー構造の出力(ネストされたコメント、AST)はXGrammarまたはllguidance(CFGベース)が必要だ。
- **巨大なenum。** 10,000オプションのenumはゆっくりコンパイルされるか、タイムアウトする。検索器に切り替える: 最初にトップk候補を予測し、それらに制約する。
- **グラマーが厳しすぎる。** `date: "YYYY-MM-DD"`正規表現を強制し、モデルは欠落した日付に対して`"unknown"`を出力できない。モデルは日付を作ることで補正される。`null`またはセンチネルを許可する。
- **早急なコミット。** 上記のフィールド順序のピットフォールを参照。常に推論を最初に配置する。
- **スキーマなしのベンダーJSONモード。** 純粋なJSONモードは有効なJSONだけを保証し、ユースケースに対して有効ではない*。常にフルスキーマを提供する。

## 使用方法

2026年スタック:

| 状況 | 選択 |
|-----------|------|
| OpenAI/Anthropic/Googleモデル、シンプルなスキーマ | ネイティブベンダー構造化出力 |
| 任意のプロバイダー、Pydanticワークフロー、再試行を容認できる | Instructor |
| ローカルモデル、100%の有効性が必要、フラットなスキーマ | Outlines (FSM) |
| ローカルモデル、再帰的スキーマ | XGrammarまたはllguidance |
| 自社ホストの推論サーバー | vLLM誘導デコーディング |
| 再試行可能なバッチ処理 | Instructor +最安モデル |

## 出荷しよう

`outputs/skill-structured-output-picker.md`として保存:

```markdown
---
name: structured-output-picker
description: 構造化出力アプローチ、スキーマ設計、検証計画を選択する。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

ユースケース(プロバイダー、レイテンシーバジェット、スキーマ複雑性、失敗容認度)を指定して、以下を出力:

1. メカニズム。ネイティブベンダー構造化出力、Instructor再試行、Outlines FSM、またはXGrammar CFG。1文の理由。
2. スキーマ設計。フィールド順序(推論が最初、答えが最後)、「不明」のためのnull許容フィールド、enumと正規表現、必須フィールド。
3. 失敗戦略。最大再試行回数、フォールバックモデル、`null`の優雅な処理、外れ値拒否。
4. 検証計画。スキーマ準拠率(目標100%)、セマンティック有効性(LLM判定)、フィールドカバレッジ率、レイテンシーp50/p99。

答えやdecisionをreasoningフィールドの前に配置する設計は拒否しなさい。スキーマなしの裸のJSONモードの使用を拒否しなさい。FSMのみのライブラリの後ろにある再帰的スキーマにフラグを立てなさい。
```

## 演習

1. **簡単。** 制約なしで小さなオープンウェイトモデル(例えば、Llama-3.2-3B)にプロンプトを与える`Review(sentiment, confidence, evidence_span)`。100レビューで有効なJSONとして解析される割合を測定する。
2. **中程度。** Outlinesの JSON モードを使用した同じコーパス。準拠率、レイテンシー、セマンティック精度を比較する。
3. **難しい。** 電話番号の正規表現制約デコーダをゼロから実装する(`\d{3}-\d{3}-\d{4}`)。1000サンプルで無効な出力がゼロであることを検証する。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 制約付きデコーディング | 有効な出力を強制する | 各生成ステップで無効トークンロジットをマスクする。 |
| ロジットプロセッサー | 制約するもの | 関数: `(logits, state) -> masked_logits`。 |
| FSM | 有限状態機械 | コンパイルされたグラマー表現; O(1)有効な次のトークンのルックアップ。 |
| CFG | 文脈自由文法 | 再帰を処理するグラマー; FSMより遅いが表現力がある。 |
| スキーマフィールド順序 | 重要? | はい — 最初のフィールドがコミット; 常に答えの前に推論を配置する。 |
| 誘導デコーディング | vLLMの名前 | 同じコンセプト、推論サーバーに統合。 |
| JSONモード | OpenAIの初期バージョン | JSON構文を保証; スキーママッチを保証しない。 |

## 参考文献

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlinesペーパー。
- [XGrammarペーパー (2024)](https://arxiv.org/abs/2411.15100) — 高速CFGベース制約付きデコーディング。
- [vLLM — 構造化出力](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推論サーバー統合。
- [OpenAI — 構造化出力ガイド](https://platform.openai.com/docs/guides/structured-outputs) — APIリファレンス + 落とし穴。
- [Instructorライブラリ](https://python.useinstructor.com/) — プロバイダー全体でのPydantic +再試行。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6つの制約付きデコーディングフレームワークのベンチマーク。
