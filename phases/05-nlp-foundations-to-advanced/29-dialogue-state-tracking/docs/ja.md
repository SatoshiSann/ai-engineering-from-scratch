# ダイアログ状態追跡

> 「安い北方のレストランが欲しい...実は中程度に...そしてイタリアンを追加して。」3ターン、3つの状態更新。DSTはスロット値の辞書を同期させておくため、予約が機能する。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 5 · 17 (チャットボット)、Phase 5 · 20 (構造化出力)
**所要時間:** 約75分

## 問題

タスク指向ダイアログシステムでは、ユーザーの目標はスロット値ペアのセットとしてエンコードされる: `{cuisine: italian, area: north, price: moderate}`。すべてのユーザーターンはスロットを追加、変更、または削除できる。システムは会話全体を読み、現在の状態を正しく出力する必要がある。

単一のスロットを誤ると、システムは間違ったレストランを予約し、間違ったフライトをスケジュール、または間違ったカードに請求する。DSTはユーザーが言ったこととバックエンドが実行することの間のヒンジだ。

2026年でもLLMにもかかわらず、それが重要な理由:

- コンプライアンス感度の高いドメイン(銀行、医療、航空会社予約)は決定論的なスロット値を必要とし、自由形式生成ではない。
- ツール使用エージェントはAPIを呼ぶ前に、スロット解決が必要だ。
- マルチターン修正は見かけより難しい: 「実際、木曜日に変更して。」

モダンパイプライン: クラシックDSTコンセプト+LLM抽出器+構造化出力ガードレール。

## コンセプト

![DST: ダイアログ履歴 → スロット値状態](../assets/dst.svg)

**タスク構造。** スキーマはドメイン(レストラン、ホテル、タクシー)とそれらのスロット(料理、エリア、価格、人数)を定義する。各スロットは空、閉集合からの値で埋められる(価格: {cheap, moderate, expensive})、または自由形式値(名前: "The Copper Kettle")。

**2つのDST定式化。**

- **分類。** 各(スロット、候補値)ペアについて、yes/noを予測。閉じたボキャブラリースロット用。2020年前の標準。
- **生成。** ダイアログを与えられたとき、自由テキストとしてスロット値を生成。オープンボキャブラリースロット用。モダンデフォルト。

**メトリック。** 結合目標精度(JGA) — すべてのスロットが*正しい*ターンの分数。全または何もない。MultiWOZ 2.4リーダーボード2026年で約83%のトップ。

**アーキテクチャ。**

1. **ルールベース(スロット正規表現+キーワード)。** 狭いドメイン用の強いベースライン。デバッグ可能。
2. **TripPy / BERT-DST。** BERTエンコーディング付きコピーベース生成。LLM前の標準。
3. **LDST (LLaMA +LoRA)。** ドメインスロットプロンプティング付き命令チューニングLLM。MultiWOZ 2.4でChatGPTレベルの品質に到達。
4. **Ontology無料(2024–26)。** スキーマをスキップ; スロット名と値を直接生成。オープンドメインを処理。
5. **プロンプト+構造化出力(2024–26)。** Pydanticスキーマ+制約付きデコーディング付きLLM。5行のコード、本番準備完了。

### クラシック失敗モード

- **ターン間での共参照。** 「最初のオプションに留まろう。」どのオプションか解決する必要がある。
- **上書き対附加。** ユーザーは「イタリアンを追加」と言う。料理を置き換えるか、追加か?
- **暗黙の確認。** 「オッケー、クール」 — それが提案された予約を受け入れたか?
- **修正。** 「実際には7時pmに変更。」時間を更新する必要があり、他のスロットをクリアしない。
- **前のシステム発話への共参照。** 「はい、あの1つ。」どの「あの」?

## ビルドしてみよう

### ステップ1: ルールベーススロット抽出器

`code/main.py`を参照。正規表現+シノニム辞書は狭いドメインの正規発話の70%をカバー:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

正規ボキャブラリーの外では脆い。決定論的なスロット確認に機能する。

### ステップ2: 状態更新ループ

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

3つの不変:

- ユーザーが接しなかったスロットをリセットしない。
- 明示的な否定(「料理を忘れて」)はクリアする必要。
- ユーザー修正(「実際...」)は上書き、追加ではない。

### ステップ3: 構造化出力付きLLM駆動DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""ターン全体でレストラン予約のスロット値を追跡します。
これまでのダイアログ:
{render(history)}

最後のユーザーターンに基づいて状態を更新します。JSONの状態だけを出力します。"""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydanticは有効な状態オブジェクトを保証。正規表現なし、スキーマ不一致なし、ハルシネートスロットなし。

### ステップ4: JGA評価

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

キャリブレート: システムがすべてのスロットを正しく取得するターンの何分のいくつか?MultiWOZ 2.4の場合、トップ2026年システム: 80-83%。ドメイン内システムはLLMベースラインの狭いボキャブラリーを超えるか、超えるべき。

### ステップ5: 修正の処理

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

検出された修正では、最後に更新されたスロットを上書きする、附加ではない。LLMヘルプなしで正しく取得するのは難しい。モダンパターン: 常にLLMに履歴から全体の状態を再生成させる — これは自然に修正を処理。

## ピットフォール

- **全履歴再生成コスト。** LLMに各ターン状態を再生成させることはO(n²)トータルトークンのコスト。履歴をキャップするか、古いターンをサマリーする。
- **スキーマドリフト。** 事後に新しいスロットを追加すると、古い訓練データが壊れる。スキーマをバージョン。
- **大文字と小文字の区別。** 「イタリア」対「イタリア」対「イタリア」 — どこでも正規化。
- **暗黙の継承。** ユーザーが以前に「4人のために」を指定したら、新しい異なる時間への要求は人を明確にしてはならない。常に履歴全体を渡す。
- **自由形式対閉集合。** 名前、時刻、住所は自由形式スロット必要; 料理とエリアは閉じている。スキーマで両方をミックス。

## 使用方法

2026年スタック:

| 状況 | アプローチ |
|-----------|----------|
| 狭いドメイン(1つ、2つの意図) | ルールベース+正規表現 |
| 広いドメイン、ラベル付きデータ利用可能 | LDST (MultiWOZスタイルデータでのLLaMA +LoRA) |
| 広いドメイン、ラベルなし、本番準備完了 | LLM + Instructor + Pydanticスキーマ |
| 音声/音声 | ASR +正規化器+LLM-DST |
| マルチドメイン予約フロー | ドメインあたりのPydanticモデルを使用したスキーマガイドLLM |
| コンプライアンス感度が高い | ルールベース主要、LLMフォールバック確認フロー付き |

## 出荷しよう

`outputs/skill-dst-designer.md`として保存:

```markdown
---
name: dst-designer
description: ダイアログ状態トラッカー — スキーマ、抽出器、更新ポリシー、評価を設計する。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

ユースケース(ドメイン、言語、ボキャブラリーオープンネス、コンプライアンス必要)を指定して、以下を出力:

1. スキーマ。ドメインリスト、ドメインあたりのスロット、スロットあたりのオープン対閉じたボキャブラリー。
2. 抽出器。ルールベース/seq2seq/LLM-with-Pydantic。理由。
3. 更新ポリシー。全体状態を再生成/増分; 修正処理; 否定処理。
4. 評価。保持ダイアログセットのJoint Goal Accuracy、スロットレベルの精度/リコール、最難スロットの混乱。
5. 確認フロー。明示的にユーザーに確認するいつ(破壊的アクション、低信頼抽出)。

コンプライアンス感度が高いスロットでのLLMのみDST用ルールベース二次チェックなしを拒否しなさい。ユーザー修正でスロットをロールバックできないDSTを拒否しなさい。バージョンタグなしのスキーマをフラグ立てなさい。
```

## 演習

1. **簡単。** `code/main.py`のルールベース状態トラッカーを3つのスロット(料理、エリア、価格)用に構築。10つの手作りダイアログでテスト。JGAを測定。
2. **中程度。** Instructor + Pydantic +小さいLLMと同じデータセット。JGAを比較。最難ターンを検査。
3. **難しい。** 両方を実装してルート: ルールベース主要、ルールベース<2スロットで信頼性を発するときのLLMフォールバック。結合されたJGAと推論コストをターンあたり測定。

## 重要な用語

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| DST | ダイアログ状態追跡 | ダイアログターン全体でスロット値の辞書を維持。 |
| スロット | ユーザー意図の単位 | バックエンドが必要とする名前が付けられたパラメータ(料理、日付)。 |
| ドメイン | タスク領域 | レストラン、ホテル、タクシー — スロットのセット。 |
| JGA | 結合目標精度 | すべてのスロットが正しいターンの分数。全または何もない。 |
| MultiWOZ | ベンチマーク | マルチドメインWOZデータセット; 標準DST評価。 |
| Ontology無料DST | スキーマなし | スロット名と値を直接生成、固定リストなし。 |
| 修正 | 「実際...」 | 以前に埋められたスロットを上書きするターン。 |

## 参考文献

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 正規ベンチマーク。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA命令チューニング用DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — コピーベースDSTワークホース。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EMベース教師なしTOD。
- [MultiWOZリーダーボード](https://github.com/budzianowski/multiwoz) — 正規DST結果。
