# チャットボット — ルールベースからニューラル、そして LLM エージェントへ

> ELIZA はパターンマッチで返答した。DialogFlow は意図をマッピングした。GPT は重みから答えた。Claude はツールを実行し、検証する。それぞれの時代は、前の時代の最悪の失敗を解決した。

**タイプ:** Learn
**言語:** Python
**前提条件:** Phase 5 · 13 (質問応答)、Phase 5 · 14 (情報検索)
**所要時間:** 約75分

## 問題

ユーザーが「フライトを変更したい」と言う。システムは、ユーザーが何を望んでいるのか、どの情報が欠けているのか、どうやってそれを得るのか、どうやってその行動を完了するのかを把握しなければならない。次にユーザーが「待って、代わりにキャンセルしたらどうなる?」と言うと、システムはコンテキストを覚えておき、タスクを切り替え、状態を保持しなければならない。

会話は ML システムにとって難しい。入力はオープンエンドである。出力は何ターンにもわたって一貫していなければならない。システムは世界に対して行動する必要があるかもしれない (フライトを変更する、カードに請求する)。あらゆる誤ったステップがユーザーに見える。

チャットボットのアーキテクチャは4つのパラダイムを巡ってきた。それぞれは、前のものがあまりに目立つ形で失敗したために導入された。このレッスンではそれらを順に辿る。2026年の本番環境の様相は、後者2つのハイブリッドである。

## コンセプト

![チャットボットの進化: ルールベース → 検索 → ニューラル → エージェント](../assets/chatbot.svg)

**ルールベース (ELIZA、AIML、DialogFlow)。** 手作業で書かれたパターンがユーザー入力にマッチし、応答を生成する。意図分類器が事前定義されたフローへ振り分ける。スロット埋めの状態機械が必要な情報を集める。設計された狭い範囲の内側では見事に機能する。その外側ではただちに失敗する。ハルシネーションが許容されない安全性が重要な領域 (銀行の認証、航空券の予約) では、今なお使われている。

**検索ベース。** FAQ 形式のシステム。(発話, 応答) のペアをすべてエンコードする。実行時に、ユーザーのメッセージをエンコードし、最も近い保存済みの応答を取得する。Zendesk の古典的な「類似記事」機能を思い浮かべるとよい。ルールよりも言い換えをうまく扱える。生成しないため、ハルシネーションがない。

**ニューラル (seq2seq)。** 会話ログで訓練されたエンコーダ・デコーダ。応答をゼロから生成する。流暢だが、一般的な出力 (「分かりません」) や事実のドリフトに陥りやすい。決して確実にトピックに沿わない。これが、Google、Facebook、Microsoft が2016〜2019年にいずれも失望させるチャットボットを抱えていた理由である。

**LLM エージェント。** 計画を立て、ツールを呼び出し、結果を検証するループに包まれた言語モデル。長いプロンプトを持つチャットボットではない。エージェントループ: 計画 → ツール呼び出し → 結果の観測 → 次のステップの決定。検索優先のグラウンディング (RAG) がハルシネーションを防ぐ。ツール呼び出しが実際に物事を行うことを可能にする。これが2026年のアーキテクチャである。

4つのパラダイムは順次的な置き換えではない。2026年の本番チャットボットは4つすべてを通って振り分ける。認証と破壊的な行動にはルールベース、FAQ には検索、自然な言い回しにはニューラル生成、曖昧でオープンエンドなクエリには LLM エージェントを使う。

## 作ってみよう

### ステップ1: ルールベースのパターンマッチング

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20行の ELIZA。リフレクションのトリック (「I feel sad」→「Why do you feel sad」) は、Weizenbaum 1966 の定番の精神療法士デモである。今なお示唆に富む。

### ステップ2: 検索ベース (FAQ)

この説明用のスニペットには `pip install sentence-transformers` (torch を引き込む) が必要である。このレッスンの実行可能な `code/main.py` は、代わりに標準ライブラリの Jaccard 類似度を使うので、外部依存なしにレッスンが実行できる。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

閾値ベースの拒否が鍵となる設計上の選択である。最良のマッチが十分に近くない場合は `None` を返し、システムにエスカレートさせる。

### ステップ3: ニューラル生成 (ベースライン)

小さな命令チューニング済みエンコーダ・デコーダ (FLAN-T5) または会話用にファインチューニングされたモデルを使う。2026年には単体では本番利用に耐えない (矛盾、トピックからの逸脱、事実的な無意味さ) が、自然な言い回しのためにハイブリッドシステムの内部に組み込まれる。DialoGPT スタイルのデコーダのみのモデルは、一貫した返答を生成するために明示的なターン区切りと EOS 処理を必要とする。FLAN-T5 の text2text パイプラインは、教育用の例としてそのまま動作する。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### ステップ4: LLM エージェントループ

2026年の本番の形:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

3つを名指ししよう。ツールは LLM が呼び出せる呼び出し可能な関数である。LLM がツール呼び出しの代わりに最終的な答えを返したとき、ループは終了する。ステップ予算は、曖昧なタスクでの無限ループを防ぐ。

実際の本番では次が加わる。検索優先のグラウンディング (各 LLM 呼び出しの前に関連文書を注入する)、ガードレール (確認なしの破壊的行動を拒否する)、可観測性 (すべてのステップをログに記録する)、評価 (エージェントの挙動が仕様どおりであり続けるかの自動チェック)。

### ステップ5: ハイブリッドルーティング

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

パターンは次のとおり。破壊的なものには決定論的ルール、定型 FAQ には検索、それ以外すべてには LLM エージェント。これが2026年のカスタマーサポートシステムで実際に使われているものである。

## 使ってみよう

2026年のスタック:

| ユースケース | アーキテクチャ |
|---------|---------------|
| 予約、決済、認証 | ルールベースの状態機械 + スロット埋め |
| カスタマーサポートの FAQ | 整備された回答に対する検索 |
| オープンエンドのヘルプチャット | RAG + ツール呼び出しを備えた LLM エージェント |
| 内部ツール / IDE アシスタント | ツール呼び出し (検索、読み取り、書き込み) を備えた LLM エージェント |
| コンパニオン / キャラクターチャットボット | ペルソナのシステムプロンプトを持つチューニング済み LLM、知識に対する検索 |

本番では常にハイブリッドルーティングを使う。単一のアーキテクチャですべてのリクエストをうまく処理できるものはない。ルーティング層それ自体は、典型的には小さな意図分類器である。

## 今なお出荷される失敗モード

- **自信に満ちた捏造。** LLM エージェントが、実際には完了していない行動を完了したと主張する。緩和策: 結果を検証し、ツール呼び出しをログに記録し、ツールの成功した戻り値なしには LLM に何かを行ったと主張させない。
- **プロンプトインジェクション。** ユーザーがシステムプロンプトを上書きするテキストを挿入する。OWASP Top 10 for LLM Applications 2025 で LLM01 にランクされた。2つの種類がある。直接インジェクション (チャットに貼り付ける) と間接インジェクション (エージェントが読む文書、メール、ツール出力に隠す)。

  攻撃の成功率はシナリオによって異なる。測定された成功率は、一般的なツール使用とコーディングのベンチマークにわたって、フロンティアモデルで約0.5〜8.5%の範囲である。特定の高リスクの構成 (AI コーディングエージェントへの適応的攻撃、脆弱なオーケストレーション) では約84%に達している。本番の CVE には EchoLeak (CVE-2025-32711、CVSS 9.3) がある。これは Microsoft 365 Copilot におけるゼロクリックのデータ漏洩の欠陥で、攻撃者が制御するメールによって起動される。

  緩和策: ループ全体を通してユーザー入力を信頼できないものとして扱う。ツール呼び出しの前にサニタイズする。ツール出力をメインプロンプトから分離する。Plan-Verify-Execute (PVE) パターンを使う。エージェントがまず計画を立て、次に実行前にその計画に対して各行動を検証する (これによりツールの結果が新しい計画外の行動を注入するのを止める)。破壊的な行動にはユーザーの確認を要求する。ツールのスコープに最小権限を適用する。

  どれだけプロンプトエンジニアリングを行っても、このリスクを完全に排除することはできない。外部のランタイム防御層 (LLM Guard、許可リストによる検証、意味的異常検知) が必要である。
- **スコープクリープ。** ツール呼び出しが間接的に関連する情報を返したために、エージェントがタスクから逸脱する。緩和策: ツールの契約を狭める。システムプロンプトに焦点を保つ。タスク逸脱率の評価を追加する。
- **無限ループ。** エージェントが同じツールを呼び出し続ける。緩和策: ステップ予算、ツール呼び出しの重複排除、「進捗しているか」についての LLM ジャッジ。
- **コンテキストウィンドウの枯渇。** 長い会話が最も古いターンをコンテキストの外へ押し出す。緩和策: 古いターンを要約する、関連する過去のターンを類似度で取得する、または長コンテキストモデルを使う。

## 仕上げよう

`outputs/skill-chatbot-architect.md` として保存する:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 演習

1. **易しい。** 上記のルールベースの応答を、コーヒーショップの注文ボット用に10個のパターンで実装する。エッジケースをテストする。二重注文、変更、キャンセル、不明瞭な意図。
2. **普通。** ハイブリッドな FAQ + LLM フォールバックを作る。SaaS 製品向けの50件の定型 FAQ エントリと、ドキュメントサイトに対する検索を備えた LLM フォールバック。100件の実際のサポート質問に対する拒否率と精度を測定する。
3. **難しい。** 上記のエージェントループを3つのツール (search、read-user-data、send-email) で実装する。プロンプトインジェクションの試みを含む50件のテストシナリオで評価を実行する。タスク逸脱率、タスク失敗率、そしてインジェクションの成功があれば報告する。

## 重要な用語

| 用語 | 人々の言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| 意図 (Intent) | ユーザーが望むこと | カテゴリラベル (book_flight、reset_password)。ハンドラへ振り分けられる。 |
| スロット (Slot) | 情報の一片 | ボットが必要とするパラメータ (日付、目的地)。スロット埋めは尋ねる一連の流れ。 |
| RAG | 検索 + 生成 | 関連文書を取得し、それで LLM の応答をグラウンディングする。 |
| ツール呼び出し | 関数の呼び出し | LLM が名前 + 引数を持つ構造化された呼び出しを発行する。ランタイムが実行し、結果を返す。 |
| エージェントループ | 計画、行動、検証 | タスクが完了するまで LLM 呼び出しとツール呼び出しを交互に実行するコントローラ。 |
| プロンプトインジェクション | ユーザーがプロンプトを攻撃する | システムプロンプトを上書きしようとする悪意ある入力。 |

## 参考文献

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 元のルールベースチャットボット論文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google の後期ニューラルチャットボット論文。LLM エージェントが主流になる直前のもの。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — エージェントループのパターンに名前を与えた論文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024年の本番向けガイダンス。2026年も通用する。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — プロンプトインジェクションの論文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — プロンプトインジェクションを最重要のセキュリティ懸念事項にしたランキング。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — Plan-Verify-Execute やユーザー確認フローを含む、実践的なオーケストレーション層の防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 間接プロンプトインジェクションによる定番のゼロクリックデータ漏洩 CVE。書き込みアクセスを持つエージェントにランタイム防御が必要な理由の参照ケース。
