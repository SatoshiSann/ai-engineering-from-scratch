# LangGraph — エージェント用ステートマシン

> 手書きのReActループは`while True:`。LangGraphで書かれたReActループはグラフで、チェックポイント、割り込み、ブランチ、時間旅行ができます。エージェントは変わっていません。周辺のハーネスが変わりました。

**タイプ:** ビルド
**言語:** Python
**前提条件:** Phase 11 · 09 (Function Calling)、Phase 11 · 14 (Model Context Protocol)
**所要時間:** 約75分

## 問題

関数呼び出しエージェントを出荷。3ターン機能、その後何か間違い：モデルが500を返すツール試み、ユーザーは中間タスク変え、またはエージェントはユーザー署名なしで返金を決定。`while True:`ループはフックありません。一時停止できず、巻き戻せず、「モデルが他のツール選んでいたら」分岐できません。デモを過ぎて出荷する瞬間、エージェントはブラックボックスになります。機能したか機能しなかったか。

次のステップは見えるとき明白。エージェントは既にステートマシン — システムプロンプト + メッセージ履歴 + 保留中ツール呼び出し + 次アクション。ステートマシンを明示：「モデルが考える」、「ツール実行」、「人間が承認」、それら間の条件遷移のエッジのノード。グラフ明示、ハーネスは4ノード無料を取得：チェックポイント(ステップ間の保存状態)、割り込み(人間のために一時停止)、ストリーミング(トークンと中間イベント)、時間旅行(前のステートに巻き戻して異なるブランチ試み)。

LangGraphはこの抽象化を出荷するライブラリ。LangChain感覚のエージェント フレームワークではない(「AgentExecutor、頑張れ」)。一級ステート、一級持続、一級割り込みを持つグラフランタイム。エージェントループは何かを手書きではなく絵です。

## コンセプト

`StateGraph`は3つのもの持つ。

1. **State.** グラフをフロー型辞書(TypeDictまたはPydanticモデル)。ノードリセット状態受け取る、部分アップデートを返す、LangGraph各フィールドでリデューサーを使って統合 — `operator.add`リストは蓄積、デフォルト上書きすべき。
2. **Nodes.** Python関数`state -> partial_state`。各離散ステップ：「モデル呼ぶ」、「ツール実行」、「要約」。
3. **Edges.** ノード間遷移。静的エッジは1場所行く。条件エッジ`state -> next_node_name`ルーター関数を実行して、グラフがモデル出力分岐可能。

グラフをコンパイル。コンパイル集約トポロジー、チェックポイントを挟む(テストに任意だがオプション本番)、実行不可能を返す。初期状態と`thread_id`を持つ呼び出し。実行のすべてのステップは`(thread_id, checkpoint_id)`でキー付けされるストアにチェックポイント永続化。

### 4つの超能力

**チェックポイント。** 各ノード遷移はストア(インメモリテスト、本番用Postgres/Redis/SQLite)に新しいステートを書く。`thread_id`を持つグラフ再度呼ぶことで再開。グラフは一時停止したところから拾う。

**割り込み。** ノードに`interrupt_before=["human_review"]`をマーク、実行はそのノード前停止。ステート永続化。API は「承認待ち」でユーザーにレスポンス。`Command(resume=...)`を持つ同じ`thread_id`への後のリクエスト実行再開。

**ストリーミング。** `graph.stream(state, mode="updates")`はステートデルタが起こるとき取得。`mode="messages"`はモデルノード内LLMトークンをストリーム。`mode="values"`は完全スナップショットをストリーム。UI で何を表示するか選ぶ。

**時間旅行。** `graph.get_state_history(thread_id)`は完全チェックポイントログを返す。前のいずれかの`checkpoint_id`を`graph.invoke`に渡し、そのポイントからフォーク。デバッグに(「モデルが他のツール選んでいたら？」)、本番トレースを再生するレグレッションテスト用に。

### リデューサーはポイント

ステートフィールドすべてがリデューサー。ほとんどデフォルトは良好 — 新値が古いを上書き。しかしメッセージリストは`operator.add`が必要で新しいメッセージは追加ではなく上書きでなく。並列エッジはリデューサーを通じアップデートをマージ。2つのノードが両方`messages`更新し、`Annotated[list, add_messages]`をリデューサーを忘れた場合、2番目が沈黙で勝利、ターン半分を失う。リデューサーは唯一微妙なもの；それを右に取得、残り構成。

### 4ノードのReActグラフ

本番ReActエージェントは4ノード、2エッジ：

1. `agent` — 現在のメッセージ履歴でLLM呼ぶ。助言メッセージを返す(tool_callsを含み得る)。
2. `tools` — 最後の助言メッセージのいずれかのtool_callsを実行、ツール結果をツールメッセージとして追加。
3. 条件エッジ`agent`から：最後メッセージがtool_callsなら`tools`にルート、さもなくば`END`。
4. 静的エッジ`tools`から`agent`へ戻る。

それはそれです。完全なReActループ(Thought → Action → Observation → Thought → …)、チェックポイント、割り込み、ストリーミング、大まかに40行コード。

## ビルド

### ステップ1：ステートとノード

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`はメッセージリストが上書きではなく蓄積するようにするリデューサー。これを忘れるのは最も一般的なLangGraph バグ。

### ステップ2：スレッドで実行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

各アップデートは辞書`{node_name: state_delta}`。フロントエンドがこれらをUIにストリーム、ユーザーは「エージェント考える… search_webを呼ぶ… 結果得た… 答える」を見ます。

### ステップ3：人間ループ割り込みを追加

実行前にノードが一時停止するようマーク。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

ステート、チェックポイント、スレッドはすべて割り込み全体で持続。メモリのみ実行中。

### ステップ4：デバッグ用の時間旅行

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

`None`をインプットとして渡すと与えられたチェックポイントから再生；値を渡すとそのチェックポイントのステートに更新が追加され、前に再開される前。生エージェント実行全体を再実行なしに悪い実行を再生する方法。

### ステップ5：本番チェックポインターをスワップ

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis、Postgresが出荷。`MemorySaver`はテスト用。リスタート全体で持続する何かがリアルストアを望み。

## スキル

> グラフとしてエージェント構築、`while True:`ループでなく。

LangGraphに到達前、60秒デザイン：

1. **ノードを名づけ。** すべての離散判断または副作用アクションはノード。「エージェント考える」、「ツール実行」、「復習者承認」、「レスポンスストリーム」。リストできれなら、タスクはまだエージェント形ではない。
2. **ステート宣言。** すべてのリストフィールドのリデューサー付き最小TypedDict。すべてを`messages`に詰めない；タスク固有フィールドを昇進(働き`plan`、予算カウンター、`retrieved_docs`リスト)トップレベル。
3. **エッジを描く。** 静的次のステップが依存でなければ。すべての条件エッジはルーター関数ある命名ブランチ。
4. **チェックポインターを選ぶ前から出す。** テスト用`MemorySaver`、他すべてPostgres/Redis/SQLite。なしで出荷しない — チェックポイントなし = 再開なし、割り込みなし、時間旅行なし。
5. **ツール前割り込みを決め。** 承認はサイド効果ノード前のエッジで起こることで悪い前キャンセルできて；検証はモデルから出るエッジでで悪い呼び出しを安く拒否できるよう。
6. **デフォルトでストリーム。** UI用`mode="updates"`、モデルノード内トークンレベルストリーミング用`mode="messages"`、評価中フルスナップショット用`mode="values"`。

チェックポインターなしLangGraphエージェント出荷拒否。副作用後割り込み出荷拒否。`messages`フィールド無し`add_messages`リデューサー出荷拒否。

## 演習

1. **簡単。** 上の4ノードReActグラフを計算機ツール、ウェブ検索ツール実装。`list(app.get_state_history(config))`が2ターン会話のための少なくとも4チェックポイント返す確認。
2. **中程度。** 実行前`planner`ノード追加、構造`plan: list[str]`をステートに書く。`agent`が完了としてプラン段階をマーク。テスト失敗チェックポイント再開全体`plan`が失われた場合(リデューサー誤り)。
3. **難しい。** 3つのサブグラフ(`researcher`、`writer`、`reviewer`)のいずれかにルートする監督グラフを構築、`Send`で。各サブグラフは独自ステート、チェックポインター。外グラフ上`interrupt_before=["writer"]`追加、人間がリサーチ概要を承認できて。前のチェックポイントからの時間旅行は分岐ブランチのみ再実行確認。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraphグラフ" | ノードと コンパイル前に追加するエッジビルダーオブジェクト。 |
| Reducer | "フィールドマージ方法" | 関数`(old, new) -> merged`ノードがそのフィールド更新返すのに適用；デフォルト上書き、`add_messages`追加。 |
| Thread | "会話ID" | `thread_id`文字列がスコープをまたはセッションのため。 |
| Checkpoint | "一時停止ステート" | ノード遷移後ノード遷移した完全グラフステートのスナップ、`(thread_id, checkpoint_id)`でキー。 |
| Interrupt | "人間のために一時停止" | `interrupt_before` / `interrupt_after`実行をノード境界に止める；`Command(resume=...)`で再開。 |
| Time-travel | "前のステップからフォーク" | `graph.invoke(None, config_with_old_checkpoint_id)`そのチェックポイント前に再生。 |
| Send | "並列サブグラフディスパッチ" | ノードが返す初期化者、ターゲットノードのN並列実行をスポーン。 |
| Subgraph | "ノードとしてコンパイルグラフ" | 別グラフで使用コンパイルStateGraph；保持スコープ。 |

## 参考文献

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、リデューサー、チェックポインター、割り込み。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — このレッスンが使う心的モデル、直ソースから。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis店、チェックポイント名前空間、スレッドIDの詳細。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)`、編集ステートパターン。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"(ICLR 2023)](https://arxiv.org/abs/2210.03629) — すべてのLangGraphエージェント実装パターン；推論トレース根拠について読む。
- [Anthropic — Building effective agents(Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — グラフ形(チェーン、ルーター、オーケストレーター労働、評価器最適化)を優先及び時間。
- Phase 11 · 09 (Function Calling) — すべてのLangGraphエージェント ノードが再利用するツール呼び出しプリミティブ。
- Phase 11 · 14 (Model Context Protocol) — MCPアダプター経由LangGraphツールノードにプラグインするツール検出。
- Phase 11 · 17 (Agent framework tradeoffs) — LangGraphとCrewAI、AutoGen、Agnoの時間に選ぶ。
