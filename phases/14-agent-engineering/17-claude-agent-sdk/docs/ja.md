# Claude Agent SDK：サブエージェントとセッションストア

> Claude Agent SDKはClaude Codeハーネスのライブラリ形。組み込みツール、コンテキスト分離向けのサブエージェント、フック、W3Cトレース伝播、セッションストアパリティ。Claude Managed Agentsは長走型非同期作業の有人代替。

**タイプ:** 学習 + ビルド
**言語:** Python（stdlib）
**前提条件:** Phase 14 · 01（エージェントループ）、Phase 14 · 10（スキルライブラリ）
**所要時間:** 約75分

## 学習目標

- Anthropic Client SDK（生API）とClaude Agent SDK（ハーネス形）の違いを説明できる。
- サブエージェントを説明できる—並列化とコンテキスト分離—そしていつ到達するかを説明。
- Python SDKのセッションストア表面を名前付けできる（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）と`--session-mirror`の役。
- 組み込みツール、分離コンテキスト付きサブエージェント生成、ライフサイクルフック、セッションストアを持つstdlibハーネスを実装。

## 問題

生LLM APIはワンラウンドトリップを得る。本番エージェントはツール実行、MCPサーバー、ライフサイクルフック、サブエージェント生成、セッション永続化、トレース伝播が必要。Claude Agent SDKはこの形をライブラリとして出荷—Claude Codeが使用する同じハーネス、カスタムエージェントのために公開。

## コンセプト

### Client SDK 対 Agent SDK

- **Client SDK（`anthropic`）。** 生Messages API。ループ、ツール、状態を所有。
- **Agent SDK（`claude-agent-sdk`）。** 組み込みツール実行、MCP接続、フック、サブエージェント生成、セッションストア。Claude Codeループをライブラリとして。

### 組み込みツール

SDKは既製で10以上ツール出荷：ファイル読み取り/書き込み、シェル、grep、glob、ウェブ取得、その他。カスタムツールは標準ツールスキーマインターフェースを通じて登録。

### サブエージェント

Anthropicが文書化する2つの目的：

1. **並列化。** 独立した作業を同時実行。「これら20モジュールのそれぞれのテストファイルを見つける」は20の並列サブエージェントタスク。
2. **コンテキスト分離。** サブエージェントはそれら独自のコンテキストウィンドウ使用；結果のみオーケストレータに復帰。オーケストレータの予算は保全。

Python SDK最近追加：`list_subagents()`、`get_subagent_messages()`、サブエージェントトランスクリプト読み込み。

### セッションストア

TypeScriptのプロトコルパリティ:

- `append(session_id, message)` — ターンを追加。
- `load(session_id)` — 会話を復帰。
- `list_sessions()` — 列挙。
- `delete(session_id)` — サブエージェントセッション対策も削除。
- `list_subkeys(session_id)` — サブエージェントキーをリスト。

`--session-mirror`（CLIフラグ）はストリーム時にトランスクリプトを外部ファイルにミラー、デバッグのため。

### フック

登録できるライフサイクルフック:

- `PreToolUse`、`PostToolUse` — ツール呼び出しをゲート、監査。
- `SessionStart`、`SessionEnd` — セットアップと破棄。
- `UserPromptSubmit` — モデルが見る前にユーザー入力に作用。
- `PreCompact` — コンテキスト圧縮前に実行。
- `Stop` — エージェント出口のクリーンアップ。
- `Notification` — サイドチャネルアラート。

フックはpro-workflow（Phase 14カリキュラム参照）および類似システムがクロスカッティング振る舞いを追加する方法。

### W3C トレースコンテキスト

呼び出し元上のアクティブなOTelスパンはW3Cトレースコンテキストヘッダを通じてCLIサブプロセスに伝播。マルチプロセス全体トレースはバックエンド内の1つのトレースとして表示。

### Claude Managed Agents

有人代替（ベータヘッダ`managed-agents-2026-04-01`）。長走型非同期作業、組み込みプロンプトキャッシング、組み込み圧縮。制御を管理インフラのためにトレード。

### このパターンがうまくいかない場合

- **サブエージェント過度生成。** 100個のちっぽけタスク向けに100サブエージェントをスポーン。オーバーヘッドが支配。代わりにバッチ。
- **フック蔓延。** すべてのチームがフックを追加；スタートアップ時間が膨らむ。フック四半期ごと見直し。
- **セッションブロート。** セッション累積；サイズが成長。`list_sessions` + 有効期限ポリシーを使用。

## ビルドする

`code/main.py`はstdlibでSDK形状を実装：

- `Tool`、`ToolRegistry`が組み込み`read_file`、`write_file`、`list_dir`。
- `Subagent` — プライベートコンテキスト、分離実行、復帰結果。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- デモ：メインエージェントが3つのサブエージェントを並列生成（各分離）、結果を集約、セッションを永続化。

実行する:

```
python3 code/main.py
```

トレースはサブエージェントコンテキスト分離（オーケストレータコンテキストサイズはバウンド保持）、フック実行、セッション永続化を示す。

## 使う

- **Claude Agent SDK**Claude1次製品のため、Claude Codeハーネス形を望むもの。
- **Claude Managed Agents**有人長走型非同期作業のため。
- **OpenAI Agents SDK**（レッスン16）OpenAI1次相対物のため。
- **LangGraph + カスタムツール**グラフ形状状態機械を代わりに望むなら。

## 出荷

`outputs/skill-claude-agent-scaffold.md`はClaude Agent SDKアプリをスキャフォールド、サブエージェント、フック、セッションストア、MCP サーバー添付、W3Cトレース伝播。

## 演習

1. 20タスクを5つの並列サブエージェントのグループにバッチするサブエージェント生成器を追加。オーケストレータコンテキストサイズを1タスクごと測定対比。
2. `PreToolUse`フック実装、`write_file`呼び出しをレート制限（セッションごと分ごと5回）。振る舞いをトレース。
3. `list_subkeys`をサブエージェント木をレンダーにワイヤー。深いネストはどう見えるか？
4. おもちゃをリアル`claude-agent-sdk` Pythonパッケージにポート。ツール登録についてなにが変わるか？
5. Claude Managed Agentsドキュメント読む。自己有人から管理に切り替えるとき？

## キー用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| エージェント SDK | 「ライブラリとしてのClaude Code」 | ハーネス形：ツール、MCP、フック、サブエージェント、セッションストア |
| サブエージェント | 「子エージェント」 | 別のコンテキスト、独自予算；結果はバブルアップ |
| セッションストア | 「会話DB」 | ターンを永続化、ロード、リスト、削除、サブエージェント対策で |
| フック | 「ライフサイクルコールバック」 | ツール前後、セッション、プロンプト投稿、圧縮、停止 |
| W3C トレースコンテキスト | 「クロスプロセストレース」 | 親スパンがCLIサブプロセスに伝播 |
| Managed Agents | 「有人ハーネス」 | Anthropic有人長走型非同期作業 |
| `--session-mirror` | 「トランスクリプトミラー」 | ターンをストリーム時に外部ファイルに書き込み |
| MCPサーバー | 「ツール表面」 | エージェントに添付された外部ツール/リソースソース |

## 参考文献

- [Claude Agent SDK の概要](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Codeのライブラリ形
- [Anthropic、Claude Agent SDKでエージェントを構築](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 本番パターン
- [Claude Managed Agents の概要](https://platform.claude.com/docs/en/managed-agents/overview) — 有人代替
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)：相対物
