# スキルとエージェント SDK — Anthropic スキル、AGENTS.md、OpenAI Apps SDK

> MCPは「どのツールが存在するか」を言います。スキルは「タスクをどのように行うか」を言います。2026年スタックは両方をレイヤー化します。Anthropic のエージェントスキル（オープン標準、2025年12月）は段階的な開示を伴う SKILL.md として配信されます。OpenAI の Apps SDK は MCP プラスウィジェットメタデータ。AGENTS.md（現在60,000以上のリポジトリにあります）はプロジェクトレベルのエージェントコンテキストとしてリポルートに位置します。このレッスンは各々がカバーするものに名前を付け、エージェント全体を移動する最小限の SKILL.md + AGENTS.md バンドルを構築します。

**タイプ:** Learn
**言語:** Python (stdlib, SKILL.md パーサおよびローダー)
**前提条件:** Phase 13 · 07 (MCP サーバー)
**所要時間:** 約45分

## 学習目標

- 3つのレイヤーを区別：AGENTS.md（プロジェクトコンテキスト）、SKILL.md（再利用可能な知識）、MCP（ツール）。
- YAML フロントマターと段階的開示を伴う SKILL.md を作成。
- スキルをエージェントランタイムにファイルシステムスタイルで読み込みます。
- 1つのパッケージがClaude Code、Cursor、Codexで機能するように、スキルをMCPサーバーと AGENTS.md で構成します。

## 問題

エンジニアは、リリースノートを書くワークフローを多段階プロンプトに蒸留します：「最新のマージされたPRを読みます。領域でグループ。各々を要約します。チームのスタイルに従ってチェンジログエントリを作成します。Slack ドラフトに投稿します。」彼らはそれをチームのノーションドキュメントに入れます。

現在彼らはこのワークフローを Claude Code、Cursor、Codex CLI から使用したいと思っています。各エージェントは指示を読み込むための異なる方法があります：Claude Code スラッシュコマンド、Cursor ルール、Codex `.codex.md`。エンジニアはワークフローを3回コピーし、3つのコピーを保守します。

AGENTS.md と SKILL.md が一緒に修正します：

- **AGENTS.md** がリポルートに位置します。互換性のあるすべてのエージェントはセッション開始時にそれを読みます。「このプロジェクトはどのように機能しますか？規約とは何ですか？どのコマンドがテストを実行しますか？」
- **SKILL.md** は移植可能なバンドル：YAML フロントマター（名前、説明）+マークダウン本体+オプションリソース。スキルをサポートするエージェントは必要に応じて名前でそれらを読み込みます。
- **MCP**（Phase 13 · 06-14）はスキルが呼び出す必要があるツールを処理します。

3つのレイヤー、1つの移植可能なアーティファクト。

## コンセプト

### AGENTS.md (agents.md)

2025年後期に起動、2026年4月までに60,000以上のリポジトリで採択。リポルートの1つのファイル。フォーマット：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

エージェントはセッション開始時にこれを読み、そのプロジェクトに対する動作を調整するために使用します。2026年のすべてのコーディングエージェントは AGENTS.md をサポートします：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md フォーマット

Anthropic のエージェントスキル（2025年12月にオープン標準として公開）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

フロントマターはスキルのアイデンティティを宣言します。本体は、スキルがロードされたときにモデルに表示されるプロンプトです。

### 段階的な開示

スキルは、エージェントが必要な場合にのみ取得するサブリソースを参照できます。例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md は「style-guide.md のスタイルルールを参照」と言います。エージェントはスキルがアクティブに実行されているときにのみ style-guide.md をプルします。これはモデルが必要としない可能性があるプロンプトを詳細で膨らませることを避けます。

### ファイルシステム検出

エージェントランタイムは SKILL.md ファイルの既知ディレクトリをスキャンします：

- `~/.anthropic/skills/*/SKILL.md`
- プロジェクト `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

ロード方法は、フォルダ名とフロントマター`name`による。Claude Code、Anthropic Claude Agent SDK、SkillKit（クロスエージェント）はすべてこのパターンに従います。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）および`claude-agent-sdk`（Python）はセッション開始時にスキルを読み込み、ランタイム内で呼び出し可能な「エージェント」として公開します。エージェントループは、ユーザーがスキルを呼び出すときにスキルに送信します。

### OpenAI Apps SDK

2025年10月に起動。MCPの上に直接構築。OpenAI の前の Connectors および Custom GPT Actions を単一の開発者表面の下に統合。Apps SDK アプリは：

- MCP サーバー（ツール、リソース、プロンプト）。
- プラス ChatGPT UI のウィジェットメタデータ。
- プラス対話的表面のためのオプション MCP Apps `ui://`リソース。

同じプロトコル、豊かなUX。

### SkillKit を通じたクロスエージェント移植性

SkillKit および同様のクロスエージェント配信層のようなツールは、単一の SKILL.md を 32以上のAIエージェント（Claude Code、Cursor、Codex、Gemini CLI、OpenCode など）各々のネイティブ形式に変換します。1つの事実上の源。多くの消費者。

### 3層スタック

| レイヤー | ファイル | 読み込み時期 | 目的 |
|--------|---------|-----------|------|
| AGENTS.md | リポルート | セッション開始 | プロジェクトレベルの規約 |
| SKILL.md | スキルディレクトリ | スキル呼び出し | 再利用可能なワークフロー |
| MCP サーバー | 外部プロセス | ツール必要 | 呼び出し可能な操作 |

すべての3つが構成：エージェントはセッション開始時に AGENTS.md を読み、ユーザーはスキルを呼び出し、スキルの指示は MCP ツール呼び出しを含み、エージェントは MCP クライアント経由で送信します。

## 使用方法

`code/main.py`は stdlib SKILL.md パーサおよびローダーを配信します。`./skills/`の下でスキルを検出し、YAML フロントマタープラスマークダウン本体を解析し、スキル名でキー付けされた辞書を生成します。次に、名前で`release-notes-writer`を呼び出すエージェントループをシミュレートします。

確認すべき点：

- YAML フロントマターは最小限の stdlib パーサで解析（`pyyaml`依存関係なし）。
- スキル本体は逐語的に保存。エージェントは呼び出し上のシステムプロンプトにそれを前に置きます。
- 段階的開示は、参照されたファイルをオンデマンドでプルする`read_subresource`関数を介してデモンストレーションされます。

## リリース

このレッスンは`outputs/skill-agent-bundle.md`を生成します。ワークフローが与えられた場合、スキルはエージェント全体に移植可能な結合された SKILL.md + AGENTS.md + MCP サーバーブループリントバンドルを生成します。

## 演習

1. `code/main.py`を実行します。`skills/`の下に2番目のスキルを追加し、ローダーがそれをピックアップすることを確認します。

2. このコースリポジトリ用の AGENTS.md を作成します。テストコマンド、スタイル規約、Phase 13 の精神モデルを含めます。

3. チームの内部ドキュメントから SKILL.md に複数ステップワークフローをポート。Claude Code で読み込まれることを確認します。

4. スキルを Cursor および Codex のネイティブルール形式に手動で変換。フォーマット間の差分をカウント — これが SkillKit が自動化する変換表面です。

5. Anthropic Agent Skills ブログポストを読みます。Claude Agent SDK のこのレッスンのローダーがカバーしない1つの機能を特定します。（ヒント：エージェントサブ呼び出し。）

## キー用語

| 用語 | 人々が言うこと | それが実際に意味すること |
|------|------------------|---------------------------|
| SKILL.md | "スキルファイル" | YAML フロントマタープラスマークダウン本体、エージェントランタイムで読み込み |
| AGENTS.md | "リポルートエージェントコンテキスト" | セッション開始時に読まれるプロジェクトレベル規約ファイル |
| Progressive disclosure | "遅延読み込みサブリソース" | スキル本体は必要な場合にのみプルされるファイルを参照 |
| Frontmatter | "上部のYAMLブロック" | `---`デリミタ内のメタデータ（名前、説明） |
| Claude Agent SDK | "Anthropic のスキルランタイム" | `@anthropic-ai/claude-agent-sdk`、スキル読み込みおよびルート |
| OpenAI Apps SDK | "MCP プラスウィジェットメタ" | MCP プラス ChatGPT UI フックに構築された OpenAI の開発表面 |
| Skill discovery | "ファイルシステムスキャン" | 既知ディレクトリを歩行 SKILL.md、名前でキー |
| Cross-agent portability | "1つのスキル、多くのエージェント" | 1つの SKILL.md を SkillKit スタイルツール経由で 32以上のエージェントに変換 |
| Agent Skill | "移植可能な知識" | MCP のツール概念外の再利用可能タスクテンプレート |
| Apps SDK | "MCP プラス ChatGPT UI" | Connectors と Custom GPTs は MCP で統合 |

## 参考文献

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025年12月起動
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md フォーマット参照
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — ChatGPT 用の MCP ベース開発者プラットフォーム
- [agents.md](https://agents.md/) — AGENTS.md フォーマットおよび採択リスト
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 公式スキル例
