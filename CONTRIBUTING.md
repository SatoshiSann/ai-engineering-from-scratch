# コントリビューション

レッスン、翻訳、バグ修正、成果物 — いずれも歓迎します。プルリクエストは1件につき1つの貢献にまとめると、レビューが迅速になり、コントリビューター数とクレジットが正しく機能します。

## 重要: README と ROADMAP はウェブサイトのソースです

`site/build.js` は `README.md`、`ROADMAP.md`、`glossary/terms.md` を解析して
`site/data.js` を生成します。これらのファイルに触れるプルリクエストでは、以下の2つのパターンを必ず維持してください:

- フェーズヘッダーは `### Phase N: Name \`X lessons\`` の形式か、
  `<details><summary><b>Phase N — Name</b> ... <code>X lessons</code> ... <em>Description</em></summary>` の形式のいずれかであること。
- レッスンテーブルは `| # | Lesson | Type | Lang |` の列構成 (または
  まとめテーブルは `| # | Project | Combines | Lang |`) であること。`Lang` 列は
  プレーンテキスト (`Python, TypeScript`) またはレガシー絵文字フラグ
  (`🐍 🟦 🦀 🟣 ⚛️`) のどちらも受け付けます。パーサー上は同等です。
- ROADMAP のステータスグリフ (`✅`, `🚧`, `⬚`) をフェーズヘッダーとレッスン行に使用すること。
  テキストに置き換えないでください — パーサーは正確な文字をキーとして使用します。

これらのファイルを編集した後は `node site/build.js` を実行してください。編集が構造的に安全であれば、`git diff site/data.js` にはタイムスタンプの変更のみが表示されるはずです。

## コントリビューションの方法

### 1. 新しいレッスンを追加する

各レッスンは `phases/XX-phase-name/NN-lesson-name/` に以下の構成で配置します:

```
NN-lesson-name/
├── code/           少なくとも1つの実行可能な実装
├── notebook/       実験用の Jupyter ノートブック (任意)
├── docs/
│   └── en.md       レッスンドキュメント (必須)
└── outputs/        このレッスンが生成するプロンプト、スキル、またはエージェント (該当する場合)
```

**レッスンドキュメントの形式** (`en.md`):

```markdown
# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

Why does this matter? What can't you do without this?

## The Concept

Explain with diagrams, visuals, and intuition. Code comes later.

## Build It

Step-by-step implementation from scratch.

## Use It

Now use a real framework or library to do the same thing.

## Ship It

The prompt, skill, agent, or tool this lesson produces.

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise
```

### 2. 翻訳を追加する

任意のレッスンの `docs/` フォルダに新しいファイルを作成します:

```
docs/
├── en.md    (英語 — 常に必須)
├── zh.md    (中国語)
├── ja.md    (日本語)
├── es.md    (スペイン語)
├── hi.md    (ヒンディー語)
└── ...
```

英語版と同じ構成を維持してください。コードではなくコンテンツを翻訳します。

### 3. 成果物を追加する

レッスンが再利用可能なプロンプト、スキル、エージェント、または MCP サーバーを生成する場合:

1. レッスンの `outputs/` フォルダに作成する
2. トップレベルの `outputs/` インデックスに参照を追加する

**プロンプトの形式:**

```markdown
---
name: prompt-name
description: What this prompt does
phase: 14
lesson: 01
---

[System prompt or template here]
```

**スキルの形式:**

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]
```

### 4. バグを修正するか既存のレッスンを改善する

- 実行できないコードを修正する
- 説明を改善する
- より良い図を追加する
- 古くなった情報を更新する

### 5. 演習またはプロジェクトを追加する

演習とプロジェクトは常に歓迎します。特に複数のフェーズをつなぐものは大歓迎です。

## ガイドライン

- **コードは必ず動くこと。** すべてのコードファイルは、記載された依存関係でエラーなく実行できること。
- **コードにコメントを書かないこと。** コードは自己説明的であるべきです。説明はドキュメントに書いてください。
- **作業に最適な言語を選ぶこと。** TypeScript や Rust の方が適切な場面で Python を無理に使わないこと。
- **まずスクラッチで構築すること。** フレームワーク版を示す前に、必ず基本原理から概念を実装すること。
- **実践的に保つこと。** 理論は実践に奉仕するものであり、その逆ではありません。
- **AI の粗雑な出力はNG。** 人間らしく書いてください。直接的に。無駄を省いてください。

## プルリクエストのプロセス

1. リポジトリをフォークする
2. フィーチャーブランチを作成する (`git checkout -b add-lesson-phase3-gradient-descent`)
3. 変更を加える
4. すべてのコードが動作することを確認する
5. 明確な説明を添えてプルリクエストを提出する

## 行動規範

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) を参照してください。親切に、助け合い、建設的に。

## スタイル

- 直接的な文章。無駄を省く。マーケティングコピーではなく、マニュアルのトーンに合わせること。
- 見出しに装飾的な絵文字を使わないこと。Lang 列の絵文字フラグは唯一の例外ですが、それはパーサーがマッピングに使用するためです。
- コードはレッスンに記載された依存関係でそのまま動作すること。
- まずスクラッチで構築し、次にフレームワークを使用すること。
