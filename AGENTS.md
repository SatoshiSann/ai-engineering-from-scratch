# AGENTS.md

このリポジトリに関わるコントリビューターおよびAIエージェント向けの操作マニュアルです。PRを作成する前に必ずお読みください。

このリポジトリはカリキュラムであり、SaaSアプリではありません。レッスンが製品です。以下のルールはすべて、435のレッスンを長期にわたって一貫したものに保つためのものです。

---

## 哲学

435レッスン。20フェーズ。すべてのアルゴリズムは、フレームワークを一つもインポートする前に生の数学から構築されます。バックプロパゲーション、トークナイザー、アテンション機構、エージェントループを、Python、TypeScript、Rust、またはJuliaで手書きします。その後、同じ操作を本番ライブラリで実行し、フレームワークがブラックボックスでなくなるようにします。「自分で作る / 使ってみる」の分割が骨格です。各レッスンは、日々のワークフローに組み込める再利用可能な成果物を提供します。

---

## リポジトリ構成

```
phases/
  NN-phase-slug/
    NN-lesson-slug/
      docs/en.md              # レッスンの説明
      code/                   # 実装 + テスト
      quiz.json               # 6問
      outputs/                # 再利用可能な成果物（スキル / プロンプト / エージェント / MCPサーバー）
README.md                     # 公開ページ；レッスン数は自動同期
ROADMAP.md                    # フェーズ/レッスンのステータス
glossary/terms.md             # 用語の正規定義
site/
  build.js                    # README + ROADMAP + glossaryを解析 -> data.js を生成
  data.js                     # 生成ファイル；mainへのプッシュ時にCIが再ビルド
scripts/                      # 自動化スクリプト
.github/workflows/
  curriculum.yml              # 不変条件チェック + 自動同期ワークフロー
```

---

## 絶対ルール

1. **レッスンディレクトリごとに1コミット。** 複数のレッスンを1つのコミットにまとめないでください。10レッスンのPRには10コミットが必要です。
2. **コンベンショナルコミットの件名**は72文字以内: `feat(phase-NN/MM): <slug>`。本文は「何を」ではなく「なぜ」を説明します。
3. **図はMermaidまたはSVGのみ。** ASCII / Unicodeのボックス描画は使用不可。
4. **すべてのコードブロックには言語タグが必要です。** 適切に `text`、`json`、`python`、`typescript`、`rust`、`julia`、`bash`、`console`、`mermaid`、`yaml` を使用してください。
5. **独自の実装のみ。** ドキュメント、コードコメント、コミットテキストに外部カリキュラムリポジトリを引用しないでください。正規のソースである場合は、RFC、公式仕様、学術論文を引用してください。
6. **依存関係の許可リスト**（下記 `Dependencies` 参照）。標準ライブラリ優先。
7. **生成ファイルは絶対にコミットしない**: `catalog.json` は .gitignore 対象、`site/data.js` はCIが再ビルド、`package-lock.json` は追跡しない。

---

## 依存関係

| 言語       | 使用可能なもの                                                                  |
|------------|--------------------------------------------------------------------------|
| Python     | `numpy`, `torch`, `h5py`, `zstandard`, `safetensors`, stdlib              |
| TypeScript | `hono`, `zod`, `ws`（WebSocketが必要な場合のみ）, `@hono/node-server`, Node 20+ stdlib |
| Rust       | stdlib のみ（シングルファイル `rustc --edition 2021`）                          |
| Julia      | `Random`, `Statistics`, `LinearAlgebra`, `Printf`（Julia stdlib）          |

禁止された依存関係が提案された場合は、「教育的な明確さのために標準ライブラリ優先を維持する」という理由でスキップしてください。

---

## レッスン規約

### docs/en.md フロントマター

```markdown
# <Title>

> <One-line hook>

**Type:** <Learn | Build | Reference>
**Languages:** <comma-list matching the main.* files in code/>
**Prerequisites:** <comma-list of upstream lessons, or "None">
**Time:** ~<estimate in minutes>

## Learning Objectives
- <4-6 bullet points starting with a verb>
```

`**Languages:**` フィールドは `code/` 内の `main.*` ファイルの言語と一致している必要があります。

### quiz.json スキーマ

```json
{
  "lesson": "<dir-slug>",
  "title": "<Lesson Title>",
  "questions": [
    {"stage": "pre",   "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 2, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 3, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""}
  ]
}
```

厳密に6問: pre 1問 + check 3問 + post 2問。`correct` はゼロインデックスです。サイトのレンダラーはこの形式のみを認識します — レガシーの `q/choices/answer` スキーマはサイレントにクラッシュします。

### code/

- 各言語の標準コマンドでエンドツーエンドに実行され、終了コード0で終了すること。
- 自己完結型デモ。無限の標準入力ループや、APIキーが不足している場合のハングは不可。
- レッスンの `docs/en.md` パスおよび仕様やRFCソースを引用する4〜6行のヘッダーコメント。

### code/tests/

- 最低5つのユニットテスト。
- 各言語の標準ライブラリランナーで実行（`python3 -m unittest discover`、`npx tsx --test`、Rust/Julia インライン）。

---

## PR前の検証

プッシュ前にローカルで実行:

```bash
python3 scripts/audit_lessons.py
python3 scripts/check_readme_counts.py        # 参考情報 — CIがマージ時に修正

# 変更した各レッスンについて:
cd phases/NN-phase/MM-lesson/code
python3 main.py && python3 -m unittest discover tests -v   # または各言語に対応するコマンド
```

CIゲート (`.github/workflows/curriculum.yml`):

| ジョブ                            | トリガー      | 動作                                              |
|----------------------------------|--------------|-------------------------------------------------------|
| `audit`                          | push + PR    | `audit_lessons.py` を実行。ブロッキング。                    |
| `readme-counts-sync`（mainのみ） | mainへのプッシュ | カタログを再ビルドし、READMEのカウントを自動修正。         |
| `site-rebuild`（mainのみ）       | mainへのプッシュ | `node site/build.js` を再実行し、`site/data.js` をコミット。 |
| `readme-counts-drift`            | PR           | 参考情報のみ — mainはマージ時に自己修正。             |

---

## 自動化の規約

**CIが自動で処理するもの — PRで変更しないでください:**

| 対象                 | ボット                          | タイミング              |
|----------------------|--------------------------------|---------------------|
| `catalog.json`       | オンデマンドで再ビルド（.gitignore対象） | すべてのCIジョブ        |
| `README.md` のカウント | `readme-counts-sync`           | mainへのプッシュ時     |
| `site/data.js`       | `site-rebuild`                 | mainへのプッシュ時     |

**あなたが対応するもの:**

| 対象                          | タイミング                                                             |
|-------------------------------|------------------------------------------------------------------|
| `README.md` のレッスンリンク行  | 新しいレッスンを追加するとき — `[Title](phases/NN-phase/MM-lesson/)` をリンクする |
| `ROADMAP.md` のステータス      | レッスンを完了またはWIPとしてマークするとき                            |
| `glossary/terms.md`           | 複数のレッスンで使用される用語を導入するとき                           |

**よくあるバグ**: マージ後に `grep -c 'tree/main/phases/NN-' site/data.js` が0の場合、フェーズNN のREADME行がプレーンテキストになっており、`[Title](phases/NN-...)` というMarkdownリンクが欠けています。`site/build.js` はそのリンクからURLを取得します。

---

## コンフリクトの解消

```bash
git fetch origin main
git merge --no-edit origin/main

# カタログのコンフリクト（レガシーブランチのみ — catalog.json は現在 .gitignore 対象）:
git rm catalog.json
git commit --no-edit

# READMEカウントのコンフリクト:
git checkout --theirs README.md
python3 scripts/build_catalog.py
python3 scripts/check_readme_counts.py --fix
git add README.md && git commit --no-edit

# site/data.js のコンフリクト:
git checkout --theirs site/data.js
node site/build.js
git add site/data.js && git commit --no-edit

git push origin <your-branch>
```

レビューコメントが付いたブランチへの `git push --force` は避けてください。強制プッシュはコメントの関連付けを切り離します。

---

## 新レッスンのオンボーディング

```bash
mkdir -p phases/NN-phase-slug/MM-new-lesson/{docs,code/tests,outputs}

# 1. 上記のフロントマターを含む docs/en.md を作成。
# 2. 4〜6行のヘッダーを含む code/main.<lang> を作成。
# 3. 5つ以上のテストを含む code/tests/test_main.* を作成。
# 4. 上記のスキーマに従って quiz.json を作成。
# 5. （任意）レッスンがスキルを提供する場合は outputs/skill-<slug>.md を追加。

# 6. README.md に追加:
#    | MM | [Lesson Title](phases/NN-phase-slug/MM-new-lesson/) | Type | Lang |

# 7. ROADMAP.md のステータス行を更新。

# 8. ローカルで検証。

# 9. アトミックコミット:
git add phases/NN-phase-slug/MM-new-lesson README.md ROADMAP.md
git commit -m "feat(phase-NN/MM): add <slug>"
git push -u origin <your-branch>
gh pr create --title "feat(phase-NN/MM): add <slug>" --body "<5行の概要>"
```

`site/data.js` はマージ時に再生成されます — CIに任せてください。

---

最終レビュー: 2026-05-27.
