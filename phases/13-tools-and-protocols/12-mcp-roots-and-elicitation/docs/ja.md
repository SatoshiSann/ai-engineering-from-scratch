# ルートと引き出し — スコープと mid-flight ユーザー入力

> ハードコードパスはユーザーが異なるプロジェクトを開いた瞬間に壊れます。事前記入ツール引数はユーザーが不完全指定の場合に壊れます。ルートはサーバーをユーザーコントロール URI セットにスコープ；引き出しは mid-tool-call を pause してユーザーに form または URL 経由の構造化入力をリクエスト。2 つのクライアントプリミティブ、一般的な MCP 障害モードの 2 つの修正。SEP-1036（URL-mode 引き出し、2025-11-25）は H1 2026 を通じて実験的 — それに依存する前に SDK バージョンをチェック。

**タイプ:** Build
**言語:** Python（stdlib、roots + elicitation demo）
**前提条件:** Phase 13 · 07（MCP サーバー）
**所要時間:** 約45分

## 学習目標

- `roots` を宣言し、`notifications/roots/list_changed` に応答する。
- サーバーファイル操作を宣言ルートセット内の URI に制限する。
- mid-tool-call で `elicitation/create` を使用してユーザーに form または URL 経由の構造化入力をリクエスト。
- form-mode と URL-mode elicitation の選択（後者は実験的；ドリフトリスク注記）。

## 問題

2 つの具体的な障害、プロダクション内のノート MCP サーバー hit。

**壊れたパス仮定。** サーバーが `~/notes` に対して作成。別マシンのユーザーが `~/Documents/Notes` にノートを持つ場合、tool call が silent fail（no file found）または worse、間違った place に wrote。

**ユーザーが知っているであろう引数を逃す。** ユーザーが「古い TPS report ノートを delete」をリクエスト。モデルが `notes_delete(title: "TPS report")` を呼び出す、しかし 3 つのマッチングノートがあります 2023、2024、2025 から。ツールは guess できません。「ambiguous」で fail は annoying；すべての 3 つで running は catastrophic。

ルートが最初を修正：クライアント `initialize` で宣言、サーバーが touch して良い URI セット。引き出しが 2 番目を修正：サーバーが tool call を pause、ユーザーをリクエストする `elicitation/create` 送信、どれを pick するか。

## コンセプト

### ルート

クライアントは `initialize` で root リストを宣言：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

サーバーは `roots/list` を呼び出し：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

サーバーは roots を境界として扱う必須：ルートセット外の任意のファイル read または write は rejected。これは client で enforce されません（サーバーはまだユーザーが trust したコード）、しかし spec-compliant サーバーはそれを honor。

ユーザーが root を add または remove の場合、クライアントが `notifications/roots/list_changed` を送信。サーバーが `roots/list` を再度呼び出し、境界を update。

### なぜ roots はクライアントプリミティブか

ルートはクライアントで宣言。それはユーザーの consent model を表す。ユーザーが Claude Desktop に「give this notes server access to these two directories」を伝える。サーバーはそのスコープを widen できません。

### 引き出し：form-mode デフォルト

`elicitation/create` は form schema plus natural-language prompt をとる：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

クライアントは form をレンダリング、ユーザー answer を collect、返す：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

3 つの可能な action：`accept`（ユーザーがそれを填充）、`decline`（ユーザーが close）、`cancel`（ユーザーが abort whole tool call）。

Form スキーマはフラット — nested objects は v1 ではサポートされません。SDK は通常、単一レイヤーより complex な何でも reject。

### 引き出し：URL mode（SEP-1036、experimental）

2025-11-25 の新機能。Schema ではなく、サーバーが URL を送信：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

クライアント browser で URL を開く、完成待つ、ユーザーが戻ってきたら return。OAuth flows、支払い authorization、document signing に useful、form 不十分な場合。

ドリフトリスク注記：SEP-1036 response shape はまだ settle 中；一部 SDK が callback URL を return、他が completion token を return。production で URL mode を使用する前に SDK のリリースノートを読む。

### 引き出しが正しいツール

- ユーザー確認 before 破壊的アクション（destructive hint + elicitation）。
- disambiguation（N マッチから 1 つ pick）。
- first-run setup（API keys、directories、preferences）。
- OAuth スタイル flow（URL mode）。

### 引き出しが wrong

- ツール required 引数を fill、モデルが prose で尋ねた可能性。Normal re-prompt を使用、elicitation dialog ではなく。
- high-frequency 呼び出し。Elicitation は conversation を interrupt；loop 内で fire しないで。
- サーバーが fact 後に validate できる何でも。Validate、error を return、モデルを text でユーザーをリクエストするようにさせる。

### Human-in-the-loop ブリッジ

Elicitation plus sampling together は MCP の「human-in-the-loop」モデルを enable。サーバーエージェントループは user input（elicitation）または model reasoning（sampling）に対して pause 可能。Phase 13 · 11 はサンプリングをカバー；このレッスンは elicitation をカバー。一緒に put。

## Use It

`code/main.py` はノートサーバーを以下で拡張：

- `roots/list` response で、root-list-changed 通知後にサーバーが再度 query。
- `notes_delete` ツール、複数ノート match の場合 elicitation/create を使用して disambiguate。
- `notes_setup` ツール、URL-mode elicitation を使用して first-run config page を open（simulated）。
- 宣言ルートセット外の URI での操作を refuse する boundary check。

デモが 3 つのシナリオを実行：happy path（1 マッチ）、disambiguation（3 マッチ、elicitation fires）、out-of-root-write（rejected）。

## Ship It

このレッスンは `outputs/skill-elicitation-form-designer.md` を作成。ユーザー確認または disambiguation が必要な可能性のあるツールが与えられた場合、スキルが elicitation form schema と message template を設計。

## 演習

1. `code/main.py` 実行。disambiguation path をトリガー；simulated ユーザー answer が tool に route-back されることを確認。

2. 新しいツール `notes_archive` を追加、毎回 elicitation 確認を require（destructive hint）。UX をチェック：これはモデルが text で再度 ask と比較してどう？

3. first-run OAuth flow 用に URL-mode elicitation を実装。ドリフトリスク注記をして SDK-version guard を追加。

4. `roots/list` handling を拡張：通知が arrive、サーバーは atomically 再読み込みと rescan open file handles、scope 外になった可能性。

5. GitHub で SEP-1036 issue discussion thread を読む。URL-mode callbacks をサーバーがどう handle すべきかに影響する 1 つ open question を識別。

## 主要用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| Root | 「Consent 境界」 | URI、client がサーバーが touch を許可 |
| `roots/list` | 「サーバーが scope をリクエスト」 | Client が現在の root set を return |
| `notifications/roots/list_changed` | 「ユーザーが scope を変更」 | Client signal、root set が mutate |
| 引き出し | 「mid-call でユーザーに ask」 | サーバー起動 structured ユーザー入力 request |
| `elicitation/create` | 「メソッド」 | Elicitation リクエスト用 JSON-RPC メソッド |
| Form mode | 「Schema-driven form」 | Flat JSON Schema が client UI で form にレンダリング |
| URL mode | 「Browser redirect」 | SEP-1036 experimental；URL を開く wait |
| `accept` / `decline` / `cancel` | 「ユーザー応答結果」 | 3 つ branch、サーバー handle |
| Disambiguation | 「1 つ pick」 | ツールが N 候補の場合の一般的 elicitation ユースケース |
| Flat form | 「トップレベルプロパティのみ」 | Elicitation スキーマは nest できない |

## 参考文献

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — canonical roots リファレンス
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — canonical elicitation リファレンス
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 追加 walkthrough
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation proposal（experimental、drift-risk）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
