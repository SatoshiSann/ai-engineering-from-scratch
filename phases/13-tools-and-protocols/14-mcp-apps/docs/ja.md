# MCP アプリ — `ui://` 経由のインタラクティブ UI リソース

> テキストのみの tool 出力は agent が表示できることを cap。MCP Apps（SEP-1724、公式 January 26、2026）はツールが Claude Desktop、ChatGPT、Cursor、Goose、VS Code で inline レンダリングされた sandboxed interactive HTML を return することを許可。ダッシュボード、form、map、3D scenes、すべて 1 つの拡張経由。このレッスンは `ui://` リソーススキーム、`text/html;profile=mcp-app` MIME、iframe-sandbox postMessage プロトコル、サーバーが HTML をレンダリングする際に来るセキュリティサーフェスを walk。

**タイプ:** Build
**言語:** Python（stdlib、UI resource emitter）、HTML（sample app）
**前提条件:** Phase 13 · 07（MCP サーバー）、Phase 13 · 10（resources）
**所要時間:** 約75分

## 学習目標

- Tool call から `ui://` リソースを return、正しい MIME と metadata を set。
- Tool の関連 UI を declare、`_meta.ui.resourceUri`、`_meta.ui.csp`、`_meta.ui.permissions`。
- Iframe sandbox postMessage JSON-RPC を implement、UI-to-host communication 用。
- UI 起動攻撃に防御する CSP と permissions-policy defaults を apply。

## 問題

2025 era `visualize_timeline` ツールは「ここは 14 ノート chronologically に organize：...」を return できる。それは paragraph。ユーザーは actually interactive timeline が want。MCP Apps 前、options は：client-specific widget APIs（Claude artifacts、OpenAI Custom GPT HTML）、または no UI at all。

MCP Apps（SEP-1724、January 26、2026 shipped）は contract を standardize。Tool result は URI が `ui://...` with MIME が `text/html;profile=mcp-app` のリソースを contain。ホスト limited CSP と no network access unless explicitly granted の sandboxed iframe にレンダリング。Iframe 内の UI は postMessage JSON-RPC tiny dialect 経由でホストに messages を post。

すべての compatible client（Claude Desktop、ChatGPT、Goose、VS Code）は same `ui://` リソースを same 方法でレンダリング。1 つのサーバー、1 つの HTML bundle、universal UI。

## コンセプト

### `ui://` リソーススキーム

ツール return：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

ホスト then で `ui://notes/timeline` URI の `resources/read` を呼び出し、back を get：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

ホスト HTML を sandboxed `<iframe>` 内でレンダリング：

- `sandbox="allow-scripts allow-same-origin"`（または server declaration per stricter）
- Server-declared CSP が response headers で apply。
- No cookies、no localStorage ホスト origin から。
- Network access は CSP の `connectSrc` に limited。

### postMessage プロトコル

Iframe は `window.postMessage` 経由でホストと communicate。Tiny JSON-RPC 2.0 dialect：

常に `targetOrigin` を peer の exact origin にpinして、receiving side は payload を process 前に allowlist に対して `event.origin` を validate。`"*"` を either side で use しない — body は tool call とリソース read を carry。

```js
// iframe to host  (host origin にpin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (iframe origin にpin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// both sides 上の receiver
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

Available host-side methods UI は呼び出し可能：

- `host.callTool(name, arguments)` — サーバーツール invoke。
- `host.readResource(uri)` — MCP リソース read。
- `host.getPrompt(name, arguments)` — prompt template fetch。
- `host.close()` — UI を dismiss。

すべてのコール MCP プロトコル経由でまだ go、server の permissions を inherit。

### Permissions

`_meta.ui.permissions` list request extra capabilities：

- `camera` — ユーザーのカメラにアクセス（document-scan UI 用）。
- `microphone` — voice input。
- `geolocation` — location。
- `network:*` — `connectSrc` alone が許可より wider network access。

各 permission は UI がレンダリング前にユーザーが see する prompt。

### セキュリティリスク

HTML iframe 内の HTML はなお HTML。新しいattack surface：

- **UI 経由の Prompt-injection。** 悪意のあるサーバー UI はシステムメッセージのような look text を show、ユーザーをトリック可能。ホストレンダリング should visibly distinguish server UI からホスト UI。
- **`connectSrc` 経由の exfiltration。** CSP permit の場合 `connect-src: *`、UI は anywhere にデータを send できます。Default should strict。
- **Clickjacking。** UI はホスト chrome を overlay。ホスト must prevent z-index manipulation と enforce opacity rules。
- **Focus を steal。** UI はキーボード focus を take、次のメッセージを capture。ホスト must intercept。

Phase 13 · 15 は MCP セキュリティの一部として deep でこれらをカバー；このレッスンはそれら intro。

### `ui/initialize` ハンドシェイク

Iframe が load 後、それが postMessage 経由で `ui/initialize` を send：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

ホスト capabilities と session token で respond。UI はすべての後続ホスト呼び出しで session token を use。

### AppRenderer / AppFrame SDK プリミティブ

Ext-apps SDK は 2 つの convenience プリミティブを expose：

- `AppRenderer`（server side）— React / Vue / Solid component をラップ、correct MIME と metadata を emit する `ui://` リソース。
- `AppFrame`（client side）— リソース receive、iframe mount、postMessage mediate。

これをまたは hand-roll HTML と JSON-RPC を use できます。

### Ecosystem status

MCP Apps は January 26、2026 ship。Client support as of April 2026：

- **Claude Desktop.** January 2026 から Full support。
- **ChatGPT.** Apps SDK 経由 Full support（same underlying MCP Apps プロトコル）。
- **Cursor.** Beta；settings 経由 enable。
- **VS Code.** Insider builds のみ。
- **Goose.** Full support。
- **Zed、Windsurf.** Roadmapped。

Servers in production：dashboards、map visualizations、data tables、chart builders、sandbox IDE previews。

## Use It

`code/main.py` は Lesson 07 ノートサーバーを `visualize_timeline` ツール with `ui://notes/timeline` リソース、plus handler for `resources/read` on that URI which return small but complete HTML bundle with SVG timeline で拡張。HTML は stdlib-templated — no build system。postMessage は JS comments sketch、stdlib cannot drive browser ため。

見るべきこと：

- `_meta.ui` on tool response は resourceUri、CSP、permissions を carry。
- HTML は network access なしでレンダリング；すべてのデータは inlined。
- JS は `host.callTool` を `window.parent.postMessage` 経由で call（documented but inert in this stdlib demo）。

## Ship It

このレッスンは `outputs/skill-mcp-apps-spec.md` を作成。Interactive UI から benefit する tool が与えられた場合、スキルは full MCP Apps contract を produce：`ui://` URI、CSP、permissions、postMessage entrypoints、security checklist。

## 演習

1. `code/main.py` 実行、emitted HTML を inspect。HTML を直接 browser で open；SVG がレンダリングを verify。Then sketch postMessage contract UI が `host.callTool("notes_update", ...)` を呼び出し用に use。

2. CSP をtighten：`'unsafe-inline'` remove、nonce-based script policy を use。HTML generation code で何が change？

3. 2 番目 UI リソース `ui://notes/editor` を add、in-place でノート edit のための form。ユーザー submit のとき、iframe は `host.callTool("notes_update", ...)` を call。

4. UI のattack surface を audit。Malicious サーバーはどこで content inject できる可能性？Iframe sandbox は何に防御して、何に防御しない？

5. SEP-1724 spec を読む、toy implementation がこれを use しないしない MCP Apps SDK の 1 つ capability を identify。（Hint：component-level state sync.）

## 主要用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| MCP Apps | 「インタラクティブ UI リソース」 | SEP-1724 extension 2026-01-26 shipped |
| `ui://` | 「App URI スキーム」 | UI bundles のリソーススキーム |
| `text/html;profile=mcp-app` | 「MIME」 | MCP App HTML の Content-type |
| Iframe sandbox | 「Render コンテナ」 | CSP と permissions の browser sandboxing UI |
| postMessage JSON-RPC | 「UI-to-host wire」 | Tiny JSON-RPC-over-postMessage dialect host call 用 |
| `_meta.ui` | 「Tool-UI binding」 | Tool result をUI リソースにリンク metadata |
| CSP | 「Content-Security-Policy」 | Scripts、network、styles の許可 source declare |
| AppRenderer | 「Server SDK プリミティブ」 | Framework component を `ui://` リソースに convert |
| AppFrame | 「Client SDK プリミティブ」 | Iframe mount helper で postMessage mediate |
| `ui/initialize` | 「Handshake」 | UI からホストへの最初の postMessage |

## 参考文献

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation と SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — formal spec document
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — high-level documentation
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — January 2026 launch post
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc-style SDK reference
