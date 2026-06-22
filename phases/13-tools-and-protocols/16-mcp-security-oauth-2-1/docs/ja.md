# MCP セキュリティ II — OAuth 2.1、リソースインジケータ、段階的スコープ

> リモート MCP サーバーは authorization が need、認証ではなく。2025-11-25 仕様は OAuth 2.1 + PKCE + リソースインジケータ（RFC 8707）+ protected-resource metadata（RFC 9728）に align。SEP-835 は step-up authorization on 403 WWW-Authenticate で段階的スコープ consent add。このレッスンは step-up flow を state machine として実装、すべての hop を see できるよう。

**タイプ:** Build
**言語:** Python（stdlib、OAuth state machine simulator）
**前提条件:** Phase 13 · 09（transports）、Phase 13 · 15（security I）
**所要時間:** 約75分

## 学習目標

- Resource server を authorization server 責任から区別。
- PKCE-protected OAuth 2.1 authorization code flow を walk。
- `resource`（RFC 8707）と protected-resource metadata（RFC 9728）使用を confused-deputy attacks prevent。
- Step-up authorization を実装：server は 403 で higher scope をリクエスト WWW-Authenticate；client は re-prompt user consent と retry。

## 問題

Early MCP（pre-2025）はリモートサーバーを ad-hoc API keys またはなお auth なしで ship。2025-11-25 仕様はそのギャップを full OAuth 2.1 profile で close。

3 つの real-world needs：

- **Ordinary リモートサーバー。** ユーザーが remote MCP サーバーを install、their Notion / GitHub / Gmail にアクセス。OAuth 2.1 with PKCE は right shape。
- **Scope escalation。** ノートサーバーは `notes:read` grant can later need `notes:write` specific action。Whole flow を re-do ではなく、step-up（SEP-835）は additional scope をリクエスト。
- **Confused deputy prevention。** クライアントは token audience-scoped hold Server A。Server A は malicious、token を Server B に present try。Resource indicators（RFC 8707）は token を intended audience へpin。

OAuth 2.1 は新しくない。何が new は MCP's profile：specific required flows（authorization code + PKCE のみ；no implicit、no client credentials by default）、すべてのトークンリクエストで resource indicators mandatory、protected-resource metadata published クライアントが know where go のため。

## コンセプト

### Roles

- **Client。** MCP client（Claude Desktop、Cursor など）。
- **Resource server。** MCP server（notes、GitHub、Postgres、whatever）。
- **Authorization server。** Issue tokens。May be same service as resource server または separate IdP（Auth0、Keycloak、Cognito）。

MCP's profile では、resource と authorization servers CAN be same host しかし SHOULD be URLs で distinguished。

### Authorization code + PKCE

Flow：

1. Client が `code_verifier`（random）と `code_challenge`（SHA256）を generate。
2. Client がユーザーを `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com` にredirect。
3. ユーザーが consent。Authorization server は `redirect_uri?code=...` に redirect。
4. Client が `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` に POST。
5. Authorization server が verifier's hash を stored challenge に対して validate、access token を issue。
6. Client がtoken を use：resource server のすべてのリクエストで `Authorization: Bearer ...`。

PKCE は authorization-code interception attacks を prevent。Resource indicators はtoken が他のどこでも valid でないよう prevent。

### Protected-resource metadata（RFC 9728）

Resource server は `.well-known/oauth-protected-resource` document を publish：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Client は resource server から authorization server を discover。Configuration を reduce — client は resource URL のみ need。

### リソースインジケータ（RFC 8707）

Token request の `resource` parameter はtoken の intended audience へ pin。Issued token は `aud: "https://notes.example.com"` を contain。別 MCP server がこのtoken receive、`aud` を check、reject。

### Scope model

Scope は space-separated strings。Common MCP conventions：

- `notes:read`、`notes:write`、`notes:delete`
- Admin capabilities（sparingly use）の `admin:*`
- Identity の `profile:read`

Scope selection は least-privilege であるべき：now に need something をリクエスト、more を need の際にstep up。

### Step-up authorization（SEP-835）

User は `notes:read` grant。彼ら later ノート delete をagent にリクエスト。Server は respond：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Client は insufficient_scope error を see、ユーザーに prompt additional scope に対して consent dialog、mini OAuth flow を perform、new token で request を retry。

### Token audience validation

すべてのリクエスト：server は check `token.aud == self.resource_url`。Mismatch = 401。これは cross-server token reuse を stop。

### Short-lived tokens と rotation

Access tokens は short-lived（1 hour default）であるべき。Refresh tokens rotate every refresh に。Client は background で silent refresh を handle。

### No token passthrough

Sampling servers（Phase 13 · 11）はクライアント token を pass through しない必須、other services に。Sampling request は boundary。

### Confused deputy prevention

Token は `aud` へbind。Client は `client_id` へbind。すべてのリクエストは both に対して validate。Spec は explicit に ban old「pass-the-token」pattern pre-MCP remote tool ecosystems に common だった。

### Client ID discovery

各 MCP client はその metadata を fixed URL で publish。Authorization servers は fetch できます client metadata document へ discover redirect URIs と contact info を。これは remove manual client registration。

### Gateways と OAuth

Phase 13 · 17 は企業ゲートウェイが OAuth をどう handle するか show：gateway は upstream servers の credentials を hold、client へのtoken は gateway-issued、upstream tokens は gateway を never leave。これは flip trust model — users は once gateway で authenticate；gateway は N server authorizations を handle。

## Use It

`code/main.py` はfull OAuth 2.1 step-up flow を state machine として simulate。それは実装：

- PKCE code-verifier / challenge generation。
- Authorization code flow with resource indicator。
- Protected-resource metadata endpoint。
- Token validation with audience check。
- Step-up on `insufficient_scope`。

No HTTP server in this lesson；state machine はメモリで実行、すべての hop を trace できるよう。Phase 13 · 17's gateway lesson はそれを actual transport にwire。

## Ship It

このレッスンは `outputs/skill-oauth-scope-planner.md` を作成。Tools を持つリモート MCP サーバーが与えられた場合、スキルは scope set、pinning rules、step-up policy を design。

## 演習

1. `code/main.py` 実行。2-scope step-up flow を trace。Note どの hops replay on step-up。

2. Refresh-token rotation を add：すべての refresh は新しい refresh token をissue、old one を invalidate。Stolen refresh token を rotation 後に use simulate、失敗を confirm。

3. Protected-resource metadata endpoint を real HTTP response として stdlib http.server を使用して implement。Mirror Lesson 09 の /mcp endpoint。

4. GitHub MCP サーバーの scope hierarchy を design：read repo、write PR、approve PR、merge PR、admin。各レベル間で step-up を use。

5. RFC 8707 と RFC 9728 を read。MCP が RFC's example と異なり使用する 1 つフィールドを identify。（Hint：`scopes_supported` に concern。）

## 主要用語

| 用語 | 人々が言うこと | 実際に意味すること |
|------|----------------|------------------------|
| OAuth 2.1 | 「Modern OAuth」 | Consolidated RFC が PKCE を mandate、implicit flow を forbid |
| PKCE | 「Proof-of-possession」 | Code verifier + challenge defeating authorization-code interception |
| リソースインジケータ | 「Token audience」 | RFC 8707 `resource` parameter を 1 つサーバーに pinning token |
| Protected-resource metadata | 「Discovery doc」 | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | 「段階的consent」 | SEP-835 flow dem機 on-demand scopes を add する |
| `insufficient_scope` | 「403 with WWW-Authenticate」 | Server signal を larger scope に対して re-consent |
| Confused deputy | 「Token reuse across services」 | Attack、信頼できるholder がtoken を inappropriately forward |
| Short-lived token | 「Access token TTL」 | Bearer が quickly expire；refresh token が renew |
| Scope hierarchy | 「Least privilege stack」 | Graduated scope set with step-up levels 間 |
| クライアント ID metadata | 「Client discovery doc」 | URL at which client がその own OAuth metadata を publish |

## 参考文献

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — canonical MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 changes の walkthrough
- [RFC 8707 — リソースインジケータ for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery-document RFC
- [Aembit — MCP OAuth 2.1、PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — practical step-up-flow walk-through
