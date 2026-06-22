# 本番環境のMCP認証 — DCR、JWKS回転、iii プリミティブ上のオーディエンス固定トークン

> レッスン16はOAuth 2.1ステートマシンをメモリに構築しました。2026年までに、実組織に配信するすべてのMCPサーバーは本番環境認証の背後にあります：動的クライアント登録（RFC 7591）、認可サーバーメタデータ検出（RFC 8414）、午前3時のトークン検証を破壊しないJWKS回転、および混同された代理人の再利用を拒否するオーディエンス固定トークン。このレッスンはすべてのiiiプリミティブを通じてそれをワイヤリングします — HTTPおよびcronの`iii.registerTrigger`、認証ロジックの`iii.registerFunction`、キャッシュされたキーの`state::set/get` — だからauth表面は観測可能で、再起動可能で、エンジン内の他のすべてのワークロードと同様に再生可能です。

**タイプ:** Build
**言語:** Python (stdlib, iii プリミティブはレッスン環境では模擬)
**前提条件:** Phase 13 · 16 (OAuth 2.1 ステートマシン), Phase 13 · 17 (ゲートウェイ)
**所要時間:** 約90分

## 学習目標

- RFC 8414メタデータを通じて認可サーバーを検出し、契約を検証します。
- RFC 7591動的クライアント登録を実装し、MCPクライアントが管理者の介入なしで登録できるようにします。
- cronトリガーを使用してJWKSキーをキャッシュおよび回転させ、署名検証がキーロールオーバーを生き残るようにします。
- RFC 8707リソースインジケータを使用してトークンを単一のMCPリソースにピンし、混同された代理人の再利用を拒否します。
- すべてのエンドポイントとバックグラウンドジョブをiiiプリミティブとしてワイヤリングします — HTTPトリガー、cronトリガー、名前付き関数、`state::*`読み取り — 単一の再起動がauth表面を再構築します。
- IdP機能マトリックスを読み取り、IdPがMCPのauth プロファイルを満たすことができない場合にデプロイを拒否します。

## 問題

レッスン16シミュレータは、OAuth 2.1をメモリ内で実行します。本番環境には、メモリのみのシミュレータが見えない3つの運用上のギャップがあります。

最初のギャップは登録です。実組織は数百のMCPサーバーと数千のMCPクライアントを実行します。オペレータは、OAuthクライアントとしてすべてのCursorユーザーを手動で登録しません。RFC 7591動的クライアント登録により、クライアントは認可サーバーに対して`POST /register`でき、その場で`client_id`（およびオプションで`client_secret`）を受け取ります。サーバーはそのRFC 8414メタデータで`registration_endpoint`を公開します。クライアントはアウトオブバンド設定なしでそれを検出します。

2番目のギャップはキー回転です。JWT検証は認可サーバーの署名キーに依存し、JSON Web Key Set（JWKS）として公開されます。認可サーバーはスケジュール（しばしば毎時間、時には事件対応下ではより速く）に従ってこれらを回転させます。起動時にJWKSを1回フェッチするMCPサーバーは、回転ウィンドウまで正常に検証されます — その後、再起動まですべてのリクエストが失敗します。本番環境はJWKSをキャッシュされた値としてワイヤリングします。前のキーが期限切れになる前にキャッシュを上書きする更新ジョブと、キャッシュより新しいキーで署名されたトークンが到着する場合のキャッシュミスのフォールバックフェッチを使用します。

3番目のギャップはオーディエンスバインディングです。レッスン16はRFC 8707リソースインジケータを導入しました。本番環境では、そのインジケータはすべてのリクエストでハードクレーム確認になります。MCPサーバーは`token.aud`を独自の正規リソースURLと比較し、ミスマッチをHTTP 401で拒否します。これは、上流のMCPサーバー（またはあるサーバー用に意図されたトークンを保持している悪意あるクライアント）が同じ信頼メッシュ内の別のサーバーに対してそのトークンを再生することに対する唯一の防御です。

このレッスンは、これらのギャップのそれぞれをiiiプリミティブとして扱います。メタデータドキュメントはHTTPトリガーであり、関数の出力を返します。JWKS回転はcronトリガーであり、`auth::rotate-jwks`を呼び出し、`state::set("auth/jwks/<issuer>", ...)`に書き込みます。JWT検証は、`iii.trigger("auth::validate-jwt", token)`を通じて他の呼び出しによって呼び出される関数です。MCPサーバー自体は、ディスパッチする前に検証を呼び出す別のHTTPトリガーに過ぎません。エンジンを再起動します：トリガーレジストリが再構築されます。状態は存続します。auth表面は手動による調整なしで動作可能です。

## コンセプト

### RFC 8414 — OAuth認可サーバーメタデータ

`/.well-known/oauth-authorization-server`のドキュメントは、クライアントが必要なすべてを説明します：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

MCPリソースURLが与えられたクライアントは検出をチェーンします：RFC 9728の`oauth-protected-resource`（リソースサーバーのドキュメント）は発行者を命名し、その後このRFCの`oauth-authorization-server`はすべてのエンドポイントを命名します。クライアントは認可URLをハードコードしません。

MCPの信頼できるIdPを導入する前に検証する契約：

- `code_challenge_methods_supported`は`S256`を含みます（RFC 7636ごとのPKCE）。
- `grant_types_supported`は`authorization_code`を含み、`password`と`implicit`を拒否します。
- `registration_endpoint`が存在します（RFC 7591サポート）。
- `response_types_supported`はOAuth 2.1の場合は正確に`["code"]`です。

これらのいずれかが不足している場合、MCPサーバーはこのIdPに対してデプロイを拒否します。デプロイメントマニフェストが間違っており、コードではありません。

### RFC 9728（再説）— 保護されたリソースメタデータ

レッスン16はRFC 9728をカバーしました。本番環境でのデルタ：このドキュメントは、クライアントが*このMCP*サーバーで信頼されている認可サーバーを見つけるために見る唯一の場所です。単一のMCPサーバーは複数のIdPからトークンを受け入れることができます（1つはスタッフ用、1つはパートナー用）。RFC 9728はそのセットを宣言します。RFC 8414は各IdPがサポートするものを文書化します。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 動的クライアント登録

DCRなしでは、すべてのMCPクライアント（Cursor、Claude Desktop、カスタムエージェント）がIdP管理者とのアウトオブバンド交換が必要です。DCRを使用すると、クライアントは投稿します：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

サーバーは、`client_id`と後で更新するための`registration_access_token`で応答します：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none`は、ユーザーのデバイスで実行されるMCPクライアントの正しいデフォルトです。彼らは`client_id`のみを取得します — 流出できる`client_secret`はありません。PKCEは、公開クライアントが必要とする所有証明を提供します。

3つの本番環境の落とし穴：

- 登録エンドポイントは送信元IPで速度制限する必要があります。そうしないと、悪意のあるアクターは数百万の偽の登録をスクリプト化し、`client_id`名前空間を枯渇させます。iiiはこれを簡単にします：登録HTTPトリガーは登録者に送信する前に`auth::rate-limit`関数を呼び出します。
- `software_statement`（クライアントに保証する署名付きJWT）は、一部のエンタープライズIdPで必須です。レッスンのモックはこれをスキップします。本番環境は、ローカルホストリダイレクトURI以外の署名なし登録を拒否する検証ステップをワイヤリングします。
- `registration_access_token`はプレーンテキストではなくハッシュとして保存する必要があります。このトークンの盗難は、攻撃者がクライアントのリダイレクトURIを書き換えられることを意味します。

### RFC 8707（再説）— リソースインジケータ

レッスン16は形状を確立しました。本番環境ルール：すべてのトークンリクエストには`resource=<canonical-mcp-url>`が含まれ、MCPサーバーはすべての呼び出しで`token.aud`が独自のリソースURLと一致することを検証します。MCPサーバーが`https://notes.example.com/mcp`で到達可能な場合、正規URLは`https://notes.example.com`です — パスコンポーネントは除外されるため、単一サーバーは1つのオーディエンスの下で複数のパスをホストします。

### RFC 7636（再説）— PKCE

PKCEはOAuth 2.1で必須です。レッスンの認可コードフローは常に`code_challenge`と`code_verifier`を実行します。サーバーは検証なし、または保存されたチャレンジにハッシュされない検証を持つトークンリクエストを拒否します。

### MCP仕様2025-11-25認証プロファイル

MCP仕様（2025-11-25）は、MCPサーバーの認可層が何をする必要があるかについて正確です：

- `/.well-known/oauth-protected-resource`（RFC 9728）を公開します。
- `Authorization: Bearer ...`を通じてのみトークンを受け入れます。
- リクエストごとに`aud`、`iss`、`exp`、および必要なスコープを検証します。
- すべての401および403について、`WWW-Authenticate`を`Bearer error=...`で返します。該当する場合は`scope=`および`resource=`パラメータを含みます。
- `aud`がcanonical resourceと一致しないトークンを拒否します。
- `iss`が保護されたリソースメタデータの`authorization_servers`リストにない場合はトークンを拒否します。

OAuth 2.1ドラフトは基質です。RFC 8414/7591/8707/9728 + RFC 7636が表面です。MCP仕様はプロファイルです。

### IdP機能マトリックス

すべてのIdPが完全なMCPプロファイルをサポートしているわけではありません。以下のマトリックスは、2025-11-25仕様現在のファクトベースの機能ステートメントを文書化しています。これは*デプロイメントゲート*であり、推奨ではありません。

| IdPカテゴリ | RFC 8414メタデータ | RFC 7591 DCR | RFC 8707リソース | RFC 7636 S256 PKCE | 注記 |
|---|---|---|---|---|---|
| Self-hosted (Keycloak) | はい | はい | はい (24.x以降) | はい | このレッスンのMCPプロファイルのリファレンスIdP。エンドツーエンドすべてのRFCをサポート。 |
| Enterprise SSO (Microsoft Entra ID) | はい | はい (プレミアムティア) | はい | はい | DCR可用性はテナントティアによって異なります。デプロイ前に対象テナントで検証します。 |
| Enterprise SSO (Okta) | はい | はい (Okta CIC / Auth0) | はい | はい | DCR はAuth0（現在Okta CIC）で利用可能。クラシックOktaorg は管理者事前登録が必要。 |
| Social login IdPs (汎用) | 異なる | 稀 | 稀 | はい | ほとんどのソーシャルIdP は クライアントを静的パートナーとして扱う。DCRに頼らない。IDソースとして使用。独自のMCP対応認可サーバーを上に層状に。 |
| Custom / homegrown | 依存 | 依存 | 依存 | 依存 | 独自にリリースする場合は、完全なプロファイルをリリースします。上記の4つのRFCのいずれかをスキップすると、MCP auth契約が破れます。 |

デプロイメントマニフェストの拒否ルール：選択されたIdPが`registration_endpoint`を返さず、`code_challenge_methods_supported`で`S256`をリストしていない場合、MCPサーバーは起動を拒否します。低下モードはありません。

### iiiを使用したJWKS回転パターン

本番環境の障害モードは、古いJWKSキャッシュです。cronトリガーと`state::*`キャッシュで解決します：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

6時間ごとに、cronトリガーは`auth::rotate-jwks`を呼び出し、`<issuer>/.well-known/jwks.json`をフェッチし、`state::set("auth/jwks/<issuer>", {keys, fetched_at})`に書き込みます。バリデータは`state::get`から読み取ります。`kid`がキャッシュにないトークンは、フォールバックとして同期`auth::rotate-jwks`呼び出しをトリガーします。これは2つのケースを一度に処理します：スケジュール回転（cron）とキーオーバーラップウィンドウ（同期フォールバック）。

状態形状：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

一度に2つのキーが定常状態です。認可サーバーは次のキー（`k_2026_04`）を導入してから前のキー（`k_2026_03`）を廃止することで回転させるため、古いキーの下で発行されたトークンは期限切れまで有効なままです。キャッシュは和集合を保持します。バリデータは`kid`で選択します。

### iiiプリミティブワイヤリング（このレッスンが実際に関するもの）

5つのプリミティブがauth表面を構成します：

```python
# 1. RFC 8414メタデータドキュメント
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591動的クライアント登録
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT検証を呼び出し可能な関数として（リソースサーバーがそれをトリガーします）
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 段階的スコープのためのステップアップ発行（L16 from SEP-835）
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cronドリブン JWKS回転
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCPサーバー自体は検証を直接呼び出しません。それはします：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

この間接参照はiiiの賭けです。明日、検証をファンアウトにスワップでき、2つのIdPに並行して相談するか、スパン出力側を追加するか、正の検証をキャッシュできます。MCPサーバーは変わりません。

### 混同された代理人のウォークスルー、オーディエンスバインディング付き

サーバーA（`notes.example.com`）とサーバーB（`tasks.example.com`）は両方ともスタンドアローンの同じ認可サーバーに登録します。サーバーA は侵害されました。攻撃者がユーザーのノートトークンを取得し、サーバーBに対してそれを再生します。

サーバーBのバリデータ：

1. JWTをデコードし、`kid`でJWKSをフェッチし、署名を検証します。
2. `iss`を保護されたリソースメタデータの`authorization_servers`リストと照らし合わせチェック。（成功 — 同じIdP。）
3. `aud == "https://tasks.example.com"`をチェック。（失敗 — トークンの`aud`は`https://notes.example.com`。）
4. `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`で401を返します。

オーディエンスクレームは、プロトコルレイヤーでこの攻撃に対する唯一の防御です。パフォーマンスのためにそれをスキップすることは、最も一般的な本番環境の間違いです。バリデータはすべてのリクエスト、セッション開始時だけではなく実行する必要があります。

### 障害モード

- **古いJWKS。** キー回転後、バリデータは有効なトークンを拒否します。修正は上記のcron+フォールバックパターンです。更新ジョブなしでJWKSをキャッシュしないでください。
- **不足している`aud`クレーム。** 一部のIdPは、トークンリクエストに`resource`がない限り、`aud`を省略することをデフォルトにします。バリデータは不足している`aud`のトークンを拒否する必要があります。不在をワイルドカードとして扱いません。
- **スコープアップグレードレース。** 同じユーザーに対する2つの同時ステップアップフローは両方成功でき、異なるスコープの2つのアクセストークンを生成できます。バリデータはリクエストに提示されたトークンを使用する必要があります。「ユーザーの現在のスコープ」を検索しません — これがTOCTOUウィンドウを作成します。
- **登録トークン盗難。** 漏洩した`registration_access_token`により、攻撃者はリダイレクトURIを書き換えられます。保存時にこれらをハッシュします。クライアント側で毎回クリアテキストを提示することを要求します。疑わしいことに回転します。
- **`iss`固定されていません。** 任意の`iss`を受け入れるバリデータにより、攻撃者は独自の認可サーバーを起動し、対象オーディエンスのクライアントを登録し、トークンを発行させます。保護されたリソースメタデータの`authorization_servers`リストはアロー・リスト。それを実装します。

## 使用方法

`code/main.py`は、stdlibのPythonと、`iii.registerFunction`、`iii.registerTrigger`、`iii.trigger`、`state::set/get`を模擬する小さな`iii_mock`レジストリを使用した完全な本番環境フローを実装します。フロー：

1. 認可サーバーは`/.well-known/oauth-authorization-server`でRFC 8414メタデータを公開します。
2. MCPクライアントはメタデータエンドポイントを呼び出し、登録エンドポイントを検出します。
3. MCPクライアントは`/register`（RFC 7591）に投稿し、`client_id`を受け取ります。
4. MCPクライアントは、PKCE保護された認可コードフロー（RFC 7636）を実行し、`resource`インジケータ（RFC 8707）を使用します。
5. MCPクライアントは`Authorization: Bearer ...`でMCPサーバー上のツールを呼び出します。
6. MCPサーバーが`auth::validate-jwt`をトリガーし、`state::get`からJWKSを読み取ります。
7. cronトリガーが`auth::rotate-jwks`を起動し、状態内のJWKSを置き換えます。
8. 次の呼び出しは再起動なしで新しいキーに対して検証されます。
9. 異なるMCPリソースに対する混同された代理人の試みは、オーディエンスミスマッチで401を取得します。

ここのモックJWTはHS256を共有秘密で使用します（レッスンはstdlib のみで実行されるため）。本番環境はRS256またはEdDSAをJWKSパターンで使用します。検証ロジックはそれ以外は同じです。

## リリース

このレッスンは`outputs/skill-mcp-auth-iii.md`を生成します。MCPサーバー設定とIdP機能セットが与えられた場合、スキルはiii登録に必要なプリミティブ、JWKS回転スケジュール、スコープマッピング、IdPが完全なRFCプロファイルをサポートしていない場合に適用する拒否ルールを出力します。

## 演習

1. `code/main.py`を実行します。9ステップのフローをトレースします。`state::get`が`auth::rotate-jwks`が上書きする直前に古いデータを返す場所と、次のリクエストが新しいキーに対して検証されるようになった方法に注意してください。

2. 保護されたリソースメタデータの`authorization_servers`リストに新しいIdPを追加します。新しいIdPで署名されたトークンを発行し、バリデータがそれを受け入れることを確認します。リストされていないIdPで署名されたトークンを発行し、バリデータが`WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`で拒否することを確認します。

3. `auth::rate-limit`をiii関数として実装し、登録者が実行される前に登録HTTPトリガー内から呼び出します。`state::set("auth/ratelimit/<ip>", ...)`に保持される送信元IPごとのトークンバケットを使用します。

4. RFC 7591を読み、レッスンの`/register`ハンドラーが検証しない2つのフィールドを特定します。検証を追加します。（ヒント：`software_statement`と`redirect_uris` URIスキーム。）

5. MCP仕様2025-11-25認証セクションを読みます。レッスンのバリデータが現在出力していない`WWW-Authenticate`ヘッダー上のいくつかの規範要件を見つけます。追加します。

## キー用語

| 用語 | 人々が言うこと | それが実際に意味すること |
|------|------------------|---------------------------|
| ASM | "OAuthメタデータドキュメント" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "セルフサービスクライアント登録" | RFC 7591 `POST /register`フロー |
| JWKS | "JWT検証のための公開キー" | JSON Web Key Set、`jwks_uri`から取得、`kid`でインデックス付け |
| Resource indicator | "オーディエンスパラメータ" | RFC 8707 `resource`パラメータ、トークンを1つのサーバーにピン |
| `aud`クレーム | "オーディエンス" | バリデータがcanonical resource URLと比較するJWTクレーム |
| Confused deputy | "トークン再生" | サーバーAで発行されたトークンがサーバーBに提示される攻撃 |
| `iss`許可リスト | "信頼された認可サーバー" | 保護されたリソースメタデータの`authorization_servers`で命名されたセット |
| Key rotation | "ローリングJWKS" | オーバーラップウィンドウを持つ署名キーの定期的な置換 |
| Public client | "ネイティブまたはブラウザクライアント" | `client_secret`のないOAuthクライアント。PKCEは補います |
| `WWW-Authenticate` | "401/403応答ヘッダ" | `Bearer error=...`ディレクティブを実行し、クライアント回復を駆動 |

## 参考文献

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — このレッスンで実装するMCP auth プロファイル
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 検出契約
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公開クライアント所有証明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — オーディエンスピンニング
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — リソースサーバー検出
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 統合OAuthサブストレート
