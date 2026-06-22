# なぜマルチエージェント?

> 1つのエージェントは壁に当たります。スマートな手段は、より大きなエージェントではありません - より多くのエージェント です。

**タイプ:** 学習
**言語:** TypeScript
**前提条件:** Phase 14(エージェント エンジニアリング)
**所要時間:** 約60分

## 学習目標

- 単一エージェント上限(コンテキスト オーバーフロー、混合専門知識、順序ボトルネック)を特定し、複数のエージェント に分割することが正しい手段である場合を説明します
- オーケストレーション パターン(パイプライン、平行ファンアウト、スーパーバイザー、階層的)を比較し、特定のタスク構造に対する正しいものを選択します
- 明確なロール境界、共有状態、通信契約を備えたマルチエージェント システムを設計します
- マルチエージェント複雑性(遅延、コスト、デバッグ難易度)対単一エージェント単純性のトレードオフを分析します

## 問題

Phase 14 で単一エージェントを構築しました。それは機能します。ファイルを読み、コマンドを実行し、API を呼び出し、結果を理由付けることができます。その後、あなたはそれを実際のコードベースに指します:200ファイル、3つの言語、インフラストラクチャに依存するテスト、およびコードを書く前に外部 API を研究する要件。

エージェントは窒息します。LLM が愚かであるためではなく、タスクが1つのエージェント ループが処理できるものを超えるためです。コンテキスト ウィンドウはファイル コンテンツで埋まります。エージェント は、40ツール呼び出し前に読んだものを忘れます。それは研究者、コーダー、レビュアーであり、3つすべてを貧しく実行しようとします。

これは単一エージェント上限です。タスクが次のものを必要とするたびにそれに当たります:

- **1つのウィンドウに適合させるより多くのコンテキスト** - 50ファイルを読むことは200k トークンを超えて吹きます
- **異なる段階での異なる専門知識** - 研究はコード生成とは異なるプロンプトを必要とします
- **並列に発生する可能性のある作業** - 3つのファイルを順序付きで読むのはなぜですか、同時に読むことができるとき?

## コンセプト

### 単一エージェント上限

単一エージェントは1つのループ、1つのコンテキスト ウィンドウ、1つのシステム プロンプトです。イメージ:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

3つのもの破:

1. **コンテキスト飽和** - ツール結果が積み重ねます。ターン30までに、エージェントはファイルコンテンツ、コマンド出力、以前の推論の150kトークンを消費しました。ターン5からの重大な詳細は失われます。

2. **ロール混乱** - 「あなたは研究者、コーダー、レビュアー、テスターです」と言うシステム プロンプトは、半分研究し、半分コード、決してレビューを終えることはないエージェントを生成します。

3. **順序ボトルネック** - エージェントはファイル A を読み、ファイル B を読み、ファイル C を読みます。3つの順序 LLM コール。3つの順序ツール実行。並列性なし。

### マルチエージェント ソリューション

作業を分割します。各エージェントに1つのジョブ、1つのコンテキスト ウィンドウ、そのジョブ用にチューニングされた1つのシステム プロンプトを提供します:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

各エージェントは持っています:
- フォーカスされたシステム プロンプト(「あなたはコード レビュアーです。あなたの唯一のジョブはバグを見つけることです」)
- その独自のコンテキスト ウィンドウ(他のエージェントの作業で汚染されていません)
- 明確な入出力契約(研究ノートを受け取り、コードを出力)

### 実際のシステムがこれをする

**Claude Code サブエージェント** - Claude Code が `Task` でサブエージェントをスポーンするとき、スコープされたタスクを備えた子エージェントを作成します。親はコンテキストをクリーンに保ちます。子はフォーカスされた作業を実行し、概要を返します。

**Devin** - プランナー エージェント、コーダー エージェント、ブラウザ エージェントを実行します。プランナーはステップに作業を分割します。コーダーはコードを書きます。ブラウザーはドキュメント を研究しています。各にはコンテキストを分離します。

**マルチエージェント コード チーム(SWE-bench)** - SWE-bench の最高パフォーマー システムは、コードベースを読む研究者、修正を設計するプランナー、それを実装するコーダーを使用します。単一エージェント システムはより低いスコアです。

**ChatGPT Deep Research** - 複数の検索エージェント を並列でスポーン、各異なる角度を探索、その後結果を合成します。

### スペクトラム

マルチエージェント はバイナリではありません。それはスペクトラムです:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**単一エージェント** - 1つのループ、1つのプロンプト。単純なタスク に適切です。

**サブエージェント** - 親は子を特定のサブタスク用にスポーンします。親はプランを維持します。子はレポート バック。これは Claude Code が実行することです。

**パイプライン** - エージェント は順序に実行します。エージェント A の出力はエージェント B の入力になります。ステージング ワークフロー(研究 ->コード ->レビュー ->テスト)に適切です。

**チーム** - エージェント は共有メッセージバス との平行で実行します。各にはロール があります。オーケストレーター は調整します。異なるスキルが同時に必要な場合に適切です。

**Swarm** - 多くの同等または近い同等エージェント 共有状態で。固定オーケストレーターなし。エージェント はキューから作業を拾います。高スループット並列タスクに適切です。

### 4つのマルチエージェント パターン

#### パターン1:パイプライン

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

各エージェント はデータを変換し、それを前方に渡します。簡単に理由を述べるべきです。1つのステージでの失敗は残りをブロック します。

#### パターン2:ファンアウト / ファンイン

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

並列エージェント全体にわたって作業を分割し、結果を統合します。独立したサブタスク に分解するタスク に適切です。

#### パターン3:オーケストレーター-ワーカー

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

スマート オーケストレーターは何をするかを決定し、ワーカーに委譲し、結果を合成します。オーケストレーター 自体はワーカーをスポーンするためのツールを備えたエージェント です。

#### パターン4:ピア Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

中央オーケストレーターなし。エージェント ピア-ツー-ピア通信。決定は相互作用から出現します。デバッグは難しいですが、多くのエージェント にスケール です。

### マルチエージェント を使用しない場合

マルチエージェント は複雑さを追加します。エージェント 間のすべてのメッセージは、潜在的な失敗ポイントです。デバッグは「1つの会話を読む」から「5つのエージェント 全体のメッセージをトレース」に移動します。

**単一エージェント に留めます:**
- タスクは1つのコンテキスト ウィンドウに適合します(~100k トークン未満の作業データ)
- 異なるステージ用に異なるシステム プロンプトは必要ではありません
- 順序実行は十分に高速です
- タスクは、それを分割することがより多くのオーバーヘッドを追加するのに十分な複雑です

**複雑性コスト:**
- すべてのエージェント 境界は、損失のある圧縮ステップです:エージェント A の完全なコンテキストはエージェント B へのメッセージに要約されます
- コーディネーション ロジック(誰が何を、いつ、どのような順序で)はそれ自体がバグの原因です
- 遅延増加:N エージェント は最小限 N 順序 LLM コール、それらが再度行き来する必要がある場合、より多くを意味します
- コスト倍:各エージェント 個別にトークンをバーン します

経験則:タスク が20ツール呼び出み未満であり、100kトークンに適合する場合、単一エージェント に保ちます。

## それを構築する

### ステップ1:過負荷単一エージェント

ここにすべてをしようとする単一エージェント があります。1つの大規模なシステム プロンプトと、研究、コード、レビューを保持している1つのコンテキスト ウィンドウがあります:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

このアプローチの問題:
- コンテキスト ウィンドウ は、すべてのステージで成長します。レビュー ステップで、それは研究ノート と コードと以前の推論を含みます。
- システム プロンプト は汎用です。各ステージに対してチューニングできません。
- 何も平行で実行しません。

### ステップ2:スペシャリスト エージェント

さて、それを分割します。各エージェント は1つのジョブを得ます:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

各スペシャリスト はフォーカスされたプロンプトを持っています。各はそれが必要な入力だけでクリーン コンテキスト ウィンドウを得ます。

### ステップ3:メッセージの通じてコーディネート

スペシャリスト を明示的なメッセージ パッシング でワイヤーします:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

各エージェント はそれに向けられたメッセージだけを受け取ります。コンテキスト汚染なし。研究者 の50kトークンのドキュメント読み取りはレビュアー のコンテキストに決して入りません。

### ステップ4:比較

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

マルチエージェント バージョンはより多くの合計トークンを使用します(3つのエージェント、3つの独立した LLM コール)が、各エージェント のコンテキストはクリーンに留まります。各ステージの品質はシステム プロンプトが特化しているため改善します。

## それを使う

このレッスンはマルチエージェント に移動するかどうかを決定するための再利用可能なプロンプトを生成します。`outputs/prompt-multi-agent-decision.md` を参照してください。

## 演習

1. 4番目のスペシャリスト を追加:コーダーからコードを受け取る「テスター」エージェント とレビュアーからのレビュー フィードバック、その後テストを書きます
2. パイプラインを変更して、レビュアーは修正ループ用のコーダーにフィードバックを戻すことができます(最大2ラウンド)
3. 順序パイプラインをファンアウト に変換:研究者と「要件アナライザー」エージェント を平行で実行、その後それらの出力をマージしてからコーダーに渡します

## 重要な用語

| 用語 | 人が言うこと | 実際の意味 |
|------|----------------|----------------------|
| Swarm | 「AI エージェント のハイブマインド」 | 共有状態と固定リーダーなしのピア エージェント のセット。動作はローカル相互作用から出現します。 |
| オーケストレーター | 「ボスエージェント 」 | 他のエージェント をスポーン および管理するためのツールを持つエージェント 。計画およびデリゲートしますが、実際の作業をしない場合があります。 |
| コーディネーター | 「トラフィック警察」 | 非エージェント コンポーネント(多くの場合、LLM ではなくコード)で、ルールに基づいてエージェント 間でメッセージをルーティングします。 |
| コンセンサス | 「エージェント は同意します」 | 複数のエージェント が進める前に同意に到達する必要があるプロトコル。矛盾した出力が解決を必要とする場合に使用されます。 |
| 新興動作 | 「エージェント はそれを自分で理解しました」 | エージェント の相互作用から生じるが、明示的にプログラムされなかったシステムレベルのパターン。有用または有害である可能性があります。 |
| ファンアウト / ファンイン | 「エージェント 用のマップ削減」 | 並列エージェント 全体にわたってタスクを分割(ファンアウト)、その後、それらの結果を組み合わせる(ファンイン)。 |
| メッセージ パッシング | 「エージェント は相互に話す」 | エージェント 間の通信メカニズム:1つのエージェント から別のエージェント に送信された構造化データ、共有コンテキスト ウィンドウを置き換えます。 |

## 参考文献

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - マルチエージェント パターンの調査
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft のマルチエージェント 会話フレームワーク
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code が Task で委譲する方法
- [CrewAI documentation](https://docs.crewai.com/) - ロール ベースのマルチエージェント フレームワーク
