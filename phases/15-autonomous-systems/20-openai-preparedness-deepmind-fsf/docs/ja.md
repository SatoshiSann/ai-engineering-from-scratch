# OpenAI Preparedness FrameworkとDeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2(2025年4月)は Research Categories を導入します — Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards — Tracked Categories とは異なります。Tracked Categories は Capabilities Reports と Safeguards Reports をトリガーし、Safety Advisory Group によってレビューされます。DeepMind の FSF v3(2025年9月、2026年4月17日に追加された Tracked Capability Levels)は自動性を ML R&D と Cyber ドメイン(ML R&D 自動性レベル1 =人間+AIツール対競争的コストで AI R&D パイプラインを完全に自動化)に折り畳みます。FSF v3 は明示的に instrumental-reasoning 悪用の自動化による監視を経由してしまくばりした整列に対処しています。正直なノート:PF v2 の Research Categories(Long-range Autonomy を含む)は自動的に緩和をトリガーしません。ポリシー言語は「可能」です。DeepMind 自体は自動化された監視「長期的に十分なままではない」と言った、 instrumental 推論が強化する場合。

**タイプ:** 学習
**言語:** Python(標準ライブラリ、3フレームワーク決定テーブル diff ツール)
**前提条件:** Phase 15 · 19(Anthropic RSP)
**所要時間:** 約45分

## 問題

レッスン19は Anthropic のスケーリング ポリシーを密接に読みました。このレッスンは OpenAI と DeepMind を読むことによって画像を完成させます。3つのドキュメントはいつ frontier ラボ一時停止またはモデルをゲートすべきかという同じ質問に対処する従兄弟アーティファクトで、小さいセットのカテゴリとその中で分岐する小さなカテゴリに収束します。

収束:3つはすべて long-range autonomy を追跡するカテゴリクラスとしてラベル付けします。3つはすべて欺まくような動作(配置不正、sandbagging)をリスクの特定のクラスとして認める。3つはすべて内部レビュー本体があります。分岐:OpenAI はカテゴリを「Tracked」(必須の緩和)と「Research」(自動トリガーなし)に分割します。DeepMind は自動性をそれを別に命名するのではなく2つのドメインに折り畳みます。ラボはウォームの名前付けします対 Research、または Critical 対 Moderate、または Tier-1 対 Tier-2。能力がどのバケットに住んでいるかの運用上の結果はラボ全体で異なります。

彼らを一緒に読むことが有用な運動です。同じ能力は Anthropic で「必須の緩和」、OpenAI で「監視されたが緩和をトリガーしない」、DeepMind で「特定のドメインで追跡」することができます。ポリシー姿勢が重要です。

## コンセプト

### OpenAI Preparedness Framework v2(2025年4月)

構造:

- **Tracked Categories**:Capabilities Reports(モデルが実行できるもの)と Safeguards Reports(どの緩和が順番か)をトリガーします。Safety Advisory Group によってデプロイメント前にレビューされます。
- **Research Categories**:ラボが追跡しているがまだ特定の緩和にコミットしていないモデル能力。Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards を含めます。

v2 のキーフレーミング:Research Categories は自動的に緩和をトリガーしません。ポリシー言語は「可能」緩和です。カテゴリの位置が Tracked 対 Research は「ゲート」と「観察」の間のヒンジです。

### DeepMind Frontier Safety Framework v3(2025年9月。2026年4月17日に追加された Tracked Capability Levels)

構造:

- **Critical Capability Levels(CCLs)**:5つのドメイン全体の能力閾値:Cyber、Bio、ML R&D、CBRN、Autonomy(ML R&D と Cyber に折り畳み)。
- **Tracked Capability Levels**:2026年4月に追加された追加粒度。具体例:ML R&D 自動性レベル1 =人間+AIツール対競争的コストで AI R&D パイプラインを完全に自動化する。
- **欺まくされた整列監視**:instrumental-reasoning 悪用の自動化された監視のための明示的なコミットメント。

自動性のフレーミングは OpenAI と異なります。DeepMind はトップレベルのドメインとして「Autonomy」を維持しません。それは自動性が危害を引き起こすドメイン(ML R&D と Cyber)に折り畳みます。議論はドメインなしの自動性が能力ですが、リスクなしです。カウンター議論は、ドメイン全体の自動性が フレームワークが命名すべきメタリスクであるということです。

### 3つがすべて収束する

- 内部 Safety Advisory Group(名前付けされた Anthropic SAG、OpenAI SAG、DeepMind 内部委員会)。高能力モデルのデプロイメント前にレビューします。
- 明示的に欺まくされた整列/整列偽造をリスク クラスとして言及します。
- 宣言されたスケジュール上の常設アーティファクト(Anthropic:Frontier Safety Roadmap、Risk Report。OpenAI:Capabilities と Safeguards Reports。DeepMind:FSF 更新スケジュール)。
- 監視のみの防御に上限があることの認識。DeepMind は明示的です:「自動化された監視は長期的に十分なままではありません」。

### 彼らが分岐するところ

- **Anthropic**:v3.0で一時停止コミットメント削除。AI R&D-4 閾値は命名された次のゲート。
- **OpenAI**:Tracked 対 Research 分割。Research Categories(Long-range Autonomy を含む)は自動的にゲートしません。
- **DeepMind**:自動性は他のドメインに折り畳み。Tracked Capability Levels は 2026年4月に粒度を追加します。

### Sandbagging:3つすべてを複雑にする特定の能力

Sandbagging(モデル戦略的にアンダーパフォーム評価上)は OpenAI の Research Categories にあります。Anthropic の RSP v3.0 はそれを評価コンテキストギャップを経由して対処します(レッスン1)。DeepMind はそれを FSF v3 の欺まくされた整列監視を経由して対処します。

モデルが評価上で sandbagging 場合、すべてのフレームワークの能力閾値は過小評価されます。フレームワークは測定が機能する場合にのみ機能します。これが外部測定(レッスン21、METR)と敵対的評価がラボ自己評価の他に必要な理由です。

### ポリシー読み取りスキル

- 特定場所:あなたが懸念しているすべての能力がポリシー内で見つけられるべきです。見つけられない場合、ポリシーはそれをカバーしていません。
- 分類:それは Tracked(緩和をトリガー)ですか、それとも Research(追跡されたが緩和をトリガーしない)ですか?OpenAI はこれを命名します。Anthropic と DeepMind は彼ら自身の同等のものを持っています。
- スケジュール:ポリシーは宣言されたスケジュール上で更新されますか、それとも特定のイベント後にのみ更新されますか?宣言されたスケジュールはより強い。
- 独立:外部レビューは必須ですか、それともオプションですか?Anthropic は Apollo と US AI Safety Institute とパートナー。OpenAI は METR とパートナー。DeepMind は主に内部 SAG とパートナー。

## それを使う

`code/main.py`は小さな決定テーブル diff ツールを実装します。能力(自動性、欺まくされた整列、R&D 自動性、cyber uplift など)を与えられ、3つのポリシーがそれぞれの能力を分類し、どの緩和をトリガーするかを出力します。これは読み取り支援ですが、ポリシーツールではありません。

## それを配送する

`outputs/skill-cross-policy-diff.md`は特定の能力の交差ポリシー比較を生成し、3つのフレームワークをリファレンスとして使用します。

## 演習

1. `code/main.py`を実行します。diff ツール出力が少なくとも2つの能力ごとに確認できるポリシーと一致することを確認します。

2. OpenAI Preparedness Framework v2 を全く読みます。各 Research Category を識別します。各ごとに、なぜ Tracked ではなく Research にあるかについて1つの文を書きます。

3. DeepMind FSF v3 を全く、2026年4月の Tracked Capability Levels 更新を読みます。ML R&D 自動性レベル1 の特定の評価基準を特定します。あなたはそれを外部でどのように測定しますか?

4. Sandbagging は OpenAI の Research Categories にあります。sandbagging モデルに実際の能力を明かすように強制する評価を設計します。レッスン1の評価コンテキストゲーミング議論をリファレンスします。

5. 特定の能力(あなたの選択)について3つのポリシーを比較します。どのポリシーの分類が最も厳密であり、どれが最も少ないかを命名します。ソーステキストで正当化します。

## 重要な用語

| 用語 | 人が言うこと | 実際の意味 |
|---|---|---|
| Preparedness Framework | 「OpenAI のスケーリング ポリシー」 | PF v2(2025年4月)。Tracked 対 Research カテゴリー |
| Tracked Category | 「必須の緩和」 | Capabilities + Safeguards Reports をトリガーします。SAG レビュー |
| Research Category | 「監視のみ」 | 追跡されたが自動的な緩和なし。Long-range Autonomy を含む |
| Frontier Safety Framework | 「DeepMind のスケーリング ポリシー」 | FSF v3(2025年9月) + Tracked Capability Levels(2026年4月) |
| CCL | 「Critical Capability Level」 | DeepMind ドメイン毎の閾値(Cyber、Bio、ML R&D、CBRN) |
| ML R&D 自動性レベル1 | 「R&D 自動性」 | 競争的コストで AI R&D パイプラインを完全に自動化する |
| Sandbagging | 「戦略的アンダーパフォーマンス」 | モデルはアンダーパフォーム評価上。OpenAI Research Categories にあります |
| Instrumental reasoning | 「目的手段の推論」 | 目標を達成する方法についての推論。DeepMind 監視のターゲット |

## 参考文献

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 アナウンスメント。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — フル ドキュメント。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 アナウンスメント。
- [DeepMind — Updating the Frontier Safety Framework(2026年4月)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels 追加。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF フォーマットの Risk Report の例。
