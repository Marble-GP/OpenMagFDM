# 次回作業メモ（更新：2026-10-16）

## DIRECT＋Anderson比較を実行中（2026-10-16 08:00 JST開始）

- ユーザーの再開指示後、角度予測の棄却を `d82fee0` にコミットし、
  DIRECT専用の比較回帰・プロトコルを `3ec2b29` にコミットしてから起動した。
- 出力：`../bench_results/direct_anderson_20261016_01`。
  `progress.json`、`summary.json`（実行完了ごとに生成）、
  `launcher_20261016_080044.stdout.log` / `.stderr.log` を確認する。
- 起動時ランチャーPIDは2680、実Pythonは15096、最初のsolverは49192。
  venvランチャーと実Pythonの親子関係であり、比較の二重起動ではない。
  PIDは再利用されるため、停止操作時はコマンドライン・親子関係を再確認する。
- `--prepare-only` の固定条件確認は成功。最初のDIRECT／AA OFFの進行を確認。
  OFF/ON、ON/OFF、OFF/ONの計6本、12ステップずつ。予測・優先4はOFF。
- 目安30〜60分だがAAの収束次第で変わる。各実行の既存ガードは7200秒・8GiB。
  不合格時は `stopped_for_review` と診断を保持し、自動再試行しない。
  計算中に入力・ハーネス・固定バイナリを変更しない。他のソルバー／重いビルドを重ねない。
- [ ] 終了後に6本の厳密収束、AA作動・採否、反復・追加費用・総時間、波形、メモリ、背景負荷をレビュー。
  `passed=true` だけでは高速化採用としない。全周検証へは自動で進まない。

## 角度予測は棄却、次はDIRECT＋Anderson（2026-10-16）

- ユーザー指示により角度予測→Anderson併用の実験に着手。公開YAML・既定値は変更しない。
- 実装：固定座標の `2*Az_prev - Az_older` を現在材料則の真残差で検査し、
  直接ウォームスタートより悪ければ復帰。履歴は連続する厳密収束2点のみ。
  まず固定角度の単一polar bandを対象とし、格子・境界・刻み変更、失敗、再実行で無効化する。
- 別ビルド `build_nonlinear_prediction/` を作成。旧固定バイナリは保持。
  Windowsビルド、CTest 4/4、release-contract、ベンチPython 38/38が通過。
- 出力：`../bench_results/angle_prediction_20261015_01`。2026-10-15 20:11 JSTに完了。
  全6実行・72ステップが厳密収束し、Az/μのファイルハッシュ・数値履歴・鎖交磁束が一致。
  予測30候補すべてを棄却（候補残差はDIRECTの2.22～2.23倍）。全実行211外反復・3081内反復。
  中央値341.554→321.702秒という実測差はあるが、予測は一度も採用されておらず収束改善としない。
  背景負荷の差もある。`passed=true` は安全性・数値検査の通過であって採用判定ではない。
- [x] 角度予測結果をレビュー。ユーザー判断により固定beta=1の予測を棄却し、DIRECTを選抜。
  棄却した研究用実装・回帰試験は再現用に保持するが、通常経路や次のAA試験では有効化しない。
- [x] 角度予測の実装・試験・棄却報告・証跡を `d82fee0` にコミット済み。
- [ ] DIRECT＋Anderson OFF/ONを独立に3組実行（深さ3・beta 0.3固定）。
  `--stage anderson --initializer direct --predictor-reference ../bench_results/angle_prediction_20261015_01`
  と新規出力先を使う。同一バイナリ・入力・12スレッド・chunk=1・優先2+3を維持。
  優先4と角度予測はOFF。未収束・数値対照不一致・品質不合格なら停止し診断を保持する。
  完了済み角度予測の出力先で再実行や集計上書きをしない。
- Andersonの新規出力先：`../bench_results/direct_anderson_20261016_01`。
  起動前にハーネスのDIRECT専用比較・AA作動／未作動・最終残差・二重起動防止を追加検証し、
  ベンチPython 42/42が通過。ソルバー本体・固定バイナリは変更しない。
  起動後はこの出力先の `progress.json` とlauncherログを確認し、重複起動しない。
- [ ] AA採否・追加処理時間・反復数・総時間・波形・メモリをレビュー。有用な候補のみ全周／大変位へ。
  12点だけで周期EMFを合格としない。効果がなければDIRECT単独で今回の仮説検証を締める。
- ソース差分・新規ヘッダー・ハッシュは元manifestに記録。生データ・固定バイナリは変更しない。
  詳細は [研究報告](research/NONLINEAR_INITIALIZATION_EXPERIMENTS.md)、[コマンド](../benchmarks/solver_speed/README.md)。

## 完了レビュー（2026-10-15）

- 長時間ベンチは07:47:46 JSTに完了。全8実行・568ステップが厳密収束し、4組すべての
  初期化・波形・メモリ判定が合格。全周は182.34分→110.96分（39.15%短縮）。
- 30°刻みは0.59%差にとどまり、外反復589→621。背景負荷にも差があるため高速化実績としない。
- 全周の最大差は鎖交磁束RMS 0.001742%、EMF RMS 0.023169%、EMF高調波0.006172%。
  最大RSS約1.727 GiB。比較条件・限界・ハッシュは
  [実験報告](research/NONLINEAR_INITIALIZATION_EXPERIMENTS.md#completed-full-revolution-and-large-displacement-gate-2026-10-15)に保存。
- ハーネス・証跡・報告は `f3ebdfa` にコミット済み。元データ・バイナリは変更せず保持する。
  完了済み出力先での再実行や集計上書きは不要。
- この完了レビュー時点の次工程は **角度予測→安全策付きAnderson併用**（その後の着手状況は上記）。
  優先4は棄却のままOFF、磁気回路初期化は保留、公開の既定値は変更しない。
- Windowsの再試験：ベンチ28/28、C++ CTest 3/3、WebUI／MCP 68/68、
  保存済みソルバーの別コピーによるrelease-contractが通過。
  MCPの配布exe・実モデル・実ブラウザー・他OS・長時間保持量の受入試験は未完了。

## 起動時の記録（2026-10-15、計算は完了済み）

- ユーザーの長時間計算実施指示により、下記の1時間制限を今回に限り解除して起動した。
- 保存済みコマンドの `--resume --prepare-only` は `VALIDATED 4 pairs; no solver started.` で成功。
  バイナリ・DLL・入力・ハーネス・電源プランの固定条件は変更していない。
- 2026-10-15 01:20 JSTに `--resume` で全4組（全周＋大変位3条件、計8実行）を起動。
  起動時のランチャーPIDは4520、Python本体PIDは32712。PIDは将来再利用されるため、
  停止等の操作時には必ずコマンドラインと親子関係を再確認すること。
- 出力先は `../bench_results/direct_validation_20261011_01`。
  起動ログは同ディレクトリの `launcher_20261015_012027.stdout.log` と `.stderr.log`。
  進捗は `progress.json`、完了・品質判定はハーネスのsummaryと各実行記録を確認する。
- 起動時は他のMagFDMsolverなし、Cドライブ空き約1,059 GiB、電源プランは記録と同じバランス。
- MCP開発は並行するが、パッケージング・実ソルバーを使うMCP試験・重い検証は計算終了後へ延期する。
  背景負荷の計測結果を確認し、並行開発中の速度値を無条件にアイドル計測と同一視しない。

以下は起動前に作成した手順・見積もりの履歴。

## 履歴：直接ウォームスタート検証の保留判断（2026-10-11）

ユーザーの条件：1時間程度で終わる見込みなら開始し、超えるなら保留する。
今回は **計算を開始していない**。大変位だけの抜粋実行も行っていない。

### 所要時間の根拠

- 既存の接線方式（通常の線形初期化）の248点全周実績は
  **11,086.565秒＝約3時間5分**。
  出典：`research/results/selected_secant_20261009_provenance.json` の
  `full_runs / waveform_candidate`。今回の対照に近い過去版であり、
  今回のバイナリで再計測した時間ではない。
- 同じバイナリによる12点の初期化比較は、中央値
  LINEAR **531.208秒**、DIRECT **328.005秒**（約38.25%短縮）。
  出典：`research/results/direct_initialization_20261010_summary.json`。
- この比率を全周へ仮に外挿すると、DIRECTは約1時間54分、
  全周の対照＋DIRECTだけで **約5時間**。全角度で同じ短縮率とは限らない。
- 大変位3ケースの対照＋DIRECTは、通常刻みの12点実績を単純に使えば
  約43分。ただし、大変位での収束性は未検証であり、長引く可能性がある。
- 全体の暫定目安は **約6時間、夜間など十分な余裕のある時間帯向け**。
  1時間以内の完了は見込めない。失敗時は早期停止するが、これは完了ではない。

### 当時の再開TODO（実行・レビュー済み）

- [x] テストハーネス・テスト・文書の差分を確認し、`f3ebdfa` に独立コミット。
  既存の実装・3組の追試結果は `03a416f` にコミット済み。
- [x] 保存済みソルバーバイナリ・DLL・入力・ハーネスを変更しない。
  UI作業のためにソルバーを再ビルド／上書きしない。
- [x] 電源プランと十分な空きディスク容量を確認。他のソルバーと同時実行しない。
- [x] 下記コマンドで保存済みプロトコルを再検証して実行する。
- [x] 全248点・大変位各12点の厳密収束、初期化選択、時間、メモリ、背景負荷をレビュー。
- [x] 全周の鎖交磁束・周期接続EMF・高調波の品質ゲートを確認。
  大変位は同角度の鎖交磁束のみ評価し、粗い点列のEMF品質を主張しない。
- [ ] 角度予測は完了・棄却。残る既存の安全策付きAnderson併用をDIRECTと比較する。
  優先4の場再利用はOFF、磁気回路初期化は保留を継続する。

作業ディレクトリ：リポジトリの `OpenMagFDM`。
準備済み出力先：`../bench_results/direct_validation_20261011_01`。
当時は設定4件とmanifestのみ。現在は完了済みの結果があるため、この出力先を新規実験に使わない。

```powershell
.\build_solver_speed\bench_py312\Scripts\python.exe benchmarks/solver_speed/run_initialization_validation.py --exe build_nonlinear_initialization/Release/MagFDMsolver.exe --reference ../bench_results/direct_initialization_20261010_01 --output ../bench_results/direct_validation_20261011_01 --resume --prepare-only
.\build_solver_speed\bench_py312\Scripts\python.exe benchmarks/solver_speed/run_initialization_validation.py --exe build_nonlinear_initialization/Release/MagFDMsolver.exe --reference ../bench_results/direct_initialization_20261010_01 --output ../bench_results/direct_validation_20261011_01 --resume
```

ハッシュ不一致や未完了の試行がある場合、既存ディレクトリを消して再開しない。
原因を確認して記録し、必要なら別の新規出力先を準備する。
詳細：[実験報告](research/NONLINEAR_INITIALIZATION_EXPERIMENTS.md)、
[実行手順](../benchmarks/solver_speed/README.md)。

## YAML入力支援の強化（初回実装済み、以下は提案時の記録）

追記（2026-10-13）：下記提案の初回実装を実施した。
最新の範囲・テスト結果・残課題は [実装計画と結果](YAML_INTELLISENSE_PLAN.md) を参照。
以下の「現状確認」は着手前の記録として保持する。

### 現状確認

WebUIは既にAce Editor上で独自の補完・候補説明・スニペットを提供している。
`webui/public/app.js` の `initializeConfigEditor` / `yamlCompleter` と
`webui/public/yaml-schema.json` が実装箇所。
`validateYAML` は構文チェック、`validateAgainstSchema` は主に行単位の
キーワードチェックを行う。現状のschemaは `keywords` 等を持つ独自辞書で、
そのままVS Code等へ渡す標準JSON Schemaではない。

既知の改善点：親の `children` に含まれなくても、グローバルに知られた
キーなら警告しない条件があるため、キーを誤った階層に置いた場合の検出に穴がある。
また、接線試行で使用している `jacobian` / `line_search_objective` はこの辞書にない。
実験機能を無条件に公開補完へ追加せず、公開仕様と研究設定を区別して整理する。

### 最初の実装範囲の提案

1. 補完・検証ロジックをテスト可能な単位に分離し、ネスト、配列、任意材料名、
   引用符、コメントを含む回帰ケースを作る。まず誤警告と見逃しを抑える。
2. 型・列挙値・範囲・配置先の検証を、実際のC++入力仕様と照合して追加する。
   数式を受け付ける項目に一律の数値型制約をかけない。
3. `material_a` / `material_b` へ、その設定で有効な材料名を候補表示する。
   片側コイル仕様を維持し、両側必須へ戻さない。
4. `r_start < r_end` や `eta_min <= eta_max` など設定間の関係を診断する。
   警告と実行不能エラーを区別し、入力値を勝手に変更しない。
5. 説明を「単位・省略時の挙動・設定を変えるとどうなるか」中心に整理する。
   古い単一ベンチの速度値を一般的な保証として表示しない。

標準JSON Schemaへの整理とエディター外の入力支援は、その後の共通化候補。
既存のAceを使った第一段階にエディターの全面置換は不要。

## MCP連携：初期実装済み・受入検証を継続（2026-10-15）

ユーザー承認済みの組み込みMCP計画に沿って、Node 24／公式SDK v2、HTTPとstdio橋渡し、
固定ユーザー領域、接続・権限管理、リビジョン付きファイル操作、入力固定・冪等ジョブ開始、
ストリーミングバックアップ、GUIヘルプ・活動表示まで実装した。
既存49件＋MCP関連19件の軽量回帰を実施。ソルバー本体・実験バイナリは変更していない。
詳細・制限・再現コマンドは [MCP実装と受入チェック](MCP_INTEGRATION.md)。

- [x] 長時間ベンチマーク終了と記録保存を確認。パッケージング・実解析試験を再開できる。
- [ ] `webui/tools/build-mcp-package.cjs` で現行exeを再生成。
  初回生成物 `build_mcp/webui-server.exe` は後続修正を含まないため完成品とみなさない。
- [ ] `webui/tests/packaged_mcp_smoke.cjs` でWindows exeのHTTP／stdioを確認。
  `--solver` を付けた小モデル完走と、未収束警告の区別も追加確認。
- [ ] 手動専用 `mcp-validation.yml` でLinux／macOSのネイティブ接続確認（現時点で未起動）。
- [ ] 実ブラウザーの接続管理・外部変更通知・狭幅表示を確認。
- [ ] 長時間連続接続／転送／画像加工でメモリ・プロセス・一時ファイル保持を確認。
- [ ] GUI全テンプレート共通化、異常終了した転送の清掃、台帳200件到達時の管理機能を仕上げる。
- [ ] Jimp依存由来の残存moderate脆弱性4件と、大きな画像加工中のUI応答性を精査。
- [ ] 上記の未完了事項が解決するまで、全計画の完了や配布承認を主張しない。

以下は実装前の候補メモ。

最初はモデル／設定／結果の列挙、設定検証、収束結果の取得など読み取り中心にする。
計算起動・停止・ファイル削除は別段階とし、ユーザー識別、対象パス制限、
同時実行制御、実行権限と確認方法を決めてから実装する。
設定検証をUIから独立させておけば、将来MCPでも共用する設計が可能。
