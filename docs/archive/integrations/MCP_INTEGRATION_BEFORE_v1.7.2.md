> Historical integration and acceptance record. Current setup: [MCP manual](../../MCP_INTEGRATION.md).

# 組み込みMCP：接続・操作・検証メモ

更新：2026-10-19。**v1.7.0の組み込みMCP仕様・検証記録。**

### Windows Git Bash起動の修正

v1.7.0タグのWindows MCP検証では、PATH上のGit Bash版`whoami.exe`が選ばれ、
Windows専用の`/user`引数を拒否して管理機能の初期化が失敗した。
修正版は`SystemRoot/System32`の`whoami.exe`と`icacls.exe`を絶対パスで実行する。
アクセス権検査を省略する回避策や、タイムアウトの延長は行わない。
配布試験は初期化失敗コードのみを報告し、ペアリングコードを含み得る生ログは公開しない。
再発防止のためWindowsの配布試験は引き続きGit Bash経由で実行する。
最初の失敗run（39144511106、b5fbdde）はこの修正を含まないため、旧ジョブ再実行では直らない。
ユーザー承認により未公開のv1.7.0タグを修正コミットへ再発行する。
ローカル回帰91件と、Git Bash経由のWindows配布exe・HTTP/stdio実解析は成功。

基本接続と小モデルの解析受入試験は完了。後述の未検証事項は継続課題であり、
本バージョンで全ての初期計画が完了したという意味ではない。

### 起動とブラウザーの信頼

OpenCodeではプロジェクトの `opencode.json` に次の設定を既存設定とマージする。
`command` 配列はGUIのSettingsに表示された `stdio.command` と `stdio.args` を順に並べる。
下記は配布版の例。ソース版はNodeの実行ファイル、launcher.js、残りの引数の順になる。

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "openmagfdm": {
      "type": "local",
      "command": ["C:/OpenMagFDM/webui-server.exe", "--mcp-stdio", "--connection", "GUIの接続ID"],
      "enabled": true,
      "timeout": 15000
    }
  }
}
```

サーバーを起動したままOpenCodeを再接続し、`opencode mcp list`とMCPの`workspace_status`で確認する。
同じWindowsユーザーで実行し、初回はWSLを経由しない。接続IDはトークンではない。
接続設定ファイルにトークンを貼り付ける必要はない。

2026-10-19検証：軽量回帰90件成功。Windows配布exeのHTTP／stdio実解析は、
SOLVER_PATHを削除しソルバーをサーバーの隣へ配置した構成で成功（正常・未収束警告・CSV取得）。
管理Cookie属性・CSRF・信頼の復元／期限／解除・MCP待受の再起動復元も自動試験で確認。
実ブラウザーの操作感とLinux/macOSの配布確認は別途必要。

- Pair browser時に「Trust this browser for 30 days」を選ぶと、サーバー再起動後も管理画面を利用できる。
  明示選択しない場合はセッションCookieで、サーバー再起動または最長24時間で再ペアリングが必要。
- 信頼CookieはHttpOnly／SameSite=Strict、管理API専用パス。ローカルHTTPのためSecure属性は付けない。
  30日固定期限で、アクセスによる自動延長はしない。トークン本体はサーバーに保存せず、
  ハッシュ・CSRF値・期限だけをOS保護領域に保存する。Host／Origin／CSRF検証は継続する。
- Forget this browserで当該ブラウザーの管理権限を解除する。MCP接続や実行中解析には影響しない。
  Cookie削除、期限切れ、別ブラウザー、localhostと127.0.0.1の切替時は再ペアリングする。
  この保護領域はアプリの配置場所ごとのため、別フォルダーへ移動した場合も再設定が必要。
- MCPのON/OFF・ポート・接続は従来どおり復元する。管理ブラウザーのペアリングが切れても、
  有効な既存MCP接続は利用できる。初回はOFF。
- Help & IntegrationsのLocal startup settingsでソルバー検出結果を確認し、絶対パスを保存できる。
  優先順位はSOLVER_PATH環境変数、GUI保存値、自動検出。自動検出はアプリ直下、build/Release、buildの順。
  実験用ビルドディレクトリは自動選択しない。空欄保存で自動検出へ戻す。
  明示指定先が存在しないときは別のバイナリに黙って切り替えない。
  変更は次のサーバー起動から適用する。実行中解析がないことを確認して再起動する。
  GUIのパス指定はペアリング済み管理者のみ可能で、MCPツールには提供しない。

外部クライアントでの解析確認には、[OpenCode受入試験手順とZIP生成方法](../acceptance/MCP_ACCEPTANCE_TEST.md)を使用する。
`yaml_inspect` は材料ライブラリを `summary.kind: materials` として識別し、
`material_presets` の件数を返す。適用外の座標系・解析設定は要約に含めない。
利用者向けの手順は[READMEのMCP接続マニュアル](../../../README.md#mcp-setup)を参照。
Windows単体exeのHTTP／stdio接続と実ソルバーの正常・未収束警告ケースは確認済み。
初見エージェントによる操作性、実ジョブ実行中のOFF／切断、他OS等の残件は末尾に区別して記録する。

## 構成とセットアップ

既存WebUIと同じNodeプロセスにMCPを組み込む。既定URLは
`http://127.0.0.1:3001/mcp`。初回はOFFで、以後は保存したON/OFF・ポートを復元する。
ポート競合はMCPのみのエラーとし、WebUIは継続する。MCPはループバックのみで待ち受ける。

1. 開発版はNode 24で `cd webui` → `npm ci` → `npm start`。
   配布版は同梱サーバー実行ファイルを起動する（配布物の受入確認は下記参照）。
2. 操作したいファイルがあるGUIで **Help & Integrations** を開き、Assigned workspaceを確認する。
   ユーザーIDはブラウザーCookieに由来するため、別ブラウザーや別ホスト名では異なる場合がある。
3. サーバーCLIに表示される `MCP management pairing code` でブラウザーをペアリングする。
   コードは一度限り。再取得は、同じサーバー実行ファイルの `--mcp-admin-code`。
   開発版は `node launcher.js --mcp-admin-code`。再起動後は再ペアリングが必要。
4. 接続名・権限を選んで作成する。対象ユーザーは作成時のGUIワークスペースに固定される。
   初期選択は閲覧のみ。必要な権限だけを追加する。
5. マスターをONにし、必要ならポートを変更する。実際の待受状態を確認する。
6. HTTPクライアントにはURLと `Authorization: Bearer <作成時のトークン>` を設定する。
   トークンは接続作成時だけ画面に表示する。コピー後は **Clear token from this page** で消す。
   URLやコマンド引数へトークンを入れない。
7. stdioクライアントにはGUIが示す実行ファイル・引数を登録する。
   `webui-server.exe --mcp-stdio --connection <接続ID>`。
   開発版は `node <絶対パス>/webui/launcher.js --mcp-stdio --connection <接続ID>`。
   このモードは既存サーバーへの橋渡しだけを行う。WebUIの二重起動・自動有効化はしない。
8. GUIの **Test** で認証付きの閲覧要求を試す。これは実際のクライアント側設定の検証とは別。

設定例はクライアント非依存の接続情報。利用するクライアントの登録形式に合わせること。
クライアントの設定ファイルをアプリから変更する処理はない。

**OFF・接続失効・stdio切断では開始済み解析を止めない。**
GUIの活動欄から停止・結果表示を行える。WebUIサーバーそのものの終了時は、既存の
プロセス終了処理で子ソルバーを終了する。これはMCPのOFFとは異なる。

## アクセス境界

MCP接続は次の論理IDだけを扱う。ツールに `userId`・保存ルート・絶対パスの指定はない。

| 論理ID | 割当ユーザーの保存先 |
|---|---|
| `configs/model.yaml` | `configs/<user>/model.yaml` |
| `libraries/steel.yaml` | `user-libs/<user>/steel.yaml` |
| `images/model.png` | `uploads/<user>/model.png` |
| `results/output_x/Az/step_0001.tiff` | `outputs/<user>/output_x/Az/step_0001.tiff` |

トラバーサル、絶対パス、UNC、ADS、予約デバイス名、リンク・ジャンクション・ハードリンクを拒否する。
既存の親ディレクトリも実体を検査する。ZIPのリンク・不正パス・暗号化・重複ターゲットを拒否する。
ZIPに別のユーザー名が入っていても、そのユーザーへの保存先切替として解釈しない。

YAMLの `initial_az_path`、`value_profile`、`gamma_profile` も論理IDで指定する。
解析開始時に許可領域のファイルを結果フォルダー内の `.inputs` にコピーし、設定中の参照を
コピー先へ固定する。設定・画像・材料・参照ファイルのハッシュを `run-inputs.json` に記録する。
ソルバーのcwdもこの結果フォルダーに固定し、相対パスの診断CSV・DD一時ファイルを外へ出さない。
ファイル参照を含むYAMLの循環／共有オブジェクト（エイリアス）は現在拒否する。
材料則や求解アルゴリズムは変更していない。

任意シェル、任意URL取得、ユーザー切替、接続管理、権限拡張のMCPツールは提供しない。
これはMCPサービスの境界であり、同一OSユーザーで実行される任意プログラムのサンドボックスではない。
**従来GUIのユーザーIDは認証ではなく、WebUI全体を多人数・外部ネットワーク向けに安全化したものでもない。**
既存WebUIを信頼しないネットワークに公開しないこと。

## 権限と競合

- `read`：閲覧。全接続に必要。
- `edit`：設定・材料編集、名称・説明変更、バックアップ復元。
- `images`：画像取り込み・加工。既存画像の置換はさらに `replace` が必要。
- `run`／`cancel`：解析開始／停止。
- `delete`：完全削除。ごみ箱・復元機能はない。
- `replace`：既存データの置換。操作自体の `edit`／`images` も必要。

作成は `expectedRevision: null`、更新・削除・名称変更は取得済みリビジョンが必要。
ファイルは内容SHA-256、ディレクトリは子ファイルのサイズ・更新情報等から変更を検出する。
競合は `REVISION_CONFLICT`。再取得して差分を確認するまで自動上書きしない。
GUIの設定・材料保存、削除、名称・説明変更、バックアップ復元も同じ処理・ロックを使う。
GUIには外部変更通知と明示的な再読込を設け、編集中の内容を自動で差し替えない。

実行中の結果は削除・置換・名称変更・エクスポートを禁止する。現在は安全側に、
当該ユーザーのソルバー実行中はそのユーザーの結果への変更全体を禁止する経路もある。
ファイル数上限で古いファイルを自動削除する動作は廃止した。

解析開始には冪等キーが必要。同じ接続・キー・入力の再送では再起動しない。
同じキーで入力が異なる場合は `IDEMPOTENCY_CONFLICT`。解析前の失敗も記録するので、
修正した入力の再実行には新しいキーを使う。再起動後の実行中記録は `interrupted` となり、自動再開しない。

ZIP復元はプレビューで対象とリビジョンを取得してから反映する。ファイル単位で反映するため、
途中失敗時は `applied` に反映済み対象を返す。アーカイブ全体のロールバックは行わない。
GUIも同じ二段階手順を用い、既存ファイルの置換件数を確認してから進む。

## ツール群

全ツールはSDK経由の型付き入力・出力スキーマと構造化結果を持つ。エラーは
`isError: true` と `error.code`。未知の内部エラーでは物理パスを返さない。

| 用途 | ツール |
|---|---|
| 閲覧・編集 | `workspace_status`, `files_list/stat/read`, `yaml_inspect/template/write` |
| 管理 | `files_rename/delete`, `result_description` |
| 画像 | `image_info/process`, `image_import` |
| 分割転送 | `transfer_begin/append/read/release` |
| バックアップ | `backup_export/preview/import` |
| 解析 | `analysis_start/list/get/cancel` |
| 結果 | `result_summary`, `results_table` |

`files_list` は1階層ずつページングし、返されたディレクトリIDを `prefix` として辿る。
一覧は解析中の追加により変わるので、後続ページ取得だけで更新完了を推測しない。
`result_summary` は条件、Azステップ一覧、保持されているジョブの収束・警告を返す。
過去結果など収束記録がない場合は `unknown`／`converged: null` とし、正常収束を推定しない。
`results_table` は数値CSVの行・列範囲。TIFFは `files_read` のバイナリ分割転送で取得できるが、
MCPサーバー内でTIFFを浮動小数点行列へデコードする機能はまだない。
`yaml_template` は編集用の最小テンプレート。GUIの色・形状に応じた全テンプレート生成との完全な共通化は未完了。

## 保持量・サイズ制限

| 対象 | 初期実装の上限 |
|---|---|
| MCP HTTP要求 | 2 MiB、同時8要求 |
| 一覧／CSV範囲 | 1ページ200項目／200行×200列 |
| CSV単一行 | 4 Mi文字（読み取り中に検査） |
| ファイル読み出し | 1回512 KiB以下 |
| YAML保存 | 100 KiB |
| 入力画像 | PNG/JPEG/BMP、10 MiB・16メガピクセル |
| 設定／画像の保存数 | 各20。超過時は作成拒否 |
| 画像プレビュー | 10分、全体16件。保存数上限にも算入 |
| アーカイブ | 圧縮・展開後とも1 GiB、10,000エントリ |
| ディレクトリ一括操作 | 10,000エントリ・深さ64以下 |
| 転送 | 256 KiBチャンク、1時間、全体32件・ユーザー4件 |
| GUI旧JSONバックアップ | 16 MiB以下をZIPに変換して共通検査 |
| 接続／管理セッション | 100接続／16セッション（24時間） |
| MCP永続ジョブ台帳 | 200件。超過時は新規開始を拒否し、既存記録は保持 |
| 監査 | メモリ500件、ディスク約1 MiB＋1世代、書込待ち128件 |

監査の書込失敗・待ち上限でのドロップは管理状態に表示する。トークンや本文は記録しない。
`transfer_release` で不要な転送を明示解放できる。通常終了時はプレビュー・一時転送を清掃する。
異常終了で残った転送ファイルの起動時清掃、永続ジョブ台帳の管理GUIは今後の課題。

資格情報はアプリ／ユーザーデータの外側にあるOS専用領域へ保存する。
Windowsは `%LOCALAPPDATA%/OpenMagFDM/automation/<アプリ場所のハッシュ>/`、
他OSは `~/.local/share/OpenMagFDM/automation/<ハッシュ>/`。
Windows ACL／Unixパーミッションで保護する。stdio用プロファイルもここに置くため、
アプリの設置場所を移動した場合は別プロファイルとなる。ユーザーバックアップには含めない。

## 実装・検証状況

依存はNode 24対象、`@yao-pkg/pkg` 6.23.0、公式MCP SDK
`server`／`client` 2.3.1、`node` 2.1.1を固定。
参照：[SDK](https://github.com/modelcontextprotocol/typescript-sdk)、
[pkg](https://yao-pkg.github.io/pkg/)、[Nodeサポート](https://nodejs.org/en/about/previous-releases)。

2026-10-15、Windows／Node環境で既存49件＋MCP関連19件＝68件の軽量回帰を通過。
長時間ベンチ終了後のコミット前にも全68件を再実施し、19ファイルのJavaScript構文チェックも通過。
ジョブ実行試験のバックエンドはモック。保存済みソルバーの別コピーによるWindowsの
release-contractも通過したが、MCP経由の実モデル試験とは区別する。
この時点では初期Windowsパッケージ生成まで。現行exeの追試は下記2026-10-17の記録を参照。

追加試験で、ペアリングコード保存の非同期処理が終了後に残る問題を検出し、
保存を直列化して終了時の待機に含めた。再実施で解消を確認した。
2026-10-15時点では `npm audit fix` 後もJimp 0.22系経由にmoderate 4件が残った（今回の接続試験では再監査していない）。
MCP画像入力ではマジックバイトと寸法をデコード前に制限するが、依存更新・追加監査は未完了。

### 受入チェックの再実行

```powershell
cd webui
node --test --test-concurrency=1 tests/*.test.js
node tools/build-mcp-package.cjs
node tests/packaged_mcp_smoke.cjs --exe ../build_mcp/webui-server.exe
# SOLVERには検証用の実行ファイルを指定する。固定ベンチ用ファイルを再ビルドしない。
node tests/packaged_mcp_smoke.cjs --exe ../build_mcp/webui-server.exe --solver <検証用ソルバーの絶対パス>
```

単体exe試験は独立一時ディレクトリへコピーし、サーバー子プロセスのPATHから実行中Nodeの
場所を除外する。HTTP／stdio双方の入力作成・検査・読出しを試し、`--solver` 指定時のみ
小さな電流励磁モデルと意図的な未収束モデルを実行する。Windowsでは実施済み。
macOS／Linuxではexe末尾なし。`mcp-validation.yml` は3 OSのネイティブパッケージと接続を
確認する手動専用CIとして追加したが、未起動。正式リリースやタグ作成は行わない。

### Windows実接続試験 — 2026-10-17

`build_mcp/webui-server.exe` をNode 24 / pkg 6.23.0で再生成し、上記ハーネスを
`--solver ../build_nonlinear_prediction/Release/MagFDMsolver.exe` 付きで実行。終了コード0。
既存接続やユーザーデータは使用せず、一時フォルダー・専用ポート・試験用認証情報を用い、
終了後にクライアント／サーバーを閉じて試験用フォルダーを清掃した。固定ソルバーは再ビルドしていない。

| 確認対象 | 結果 |
|---|---|
| 初回OFF、管理認証、GUI Settingsのコマンド・引数、GUI Testと同じ経路 | PASS |
| HTTP／stdioの初期化・ツール一覧・説明・入出力スキーマ | PASS |
| 両経路でYAML生成・検査・保存、画像分割転送・取込・情報取得 | PASS |
| 両経路で実ソルバー正常完了、conditions・結果要約・CSV範囲取得 | PASS |
| 両経路で同じ解析開始要求の再送（jobIdが同一） | PASS |
| 両経路で非線形反復2回・tolerance=0の診断モデル | PASS：completed=true、warning=true、converged=false、exitCode=2 |
| 未収束結果の要約に警告があり、診断用出力が保持される | PASS |
| HTTPのread-only書込拒否、別ユーザーの一覧・取得拒否、接続失効 | PASS |
| 両経路の同名新規作成はREVISION_CONFLICT（上書きなし） | PASS |

合計4回の小モデル解析（HTTP／stdio × 正常／警告）であり、性能ベンチや長時間保持量試験ではない。
SDKクライアントからの試験であり、別LLMセッションが自律的に操作した試験とも区別する。
Nodeの実行ディレクトリを子PATHから外した単体exeの動作試験で、Node未導入のクリーンOS試験ではない。
pkgのSDK動的require／exportsに関する警告は残ったが、上記経路の実行時欠落は認めなかった。

試験対象SHA-256：

```text
webui-server.exe: EC8224793FE48BBB30FAFAF862340282AEFFD9ECD188958A51E130F922A3ECE7
MagFDMsolver.exe: 4A9A0F1D893234FF544FC51D46A61372B55DD347D96F6AE010144635A05906A1
```

軽量回帰も **85/85件成功**。活動欄の **Open Results** で読込対象は変わるのに
選択欄の表示名が残る箇所を修正し、選択表示と実際のロード先の一致・出力未生成時の案内を
2件の追加テストで確認した。ブラウザー実画面の操作確認を代替するものではない。

### 初見セッションでの使い勝手確認

別セッションは機能試験の必須条件ではないが、リリース前の初見テストとして有効。
READMEだけを渡し、まずread-only接続で「割当ユーザー確認→入力一覧→結果1件の条件・収束・警告を要約」を
実行してもらう。MCPを使わずシェルや直接ファイルアクセスで代用した場合はMCPの合格としない。
接続設定の登録後はクライアントの接続を再起動して、新規セッションで確認する。
クライアントのローカル設定やトークンをこの試験のために自動変更することはしない。

その後、必要な権限のみを追加し、検証用入力で保存→検査→解析→状態取得→結果取得を行う。
GUIの活動欄・結果と報告が一致することを確認する。削除・置換は初見テストでは不要。
現時点でGUIのCopy settingsは汎用JSONなので、クライアント別形式への転記はREADMEに従う。
TIFFの数値行列デコードは未提供のため、数値表を試す場合はCSV出力を用意する。

未完了の合格条件：

- 実ジョブが進行している最中のMCP OFF／stdio切断とGUI監視・停止の組合せ（現状はモックで確認）。
- Linux・macOSでの起動・接続と既存数値／release-contract回帰。
- 実ブラウザーでのデスクトップ／狭幅表示、編集競合通知、接続管理の操作確認。
- READMEだけを渡した別エージェントセッションでの初見操作試験。
- 接続・転送・画像加工・解析の連続利用による保持メモリとプロセスの長時間監視。
- 画像加工の高負荷時のGUI応答性。現在は既存の同期アルゴリズムを直列化して共用している。
- GUI全テンプレートの共通化、異常終了後の一時ファイル清掃、ジョブ台帳の満杯時運用。

上記が終わるまで「全計画・受入条件を完了」と扱わない。
