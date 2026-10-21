# v2 改善実装・運用と検証記録

更新日：2026-10-22。v1.7.0を基準とする開発中の変更であり、v2正式リリースの受入完了を示すものではない。
2Dの離散化・線形／非線形解法は変更していない。C++側の変更は解析入力の保存と検証、実行時の参照パスに限定する。

## 今回の実装

| 項目 | 現在の動作 |
|---|---|
| ジョブ | WebUIとMCPが同じ永続台帳・待機キューを使用。全体2件／領域1件が既定。領域ごとに実行機会を回す。受付時入力の保存、冪等キー、取消、再接続に対応。 |
| 再起動 | 実行途中を`interrupted`、待機中を`held`として復元。作業領域の明示的なResumeで待機を再開する。計算途中からの再開はしない。 |
| 入力 | `.inputs/`へ元画像、元YAML、実効YAML、指定材料ライブラリー、参照ファイルを保存。`run-inputs.json`にSHA-256を記録。保存・検証に失敗すれば起動しない。 |
| 保存先 | WebUI、MCP、結果ZIPが共通の保存先解決を使用。保存先設定・コピー検証付き移行をCLIで提供。 |
| 認証 | 既定はloopbackのHTTP。明示的な共有モードではHTTPSとログインが必須。アカウント、所有領域、CSRF、セッション失効を確認する。 |
| 結果編集 | 常設の複数行description、競合検出付き保存、File ManagerのMemo、極座標XYの縦横比固定（既定ON）。 |
| 磁力線 | Azから表示する磁力線でステップ入力画像が欠けても背景なしで表示。元入力は解析開始時の画像として別表示。旧結果の入力欠落を明示する。 |
| 削除 | 単一／一括削除を非同期操作IDで追跡。再接続時の照会、部分失敗、進行中結果の保護、ボタン状態復旧に対応。 |
| 描画更新 | 分布図と材料背景の磁力線は既存Plotly画面を保持して更新。Plotlyの同一領域への非同期commitを直列化し、古い待機要求を捨てる。 |
| Preload | Az/Muの隣接ステップ先読み、全ステップPreload、容量・停止理由・キャッシュ数の表示。Worker／GPU用作業量も予算に含める。 |
| GPU試作 | 単一WebGL2コンテキストの分布図、極座標XY変換、色付け。極座標磁力線はWorkerで従来のラスタ抽出を実行しGPUで表示変換する。 |
| 計測 | TIFF取得・デコード、Worker、Plotly commit、ステップ表示待ち、RAF間隔、キャッシュ・GPU見積容量、利用可能ならJSヒープをCSVに保存。 |

入力・設定フォルダーの経過日数による自動削除は行わない。各ステップの`InputImg`出力は引き続き任意。
結果ZIPには`.inputs`も含める。他の管理用隠しフォルダーは公開しない。

## ソース版の起動

リポジトリ直下のPowerShellで実行する。Node.js 24.xとソルバーの実行に必要なDLLを用意する。

```powershell
cd webui
npm ci
# 標準配置で見つからない場合だけ指定
$env:SOLVER_PATH = 'C:\absolute\path\MagFDMsolver.exe'
npm start
```

既定は `http://localhost:3000`。今回の開発用ビルドは `build_solver_v2/Release/MagFDMsolver.exe`。
この環境では依存DLLが `build_voxel_deps/Library/bin` にあり、起動プロセスのPATHへ加える必要がある。
この開発用配置を、別端末へ配布できるパッケージとして扱わない。

## CLIでの管理

以下はリポジトリ直下で実行する。配布exeでは `node webui/launcher.js` をそのexeに置き換える。
パスワードは対話端末の非表示入力で設定する（12文字以上）。

```powershell
node webui/launcher.js users create admin --admin
node webui/launcher.js users create alice
node webui/launcher.js users list
node webui/launcher.js workspaces list
node webui/launcher.js workspaces assign EXISTING_WORKSPACE_ID alice
node webui/launcher.js users disable alice
node webui/launcher.js users enable alice
node webui/launcher.js users reset-password alice
```

既存のブラウザー別領域は、共有モードでは割り当てるまで一般利用者からアクセスできない。
アカウント無効化はデータを消さず、待機ジョブの開始を止める。既に動いているジョブは継続する。
管理者は作業領域を切り替えてジョブを照会・取消できる。利用者はログイン後のChange passwordから自分のパスワードを変更できる。

サーバーを停止してから共有モードを設定する。証明書の発行・端末への信頼設定は運用側で行う。

```powershell
node webui/launcher.js network shared --cert C:\certs\server.pem --key C:\certs\server-key.pem --host 0.0.0.0
node webui/launcher.js
# HTTPのローカル専用へ戻す場合も、いったん停止する
node webui/launcher.js network local
```

共有モードではlocalhostからもログインが必要。通常セッションは無操作8時間／最長24時間、
明示的な「記憶」は最長30日。ログアウト・パスワード変更・無効化でセッションを失効する。
scrypt、サーバー側セッション、Secure/HttpOnly/SameSite Cookie、更新APIのCSRF対策とログイン試行制限を実装した。
参考：[Expressのセキュリティ指針](https://expressjs.com/en/advanced/best-practice-security/)。

アカウント・所有者・セッション・ジョブ台帳・サーバー設定は、Windowsでは
`%LOCALAPPDATA%/OpenMagFDM/automation/<アプリ基準パスのハッシュ>/` に保存する。
解析結果のZIPはこれらの管理情報を含まない。サーバー移設では管理情報も別途保全する。
ブラウザー依存ライブラリーは`webui/public/vendor`から配信し、取得元・ハッシュ・ライセンスを同梱する。

## 保存先の変更と移行

```powershell
node webui/launcher.js storage show
# 保存先だけ切り替える。既存データは移動しない。
node webui/launcher.js storage set-results-root D:\OpenMagFDM\results
# 現在の保存先から、別の空ディレクトリへコピー・ハッシュ検証して切り替える。
node webui/launcher.js storage migrate-results E:\OpenMagFDM\results
```

変更は再起動後に反映する。稼働中の変更は拒否し、移行中は起動用ロックを保持する。
移行先は元ディレクトリと重ならない空ディレクトリを指定する。元データは自動削除しない。
コピー途中の失敗では設定を切り替えない。途中までコピーされた移行先は残るため、別の空ディレクトリを指定して再試行する。
保存先が変わっても、APIでは`outputs/<workspace>/<result>`、MCPでは`results/<result>`という論理IDを使う。
入力参照は結果フォルダーからの相対パスで保存する。

## APIと移行上の変更

- `POST /api/jobs`：`userId`, `configFile`, `imageFile`, `idempotencyKey`、任意の`materialLibraryFile`。
- `GET /api/jobs?userId=...`、`GET /api/jobs/:id?userId=...`：履歴・状態・進捗・ログ。
- `DELETE /api/jobs/:id?userId=...`：取消要求。実行中はプロセス終了を確認して確定する。
- `POST /api/jobs/resume`：指定作業領域の保留キューを再開。MCPは`analysis_resume`。
- `/api/file-operations`：削除受付・一覧・IDによる照会。通信タイムアウト後もIDを照会する。
- `/api/auth/login`, `/api/auth/logout`, `/api/auth/me`, `/api/auth/password`：共有モードのセッション管理。

`/api/solve`、`/api/solve-stream`、`/api/stop-solver`はHTTP 410とし、接続寿命に依存する旧実行経路を廃止した。
旧クライアントはジョブAPIへ移行する。`/data`による基準ディレクトリ公開もHTTP 410へ変更した。
既存結果は一括変換せず、そのまま閲覧する。
WebUI/MCP受付のYAML内ファイル参照には`configs/...`等の作業領域IDを指定する。
任意のOS絶対パスは受付しない。単体ソルバーは従来のCLI入力パスを受け取り、その内容を保存する。

MCPのHTTP/stdio接続は引き続きローカル専用。既存の接続割当とアカウントの領域有効状態を確認する。
従来のHelp / Integrationsのペアリング管理はローカルHTTP向けであり、共有HTTPSからの管理UI統合は未完了。
共有運用へ切り替える前に必要なMCP接続を設定する。

## 描画・測定の使い方と現在の限界

Resultsへ結果をロードし、Rendererを選ぶ。`Auto (validated legacy)`は現時点ではPlotlyを使う。
`GPU (experimental)`だけがGPU試作経路を選択する。r–θ表示、非対応設定、容量上限、WebGL2非対応等ではPlotlyへ戻り、
理由を表示する。CSVには選択モードと実際の描画方式を別に記録する。

同じ入力・表示数・画面サイズで、Measure step changesを1〜30分実行し、Export timing CSVで保存する。
非表示タブ、結果変更、描画失敗は無効試行として記録する。FPSはRAF間隔からの推定でありGPUタイマー実測ではない。
GPU用Float32は表示用で、元の数値配列やCSVを書き換えない。GPU領域はホイール拡大縮小、ドラッグ移動、ダブルクリックで範囲復帰に対応する。

未完了の点：

- Chrome／Edgeによる実表示、画像比較、30分連続切替、旧版との中央値／p95／メモリ比較。
- GPUの周期／反周期・欠損・材料境界の実画面による受入。自動GPU選択はまだ有効化しない。
- 磁力線は既存のラスタ抽出をWorker化した段階。線分バッファとしてまとめるベクトル描画は未実装。
- GPUの目盛・数値ホバー等、Plotlyと同等の操作表示。未対応の図種や一部の複合ウィジェットは従来描画を維持。
- 全図種での画面準備後の切替、および全変換段階の独立した時間計測。現在のちらつき対策は主に分布図・材料背景磁力線へ導入。
- ネットワーク障害、端末間のログイン操作、大量実データ削除の実ブラウザー受入。
- インストーラー／配布exeの再構築とWindows以外の回帰確認。

ジョブ台帳は現在1万件で新規受付を止める。履歴の自動間引きはしない。履歴アーカイブは後続課題。
メモリ値はキャッシュ管理上の見積もりを含み、OS全体やGPUドライバーの実使用量を保証しない。

## 検証記録

今回のWindows環境で以下を実行した。

- Node回帰：100件合格（既存91件に追加）。所有権、CSRF、セッション復元／無効化、冪等受付、入力固定、再起動時保留、3領域の順番、無効化解除後の待機再開、削除の部分成功、外部保存先と入力付きZIP復元、Worker、描画commit競合を含む。
- C++ CTest：5件合格。入力のSHA-256、改変拒否、移行後の相対参照、および既存数値回帰4件。
- 単体ソルバー：32×32の受入モデルを実行し正常終了、入力スナップショット保存を確認。
- 独立したコピー上でHTTPS API統合試験：所有領域外の結果・画像・ジョブ・ZIP拒否、ログアウト失効、冪等受付、実ソルバー完了、受付後に元画像を変更しても入力保持、外部保存先、結果ZIP、稼働中の移行拒否と停止後CLI移行を確認。
- ボクセル化／CLI／VTKビューアーのCTest：3スイート合格。VTK更新で削除された`AddActor2D`を`AddViewProp`へ置換した。
- Voxel Lab：JavaScript 5件、Python/native API 3件を再実行し合格。

Node／C++のテストは実ブラウザーの描画性能を証明しない。このセッションではブラウザー操作接続が利用できず、上記の実ブラウザー受入は未実施。

```powershell
node --test webui/tests/*.test.js
ctest --test-dir build_solver_v2 -C Release --output-on-failure
ctest --test-dir build_voxelizer -C Release --output-on-failure
# DLLを検索できるPATHとOpenSSLを用意する。専用のコピーとテストアカウントを作成する。
node webui/tests/v2_server_smoke.cjs --solver C:\absolute\MagFDMsolver.exe --openssl C:\absolute\openssl.exe --artifacts C:\absolute\test-artifacts
```

統合試験は専用サーバーだけを停止し、検証用コピー・結果・ZIP・`verification.json`を残す。
ソースリポジトリのアカウント、共有モード、保存先は変更しない。
