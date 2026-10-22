# v1.7.1 Operations

See [release notes](releases/RELEASE_NOTES_v1.7.1.md) for release readiness and limitations.

## CLI help and file count limits

Packaged: `webui-server.exe help` (also `--help`, `-h`, `limits --help`).
Source: `node webui/launcher.js help`. Help exits without starting the server.

Stop the server before changing limits; start it again after saving settings.
For example, using the packaged executable:

```text
webui-server.exe limits show
webui-server.exe limits show user_example
webui-server.exe limits set-default configs 100
webui-server.exe limits set user_example images 200
webui-server.exe limits set user_example libraries unlimited
webui-server.exe limits set user_example images default
```

Categories are `configs` (YAML), `images`, and `libraries` (material YAML).
Defaults remain 20, 20, and unlimited respectively. `unlimited` removes the
quota, `0` blocks creation, and `default` removes a workspace override.
These are **per workspace**, the existing browser userId. An account assigned
multiple workspaces has separate quotas for each workspace, not a combined quota.
Nested files count too. Results, frozen analysis inputs, and transfer staging
are not input-file quotas; their existing resource safeguards still apply.
WebUI, MCP, and backup import use the same write check. Lowering a limit never
deletes files and existing files can still be updated. OS filesystem access by
the administrator is outside these application quotas.

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
