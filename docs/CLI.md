# CLIコマンドマニュアル — v1.7.2

サーバーの常駐コンソール、起動引数、単体2Dソルバーの操作を説明する。
保存先・アカウント・共有HTTPSの運用は[運用マニュアル](OPERATIONS.md)、
HTTPからの操作は[REST APIマニュアル](REST_API.md)を参照。

## 1. 起動

| 配布形態 | 起動コマンド |
|---|---|
| WindowsのNode同梱版 | `webui-server.exe` |
| Linux/macOSのNode同梱版 | `./webui-server` |
| ソース版（リポジトリ直下） | `npm --prefix webui ci`の後、`node webui/launcher.js` |
| ソース版（webui内） | `npm ci`の後、`npm start` |

端末が接続されていればサーバー起動後に`OpenMagFDM>`が表示される。
以降の例は、このプロンプトへ入力するコマンド。プロンプト文字自体は入力しない。
WebUIは既定で`http://localhost:3000`。

```text
OpenMagFDM> help
OpenMagFDM> status
OpenMagFDM> menu
OpenMagFDM> configure
```

`menu`は矢印キーによる操作選択、`configure`はファイル数制限の設定フォーム。
Tabはコマンド補完、上下キーは今回の起動中の履歴。履歴はメモリ内で100件まで保持する。
空白を含むパスは`"D:\Analysis Results"`のように引用符で囲む。
フォームのCtrl+Cは入力取消、コマンド入力途中のCtrl+Cは行の消去。
空行でのCtrl+Cは通常終了を要求する。Ctrl+D／標準入力EOFは解析を含め終了する。
パスワードは伏字の専用入力と確認入力を使用し、コマンド履歴に入れない。

## 2. 共通操作とサーバー制御

| コマンド | 動作・条件 |
|---|---|
| `help` / `/help` / `?` | 常駐コンソールのコマンド一覧。`--help`、`-h`も使用可 |
| `menu` | 選択メニュー |
| `configure` | 既定または指定作業領域のファイル数制限を対話設定 |
| `status` | PID、アドレス、初期化状態、実行状況、ジョブをJSON表示。停止中は停止状態を表示 |
| `start` | 停止中のサーバーを起動 |
| `stop [--force]` | サーバーを停止し、コンソールを残す |
| `restart [--force]` | 停止して起動 |
| `exit [--force]` / `quit [--force]` | サーバーを停止してコンソールを終了 |
| `<group> --help` | 管理コマンドの引数一覧。例：`limits --help` |

`stop`、`restart`、`exit`は解析・入力準備中には拒否する。
`--force`は実行中の解析を終了する指定。取消はプロセスの終了を確認して確定する。
待機ジョブはサーバー再起動後に保留となる。再開には`jobs resume`を使用する。
起動に失敗してもコンソールは残るため、設定やポート競合を修正して`start`できる。

## 3. ジョブ（稼働中のみ）

| コマンド | 内容 |
|---|---|
| `jobs` / `jobs list` | 全作業領域の最近100件までを表示 |
| `jobs list <workspace>` | 作業領域を絞って表示 |
| `jobs cancel <workspace> <jobId>` | 指定ジョブの取消を要求 |
| `jobs resume <workspace>` | サーバー再起動後の保留ジョブを再開 |

```text
jobs list user_example
jobs cancel user_example 01234567-89ab-cdef-0123-456789abcdef
jobs resume user_example
```

ジョブIDと作業領域は一覧から取得する。解析の新規受付はWebUI、REST API、MCPを使用する。
常駐コンソールはOS端末を使用するサーバー管理者の操作経路で、全体のジョブを管理できる。

## 4. アカウント（稼働中に反映）

| コマンド | 内容 |
|---|---|
| `users list` | アカウント名、役割、無効化状態を表示 |
| `users create <name> [--admin]` | 通常ユーザー、または管理者を作成 |
| `users disable <name>` | 新規操作と待機ジョブの開始を停止 |
| `users enable <name>` | アカウントを有効化 |
| `users reset-password <name>` | 新しいパスワードと確認入力を要求 |

パスワードは12文字以上・UTF-8で1,024バイト以下。salt付きscryptで保存する。
最後の有効な管理者は無効化できない。無効化してもデータは残り、既に実行中の解析は継続する。
パスワード変更・無効化はそのアカウントのWebセッションを失効する。

## 5. 作業領域（稼働中に反映）

| コマンド | 内容 |
|---|---|
| `workspaces list` | 既存領域と所有者を表示。未割当は`(unassigned)` |
| `workspaces assign <workspace> <account>` | 有効なアカウントへ所有者を割り当てる |

```text
users create alice
workspaces list
workspaces assign user_example alice
```

作業領域IDは既存のブラウザー別`userId`に対応する。名前は英数字、`_`、`-`を用いる。
共有モードでは所有者を割り当てるまで一般ユーザーへ公開しない。
作業領域の割当変更やアカウント無効化で、ファイルを削除・移動することはない。

## 6. ファイル数制限（稼働中に反映）

```text
limits show [workspace]
limits set-default <configs|images|libraries> <count|unlimited>
limits set <workspace> <configs|images|libraries> <count|unlimited|default>
```

| 分類 | 対象 | 既定 |
|---|---|---|
| `configs` | 設定YAML | 20件 |
| `images` | 入力画像 | 20件 |
| `libraries` | 材料ライブラリYAML | 無制限 |

`count`は0以上の整数、`unlimited`は無制限、`default`は領域別の上書きを解除する。
`default`は`limits set`だけで使用できる。

```text
limits set-default configs 100
limits set user_example images 200
limits set user_example libraries unlimited
limits set user_example images default
limits show user_example
```

制限は作業領域ごとで、サブフォルダーも数える。複数領域を持つアカウント全体の合算ではない。
WebUI・MCP・バックアップ復元に共通で適用する。
`0`は新規作成を禁止する。制限を減らしても既存ファイルは残り、上書きできる。
結果と保存済み解析入力は、この入力ファイル数の対象外。

## 7. 保存先（stop後に変更）

```text
storage show
storage set-results-root <absolute-directory>
storage migrate-results <absolute-empty-directory>
```

```text
stop
storage migrate-results "E:\OpenMagFDM Results"
start
```

`set-results-root`は保存先設定だけを切り替える。`migrate-results`はコピーし、
ファイル数とハッシュを検証してから設定を切り替える。元データは自動削除しない。
移行元と重ならない空ディレクトリを指定する。失敗途中のコピーは残るため、
原因を解決して別の空ディレクトリへ再試行する。設定は次の`start`で有効になる。
詳細は[更新・移行手順](OPERATIONS.md#更新と移設)を参照。

## 8. ネットワーク（stop後に変更）

```text
network local
network shared --cert <path> --key <path> [--host address]
```

```text
users create admin --admin
stop
network shared --cert "C:\certs\server.pem" --key "C:\certs\server-key.pem" --host 0.0.0.0
start
```

共有モードの有効化には有効な管理者、読み取り可能な証明書・秘密鍵が必要。
証明書の発行・クライアント端末への信頼設定は運用側で行う。
`--host`の既定は`0.0.0.0`。共有時はHTTPSで、localhostからもログインが必要。
`network local`はループバック限定のHTTPへ戻す。ポートは起動環境の`PORT`で指定する。

## 9. 起動引数と自動運用

次は`OpenMagFDM>`ではなく、OSのシェルから指定する。

| 引数 | 内容 |
|---|---|
| 引数なし | TTYがあれば常駐コンソール、なければサーバーのみ起動 |
| `--no-console` | 対話端末があってもサーバーのみ起動。サービス用 |
| `--console` | パイプ等でもコンソールを明示起動。フォームにはTTYが必要 |
| `help` / `--help` / `-h` | 管理コマンドのhelpを表示して終了 |
| `--runtime-info` | Nodeバージョン・OS・CPU・同梱実行の有無をJSON表示して終了 |
| `--mcp-admin-code` | 同じ配置の稼働サーバーのペアリングコードを表示 |
| `--mcp-stdio --connection <id>` | 既存サーバーへのMCPブリッジ。GUIで表示された引数を使用 |
| `users ...` / `workspaces ...` / `limits ...` / `storage ...` / `network ...` | 従来の単発管理コマンド |

単発の保存先・ネットワーク・制限変更は、別プロセスのサーバーを停止してから実行する。
稼働中の管理には常駐コンソールを使用する。`status`、`menu`、`jobs`、`stop`等は常駐コンソール専用。
単発コマンドは正常時0、エラー時は非0で終了する。

```powershell
# OSシェルでの例
.\webui-server.exe --runtime-info
.\webui-server.exe limits show
$env:PORT = '3100'
.\webui-server.exe --no-console
```

主な起動環境変数は[運用マニュアル](OPERATIONS.md#起動環境とリソース設定)を参照。
MCPのOFF・切断・ログアウトは解析取消ではない。サーバー自体の終了は子ソルバーも終了する。

## 10. 単体2Dソルバー

```text
MagFDMsolver[.exe] <config.yaml> <image.png> [output_folder]
MagFDMsolver[.exe] --version
```

Windowsの例：

```powershell
.\MagFDMsolver.exe .\sample_config.yaml .\model.png .\output_example
.\MagFDMsolver.exe --version
```

Linux/macOSは`./MagFDMsolver sample_config.yaml model.png output_example`。
出力先省略時はタイムスタンプ付きフォルダーを使用する。
引数不足時は使用方法を表示して非0で終了する。`--version`の出力は`OpenMagFDM 1.7.2`。
画像はRGB材料配置、設定仕様は[CONFIGURATION.md](CONFIGURATION.md)を参照する。
WebUI/MCPジョブ台帳と単体ソルバーの直接実行は別の起動経路である。
