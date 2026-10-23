# サーバー運用マニュアル — v1.7.2

コマンドの全引数は[CLI](CLI.md)、外部プログラムからの操作は[REST API](REST_API.md)、
エージェント接続は[MCP](MCP_INTEGRATION.md)を参照。

## 配布物と起動

Windowsはインストーラー、またはソルバーZIPとWebUI ZIPを同じフォルダーへ展開する。
Linux/macOSもソルバーとWebUIを配置し、`webui-server`を起動する。
Node同梱WebUIにはNode.jsの別途インストールは不要。ソース版はNode.js 24.22.1を使用する。
WebUI単体のアーカイブにはC++ソルバーは含まれない。

ソルバーLinuxはx86_64、ソルバーmacOSはarm64。
`OpenMagFDM-WebUI-macOS.tar.gz`はIntel x64向けで、Apple SiliconではRosetta 2が必要。
WindowsのソルバーはAVX2対応CPUと、配布物に含まれるDLLを使用する。

起動すると`OpenMagFDM>`が表示される。`help`、`menu`、`configure`から操作できる。
既定アドレスは`http://localhost:3000`。サービス運用では`--no-console`を指定する。

ソルバーの優先順位は`SOLVER_PATH`、Help & Integrationsの保存値、標準配置の自動検出。
GUIで実行ファイルの絶対パスを保存した場合は、解析終了後にサーバーを再起動する。
実行ファイルが見つからない場合は`GET /api/health`、`GET /api/solver/info`で確認できる。

## ローカル利用と共有HTTPS

既定のローカルモードはループバック限定で、WebUIログインは不要。
MCPの管理には、別途Help & Integrationsのブラウザーペアリングを使用する。

共有モードはLAN/VPN内の利用を想定する。管理者を作成し、作業領域を所有者へ割り当て、
サーバーを止めて証明書・秘密鍵を設定してから起動する。

```text
users create admin --admin
users create alice
workspaces list
workspaces assign user_example alice
stop
network shared --cert "C:\certs\server.pem" --key "C:\certs\server-key.pem" --host 0.0.0.0
start
```

共有時はlocalhostからもHTTPSログインが必要。通常セッションは無操作8時間・最長24時間。
「この端末を記憶」を選んだ場合は最長30日。ログアウト、パスワード変更、無効化で失効する。
一般ユーザーは割当領域、管理者は全領域を操作できる。未割当の既存領域は一般ユーザーへ公開しない。
無効化してもデータと実行中の解析は残り、新規操作と待機解析の開始を止める。
証明書・鍵の用意と各端末の信頼設定は運用側で行う。

MCPの接続はローカル方式を維持する。共有HTTPSのMCP管理画面統合には制約があるため、
必要なMCP接続の作成と設定表示はローカル運用中に済ませる。
共有移行後も割当・アカウント有効状態を共通で判定する。

## 保存されるデータ

| 内容 | 配置 |
|---|---|
| 設定YAML | アプリ基準ディレクトリの`configs/<workspace>/` |
| 入力画像 | `uploads/<workspace>/` |
| 材料ライブラリ | `user-libs/<workspace>/` |
| 結果 | 既定は`outputs/<workspace>/`。`storage`で変更可能 |
| アカウント・所有者・設定・ジョブ台帳・接続情報 | OS保護領域の、アプリ配置パス別ディレクトリ |

Windowsの管理情報は`%LOCALAPPDATA%/OpenMagFDM/automation/<配置パスのハッシュ>/`、
Linux/macOSは`~/.local/share/OpenMagFDM/automation/<配置パスのハッシュ>/`。
アプリの設置パスを変えると管理情報の参照先も変わる。
通常のユーザーデータZIPにはアカウント・セッション・MCPトークンは含めない。

新規解析では受付時点の入力・実効設定・参照ファイル・ハッシュを保存する。
入力保存に失敗した場合は解析を開始しない。元ファイルを編集しても待機ジョブは受付時の入力を使う。
ステップごとの`InputImg`は任意出力。元入力と移動後のステップ入力は別の画像として扱う。
旧結果は一括変換せず閲覧でき、欠落した入力や画像は表示上で明示する。

## 永続ジョブと停止

WebUIのリロード、タブ終了、ログアウト、MCP切断でジョブを停止しない。
再接続後は作業領域のジョブ一覧から進捗・ログ・取消を復元する。
既定は全体2件、作業領域あたり1件。同一受付の再試行には同じ冪等キーを使う。

サーバーの再起動では計算途中の解析は`interrupted`、待機ジョブは`held`となる。
`jobs resume <workspace>`で保留を解除する。計算途中からの自動再開は行わない。
通常の`stop`や`exit`は解析・入力準備中に拒否し、`--force`で終了を明示できる。
EOF・OS終了要求・コンソール親プロセス消失では子サーバーとソルバーを終了する。

## 更新と移設

同じ配置場所で更新する場合：

1. `jobs list`で実行状況を確認し、解析の完了または取消を待つ。
2. File Managerから必要なユーザーデータをZIPへ保存し、OS保護領域の管理情報も別途保全する。
3. `exit`でサーバーを閉じ、実行ファイル・公開アセット・マニュアルを更新する。
4. `configs`、`uploads`、`user-libs`、結果フォルダーとOS保護領域を保持して起動する。
5. `status`、`limits show`、既存結果の閲覧、ログイン／MCP設定を確認する。
6. 保留ジョブを確認し、必要な作業領域だけ`jobs resume`する。

v1.7.1からv1.7.2への更新に、既存の2D YAMLや結果の変換は不要。
v1.7.1から存在する単発CLIも利用できるが、稼働中の管理には常駐コンソールを使う。

保存先の設定だけを変更する場合は、`stop`→`storage set-results-root <絶対パス>`→`start`。
既存結果も移行する場合は`storage migrate-results <空の絶対パス>`を使う。
コピー・ファイル数・ハッシュを検証して設定を切り替え、元データは自動削除しない。
移設時の失敗では元データを保持し、設定を切り替えず、途中のコピー先を残す。

## 起動環境とリソース設定

環境変数はサーバー起動前に設定する。常駐コンソールの子サーバーへ引き継ぐ。

| 変数 | 既定・意味 |
|---|---|
| `PORT` | 3000。WebUI/REST APIのポート |
| `SOLVER_PATH` | ソルバーの絶対パス。保存設定・自動検出より優先 |
| `MAX_SOLVER_PROCESSES` | 2。ソルバープロセス数の上限 |
| `MAX_CONCURRENT_JOBS` | 2。実行ジョブ数の上限 |
| `MAX_CONCURRENT_JOBS_PER_USER` | 1。作業領域ごとの実行数 |
| `MAX_SOLVER_START_OPERATIONS` | `MAX_SOLVER_PROCESSES`。非同期入力準備の上限 |
| `MAX_SOLVER_START_OPERATIONS_PER_USER` | 1。領域ごとの入力準備の上限 |
| `MAX_JOB_RECORDS` | 200。プロセス内の完了履歴保持の上限 |
| `MAX_JOB_LOG_LINES` / `MAX_JOB_LOG_CHARS` | 500行／256 KiB相当の文字数。ジョブログの上限 |
| `JOB_TTL_MS` | 86,400,000。プロセス内完了履歴の保持時間 |
| `OMP_NUM_THREADS` | ソルバーのOpenMPスレッド数 |

永続ジョブ台帳とプロセス内の一時履歴は別の保存対象である。
入力・設定を経過日数だけで自動削除しない。
ファイル数制限は環境変数ではなく`configure`または`limits`で設定する。
実行並列数を増やす場合は、対象モデルでソルバーのピークメモリを確認する。

## 問題の確認先

| 症状 | 確認 |
|---|---|
| `OpenMagFDM>`がない | v1.7.2の実行ファイルか、TTYがあるか、`--no-console`を付けていないか |
| 起動失敗 | ポート競合、ソルバー設定、証明書・鍵・書込権限。修正後に`start` |
| 制限変更を拒否される | 稼働中なら同じ常駐コンソールで`limits`を実行 |
| 保存先変更を拒否される | `stop`後に設定し、`start` |
| 解析が再開しない | `jobs list`で`held`／`interrupted`を区別 |
| 書込が409 | 更新競合またはファイル数制限。メッセージを確認し再取得 |
| 他ユーザーの領域が見えない | `workspaces list`と所有者割当、アカウント有効状態 |

過去の診断・検討・測定結果は[archive](archive/README.md)に保管する。
