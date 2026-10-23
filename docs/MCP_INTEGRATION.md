# MCP接続マニュアル — v1.7.2

同じPCのMCPクライアントから、固定された作業領域の入力・結果を参照し、
許可した編集・画像加工・解析・取消・削除を実行する。
一般HTTPクライアント向けの仕様は[REST API](REST_API.md)を参照。

## セットアップ

1. `webui-server.exe`／`./webui-server`、またはソース版`npm start`でサーバーを起動する。
2. WebUIの**Help & Integrations**を開き、Assigned workspaceを確認する。
3. 端末の`MCP management pairing code`でブラウザーをペアリングする。
   最長30日維持する場合は**Trust this browser for 30 days**を明示選択する。
4. 接続名を付け、最初は閲覧権限だけで**Create connection**する。
5. **Enable MCP**をApplyし、`Listening`とURLを確認する。初回はOFF。
6. 接続の**Test**を実行し、**Settings**に表示された接続情報をクライアントへ登録する。

ペアリングコードを見失った場合は、同じ配置の実行ファイルを`--mcp-admin-code`付きで実行する。
配布物の別コピーや別のOSユーザーは同じ接続プロファイルを使用しない。
既定MCP URLは`http://127.0.0.1:3001/mcp`。ポートは管理画面から変更できる。
ポート競合でMCPが起動しない場合もWebUIは継続する。

## 接続方式

| 方式 | クライアント設定 |
|---|---|
| Streamable HTTP | SettingsのURL、`Authorization: Bearer <作成時のトークン>` |
| stdio | Settingsの`stdio.command`と`stdio.args`をそのクライアントの形式で登録 |

stdioの配布版は`webui-server.exe --mcp-stdio --connection <id>`に相当する。
ソース版はNode実行ファイル、launcher.jsの絶対パス、残りの引数の順。
GUIが表示する値を使用する。stdioは起動済みの同じサーバーへのブリッジで、
サーバーやMCPを自動で起動しない。

HTTPのトークン本体は接続作成時に一度だけ表示する。
接続IDはトークンではない。Settingsの再表示に秘密のトークンは含まれない。
クライアント設定を変更したら接続を再起動し、新規セッションで確認する。

## 作業領域・権限

接続は作成時の作業領域へ固定され、ツール引数で他領域を指定できない。
read、edit、images、run、cancel、delete、replaceを必要な範囲だけ付与する。
ファイル更新・削除には取得済みリビジョンを、解析開始には冪等キーを使う。
共有モードでも所有者・アカウント有効状態を共通で確認する。

最初の依頼例：

```text
OpenMagFDMのMCPでworkspace_statusを確認し、割当領域の入力と結果を一覧してください。
結果があれば条件・収束・警告を1件要約してください。
ファイル変更や解析開始は行わないでください。
```

ツール群は`workspace_status`、`files_*`、`yaml_inspect`、`image_*`、
`analysis_*`、`result_*`／`results_*`、バックアップ等。
正確な引数・出力スキーマは接続時のtool discoveryで取得する。
TIFFの生データをCSVの数値行列と同一視しない。数値表の取得には対応するCSV出力を用意する。

## ジョブとブラウザー信頼

MCP OFF、トークン失効、クライアント切断で開始済み解析を停止しない。
取消は許可された`analysis_cancel`、WebUI、サーバーCLIで要求する。
ブラウザーとMCPは共通の永続ジョブサービスを使用する。
サーバー終了では子ソルバーを終了し、再起動後の待機ジョブは保留になる。

**Forget this browser**はそのブラウザーの管理権限を解除する。
MCP接続の失効は**Revoke**で行う。アプリの配置パスを変えると管理情報の参照先も変わる。
接続情報と管理セッションはユーザーデータZIPには含まれない。

MCPはループバック専用。共有HTTPSでのMCP管理画面には未統合部分があり、
必要な接続の作成・設定表示はローカル運用中に行う。
WebUIログインのセッション／CSRFとMCP管理のペアリング／CSRFは別の仕組み。

## 問題の確認

| 症状 | 確認 |
|---|---|
| GUIのTestだけ成功 | クライアント側のURL、トークン、command/args、接続再起動 |
| 接続拒否 | サーバー稼働、MCP ON、Listening、ポート競合 |
| stdio失敗 | 同じOSユーザー・配置場所か、絶対パスと引数がSettingsと一致するか |
| ファイルが見えない | Assigned workspace、所有者割当、権限。別ブラウザーの領域と区別 |
| 409・更新競合 | 対象を再取得し、最新リビジョンで操作を組み立て直す |
| トークン紛失 | 古い接続をRevokeし、新しく作成 |

3 OSのNode同梱版のHTTP/stdio回帰はリリースCIで確認する。
長時間保持量、全ブラウザー画面の手動操作、初見エージェントの使い勝手は別の検証項目。
導入時の試験・診断は[旧MCP記録](archive/integrations/MCP_INTEGRATION_BEFORE_v1.7.2.md)、
外部クライアントの小モデル試験は[受入手順](archive/acceptance/MCP_ACCEPTANCE_TEST.md)に保管する。
