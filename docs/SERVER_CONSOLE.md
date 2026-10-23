# 常駐サーバーコンソール

この機能はv1.7.1公開後の修正で追加する。公開済みv1.7.1には含まれない。
v1.7.1で提供していた起動引数による管理コマンドも引き続き使用できる。

## 起動と操作

端末で`webui-server.exe`（macOS/Linuxは`./webui-server`）を起動すると、
WebUIサーバーとともに`OpenMagFDM>`プロンプトが開く。
ソース版はリポジトリの`webui`で`npm ci`、`npm start`を実行する。
別途、従来どおりソルバーと実行時ライブラリが必要。

```text
OpenMagFDM> help
OpenMagFDM> menu
OpenMagFDM> configure
OpenMagFDM> status
OpenMagFDM> limits show
OpenMagFDM> limits set-default configs 100
OpenMagFDM> limits set user_example images 200
OpenMagFDM> users create alice
OpenMagFDM> workspaces assign user_example alice
OpenMagFDM> jobs list
```

`menu`は矢印キーとEnterによる操作一覧、`configure`はファイル数制限の設定フォーム。
コマンド入力ではTabで補完し、上下キーで今回の起動中の履歴を呼び出せる。
履歴はファイルへ保存しない。パスワードは専用の伏字入力と確認入力で設定し、
コマンド履歴には入れない。フォーム中のCtrl+Cは入力を取り消してプロンプトへ戻る。
入力途中のCtrl+Cは行を消し、空行でのCtrl+Cは終了を要求する。

`users`、`workspaces`、`limits`の変更は稼働中のサーバーへ反映する。
アカウントを無効にしてもデータは消さず、実行中の解析は継続する。
ファイル数は従来どおり作業領域ごとに数える。アカウント全体の合算ではない。
制限を減らしても既存ファイルは消えず、上書きは可能。
引数の詳細は`limits --help`等で確認できる。

## 停止、設定変更、再起動

保存先とHTTPS待受設定の変更は、同じコンソール内でサーバーを止めて行う。
コンソール自体を閉じる必要はない。

```text
OpenMagFDM> stop
OpenMagFDM> storage set-results-root "D:\Analysis Results"
OpenMagFDM> start
OpenMagFDM> restart
OpenMagFDM> exit
```

`stop`、`restart`、`exit`は解析・入力準備中には拒否する。
`jobs cancel <workspace> <jobId>`で停止を要求するか、
`stop --force`、`restart --force`、`exit --force`で解析の終了を明示的に要求できる。
待機ジョブは保存され、再起動後は保留となる。`jobs resume <workspace>`で再開する。
サーバー起動に失敗した場合もコンソールは残り、設定修正後に`start`を試せる。
Ctrl+D（標準入力EOF）とOSからの終了要求は、解析も含めサーバーを終了する。
コンソールの強制終了時も、子サーバーはIPC切断と親プロセス消失を監視して終了する。

コンソールは、そのOS端末の利用者によるローカル管理機能である。
管理用HTTPエンドポイントや任意のシェルコマンド実行機能は追加しない。
WebUI/MCPは既存の認証・所有者管理・共通ジョブサービスを継続使用する。

## サービス、パイプ、MCP

TTYのないサービス起動や出力リダイレクトでは、従来のサーバー起動を維持する。
明示的にコンソールを無効にする場合は`--no-console`を付ける。
パイプで管理コマンドを渡す検証では`--console`を指定できるが、
パスワードやメニューなどのフォームには対話端末が必要。
`--mcp-stdio`はコンソールを開かず、従来のブリッジ専用動作を維持する。

## UIライブラリの選定（2026-10-23）

| 候補 | 実績・保守の確認 | 判断 |
|---|---|---|
| [Inquirer.js / @inquirer/prompts](https://github.com/SBoudrias/Inquirer.js) | MIT、約2.2万stars、直近の更新あり。npmの`@inquirer/password`は直近1週間で約2,189万downloads。選択、フォーム、非表示入力と中断APIを提供 | `@inquirer/prompts` **8.7.4**をバージョン固定で採用 |
| [Ink](https://github.com/vadimdemedes/ink) | MIT、約4万stars、継続保守。公式にGemini CLI、GitHub Copilot CLI、Claude Code等の利用を掲載 | 常時更新する画面の有力候補。今回のコマンド＋フォームはInquirerで実装 |
| [Commander](https://github.com/tj/commander.js) | MIT、約2.8万stars、継続保守。起動引数とhelpのパーサー | 常駐画面の描画・入力管理は別途必要なため、今回のUI依存には追加しない |

starsとdownloadsは採用規模の参考値であり、安定性の保証とはしない。
GitHub公式API、npmレジストリのバージョン・engines・license、
[npm downloads API](https://api.npmjs.org/downloads/point/last-week/@inquirer/password)を確認した。
downloadsの対象期間は2026-10-12〜2026-10-18。
Node.js **24.22.1**への対応、既存のCommonJSコードからの使用、
Node同梱EXEへの封入、実際の端末操作、依存関係の監査を実装側で検証する。
コマンド行の編集・補完・履歴はNode標準`readline`を使用する。

端末入力テストには、VS Codeで利用されるMicrosoftの
[`node-pty`](https://github.com/microsoft/node-pty) **1.1.0**を開発依存として使用する。
Linux/macOSのPTYとWindowsのConPTYでキーボード操作を再現する。
`node-pty`はサーバーの実行時依存にはしない。

## 検証

リポジトリ直下から実行する。テストは一時ディレクトリと専用ポートを使用し、
利用者のアカウント・設定・結果は変更しない。

```text
npm --prefix webui test
npm --prefix webui audit --audit-level=moderate
node webui/tests/server_console_smoke.cjs
node webui/tests/server_console_pty.cjs --artifacts build_console_qa
node webui/tools/build-mcp-package.cjs
node webui/tests/server_console_smoke.cjs --exe build_mcp/webui-server.exe
node webui/tests/server_console_pty.cjs --exe build_mcp/webui-server.exe --artifacts build_console_qa
```

macOS/LinuxはEXEのパスを`build_mcp/webui-server`へ置き換える。
Linuxで端末テスト用ネイティブモジュールが未構築の場合は、
C++ビルド環境を用意して`npm --prefix webui rebuild node-pty --ignore-scripts=false`を実行する。
`server_console_smoke.cjs`に`--solver <絶対パス>`を追加すると、実解析中の停止拒否と強制取消も確認する。

Windowsローカルでは103件のNode回帰試験、依存監査（指摘0件）、
ソース版・Node同梱EXEの端末試験、HTTP制限変更、EOF・親強制終了時の後始末、
実ソルバーの停止保護・取消、既存のEXE版HTTP/MCP/画像API試験を確認した。
端末試験は既定の対話起動、Tab、履歴、矢印メニュー、設定フォーム、
パスワードのハッシュ保存・履歴除外、取消、リサイズ、ログ出力中の行編集、終了を含む。
3 OSの同等試験は`mcp-validation.yml`で実行する。
画面の外観の人手確認と3 OSのCI完了は、自動入力試験とは分けて扱う。
