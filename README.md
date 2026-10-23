# OpenMagFDM

OpenMagFDMは、RGB画像で材料配置を定義する2D磁界解析ツールです。
C++17の有限差分ソルバーと、設定編集・画像前処理・結果表示を行うWebUIで構成されます。
直交／極座標、非線形B–H材料、永久磁石、回転を伴う過渡解析に対応します。

**Current version: v1.7.2** · [Release notes](docs/releases/RELEASE_NOTES_v1.7.2.md) ·
[Changelog](CHANGELOG.md) · [Documentation](docs/README.md)

## v1.7.2の変更

サーバー起動後にコマンドを入力できる常駐コンソールを追加しました。
`OpenMagFDM>`で`help`、`menu`、`configure`を使用でき、
ファイル数制限・アカウント・作業領域の割当を稼働中に変更できます。
Tab補完、履歴、伏字のパスワード入力、ジョブ管理、停止・再起動にも対応します。
コマンドの全引数は[CLIマニュアル](docs/CLI.md)を参照してください。

既存の2D YAML・結果は引き続き利用できます。配布WebUIのNode.jsは**24.22.1**です。
3Dソルバーと実験中のGPU描画は、この改善リリースの配布対象に含めません。

## ダウンロードと起動

[GitHub Releases](https://github.com/Marble-GP/OpenMagFDM/releases)から取得します。

| 環境 | 配布物・起動 |
|---|---|
| Windows x86_64 | インストーラー、またはWindowsソルバーZIP＋WebUI ZIPを同じフォルダーへ展開。`webui-server.exe`を起動 |
| Linux x86_64 | LinuxソルバーとWebUIを配置し、`./webui-server`を起動 |
| macOS | arm64ソルバー＋WebUIを配置。現在のWebUIアーカイブはIntel x64向けで、Apple SiliconではRosetta 2が必要 |

Node同梱WebUIにはNode.jsの別途インストールは不要です。
WebUI単体のアーカイブにはC++ソルバーが含まれません。ソルバーの依存ライブラリも用意してください。
WindowsソルバーはAVX2対応CPUを使用します。

起動後、`http://localhost:3000`を開きます。ローカルモードはログイン不要です。
サーバー端末には次のプロンプトが表示されます。

```text
OpenMagFDM> help
OpenMagFDM> menu
OpenMagFDM> configure
OpenMagFDM> status
OpenMagFDM> limits set-default configs 100
OpenMagFDM> jobs list
```

`exit`で終了します。解析中は通常の終了を拒否し、`exit --force`で終了を明示できます。
サービス起動では`--no-console`を使用してください。

## 最初の解析

1. WebUIのFile Managerへ入力画像とYAML設定を保存します。
2. 必要に応じ、画像の色の均一化、材料RGBの検出、極座標変換を行います。
3. YAMLの材料・寸法・電流・磁化・境界条件を確認し、Runで解析を受付します。
4. 進捗・ログを確認し、ResultsでAz・磁束密度・透磁率・磁力線・力などを表示します。
5. descriptionを保存し、必要な入力・結果をZIPでバックアップします。

解析はブラウザーをリロード・終了してもサーバー上で継続します。
再接続後はジョブ一覧から操作を復元します。サーバーを終了するとソルバーも終了します。
サーバー再起動後の待機ジョブは保留となり、`jobs resume <workspace>`で再開できます。

ソルバーは標準配置から自動検出します。見つからない場合は
**Help & Integrations → Local startup settings**で絶対パスを保存し、再起動してください。

## マニュアル

| 目的 | 資料 |
|---|---|
| 設定、材料、数式、境界条件、出力 | [YAML仕様](docs/CONFIGURATION.md) |
| 常駐コンソール、全コマンド、起動引数 | [CLIマニュアル](docs/CLI.md) |
| HTTPから解析・ファイル・結果を操作 | [REST APIマニュアル](docs/REST_API.md) |
| 共有HTTPS、所有者、保存先、更新・バックアップ | [運用マニュアル](docs/OPERATIONS.md) |
| エージェントからのローカル接続 | [MCPマニュアル](docs/MCP_INTEGRATION.md) |
| 過去の検討、数値実験、診断記録 | [アーカイブ](docs/archive/README.md) |

## 主な機能

- RGB材料画像と共用材料ライブラリ、変数・数式による設定
- 直交・極座標、周期／反周期境界、永久磁石の磁化分布
- 非線形B–H材料、Newton–Krylov、AMGCL、OpenMP、任意の領域分割
- 回転・移動を伴う過渡解析、鎖交磁束、電磁力・トルク評価
- TIFF／CSVの結果出力、分布・時系列表示、入力画像がない結果の磁力線表示
- WebUI・REST・MCPで共通の永続ジョブ、入力保存、更新競合検出
- ファイル数制限、アカウント・作業領域所有者、共有HTTPS、結果保存先の変更・移行

<a id="mcp-setup"></a>

## MCP接続

同じPCのエージェントから、割り当てた作業領域の入力・結果を参照し、
許可した操作だけを実行できます。初回はOFFです。

1. WebUIの**Help & Integrations**で、端末に表示されたコードを使いブラウザーをペアリングします。
2. Assigned workspaceを確認し、最初は閲覧権限で接続を作成します。
3. Enable MCPを有効にし、Testで確認します。
4. Settingsが表示するHTTP URLとBearer認証、またはstdioのcommand/argsをクライアントへ登録します。

既定のMCP URLは`http://127.0.0.1:3001/mcp`。
stdioは起動済みサーバーへのブリッジです。サーバーと同じOSユーザー・設置場所を使用します。
詳細、共有モードでの制約、トラブル対応は[MCPマニュアル](docs/MCP_INTEGRATION.md)を参照してください。

## ソースから使用する

WebUIは[.node-version](.node-version)のNode.jsを使用します。

```powershell
cd webui
npm ci
npm start
```

C++ソルバーの主な依存はEigen3、OpenCV、yaml-cpp、libtiffです。
WindowsはVisual Studioとvcpkg、Linux/macOSは各OSの開発パッケージを使用できます。
AMGCLとtinyexprはソースを同梱しています。

```text
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

単体実行は`MagFDMsolver <config.yaml> <image.png> [output_folder]`。
Windowsの例：

```powershell
.\build\Release\MagFDMsolver.exe .\sample_config.yaml .\model.png .\output_example
```

ソルバーの出力は既定でTIFF、倍精度、非同期書込です。旧CSVも選択できます。
詳細は[YAML仕様](docs/CONFIGURATION.md)と[CLI](docs/CLI.md#10-単体2dソルバー)を参照してください。

## 開発・問い合わせ

WebUIの回帰試験は`npm --prefix webui test`。
リリースでは3 OSでNode同梱版・HTTP/MCP・実端末入力を確認し、C++回帰試験も実行します。
不具合は[GitHub Issues](https://github.com/Marble-GP/OpenMagFDM/issues)へ、
使用版・OS・再現手順・認証情報を除いた設定やログを添えて報告してください。
各依存ライブラリのライセンスは同梱のLICENSE・vendor情報を参照してください。
