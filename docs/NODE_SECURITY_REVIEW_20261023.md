# Node.js セキュリティ・互換性診断（2026-10-23）

## 結論

インストール済みの Node.js v24.22.1 は、2026-10-21 のセキュリティ修正を含む24系LTSです。更新したNodeで既存の100件のNode回帰試験が通り、今回確認した範囲ではNode更新による動作不具合は検出しませんでした。

一方、更新前の配布方法と画像関連依存には問題がありました。以下の対策をリリース用ブランチに実装しました。公開済みの配布物を置き換えた記録ではありません。

| 対象 | 診断時 | 対策後 |
|---|---|---|
| PCのNode / npm | 24.22.1 / 11.19.1 | 変更なし |
| ローカルに残っていた候補EXEの内蔵Node | **24.20.0**（EXEを実行して確認） | **24.22.1**（再構築EXEの実測） |
| パッケージ方式 | pkg標準方式、`node24`指定 | SEA方式、公式Nodeの24.22.1に固定 |
| Jimp / file-type | 0.22.12 / 16.5.4 | 1.6.1 / 21.3.4 |
| Multer | 1.4.5-lts.2 | 2.5.0 |
| npm audit | Jimp由来のmoderate 4パッケージ（根本の勧告は1件） | **0件**（開発依存を含む） |

公式情報：[Node.js 2026-10-21 セキュリティ告知](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases)。DNS、TLS、Brotliなどの修正を含みます。個々のCVEの存在と、本アプリへの具体的な攻撃成立は別です。本診断では侵入や情報漏洩が起きた証拠は確認していません。

## 検出した問題と変更

### 配布EXEのNodeがPC側とは独立していた

`@yao-pkg/pkg 6.23.0` が使用する `pkg-fetch 3.6.6` の24系バイナリーは24.20.0でした。`setup-node: 24`やPCのNode更新だけでは、そこへ取り込まれる実行環境は更新されません。診断時の既存EXEに `PKG_EXECPATH=PKG_INVOKE_NODEJS` を指定し、`process.versions`から24.20.0を確認しました。

公式Nodeを利用するSEA方式へ変更し、ルートの `.node-version` をビルド元・取り込み先の共通指定にしました。現在は24.22.1です。`npm run package`、MCP検証、リリースCIを同じビルドスクリプトへ接続しています。公式配布物のチェックサム検証はpkgのSEAダウンロード処理が行います。

`webui-server.exe --runtime-info` はサーバーを起動せず、実際のNode・構成ライブラリ・OS・CPU情報をJSONで返します。同じOS/CPU向けビルドでは、ビルド後にEXEを起動して版を照合し、不一致なら失敗します。クロスビルドは固定版を指定して構築し、生成する `runtime.json` に `runtimeVerified: false` と明記します。実行可能な各OSでの受入試験は別途必要です。ファイルのSHA-256も記録します。

参照：[pkg公式の標準方式とSEA方式の説明](https://yao-pkg.github.io/pkg/guide/sea-vs-standard)。SEAへの切替に伴いJimpのサブパスを明示的に梱包しました。実行時の依存不足がないことをWindowsの配布EXE試験で確認しています。

### 画像判定とアップロード

- **CVE-2026-31808 / file-type**：不正ASFデータの判定で処理が停止し得る既知の問題です。MCP画像経路には形式の事前検査がありましたが、WebUIの材料色検出はJimpへ直接渡していました。Jimpを1.6.1へ更新し、修正版file-typeへ移行しました。材料色検出にもPNG/JPEG/BMPの署名、10 MiB、16メガピクセルの事前検査を適用しました。[開発元の勧告](https://github.com/sindresorhus/file-type/security/advisories/GHSA-5v7r-6r5c-r473)
- **CVE-2025-47935 / Multer**：旧版のストリーム処理にDoS問題があります。今回のnpm auditには表示されませんでしたが、実際の1.4.5-lts.2は公式勧告の対象です。修正を含む2.5.0へ更新しました。[開発元の勧告](https://github.com/expressjs/multer/security/advisories/GHSA-44fp-w29j-9vj5)
- 前の作業で更新したfollow-redirectsの旧依存経路は、今回のJimp更新で依存ツリーからなくなりました。
- 画像前処理の読み込みを作業領域のパス検査へ統一しました。`filename: ../other/secret.png`のような他領域参照を拒否します。認証済みであることだけを画像パスの安全性の根拠にしません。
- 不正アップロードの一時ファイルを`finally`で回収し、極座標変換の出力サイズにも16メガピクセルの上限と有限値検査を追加しました。

Jimpのメジャー更新に伴い、名前付きexport、コンストラクター、PNG出力APIを変更しました。2Dソルバーの数値計算部分、YAML形式、解析アルゴリズムは変更していません。

## 確認した動作

Windows、Node24.22.1で実施しました。

| 試験 | 結果 |
|---|---|
| 既存Node回帰 | 100 / 100成功（更新直後、依存変更後） |
| `npm audit --json` | 全severity 0 |
| HTTPS API結合 | ログイン・CSRF・2利用者間の所有権・旧URL拒否・冪等ジョブ・実ソルバー・入力保存・ZIP・保存先移行が成功 |
| 画像API（ソース起動・配布EXE） | PNG/JPEG/BMP、材料色検出、極座標変換、量子化、PNG再読込が成功。不正画像・壊れたmultipart・巨大出力・他領域パスを拒否し、サーバー継続と一時ファイル回収を確認 |
| 配布EXE（外部NodeをPATHから除外） | 内蔵24.22.1、管理設定、HTTP/stdio MCP、画像取込、実ソルバー、未収束警告、結果取得、権限・失効が成功 |
| Chrome155.0.8059.85 / Edge157.0.4341.10 | 専用のheadlessプロファイルで再接続・description保存・縦横比設定保持・Az/磁力線・各16回のステップ変更が成功。画面画像も確認 |

ブラウザー試験の待ち時間込みの値は性能ベンチマークではありません。30分の耐久試験、全モデルの数値・画像比較、手動の全操作を完了したという意味ではありません。

## 再実行

リポジトリー直下から実行します。Nodeは `.node-version` と同じ版を使用してください。

```powershell
node --version
npm --prefix webui ci
npm --prefix webui test
npm --prefix webui audit
node webui/tests/image_api_smoke.cjs
node webui/tools/build-mcp-package.cjs
.\build_mcp\webui-server.exe --runtime-info
node webui/tests/image_api_smoke.cjs --exe build_mcp/webui-server.exe
node webui/tests/packaged_mcp_smoke.cjs --exe build_mcp/webui-server.exe --solver <MagFDMsolver.exeの絶対パス>
```

実ソルバー試験では必要なDLLを利用可能にしてください。HTTPS結合試験は `webui/tests/v2_server_smoke.cjs`、ブラウザー試験は `webui/tests/browser_release_smoke.cjs` の引数説明を参照してください。名前にv2を含む前者は、今回の実用機能を検証するもので3D計算ではありません。

## 残る確認範囲

- Linux/macOSの新しい配布方式とWindowsインストーラーは、このWindows端末では受入確認していません。CIへ同じ版の指定・audit・画像API・内蔵Node検査を追加しました。GitHub上のCIを実行したという記録ではありません。
- 長いWindows保存先で、C++ソルバーが`EnergyDensity/step_0001.csv.tmp`を開けず終了するケースを検出しました。短い試験先では同一Node・同一ソルバーで成功しました。パス長制約の別課題として残ります。
- 既知のNode/npm勧告、今回読んだAPI経路、回帰・結合試験に基づく診断です。未知の脆弱性、すべての入力パターン、C++依存やブラウザー用同梱ライブラリの全件監査まで安全性を保証するものではありません。
- v1.7.1の3D除外方針を維持します。タグ・GitHub Releaseの公開、公開済み全EXEの検査・差替えは本作業では行っていません。
