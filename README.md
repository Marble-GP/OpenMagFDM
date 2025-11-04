# OpemMagFDM

OpenMagFDM は、画像で定義された矩形一次メッシュ空間に対して磁界計算を行うツール群です。
コアは C++ で書かれた数値ソルバー(MagFDMsolver)と、結果の可視化や操作を行う Node.js ベースの WebUI から構成されています。  

## 概要
本プロジェクトは上記の二つの主要コンポーネント（C++ ソルバーと WebUI）で構成されています。

- コア: C++ ソルバー（C++11/C++13 準拠を想定）
- ビルド: CMake を使用
- 同梱ライブラリ: `tinyexpr`、`amgcl`（ソースをリポジトリ内に含む）
- Web UI: Node.js サーバを起動してブラウザから `http://localhost:3000/integrated.html` にアクセス



## 特徴
- 画像からメッシュを生成して有限差分（FDM）法で磁界解析を行う
- 高速な反復ソルバーに `amgcl` を利用
- 軽量数式評価に `tinyexpr` を利用
- WebUI によるインタラクティブな結果確認（ローカルホストで起動）

## 必要環境
- C++ コンパイラ（C++13 相当の機能が使えるもの）
- CMake
- make / Ninja などのビルドツール
- Node.js (WebUI を使う場合)

## ビルド (C++ ソルバー)
1. リポジトリのルートでビルドディレクトリを作成します。

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

2. ビルドが成功すると、実行ファイル（例: `MagFDMsolver`）が生成されます。起動方法はビルド設定やターゲットに依存します。詳細は `CMakeLists.txt` を参照してください。
3. ソルバーはコマンドライン引数でPNG画像ファイルとYAMLファイルのパスを要求します。
実行例: 
'''bash
build/MagFDMsolver ./sample_config.yaml　./uploads/sample_PMSM.png 
'''


## Web UI の起動
WebUI は `webui/` ディレクトリにあります。サーバを起動してブラウザで `http://localhost:3000` にアクセスしてください。

簡単な起動手順の例:

```bash
cd webui
npm install    # 初回のみ
node server.js # または package.json に start スクリプトがあれば npm start
```

サーバはデフォルトでポート 3000 をリッスンする想定です。必要に応じて `server.js` の設定を変更してください。　　

自由度の高い解析結果表示ツールは`http://localhost:3000/dashboard.html`で利用可能です。　　
`http://localhost:3000/integrated.html`画面上部の"カスタマイズダッシュボード"からもアクセス可能です。  

## 設定ファイル
- サンプル設定ファイル: `sample_config.yaml`
	- 実行時のメッシュ・境界条件・解析設定などを記述します。

## 同梱ライブラリ
- `amgcl/` — 高性能な多重格子前処理器（ソースを同梱）
- `tinyexpr/` — 軽量な数式評価ライブラリ（ソースを同梱）

これらは外部依存として別途インストールする必要はなく、プロジェクト内でビルドに組み込まれる想定です。依存関係や CMake のオプションは `CMakeLists.txt` を確認してください。

## サンプル実行フロー
1. `build/` でソルバーをビルド
2. `sample_config.yaml` を編集して入力設定を用意
3. ソルバーを実行して結果ファイルを生成
4. `webui/` を起動して結果をブラウザで可視化

## ライセンス
プロジェクト全体のライセンスはリポジトリの `LICENCE` ファイルを参照してください。

## 貢献・バグ報告
- バグや機能要望は Issue を立ててください。

#### 既知の不具合など（2025-11-3）
- [] Jzマップ表示
- [] 磁気マクスウェル応力の評価

#### 将来検討中の機能
- [] 非線形透磁率材料の計算対応
- [] 非空気界面上のマクスウェル応力線積分の除外
- [] 表示UIの改良（integrated.htmlとdashboard.html, 計算のプログレスバー）
- [] YAML-Lint、キーワード補完
- [] 入力画像の設計支援ツールの統合


## 連絡先
詳細・質問があれば Issue を立てるか、リポジトリのメンテイナーにお問い合わせください。

#### SNS
- X(Twitter): @scalar_subby  
