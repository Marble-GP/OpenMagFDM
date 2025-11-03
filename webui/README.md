# MagFDM Visualizer - Web UI

MagFDMsolverの出力CSVファイルをブラウザで可視化するWebアプリケーションです。

## 特徴

- 🌐 **ブラウザベース**: インストール不要、ブラウザで実行
- 📊 **インタラクティブ**: Plotly.jsによるズーム・パン可能な可視化
- 🎨 **カラーマップ選択**: 複数のカラーマップから選択可能
- 📈 **4つの表示**: Az, μ, |B|, |H|を同時表示
- 💾 **ドラッグ&ドロップ**: CSVファイルを簡単に読み込み

## セットアップ

### 初回のみ

```bash
cd webui
npm install
```

## 起動方法

```bash
npm start
```

または

```bash
node server.js
```

サーバーが起動したら、ブラウザで以下にアクセス：

```
http://localhost:3000
```

## 使い方

1. **サーバー起動**
   ```bash
   cd webui
   npm start
   ```

2. **ブラウザでアクセス**
   - `http://localhost:3000` を開く

3. **CSVファイルを読み込み**
   - "ベクトルポテンシャル (Az) CSV" で Az ファイルを選択
   - "透磁率分布 (μ) CSV" で Mu ファイルを選択

4. **パラメータ設定**
   - dx, dy: メッシュ間隔（デフォルト: 0.001 m）
   - カラーマップ: 表示色を選択

5. **可視化実行**
   - "📊 可視化実行" ボタンをクリック

## 表示される物理量

### 1. ベクトルポテンシャル Az [Wb/m]
- 等高線表示
- カラーマップで値を表示

### 2. 透磁率分布 μ [H/m]
- ヒートマップ表示
- 材質分布を確認

### 3. 磁束密度 |B| [T]
- ヒートマップ表示
- Azから自動計算（Bx = ∂Az/∂y, By = -∂Az/∂x）

### 4. 磁界強度 |H| [A/m]
- ヒートマップ表示
- H = B/μ で計算

## インタラクティブ機能

Plotly.jsの機能により、以下が可能：

- **ズーム**: マウスドラッグで範囲選択
- **パン**: ダブルクリック後ドラッグで移動
- **リセット**: ホームアイコンでリセット
- **値の確認**: カーソルを合わせると座標と値を表示
- **画像保存**: カメラアイコンでPNG保存

## ポート変更

デフォルトはポート3000です。変更する場合：

```bash
PORT=8080 npm start
```

## ファイル構成

```
webui/
├── package.json          # npm設定
├── server.js             # Expressサーバー
├── public/
│   ├── index.html       # UIのHTML
│   └── app.js           # 可視化ロジック
└── README.md            # このファイル
```

## 技術スタック

- **バックエンド**: Node.js + Express
- **可視化ライブラリ**: Plotly.js
- **UI**: HTML5 + CSS3 + Vanilla JavaScript

## トラブルシューティング

### ポートが使用中

```
Error: listen EADDRINUSE: address already in use :::3000
```

→ ポート番号を変更：
```bash
PORT=8080 npm start
```

### CSVファイルが読み込めない

- ファイル形式がCSV（カンマ区切り）か確認
- 数値データのみか確認（ヘッダー行がある場合は除去）
- ファイルサイズが大きすぎないか確認

### ブラウザで表示されない

- サーバーが起動しているか確認
- ブラウザのコンソールでエラーを確認（F12キー）
- キャッシュをクリアして再読み込み（Ctrl+Shift+R）

## 開発者向け

### コードの構造

**app.js の主要関数:**
- `parseCSV()`: CSV読み込みと2D配列への変換
- `calculateMagneticField()`: 磁束密度の計算
- `plotHeatmap()`: ヒートマップ描画
- `plotContour()`: 等高線描画
- `visualize()`: メイン可視化処理

### カスタマイズ

カラーマップの追加:
```javascript
// app.js の colormap select に追加
<option value="YourColormap">Your Colormap</option>
```

利用可能なPlotlyカラーマップ:
- Viridis, Hot, Jet, Portland, Blackbody, Electric
- Greys, Blues, Greens, Reds
- など（Plotly.jsドキュメント参照）
