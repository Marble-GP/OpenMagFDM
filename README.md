# OpenMagFDM

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
- **非線形透磁率材料対応**（Newton-Krylov法による高速収束）
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
- 非線形材料テスト: `test_nonlinear.yaml`
	- 非線形透磁率材料の設定例とNewton-Krylov法のパラメータ設定

## 非線形材料対応
OpenMagFDM は、磁界強度依存の非線形透磁率 μ(H) をサポートしています。

### 定義方法
```yaml
materials:
  core_nonlinear:
    mu_r: 5000 / (1 + ($H / 5e4)^2)  # 実効透磁率 μ_eff = B/H を数式で定義
    # または
    # mu_r: [[H値列], [μ_eff値列]]  # テーブルで定義（カタログデータ形式）
```

**重要**: YAMLで定義する `mu_r` は**実効透磁率** μ_eff = B/H です（カタログで提供される形式）。ソルバーは内部で解析的に微分透磁率に変換します。

### ソルバー
- **Picard法**: 固定点反復（基本手法）
- **Anderson加速**: Picard法の高速化
- **Newton-Krylov法**: Quasi-Newton + Backtracking line search（推奨）
  - 適応的ステップ長調整で安定収束
  - YAMLで詳細なパラメータ調整が可能

詳細は `nonlinear_material_spec.yaml` を参照してください。

### 非線形材料の可視化
非線形材料を含む解析では、出力に以下のファイルが含まれます：

- **Az/**: 磁気ベクトルポテンシャル A_z [Wb/m]
- **Mu/**: 透磁率 μ [H/m] （各位置でμ_r(H)を評価した結果）
- **H/**: 磁界強度 |H| [A/m] （Newton-Krylov収束後の値）
- **conditions.json**: 解析条件（`has_nonlinear_materials`フラグを含む）

#### B-H関係の計算
```
線形材料の場合:
  B = ∇×Az, H = B/(μ₀*μ_r), μ_rは定数

非線形材料の場合（μ_rは実効透磁率 μ_eff = B/H として定義）:
  B = ∇×Az  （常に正確、ソルバーが直接計算）
  H = H.csvから読み込み （Newton-Krylov収束後の値）

  B-H関係式: B(H) = μ_eff(H) * μ₀ * H
  　　　　　（μ_effはカタログで提供される実効透磁率）

  微分透磁率: dB/dH = μ₀ * (μ_eff + H * dμ_eff/dH)
  　　　　　（Jacobian計算用に内部で解析的に変換）
```

**重要**:
1. YAMLで定義する `mu_r` は**実効透磁率 μ_eff = B/H** です（カタログデータ形式）
2. ソルバーはB-H曲線を `B = μ_eff * μ₀ * H` で直接計算します（[MagneticFieldAnalyzer_nonlinear.cpp](MagneticFieldAnalyzer_nonlinear.cpp#L310)参照）
3. Newton-Krylov法のJacobian計算では、解析的に微分透磁率 dB/dH に変換します（[MagneticFieldAnalyzer.h](MagneticFieldAnalyzer.h#L180-L198)参照）
4. WebUIでは、`conditions.json` の `has_nonlinear_materials` フラグを確認し、
   `true` の場合は `H/` フォルダから磁界強度分布を読み込んでください
5. B-Hプロット作成時は、エクスポートされたBとH値を直接使用してください

#### 座標系の注意点
画像座標系（y下向き）と解析座標系（y上向き）では上下反転しています。
WebUIで可視化する際は、csvデータを垂直反転（flip）すると元の画像と同じ向きになります。
詳細は「Field Lines + Boundary」の可視化結果を参照してください。

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

#### 実装済み機能（2024-12-3 更新）
- [ x ] 非線形透磁率材料の計算対応（Newton-Krylov法）
- [ x ] 磁気マクスウェル応力の評価

#### 既知の不具合など
- [ ] 非線形ソルバーの収束が遅い
- [ ] 透磁率の定義範囲外の取り扱い

#### 将来検討中の機能
- [ ] 非空気界面上のマクスウェル応力線積分の除外
- [ ] 入力画像の設計支援ツールの統合


## 連絡先
詳細・質問があれば Issue を立てるか、以下お問い合わせください。

#### SNS
- X(Twitter): @scalar_subby  
