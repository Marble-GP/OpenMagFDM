# OpenMagFDM

OpenMagFDM は、画像で定義された矩形一次メッシュ空間に対して磁界計算を行うツール群です。
コアは C++ で書かれた数値ソルバー(MagFDMsolver)と、結果の可視化や操作を行う Node.js ベースの WebUI から構成されています。

<!-- <img width="1452" height="914" alt="thumbnail" src="https://github.com/user-attachments/assets/5b6d7e0b-32bf-48f7-bb47-30206aad8b22" /> -->
[!['チュートリアル動画'](https://github.com/user-attachments/assets/5b6d7e0b-32bf-48f7-bb47-30206aad8b22)](https://youtu.be/YvBkXMhFDTs)


## 概要

本プロジェクトは上記の二つの主要コンポーネント（C++ ソルバーと WebUI）で構成されています。

- コア: C++ ソルバー（C++17 準拠）
- ビルド: CMake を使用
- 同梱ライブラリ: `tinyexpr`、`amgcl`（ソースをリポジトリ内に含む）
- Web UI: Node.js サーバを起動してブラウザから `http://localhost:3000` にアクセス

## 特徴

- 画像からメッシュを生成して有限差分（FDM）法で磁界解析を行う
- **非線形透磁率材料対応**（Newton-Krylov法 + Anderson加速による高速収束）
- **複数の電磁力評価手法**（束縛電流法・仮想仕事法など）
- **ユーザ定義変数**による柔軟な数式記述
- 直交座標系・極座標系の両対応
- 周期境界条件・過渡解析（回転機シミュレーション等）
- 高速な反復ソルバーに `amgcl` を利用
- 軽量数式評価に `tinyexpr` を利用
- WebUI によるインタラクティブな結果確認

## ダウンロード

### プリビルドバイナリ（推奨）

[Releases](https://github.com/Marble-GP/OpenMagFDM/releases) から各プラットフォーム向けのビルド済みパッケージをダウンロードできます：

- **Linux** (x86_64): `OpenMagFDM-Linux-x86_64.tar.gz`
- **Windows** (x86_64): `OpenMagFDM-Windows-x86_64.zip` または `OpenMagFDM-Installer-Windows-x86_64.exe`（インストーラ）
- **macOS** (x86_64): `OpenMagFDM-macOS-x86_64.tar.gz`
- **WebUI Standalone**: Node.js不要の単体実行可能ファイル（全プラットフォーム対応）

### 依存関係

プリビルドバイナリを使用する場合、以下のライブラリが必要です：
- Linux: `libeigen3-dev`, `libopencv-dev`, `libyaml-cpp-dev`（apt経由でインストール）
- Windows: インストーラ使用時はDLL同梱
- macOS: `eigen`, `opencv`, `yaml-cpp`（Homebrew経由でインストール）

## ソースからビルド

### 必要環境

- C++ コンパイラ（C++17 以降）
- CMake 3.5 以降
- Eigen3
- OpenCV
- yaml-cpp
- Node.js 18.x 以降 (WebUI を使う場合)

### ビルド方法 (C++ ソルバー)

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

ビルド後、`build/MagFDMsolver` が生成されます。

### 実行例

```bash
./build/MagFDMsolver ./sample_config.yaml ./uploads/sample_PMSM.png
```

## Web UI の起動

```bash
cd webui
npm install    # 初回のみ
node server.js
```

ブラウザで `http://localhost:3000` にアクセスしてください。
すべての機能（設定編集・解析実行・結果可視化・ダッシュボード）が統合されています。

---

## YAML設定ファイル

### 基本構造

```yaml
# 座標系設定
coordinate_system: cartesian  # または polar

# メッシュ設定
mesh:
  dx: 0.2e-3  # [m]
  dy: 0.2e-3  # [m]

# 境界条件
boundary_conditions:
  left:   { type: dirichlet, value: 0.0 }
  right:  { type: dirichlet, value: 0.0 }
  bottom: { type: periodic }
  top:    { type: periodic }

# 材料定義
materials:
  air:
    rgb: [255, 255, 255]
    mu_r: 1.0
    jz: 0.0
    calc_force: false
```

### ユーザ定義変数

`variables:` セクションでユーザ変数を定義できます。数式内で `$変数名` として参照可能です。

```yaml
variables:
  freq: 60              # 周波数 [Hz]
  omega: 2*pi*freq      # 角周波数（他の変数を参照可能）
  J0: 1.0e6             # 電流密度振幅 [A/m²]
  poles: 4              # 極数

materials:
  coil_U:
    rgb: [255, 0, 0]
    mu_r: 1.0
    jz: $J0 * cos($omega * $step / 100)  # 変数を使用
```

### 予約変数一覧

| 変数名 | 説明 | 使用例 |
|--------|------|--------|
| `$step` | ステップ番号（過渡解析時） | `jz: 1e6*sin(2*pi*$step/100)` |
| `$H` | 磁界強度 [A/m]（非線形材料用） | `mu_r: 5000/(1+($H/1e4)^2)` |
| `$dx`, `$dy` | メッシュサイズ [m]（直交座標） | 数式内で使用可能 |
| `$dr`, `$dtheta` | メッシュサイズ（極座標） | 数式内で使用可能 |
| `$N` | 材料ピクセル数 | 総電流計算等 |
| `$A` | 材料面積 [m²] | 電流密度計算等 |
| `pi` | 円周率 | `theta_range: pi/6` |
| `e` | 自然対数の底 | 数式内で使用可能 |

---

## 非線形材料対応

### 定義方法

```yaml
materials:
  iron_core:
    rgb: [128, 128, 128]
    # 方法1: 数式で定義（μ_eff = B/H）
    mu_r: 5000 / (1 + ($H / 5e4)^2)

    # 方法2: B-Hテーブルで定義（カタログデータ形式）
    # mu_r: [[H値列], [μ_eff値列]]
    # mu_r: [[0, 100, 500, 1000, 5000], [5000, 4800, 3000, 1500, 200]]
```

**重要**: `mu_r` は**実効透磁率** μ_eff = B/H です（カタログで提供される形式）。

### ソルバー設定

```yaml
nonlinear_solver:
  enabled: true
  solver_type: newton-krylov  # picard, anderson, newton-krylov
  max_iterations: 50
  tolerance: 1.0e-5

  # Anderson加速（Picard法と併用可能）
  anderson:
    enabled: true
    depth: 8       # 履歴の深さ
    beta: 0.3      # 緩和係数

  # Line search（Newton-Krylov用）
  line_search_adaptive: true
  line_search_alpha_init: 1.0
  line_search_alpha_min: 1.0e-4

  verbose: true
  export_convergence: true
```

---

## 電磁力評価

### 概要

OpenMagFDMは複数の電磁力評価手法を実装しています：

| 手法 | 説明 | 特徴 |
|------|------|------|
| **束縛電流法（Distributed Amperian）** | F = ∫ J_b × B dV | **推奨**。ギザギザ境界に強い |
| Maxwell応力テンソル法 | F = ∮ T·n dS | 古典的手法 |
| 体積積分法 | F = ∫ (J×B + (M·∇)B) dV | 磁化力を含む |
| **仮想仕事法** | F = -dW/dx | エネルギー微分による評価 |

### 使用方法

```yaml
materials:
  rotor:
    rgb: [100, 100, 100]
    mu_r: 500.0
    jz: 0.0
    calc_force: true  # この材料の電磁力を計算
```

WebUIのダッシュボードで「Force」「Torque」「Virtual Work」プロットを追加して結果を確認できます。

---

## 極座標系解析

回転機など円筒形状の解析に対応しています。

```yaml
coordinate_system: polar

polar_domain:
  r_start: 0.05    # 内半径 [m]
  r_end: 0.10      # 外半径 [m]
  r_orientation: horizontal  # r方向
  theta_range: pi/6          # θ範囲（1極分など）

polar_boundary_conditions:
  inner:    { type: dirichlet, value: 0.0 }
  outer:    { type: dirichlet, value: 0.0 }
  theta_min: { type: periodic }
  theta_max: { type: periodic }
```

---

## 過渡解析（回転シミュレーション）

```yaml
transient:
  enabled: true
  enable_sliding: true        # 画像スライド有効
  total_steps: 100            # 総ステップ数
  slide_direction: vertical   # スライド方向
  slide_region_start: 110     # スライド領域開始 [pixel]
  slide_region_end: 390       # スライド領域終了 [pixel]
  slide_pixels_per_step: 1    # ステップあたり移動量 [pixel]
```

---

## 出力ファイル

解析結果は `results/` ディレクトリに保存されます：

| ファイル | 内容 |
|----------|------|
| `Az/step_XXX.csv` | 磁気ベクトルポテンシャル [Wb/m] |
| `Mu/step_XXX.csv` | 透磁率分布 [H/m] |
| `H/step_XXX.csv` | 磁界強度 \|H\| [A/m]（非線形時） |
| `conditions.json` | 解析条件 |
| `force_results.json` | 電磁力結果 |
| `energy_results.json` | 磁気エネルギー |

---

## 同梱ライブラリ

- `amgcl/` — 高性能な多重格子前処理器
- `tinyexpr/` — 軽量な数式評価ライブラリ

外部依存として別途インストールする必要はありません。

---

## 貢献・バグ報告

バグや機能要望は [Issue](https://github.com/Marble-GP/OpenMagFDM/issues) を立ててください。

### 実装済み機能

- [x] 非線形透磁率材料の計算対応（Newton-Krylov法 + Anderson加速）
- [x] 複数の電磁力評価手法（束縛電流法・仮想仕事法など）
- [x] ユーザ定義変数と予約変数
- [x] 極座標系解析
- [x] 過渡解析（回転機シミュレーション）
- [x] WebUIダッシュボード

### 将来検討中の機能

- [ ] 3次元解析への拡張
- [ ] 入力画像の設計支援ツールの統合

---

## ライセンス

プロジェクト全体のライセンスは `LICENCE` ファイルを参照してください。

## 連絡先

- X(Twitter): [@scalar_subby](https://twitter.com/scalar_subby)
