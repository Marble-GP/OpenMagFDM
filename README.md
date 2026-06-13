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
- **永久磁石磁化モデル**（parallel / Halbach / polar anisotropy / custom）
- **適応粗大化メッシュ**（均一領域を自動的に粗くし、境界は高解像度を維持）
- **複数の電磁力評価手法**（束縛電流法・仮想仕事法など）
- **材料ライブラリ**（B-Hカーブ等のプリセットを別YAMLで管理・再利用）
- **ユーザ定義変数**による柔軟な数式記述
- **OpenMP 並列化**による高速計算
- 直交座標系・極座標系の両対応
- 周期境界条件・過渡解析（回転機シミュレーション等）
- アンチエイリアス補間（材料境界の透磁率を自動的に調和平均で補間）
- 高速な反復ソルバーに `amgcl` を利用
- 軽量数式評価に `tinyexpr` を利用
- WebUI によるインタラクティブな結果確認（REST API による外部制御にも対応）

## v1.4 移行ガイド（出力フォーマット変更）

v1.4 から **デフォルトの結果出力フォーマットが CSV から TIFF (IEEE 754 binary) に切り替わりました**。同時に、書き込みも非同期化されています。

### 主な変更点

- **default `format: tiff`**（CSV 比でファイルサイズが約 1/4〜1/5、step あたりの export 時間も短縮）
- **default `async: true`**（書き込みをワーカースレッドに移し、ソルバーが I/O を待たない）
- **`precision: double`** はそのまま（IEEE 754 64-bit、情報落ちなし）
- **WebUI は両形式に対応**（`/api/load-field` が TIFF / CSV を自動判別。format=both のときは TIFF 優先・CSV フォールバック）

### 既存ユーザの選択肢

| やりたいこと | yaml 設定 |
|---|---|
| 新 default（推奨）でそのまま使う | `export:` ブロックなし |
| 旧挙動（CSV のみ）に戻す | `export:\n  format: csv` |
| 過渡的に CSV と TIFF の両方を書く | `export:\n  format: both` |
| ファイルサイズ最優先（精度 32 bit に丸める） | `export:\n  precision: float` |
| 非同期書き込みを切る（同期 I/O） | `export:\n  async: false` |

### CSV を直接消費する外部スクリプトをお使いの方へ

- WebUI の `/api/load-csv` エンドポイントは引き続き使えますが、ログに deprecation warning が出ます。`/api/load-field` への移行を推奨します（クエリは互換、レスポンスに `format` / `precision` フィールドが追加されているのみ）。
- 過渡期間中は `export: { format: both }` を yaml に書くことで CSV と TIFF の両方を出力できます。

### TIFF ファイルの可視化

TIFF (32/64-bit float, FP predictor + DEFLATE) は ImageJ / ParaView / Python `tifffile` などの標準ツールでそのまま読めます。Python 例:

```python
import tifffile
arr = tifffile.imread("output_xxx/Az/step_0001.tiff")
print(arr.dtype, arr.shape)  # float64 (500, 500)
```

## v1.5 リリースノート（WebUI 画像処理パイプライン）

v1.5 では、CAD モータ断面の **スクリーンショットを直接 OpenMagFDM の解析にかけられる状態まで整える前処理 WebUI** が追加されました。コアソルバーには変更ありません。

### Uniform-colour Filter — `/api/preprocess-filter/quantize`

JPEG 由来 / アンチエイリアスで何万色にも分かれた CAD 画像を、少数の "材料色" に圧縮します。

- Input Image パネルで `uniqueColors > 1000` が検出されたら黄色の警告バナーが自動表示
- 「Apply Color Uniformization Filter」ボタンでフィルタ用モーダルを起動
- スライダー: rare threshold (%) / Top N targets / Minimum colour distance / Despeckle radius / Min island size / Bilateral σ-spatial / σ-color
- **Auto-tune** ボタン — N と rare threshold をユーザーが固定し、残りを Nelder-Mead で boundary-noise 最小化方向に探索 (subsample 256 px, 60-80 評価, 1 秒未満)
- 中心ボタンクリック → 即時プレビュー / 1.5 秒 debounce 自動プレビュー / Apply で元画像を量子化済みに差し替え
- ホイールズーム + 中ボタンドラッグでプレビューをパン

### Polar Preprocess — `/api/preprocess-polar/{detect,warp}`

回転機の断面画像から、中心 / 内径 / 外径 / セクタ角 / 周期 N を自動検出し、極座標 warp を出力。

- 自動検出パイプライン (Stage 1–4): 前景マスク + 形状分類 → ヒストグラム精密化 + Hough 補正 → エアギャップ dip 検出 → 360-pt DFT で N-fold 周期検出 (grayscale + RGB のデュアル) + Jacobsen 補間
- インターラクティブ編集: SVG オーバーレイ上の中心 / 内径 / 外径 / θ ハンドルをマウスドラッグ、数値 input は同期、ホイール ±step (Shift = ±10×)、矢印キーで中心を ±1 px ナッジ
- **Air-gap candidate dropdown** — 多重ギャップ構造 (mid-yoke 補助エアギャップ等) で上位スコアが物理エアギャップでない場合、トップ N 候補から手動選択
- **Color grouping** — カラーチップに dropdown (None / A / B / C / D)、Recompute で grouped 周期検出 (3 相 UVW グループ化 → 8-fold 対称性のような色ベースの幾何対称を検出)
- Hybrid preview: 1.5 秒 debounce 自動 warp + Apply Transform 即時実行 + 黄色 dirty バッジ + 160×160 サムネ
- **Save & Insert** — Section 6 ラジオで polar / cartesian を選択
  - polar: `coordinate_system: polar` + `polar_domain: { r_start, r_end, theta_range, r_orientation }` + `image_path` = warp 出力ファイル
  - cartesian: `coordinate_system: cartesian` + `image_path` = 元画像 (`mesh.dx/dy` のデフォルトを補完)

### 既知の制約

- 大画像 (4MP 超) で warp は 1–2 秒 / 回。Polar Preprocess Modal の右上にヒント表示
- 矩形 stator は Hough が rotor 内縁を捕捉、 r_inner は手動補正前提
- Auto-tune は完全に AA に支配された JPEG 系画像 (EEEEpaper outerSPMSM 等) では完璧ではなく、Bilateral σ-spatial を上げる余地あり

### v1.5 追加機能 (Phase A + B)

WebUI 前処理に続き、ソルバー側 + WebUI YAML 生成を強化しました。

**Insert YAML テンプレート自動生成 (Phase A.1)** — Polar Preprocess Modal の "Insert YAML" が `Δθ / Θp` 比 (Θp = 2π/N、N は検出された極対数) で θ-BC を自動選択するようになりました。整数倍偶数 → `periodic (value: 1.0)`、整数倍奇数 → `anti-periodic (value: -1.0)`、整数倍でない → `dirichlet`。cartesian 保存時は `mesh.dx = dy = r_outer_physical / r_outer_px` を自動算出。生成 YAML には選択理由のコメントが付きます。

**Polar Modal UI 整理 (Phase A.2)** — 利用頻度の低い "External shape" / "Color grouping" セクションを削除し、Polar Preprocess Modal は中心・半径・セクタ・出力 nr/nθ + Save target の 4 セクションにフォーカスしました。

**`$var` グローバル展開 (Phase B.1)** — YAML の `variables:` で定義した `$name` トークンが、`materials:` だけでなく `mesh`、`polar_domain`、`polar_boundary_conditions`、`transient`、`flux_linkage`、`nonlinear_solver`、`magnetization` など全フィールドで展開されるようになりました。`mesh: { dx: $cell_size }` や `transient: { total_steps: $N_steps }` がそのまま動きます。Reserved コンテキストトークン (`$step`, `$H`, `$N`, `$A`, `$dx`, `$dy`, `$dr`, `$dtheta`) はこれまで通り材料/フォーミュラ評価時に展開されます。

**Multi-slide for transient (Phase B.2)** — 1 つの過渡解析設定で複数の独立スライド領域を指定できます。各領域は `name / direction / region_start / region_end / pixels_per_step` を持ち、各ステップで独立に circular shift されます。磁気ギア / 多段ロータなどに有用。後方互換あり (旧 `slide_*` キーは 1 要素 `slides` に自動変換)。

```yaml
transient:
  enabled: true
  total_steps: 100
  slides:
    - name: rotor_outer
      direction: vertical
      region_start: 110
      region_end: 390
      pixels_per_step: 5
    - name: rotor_inner
      direction: vertical
      region_start: 50
      region_end: 100
      pixels_per_step: -3   # 逆方向回転
```

Cartesian は完全対応。Polar は `slides[0]` のみ (Multi-slide polar permutation は v1.6 予定、load 時に警告)。

**スライド時の wrap モード (Phase B.5)** — スライドして反対側に出ていった領域の扱いを `wrap_mode` で 3 モードから選べます。デフォルトの `auto` は対応する境界条件タイプから物理的に自然なものを選択します:

- `periodic`: 旧仕様。content を環状に巻き戻す。
- `antiperiodic`: 反周期境界条件と一致。シームを跨いだ画素は `jz` と magnetisation の符号が反転 — 電気機械的に「次のポール」は反対極性、を再現。
- `vacuum`: Dirichlet 境界条件と一致。巻き戻しせず、空白部に `vacuum_rgb` (デフォルト [255,255,255] = air) を埋める。

例: 反周期 θ 境界 + 単一ポール解析

```yaml
polar_boundary_conditions:
  theta_min: { type: periodic, value: -1.0 }   # anti-periodic
  theta_max: { type: periodic, value: -1.0 }
transient:
  enabled: true
  slides:
    - direction: vertical
      region_start: 0
      region_end: 360
      pixels_per_step: 5
      wrap_mode: auto   # → "antiperiodic" が選ばれる
```

実装は per-cell `slide_sign_map` (符号トラッカー) として常駐し、`jz_map` と `(Mx_map, My_map)` 更新時に符号を掛けます。

**矩形領域スライド (Phase B.6)** — Band タイプ (縦/横の短冊) に加えて、2D 矩形領域を切り取って `(dx, dy)` 方向に毎ステップ平行移動させる `rectangle` タイプを追加しました。

```yaml
transient:
  slides:
    - name: linear_mover
      kind: rectangle
      rect: [100, 50, 200, 150]   # [x0, y0, x1, y1] 画像座標 (origin = 左上)
      dx: 2                        # 数値リテラルまたは tinyexpr 数式
      dy: "$omega * cos(2*pi*$step/$N_step)"
      vacuum_rgb: [255, 255, 255]  # 空白部の色 (デフォルト白)

    - name: counter_mover
      kind: rectangle
      rect: [300, 50, 400, 150]
      dx: -1                       # 反対方向に移動
      dy: 0
```

**特徴**:

- **per-region 独立 `dx` / `dy`** — 各矩形が独自の速度を持つ
- **tinyexpr 数式対応** — `$omega`, `$N_step` 等の `$name` は load 時に Phase B.1 で展開済、`$step` のみ実行時に評価
- **画像外はカット** — 矩形が境界を跨いだ場合、超過分の content は破棄 (周期 wrap なし)
- **カット元領域は vacuum** — `vacuum_rgb` (デフォルト [255,255,255] = air, jz=0, mu_r=1) で埋める
- **マルチ矩形の重ね合わせ** — `slides:` リスト後方の矩形が前方の矩形を上書き (overlay)
- **初期位置の重なり警告** — yaml load 時に初期 `rect` 同士が重なってると `WARNING:` で通知
- **小数速度対応** — `dx: 0.5` 等の小数値は float 累積で内部管理し、毎ステップの離散シフトは累積値の round の差分。`0.5` なら 0, 1, 0, 1, ... と交互にシフト

Band と rectangle は同一 `slides:` リスト内で混在可能 (それぞれ独立に処理)。

**材料ペア flux linkage (Phase B.3)** — 各 `flux_linkage` エントリで `material_a` / `material_b` (材料名) を指定すると、`mean(Az over material A pixels) - mean(Az over material B pixels)` を計算します。太い導体や多数巻コイルで「点間の Az 差」が物理的に不適切なケース向け。既存の path variant (`start` + `end`) と同じリスト内で混在可。Cartesian 専用。

```yaml
flux_linkage:
  - name: phase_U                    # 新: 材料ペア variant
    material_a: coil_U_pos
    material_b: coil_U_neg
  - name: phase_V_legacy             # 既存: path variant
    start: [0.0, 0.05]
    end:   [0.1, 0.05]
```

WebUI に `GET /api/get-flux-linkage?result=<folder>` を追加。`FluxLinkage/flux_linkage.csv` を `{ steps: [...], series: { name: [values] } }` 形式で返します。

**反周期 BC 数値検証 (Phase B.4)** — `nonlinear_benchmark/anti_periodic_test/` に AP-BC の数値リグレッションテストを追加。Half-circle AP-BC vs Full-circle periodic + 反対符号コイルの等価性を `L2(diff) / ||Az_ref|| < 1e-3` で確認します。`bash run_test.sh /path/to/MagFDMsolver` で実行可、終了コード 0=PASS、1=FAIL。

```bash
cd nonlinear_benchmark/anti_periodic_test
bash run_test.sh ../../build/MagFDMsolver
```

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
| **束縛電流法（Distributed Amperian）** | F = ∫ J_b × B dV | **デフォルト**。ギザギザ境界に強い |
| **仮想仕事法（Co-Energy）** | F_q = +∂W'/∂q | 電流源系用。非線形材料対応 |
| ~~Maxwell応力テンソル法~~ | ~~F = ∮ T·n dS~~ | 非推奨（ゴースト力問題） |

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

## 永久磁石磁化モデル

YAML に `magnetization` ブロックを追加するだけで永久磁石を定義できます。

```yaml
materials:
  magnet_n:
    rgb: [255, 200, 0]
    mu_r: 1.05
    magnetization:
      Hc: 900000        # 保磁力 [A/m]
      pattern: parallel  # parallel, halbach_continuous, polar_anisotropy, custom
      angle: 90          # 磁化方向 [deg]、0=+x
```

磁化 **M** から等価磁化電流 `Jz_mag = ∂My/∂x - ∂Mx/∂y` を計算し、ソース項に加算します。
Cartesian・Polar 両座標系に対応。詳細は [`permanent_magnet_guide.md`](permanent_magnet_guide.md) を参照してください。

---

## 材料ライブラリ

B-Hカーブや永久磁石パラメータを専用 YAML ファイルで管理し、複数の解析設定から再利用できます。

```yaml
# my_materials.yaml
material_presets:
  silicon_steel_m19:
    mu_r:
      type: bh_curve
      H: [0, 200, 500, 1000, 2000, 5000, 10000]
      B: [0, 0.6,  1.0,  1.3,  1.55, 1.75, 1.85]
```

解析設定から `preset:` キーで参照します。

```yaml
materials:
  core:
    rgb: [128, 128, 128]
    preset: silicon_steel_m19
```

WebUI の Material Library Manager から YAML ファイルの管理・B-H カーブの可視化が可能です。

---

## 適応粗大化メッシュ

均一材料領域のメッシュを自動的に粗くし、材料境界は高解像度を維持します。

```yaml
materials:
  iron_stator:
    rgb: [128, 128, 128]
    mu_r: 1000
    coarsen: true       # 粗大化を有効化
    coarsen_ratio: 8    # 8×8 → 1セル
```

非線形材料でも使用可能（Newton-Krylov の Defect Correction 方式により、物理精度はファイングリッド残差で保証）。

### 既知の制限 (v1.5.1)

**`coarsen_ratio: 4` 以下が silent no-op になる場合があります**。極座標系で物理セル
アスペクト比 `dr / (r_mid · dtheta)` が 1.5 以上ある場合、内部の skip 比計算が
`min(skip_x, skip_y) = 1` を生成し、gradient coarsening の level cap が 1 になって
**全 cell が active 化** → coarsening は機能しない経路に landing します。
実例: IEEJ-D polar IPMSM (nr=450, ntheta=2976, aspect≈1.89) では `coarsen_ratio: 4`
で 0 inactive cells、`coarsen_ratio: 8` で 42.6% reduction。

**v1.5.1 ではこの状況で WARNING を出力**します:

```
WARNING: Coarsening enabled in YAML but 0 inactive cells generated.
The full-grid solve path will run (this is silently equivalent to coarsening:disabled).
Try a larger coarsen_ratio (>= 5 for non-square aspect meshes) or set coarsen:false.
```

WARNING が出たら `coarsen_ratio` を増やすか、`coarsen: false` で full-grid 経路を
明示してください。

### Solver 経路の選択 (v1.5.1)

coarsening の effective 化に応じて 2 つの NK loop 経路に分岐します:

| 条件 | 経路 | 内側 solver | Eisenstat-Walker (Phase BC) |
|---|---|---|---|
| `using_coarsening == false` | Standard direct solve | AMGCL-CG (iterative) | ✅ 適用 |
| `using_coarsening == true` + `use_phase6_precond_jfnk: true` | Phase 6 + Galerkin | AMGCL-CG (Phase BJ-3 で統合) | ✅ 適用 |

v1.5.1 以前は Phase 6 経路が SparseLU 直接呼出で EW (Phase BC) を bypass していました
が、v1.5.1 で両経路が AMGCL+EW で統一されました ([Phase BJ-3](https://github.com/Marble-GP/OpenMagFDM/commits/main))。

### 適応粗大化が IEEJ-D class motor で有効でない理由 (技術ノート)

Gradient coarsening の level 上限 `max_skip_iso` は `min(skip_x, skip_y)` で決まります。
ここで `skip_x, skip_y` は物理セルを概ね正方形にするための rounding 後の値:

```
skip_x = round(sqrt(ratio / aspect))
skip_y = round(sqrt(ratio * aspect))
```

`aspect > 1.5` (radial direction が広い細長メッシュ) では `skip_x` が 1 に rounding され、
`max_skip_iso = min(1, skip_y) = 1` で coarsening は無効化されます。

回避策:
1. `coarsen_ratio` を上げる (例: 8, 16)
2. radial 方向 (`nr`) を増やしてセル aspect を 1 に近づける
3. 直交座標系を使う (Cartesian なら aspect=1 が自然)

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
- [x] 永久磁石磁化モデル（parallel / Halbach / polar anisotropy / custom）
- [x] 適応粗大化メッシュ（coarsen / coarsen_ratio）
- [x] アンチエイリアス補間（材料境界の調和平均）
- [x] 材料ライブラリ（B-Hカーブプリセットの管理・再利用）
- [x] REST API（外部プログラムからのソルバー制御・自動化）
- [x] OpenMP 並列化（マルチコア対応）
- [x] カラー検出 UI（画像から材料色を自動検出）

### 将来検討中の機能

- [ ] 3次元解析への拡張
- [ ] 入力画像の設計支援ツールの統合

---

## ライセンス

プロジェクト全体のライセンスは `LICENCE` ファイルを参照してください。

## 連絡先

- X(Twitter): [@scalar_subby](https://twitter.com/scalar_subby)
