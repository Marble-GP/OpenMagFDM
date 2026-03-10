# OpenMagFDM v1.3.0 Release Notes

## Overview

v1.3.0 は5つの主要機能強化を含みます。

1. **REST API 拡張** — WebUI の外部プログラムからの制御・自動化に対応
2. **OpenMP 並列化** — ソルバーのマルチコア対応による高速化
3. **永久磁石磁化モデル** — 永久磁石を YAML 設定で定義可能に
4. **カラー検出 UI** — 入力画像から材料色を自動検出し YAML テンプレートを生成
5. **材料ライブラリ** — 材料プリセット（B-H カーブ等）を別 YAML ファイルで管理・再利用

---

## New Features

### 1. REST API 拡張 (A1–A5)

WebUI サーバー (`server.js`) に新しいエンドポイントを追加。外部スクリプトやツールからソルバーを制御できます。

#### インフラ系エンドポイント (A1)

| エンドポイント | 機能 |
|-------------|------|
| `GET /api/health` | ソルバーバイナリ存在確認・ディレクトリ書き込み権限チェック |
| `GET /api/solver/info` | バージョン・対応座標系・機能一覧を取得 |

#### 設定バリデーション (A2)

```
POST /api/validate-config
```

YAMLテキストをサーバー側で解析し、必須フィールドの欠落・不正値を事前チェック。
`coordinate_system`、`materials`、`mesh`（Cartesian）、`polar_domain`（Polar）を検証。

#### 素材色検出・YAMLテンプレート自動生成 (A3)

```
POST /api/materials/detect  (multipart: image file)
     ?rareThreshold=0.05&blendTolerance=8&minColorDist=30
```

PNG/BMP を解析してユニーク色を列挙し、`materials:` セクションの YAML テンプレートを自動生成。
人手でRGB値を書く手間を省きます。

**アンチエイリアス (AA) 検出**（`3c52ab6`）:
画素出現率が `rareThreshold`（デフォルト 5%）以下の色について、出現率上位色のペアの加重平均で表現できるか検証します。
判定基準は以下の通りです。

| チェック項目 | 内容 |
|-----------|------|
| ペア距離 | A, B 間のユークリッド距離 ≥ `minColorDist`（デフォルト 30） |
| t 値の一貫性 | 各チャンネルで推定した補間比 t の最大・最小差 ≤ 0.15 |
| t 値の範囲 | t ∈ [-0.05, 1.05]（境界値を許容） |
| 残差 | 各チャンネルの再構成誤差 ≤ `blendTolerance`（デフォルト 8/255） |

AA と判定された色はレスポンスの `aaBlends` 配列に格納され、生成 YAML には含まれません。
AA の基底となった2色は `antialias: true` フラグ付きで `materials:` セクションに出力されます。

**レスポンス形式:**
```json
{
  "colors":    [{ "rgb": [R,G,B], "ratio": 0.45, "antialias": true }, ...],
  "allColors": [{ "rgb": [R,G,B], "count": 1234, "ratio": 0.45 }, ...],
  "aaBlends":  [{ "rgb": [R,G,B], "ratio": 0.01, "baseA": [R,G,B], "baseB": [R,G,B], "t": 0.5 }, ...],
  "yamlTemplate": "# Auto-generated...\nmaterials:\n  ..."
}
```

#### 非同期ジョブキュー (A4)

| エンドポイント | 機能 |
|-------------|------|
| `POST /api/jobs` | ジョブ投入（即時 `jobId` 返却） |
| `GET /api/jobs?userId=X` | ユーザーのジョブ一覧 |
| `GET /api/jobs/:jobId` | ジョブ状態・ログ末尾取得 |
| `DELETE /api/jobs/:jobId` | 実行中ジョブのキャンセル |

SSEによるリアルタイムストリーミングに加え、ポーリング型の非同期APIが利用可能になりました。

#### フィールド値クエリ (A5)

```
GET /api/results/:resultFolder/field-at-point?x=0.01&y=0.05&step=0&userId=X
```

物理座標 (x, y) [m] を指定して Az・Bx・By・|B| を双線形補間で取得。
Cartesian・Polar 両座標系に対応。

---

### 2. OpenMP 並列化 (B1–B4)

ソルバーをマルチコアで実行できるようになりました。OpenMP が利用できない環境では従来通りシングルスレッドでビルドされます。

#### 並列化対象

| 関数 | 手法 | 効果 |
|------|------|------|
| `interpolateToFullGrid()` | flat index + `static` schedule | 補間の高速化 |
| `interpolateInactiveCells()` | flat index + `static` schedule | 同上 |
| `interpolateInactiveCellsPolar()` | flat index + `static` schedule | 極座標版補間 |
| `generateCoarseningMaskCartesian()` | flat index + `reduction` | マスク生成 |
| `generateCoarseningMaskPolar()` | flat index + `reduction` | 極座標版マスク |
| `calculateBFieldAtActiveCells()` | active-cell loop + `static` | B場計算 |
| `buildMatrix()` | thread-local triplets + `critical` merge | 行列構築 |
| `buildMatrixCoarsened()` | 同上 | coarsening版行列構築 |
| `buildMatrixPolar()` | 同上 | 極座標版行列構築 |
| `buildMatrixPolarCoarsened()` | 同上 | 極座標coarsening版 |
| NK Az コピーループ | `static` schedule | Newton-Krylov 前処理 |

#### Windows / MSVC 互換

`collapse(2)` 句（MSVC OpenMP 2.0 未対応）は使用せず、ループを `k = j*nx + i` でフラット化することで Windows ビルドとの互換性を維持しています。

#### 注意事項

- Eigen SparseLU / AMGCL ソルバー本体は並列化されていません（スレッドセーフの制約）
- `calculateHFieldAtActiveCells()` は内部で YAML ノードアクセスがあるため並列化対象外

---

### 3. 永久磁石磁化モデル (C1–C6)

YAML に `magnetization` ブロックを追加するだけで永久磁石を定義できます。

#### 物理モデル

磁化 **M** から等価磁化電流 J_z_mag を計算し、ソース項に加算します。

**Cartesian:**
```
Jz_mag = ∂My/∂x - ∂Mx/∂y
```

**Polar:**
```
Mr  = Mx·cos(θ) + My·sin(θ)
Mθ  = -Mx·sin(θ) + My·cos(θ)
Jz_mag = (1/r)·∂(r·Mθ)/∂r - (1/r)·∂Mr/∂θ
```

均一磁化領域の内部では ∇×M ≈ 0 となり、境界で自然に表面磁化電流が生じます。
磁化は Az に依存しないため Newton 反復でも RHS 定数として扱われます。

#### 対応磁化パターン

| pattern | 説明 |
|---------|------|
| `parallel` | 均一方向磁化（角度 `angle` で指定） |
| `halbach_continuous` | 連続 Halbach 配列 |
| `polar_anisotropy` | ピッチ円上の集中電流の重ね合わせに沿った磁化 |
| `custom` | ユーザー定義式 (tinyexpr) |

#### YAML 設定例

```yaml
materials:
  magnet_n:
    rgb: [255, 200, 0]
    mu_r: 1.05
    magnetization:
      Hc: 900000        # 保磁力 [A/m]
      pattern: parallel
      angle: 90         # 磁化方向 [deg]、0=+x
```

詳細は [`permanent_magnet_guide.md`](permanent_magnet_guide.md) を参照してください。

---

### 4. カラー検出 UI (D1–D3)

Run & Preview タブの Input Image パネルから `/api/materials/detect` を直接呼び出し、結果をモーダルで確認・編集に反映できます。

#### 操作フロー

```
1. Run & Preview タブで画像をアップロードまたは選択
2. 「Detect Colors & Generate YAML Template」ボタンが表示される
3. ボタンをクリック → 検出結果モーダルが開く
4. 必要に応じてしきい値を調整して「Re-detect」
5. 「Insert materials: section into Editor」で Config Editor に反映
```

#### 検出結果モーダル

| 要素 | 内容 |
|------|------|
| **Rare threshold** | 占有率がこの値 [%] 以下の色を AA 候補とみなす（デフォルト: 5） |
| **Blend tolerance** | AA 判定の再構成誤差許容値 [/255]（デフォルト: 8） |
| **Detected Colors** | 支配色のスウォッチ一覧（hex + 占有率 + "AA base" バッジ） |
| **AA Blends** | AA ブレンド色の一覧（ブレンド式とブレンド比を表示） |
| **Generated YAML Template** | `materials:` セクションのプレビュー |

#### エディタへの挿入

「**Insert materials: section into Editor**」ボタンは、現在のエディタ内容を js-yaml でパースし、`materials:` キーのみを検出結果で置き換えます。他のセクション（`coordinate_system`、`boundary_conditions`、`transient` 等）は保持されます。

```
実行前: { coordinate_system: ..., materials: { old_iron: ... }, transient: ... }
実行後: { coordinate_system: ..., materials: { material_ffffff: ..., material_808080: ... }, transient: ... }
```

挿入後は自動的に Config Editor タブへ遷移します。

---

### 5. 材料ライブラリ (E1–E3)

材料の透磁率特性（B-H カーブ）や永久磁石パラメータを専用の YAML ファイル（材料ライブラリ）として管理し、複数の解析設定から再利用できます。ソルバー（C++ バイナリ）への変更は不要です。

#### 材料ライブラリ YAML の構造

```yaml
# my_materials.yaml

material_presets:
  silicon_steel_m19:
    mu_r:
      type: bh_curve
      H: [0, 200, 500, 1000, 2000, 5000, 10000]
      B: [0, 0.6,  1.0,  1.3,  1.55, 1.75, 1.85]

  neodymium_n42:
    mu_r: 1.05
    magnetization:
      Hc: 950000
      pattern: parallel
      angle: 0

materials:           # rgb なし → preset として扱われる
  air_standard:
    mu_r: 1.0
    jz: 0.0
```

解析設定 YAML からは `preset:` キーでプリセットを参照します。

```yaml
# analysis_config.yaml
materials:
  core:
    rgb: [128, 128, 128]
    preset: silicon_steel_m19   # ← ライブラリのプリセット名
  magnet:
    rgb: [255, 200, 0]
    preset: neodymium_n42
    magnetization:
      angle: 90                 # ← 解析設定側の値が優先される
```

#### マージ動作（サーバー側）

ソルバー起動前にサーバーが自動でマージを実行します。

```
ライブラリの material_presets
+ ライブラリの materials（rgb フィールドを除去）
→ 解析設定の material_presets にマージ
  （同一キーは解析設定側が優先）
→ 一時 YAML ファイルとしてソルバーに渡す
→ ソルバー終了後に一時ファイルを削除
```

#### Material Library Manager UI

Config Editor タブの下部に「**Material Library Manager**」ボタンを配置。クリックするとモーダルが開きます。

**左ペイン — ファイル一覧:**
- ユーザーのライブラリファイル一覧を表示
- 「**Upload**」: ローカルの `.yaml`/`.yml` ファイルをアップロード
- 「**New**」: ファイル名を入力してデフォルトテンプレートから新規作成

**右ペイン — Edit タブ:**
- Ace Editor（YAML モード、monokai テーマ）でファイル内容をインライン編集
- 「**Save**」: 編集内容をサーバーに保存

**右ペイン — B-H Curves タブ:**
- `material_presets` または `materials` 内の `mu_r.type: bh_curve` エントリを自動検出
- Plotly.js で B [T] vs H [A/m] カーブを重ね描き表示

**フッターボタン:**

| ボタン | 動作 |
|--------|------|
| **Delete File** | 選択中のファイルを削除（確認ダイアログあり） |
| **Use This Library** | 選択中のファイルをアクティブライブラリに設定 |
| **Save** | 編集内容を保存 |
| **Close** | モーダルを閉じる |

アクティブなライブラリは Config Editor タブにバッジとして表示されます（`Library: my_materials.yaml`）。バッジの × をクリックすると解除されます。

#### Material Library API

| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/api/material-libraries?userId=X` | ライブラリファイル一覧 |
| POST | `/api/material-libraries` | ファイルアップロード（multipart） |
| GET | `/api/material-libraries/:filename?userId=X` | ファイル内容取得 |
| PUT | `/api/material-libraries/:filename` | 内容保存（JSON: `{content, userId}`） |
| DELETE | `/api/material-libraries/:filename?userId=X` | ファイル削除 |

すべてのエンドポイントでパストラバーサル対策と YAML 構文検証を実施します。

---

## Bug Fixes

- WebUI: bulk delete ルートが parameterized ルートに隠れていた問題を修正 (`7fb8c0e`)
- WebUI: surface concentration の B 計算精度向上（全グリッドステンシル使用）
- Solver: 非線形材料（35H300 等）+ 適応粗大化での Newton-Krylov 収束失敗を修正 (`c69f376`)
  - **問題**: Picard 反復のスペクトル半径 ρ = μ_eff/μ_diff > 1（Si-steel 飽和域）でソルバーが発散または周期2振動に陥っていた
  - Picard → **Newton ステップ**: A(μ_eff) の代わりに微分透磁率 A(μ_diff) で反復行列を構成し ρ < 1 を保証
  - **周期2振動対策**: Anderson 加速 m=2 を有効化し、交互に正負を繰り返す反復を固定点へ外挿
  - **台地検出**: 直近5反復の残差が2倍幅に収まった時点で粗フェーズ収束を宣言（粗大化誤差フロア対応）
  - **ファイン仕上げ Newton**: 粗収束後の全グリッド補正を Picard → Newton に変更（Picard 発散を防止）
  - テスト結果: 35H300 + polar + coarsen_ratio=8 → 10 粗反復 + 2 ファイン反復 = R=8.14×10⁻⁶

## Technical Improvements

- `getJzPolar()` の再利用により極座標版行列構築でも Jz_mag を統一的に扱える実装
- `setupMaterialPropertiesForStep()` に `computeMagnetizationGrids()` を統合し、スライディングメッシュ時も磁化を逐次更新
- `updateMuDiffAtActiveCells()`: 粗活性セルの微分透磁率 μ_diff = dB/dH / μ₀ を中心差分で計算（Newton 粗フェーズ用）
- `updateMuDiffDistribution()`: 全グリッド版 μ_diff 更新（H_map[j,i] 使用、ファイン仕上げ Newton 用）
- Phase 6 収束判定: Az ステップサイズ + 台地検出の2段階フォールバック追加

## API Endpoints (新規追加)

| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/api/health` | ヘルスチェック |
| GET | `/api/solver/info` | ソルバー情報 |
| POST | `/api/validate-config` | YAML バリデーション |
| POST | `/api/materials/detect` | 素材色検出・AA 検出・テンプレート生成 |
| POST | `/api/jobs` | 非同期ジョブ投入（`materialLibraryFile` 対応） |
| GET | `/api/jobs` | ジョブ一覧 |
| GET | `/api/jobs/:jobId` | ジョブ状態取得 |
| DELETE | `/api/jobs/:jobId` | ジョブキャンセル |
| GET | `/api/results/:folder/field-at-point` | 指定座標のフィールド値 |
| GET | `/api/material-libraries` | ライブラリファイル一覧 |
| POST | `/api/material-libraries` | ライブラリファイルアップロード |
| GET | `/api/material-libraries/:filename` | ライブラリファイル内容取得 |
| PUT | `/api/material-libraries/:filename` | ライブラリファイル内容保存 |
| DELETE | `/api/material-libraries/:filename` | ライブラリファイル削除 |

## Commits

```
9192934 Test+Config: Add Cartesian coarsening test and document coarsening settings
c69f376 Solver: Fix Phase 6 convergence with Newton fine finishing + plateau detection
1bd46c5 Docs: Add Phase 6 Newton convergence report and remaining tasks list
e91334b Solver: Replace Phase 6 FVM-Picard with True Defect Correction
8214b62 Solver: Add fine-finishing full-grid Picard after coarse convergence
d823591 WebUI: Add material library manager UI and solver integration
5762d42 WebUI: Add material library API and merge logic
7ee1334 WebUI: Add Detect Colors & Generate YAML Template feature
3c52ab6 WebUI: Anti-aliasing detection in /api/materials/detect
d74ab45 Permanent Magnet: C1-C6 magnetization model implementation
9bbfe9e OpenMP B2-B4: Parallel loops for solver hot paths
d78245a Build: Add optional OpenMP support to CMakeLists.txt
4e2d1ad WebUI: Add /api/results/:folder/field-at-point endpoint
1ef3165 WebUI: Add async job queue API (/api/jobs)
d546b8a WebUI: Add /api/materials/detect endpoint for YAML template auto-generation
c56cdf6 WebUI: Add /api/validate-config endpoint
bd0a109 WebUI: Add /api/health and /api/solver/info endpoints
7fb8c0e WebUI: Fix bulk delete route shadowed by parameterized route
```

## Breaking Changes

なし。v1.2.x と完全な後方互換性があります。

- `magnetization` ブロックなしの既存設定は従来通り動作します。
- 材料ライブラリを指定しない場合（`materialLibraryFile: null`）は従来通りのソルバー起動になります。
- Detect Colors ボタンは画像未選択時は非表示のため、既存 UI フローに影響しません。

## Files Modified

| ファイル | 変更内容 |
|--------|---------|
| `CMakeLists.txt` | OpenMP optional find_package 追加 |
| `MagneticFieldAnalyzer.h` | MagnetizationConfig struct、メンバ変数、メソッド宣言 |
| `MagneticFieldAnalyzer.cpp` | OpenMP 並列化、永久磁石実装（computeMagnetizationGrids 等）、buildMatrix RHS 統合、updateMuDiffDistribution / updateMuDiffAtActiveCells 追加 |
| `MagneticFieldAnalyzer_nonlinear_newton.cpp` | NK Az コピーループ並列化、Phase 6 Newton-Picard + Anderson m=2 + 台地検出 + ファイン仕上げ Newton |
| `webui/server.js` | REST API 9本、AA 検出強化、材料ライブラリ API 5本、solve-stream/jobs マージ対応 |
| `webui/package.json` | jimp 依存追加 |
| `webui/public/index.html` | Detect Colors ボタン・モーダル、Material Library Manager モーダル、lib-tab CSS |
| `webui/public/app.js` | detectColors/insertMaterialsSection 等 D 系 6 関数、openLibraryManager 等 E 系 12 関数、runSolver 更新 |

## Installation

Download the appropriate package for your platform:
- **Windows**: `OpenMagFDM-Installer-Windows-x86_64.exe` (recommended) or `OpenMagFDM-Windows-x86_64.zip`
- **Linux**: `OpenMagFDM-Linux-x86_64.tar.gz`
- **macOS**: `OpenMagFDM-macOS-x86_64.tar.gz`

---

**Full Changelog**: https://github.com/Marble-GP/OpenMagFDM/compare/v1.2.0...v1.3.0
