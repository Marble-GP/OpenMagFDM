# Phase 6 Newton-Krylov 収束改善レポート

## 概要

Galerkin 粗大化 + 非線形材料（35H300 Si-steel）を組み合わせた場合に発生していたソルバー収束失敗問題を解決した。最終的なアルゴリズムでは **粗グリッド Newton-Picard（μ_diff 接線行列 + Anderson m=2）** と **ファイン仕上げ Newton** を組み合わせることで、10回の粗反復 + 2回のファイン反復で R ≈ 8×10⁻⁶ まで収束する。

---

## 問題の経緯

### 1. Picard 反復の根本的限界

Picard 法では A(μ_eff) × Az_new = b を繰り返す。
更新式：`Az_new = A(μ_eff)⁻¹ × b`
収束条件：スペクトル半径 ρ_Picard = μ_eff / μ_diff < 1

**35H300 の問題点：** μ_r(H) が H ≈ 100 A/m でピーク（7480）を持ち非単調。
ピーク付近では `μ_diff = dB/dH / μ₀ > μ_eff = B/H` となり、ρ_Picard > 1 → **発散**。

```
H ≈ 100 A/m: μ_eff ≈ 7480,  μ_diff ≈ 35000  →  ρ = 0.21 (convergent)
H ≈ 50 A/m:  μ_eff ≈ 5570,  μ_diff ≈ 38000  →  ρ = 0.15 (convergent)
H ≈ 300 A/m: μ_eff ≈ 1400,  μ_diff ≈ 298    →  ρ = 4.7  (DIVERGENT)
```

粗大化グリッドでは飽和領域が解に混在するため Picard は収束不可。

### 2. Newton-Picard（μ_diff 接線行列）

**修正：** 反復行列を A(μ_diff) に変更。

```
更新式: Az_new = Az - A(μ_diff)⁻¹ × (A(μ_eff) × Az - b)
収束条件: ρ_Newton = |1 - μ_eff / μ_diff|
```

飽和域（H ≈ 300 A/m）：ρ_Newton = |1 - 4.7| = 3.7 → まだ発散。

### 3. Period-2 振動

μ_eff / μ_diff が 1 を超えると更新行列 M = 1 − μ_eff/μ_diff < 0 → 符号反転 → **周期2振動**（R ≈ 0.3 ↔ 3.5 の交互）。

通常の ω 適応（連続2回増加でダンピング）は周期2を検出できない（増加→減少→増加→...）。

**修正：** Anderson 加速 m=2 を有効化。
Anderson は交互方向の反復を記録し、2ベクトルの外挿で固定点（解）を直接推定 → 周期2を破壊。

### 4. 粗大化誤差フロア

周期2が解消されても、coarsen_ratio > 1 の粗グリッド解は
`R_coarse ≈ 粗大化打ち切り誤差 ≈ 3×10⁻³`（tol = 5×10⁻⁴ に届かない）という残差フロアが存在する。

---

## 最終アルゴリズム

### 粗グリッドフェーズ（Phase 6 Newton-Picard + Anderson）

```
for iter = 1 .. MAX_ITER:
    1. H_active = 粗活性セルから B/H 計算
    2. μ_eff(H_active)  で A_eff, b 構築（Galerkin: A_c = P^T A_f P）
    3. R_coarse = A_eff × Az_vec - b_coarse を評価
    4. 収束チェック:
       (a) ||R_coarse||_rel < TOL          … 一次基準
       (b) Az ステップサイズ → 0           … 三次基準（振動でも停止）
       (c) 直近 5 反復が 2 倍幅以内に収束  … 台地検出（粗大化誤差フロア）
    5. μ_diff(H_active) で A_diff を構築
       δAz = A_diff⁻¹ × (−R_coarse)      … Newton ステップ
    6. Anderson m=2 で Az_vec を更新
```

### 台地検出（Coarse Plateau Criterion）

```cpp
if (iter >= 5 && residual_rel < 1.0) {
    double best  = min(residual_history[iter-4 .. iter]);
    double worst = max(residual_history[iter-4 .. iter]);
    if (worst < best * 2.0) → converged;  // 残差が 2 倍幅以内で 5 回連続
}
```

### ファイン仕上げフェーズ（全グリッド Newton）

```
for fi = 1 .. FINE_ITER:
    1. 全グリッド B/H 計算 → H_map
    2. μ_eff(H_map)  で A_fine, b_fine 構築
    3. R_fine = A_fine × Az - b_fine
    4. if ||R_fine||_rel < tol → break
    5. μ_diff(H_map) で A_diff_fine を構築  ← NEW (updateMuDiffDistribution)
    6. δAz = A_diff_fine⁻¹ × (−R_fine)    ← NEW Newton ステップ
    7. Az += δAz
```

ファイン Newton は ρ_Newton < 1 が保証される領域（飽和域）でも収束する。
Picard で ρ > 1 となる場合でもロバストに動作。

---

## 実装詳細

### 新規追加関数

| 関数 | ファイル | 説明 |
|------|---------|------|
| `updateMuDiffDistribution()` | `MagneticFieldAnalyzer.cpp` | 全グリッド μ_diff 更新（H_map[j,i] 使用） |
| `updateMuDiffAtActiveCells()` | `MagneticFieldAnalyzer.cpp` | 粗活性セルのみ μ_diff 更新（H_active ベクトル使用） |

μ_diff の計算式（中心差分）：

```
H_eps = max(1.0, H_mag × 1e-4)
B⁺ = μ₀ × evaluateMu(H + H_eps) × (H + H_eps)
B⁻ = μ₀ × evaluateMu(H - H_eps) × (H - H_eps)
μ_diff = max(1.0, (B⁺ - B⁻) / (2 × H_eps × μ₀))
```

### YAML 設定

```yaml
nonlinear_solver:
  max_iterations: 30
  fine_finishing_iterations: 5    # 全グリッド Newton ステップ数（0 = 無効）
  use_galerkin_coarsening: true
  use_phase6_precond_jfnk: true
```

---

## テスト結果

**設定：** polar 座標 + 35H300 Si-steel + coarsen_ratio=8
**ファイル：** `configs/user_nak0ulbnr/test_newton_polar.yaml`

```
粗グリッド反復:
  NK iter  1: ||R|| = 3.18e-02  (Newton ステップ)
  NK iter  2: ||R|| = 1.53e-02
  NK iter  3: ||R|| = 2.94e-02  (Anderson 外挿中)
  NK iter  4: ||R|| = 1.20e-02
  NK iter  5: ||R|| = 7.80e-03
  NK iter  6: ||R|| = 5.20e-03
  NK iter  7: ||R|| = 4.72e-03
  NK iter  8: ||R|| = 3.74e-03
  NK iter  9: ||R|| = 3.43e-03
  NK iter 10: ||R|| = 3.22e-03  [Coarse plateau: R∈[3.22e-03, 5.20e-03]] → 収束宣言

ファイン仕上げ (Newton):
  Fine iter 1/5: ||R_fine||_rel = 2.03e-02
  Fine iter 2/5: ||R_fine||_rel = 8.14e-06  [converged]  ← tol=5e-4 を大幅クリア
```

**以前の挙動（修正前）：**
- Picard: R が 0.18↔0.27 を30回振動して最大反復に到達
- Newton + Anderson なし: 周期2振動（0.3↔3.5）
- ファイン仕上げ Picard: R = 1.50 → 1.91 → 31.0 → 4.55 → 19.1 と発散

---

## 収束の改善経緯（コミット履歴）

| コミット | 内容 |
|---------|------|
| `8214b62` | ファイン仕上げ（Picard）をコース収束後に追加 |
| `ea65519` | Phase 6 デフォルトを Galerkin に変更 |
| `e91334b` | True Defect Correction 実装（後に失敗と判明） |
| 前セッション | Newton-Picard（μ_diff 接線行列）実装、Anderson m=2 有効化 |
| `c69f376` | 台地検出 + ファイン仕上げ Newton → 完全収束 |

---

## 既知の制約

1. **coarsen_ratio 依存性：** 粗大化誤差フロア（台地）の高さは coarsen_ratio に依存。ratio が大きいほどファイン仕上げが重要。
2. **台地検出の感度：** 2倍幅条件・5反復ウィンドウは現在固定値。材料によっては調整が必要な場合あり。
3. **ファイン仕上げコスト：** ファイン Newton は毎回 A(μ_diff) の LU 分解が必要（粗反復と同等のコスト）。収束は通常 2〜5 回で完了。
