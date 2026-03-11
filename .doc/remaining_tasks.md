# OpenMagFDM 残タスク一覧

最終更新: 2026-03-10（High Priority 3件完了後）

---

## 完了済み（参考）

- [x] v1.3.0 Feature 1-5（Robin BC、Anti-aliasing、Flux Linkage、Material Presets、Adaptive Mesh）
- [x] v1.3.0 Feature A-E（REST API 拡張、OpenMP、永久磁石、カラー検出、材料ライブラリ）
- [x] Phase 6 Newton-Picard + Anderson m=2（周期2振動対策）
- [x] Phase 6 台地検出（粗大化誤差フロア対応）
- [x] ファイン仕上げ Newton（Picard 発散問題の解決）
- [x] Cartesian + coarsening 回帰テスト（`test_newton_cartesian.yaml` 作成、動作確認）
- [x] smooth_iterations デフォルト整合（テスト設定 0 に統一、sample_config にドキュメント追加）
- [x] RELEASE_NOTES_v1.3.0.md への Phase 6 修正追記

---

## Medium Priority（機能改善）

### 4. 台地検出パラメータの設定可能化

**現状：** `plateau_window = 5`、`plateau_factor = 2.0` はハードコーディング。
**改善案：** `nonlinear_solver.plateau_window / plateau_factor` として YAML で設定可能に。

```yaml
nonlinear_solver:
  plateau_window: 5       # デフォルト
  plateau_factor: 2.0     # デフォルト
```

---

### 5. ファイン仕上げ時の収束ログ改善

**現状：** ファイン仕上げの収束メッセージは最終 R_fine のみ表示。
**改善案：** コース残差からの改善比（R_fine / R_coarse）を表示して改善幅を可視化。

```
Fine finishing: up to 5 full-grid Newton iter(s), tol=5.00e-04
  Fine iter 1/5: ||R_fine||_rel = 2.03e-02  (vs coarse 3.22e-03: 6.3x worse)
  Fine iter 2/5: ||R_fine||_rel = 8.14e-06  [converged]  (improvement: 2500x)
```

---

### 6. `coarsen_ratio` と実効粗大化率の表示

**現状：** `coarsen_ratio: 8` を指定しても境界保護により実際の削減率は 5〜10% 程度にとどまるケースがある。
**改善案：** 粗大化後に活性セル削減率と実効粗大化率を明示するメッセージを追加。

---

## Low Priority（将来機能）

### 7. Nonlinear Solver の v1.4.0 機能

現在 CLAUDE.md に計画なし。候補として：

- **Nested Newton（ネスト Newton）：** 粗グリッドで Newton を outer loop、ファインで inner 修正（より理論的な多重グリッド非線形解法）
- **GMRES Krylov 加速：** Newton ステップを GMRES で近似求解（大規模問題向け）
- **過渡解析の性能改善：** 各ステップで前回の μ 分布を初期値として再利用（hot start）

### 8. GitHub Actions CI への coarsening テスト追加

**現状：** CI はビルドのみ確認。
**改善案：** `test_newton_polar.yaml` を CI テストケースとして実行し収束を検証。

---

## 参考: アルゴリズム変更履歴

| フェーズ | 問題 | 解決手段 |
|---------|------|---------|
| Picard (初期) | ρ > 1 → 発散 | Newton-Picard (μ_diff) |
| Newton (μ_diff) | μ_eff/μ_diff > 1 → 符号反転 → 周期2 | Anderson m=2 |
| Anderson | 粗大化誤差フロアで停滞 | 台地検出 |
| ファイン Picard | ρ > 1 → 発散 | ファイン Newton |
| **現在** | **全部解決** | 10+2 反復で R=8e-6 |
