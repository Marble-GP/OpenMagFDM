# Adaptive Mesh Coarsening: Convergence Improvement for Nonlinear Solvers

## 1. Background

### 1.1 Project Overview

**OpenMagFDM** is a 2D magnetic field analysis solver using the Finite Difference Method (FDM). It supports:
- Cartesian and Polar coordinate systems
- Linear and nonlinear magnetic materials (B-H curves)
- Multiple solver methods: Picard iteration, Anderson acceleration, Newton-Krylov

### 1.2 Adaptive Mesh Coarsening

To reduce computational cost, we implemented **adaptive mesh coarsening**:

```
Original grid (500×500 = 250,000 cells)
    ↓ Coarsening (ratio=4) in uniform regions
Coarsened grid (~80,000 active cells)
    ↓ Solve reduced system
    ↓ Interpolate back to full grid
Result on original grid
```

**Key data structures:**
```cpp
Eigen::Matrix<bool, Dynamic, Dynamic> active_cells;  // true = active (solved), false = interpolated
std::vector<std::pair<int,int>> coarse_to_fine;      // coarse_idx → (i, j)
std::map<std::pair<int,int>, int> fine_to_coarse;    // (i, j) → coarse_idx
int n_active_cells;                                   // number of DOFs in coarsened system
```

**Coarsening rules:**
- Only applied to materials with `coarsen: true` in YAML config
- Boundary cells (detected by Canny edge detection) are always kept active
- Interior cells are decimated by `coarsen_ratio` (e.g., keep every 4th cell)

### 1.3 Non-uniform FDM Stencil

For coarsened grids, the standard 5-point Laplacian stencil is replaced with a non-uniform stencil:

**Standard (uniform spacing h):**
```
∂²u/∂x² ≈ (u_{i-1} - 2u_i + u_{i+1}) / h²
```

**Non-uniform (spacing h₋ to left, h₊ to right):**
```
∂²u/∂x² ≈ 2/(h₋(h₋+h₊))·u_{i-1} - 2/(h₋h₊)·u_i + 2/(h₊(h₋+h₊))·u_{i+1}
```

This ensures second-order accuracy even with non-uniform mesh spacing.

---

## 2. Problem Statement

### 2.1 Observed Behavior

| Scenario | Solver | Iterations to R < 10⁻³ |
|----------|--------|------------------------|
| No coarsening | Newton-Krylov | ~50 |
| With coarsening (ratio=4) | Newton-Krylov | >50 (R still > 10⁻¹) |

**The coarsened system converges ~10x slower than the full system.**

### 2.2 Previous Issue (Resolved)

Initially, the solver **diverged** with coarsening enabled. The root cause was:
- During Newton-Krylov line search, only active cells were updated
- Inactive cells retained OLD values
- B/H/μ calculations used inconsistent gradients at active/inactive boundaries
- This caused incorrect μ updates → divergence

**Fix implemented:** After every Az update, interpolate inactive cells:
```cpp
// After updating active cells
if (using_coarsening) {
    for (int idx = 0; idx < n_active_cells; idx++) {
        auto [i, j] = coarse_to_fine[idx];
        Az(j, i) = Az_vec(idx);  // Update active cell
    }
    // CRITICAL: Interpolate inactive cells
    interpolateInactiveCells(Az_vec);  // Bilinear interpolation
}
```

This fixed the divergence, but convergence is still slow.

---

## 3. Root Cause Analysis

### 3.1 Matrix Condition Number Degradation

The non-uniform stencil creates coefficient imbalance at coarse/fine boundaries:

```
Fine region:     h₋ = dx,  h₊ = dx     → coefficients balanced
Transition:      h₋ = dx,  h₊ = 4*dx   → coefficient ratio = 4:1
Coarse region:   h₋ = 4*dx, h₊ = 4*dx  → coefficients balanced
```

At `coarsen_ratio=4`, the transition zones have coefficient ratios up to 4:1, which:
- Increases matrix condition number
- Slows iterative solver convergence
- May cause numerical instability

### 3.2 Jacobian Approximation Error

In Newton-Krylov, the Jacobian includes nonlinear corrections based on H field:

```cpp
// Jacobian correction for nonlinear materials
double dmu_dH = evaluateMuDerivative(mu_val, H_val);
J_matrix.coeffRef(idx, idx) += r * dmu_dH * correction_term;
```

**Problem:** The H field at active cells near boundaries depends on interpolated Az values from inactive neighbors. Interpolation introduces error → Jacobian is inaccurate → Newton step direction is suboptimal.

### 3.3 Interpolation Error Accumulation

Each nonlinear iteration:
1. Solve coarse system → Az_coarse
2. Interpolate inactive cells (bilinear)
3. Calculate B = ∇×Az (uses interpolated values at boundaries)
4. Calculate H from B
5. Update μ(H) for next iteration

The interpolation error in step 2 propagates through steps 3-5 and affects the next iteration's matrix, potentially accumulating over iterations.

### 3.4 Residual Definition Mismatch

The residual is computed on the **coarsened system**:
```cpp
Eigen::VectorXd residual = A_matrix * Az_vec - b_vec;  // Size: n_active_cells
```

But the physical accuracy should be measured on the **full grid**. The coarse residual may not reflect true solution quality.

---

## 4. Proposed Solutions

### 4.1 Hybrid Iteration Strategy

**Concept:** Use full grid for initial convergence, then switch to coarsened grid.

```cpp
int initial_fine_iterations = 5;  // Configurable

for (int iter = 0; iter < MAX_ITER; iter++) {
    bool use_coarse = using_coarsening && (iter >= initial_fine_iterations);

    if (use_coarse) {
        buildMatrixCoarsened(A_matrix, b_vec);
    } else {
        buildMatrix(A_matrix, b_vec);
    }
    // ... rest of iteration
}
```

**Pros:**
- Simple to implement
- Gets good initial convergence from full grid
- Still benefits from coarsening speedup in later iterations

**Cons:**
- First N iterations are slow (no coarsening benefit)
- May have transition issues when switching

**Expected improvement:** Moderate (addresses poor initial convergence)

### 4.2 Two-Grid Defect Correction (Multigrid-lite)

**Concept:** Solve on coarse grid, compute residual on fine grid at boundaries, correct.

```
Algorithm:
1. Solve coarse system: A_c · x_c = b_c
2. Prolongate to fine grid: x_f = P · x_c
3. Compute fine residual at boundary cells: r_f = A_f · x_f - b_f
4. Restrict residual: r_c = R · r_f
5. Solve correction: A_c · e_c = r_c
6. Update: x_c ← x_c + e_c
```

**Pros:**
- Maintains boundary accuracy
- Theoretically optimal convergence rate
- Standard multigrid technique

**Cons:**
- More complex implementation
- Requires fine grid matrix at boundaries
- Additional memory for restriction/prolongation operators

**Expected improvement:** High (addresses boundary accuracy issue)

### 4.3 Adaptive Coarsening Ratio

**Concept:** Use weaker coarsening near material boundaries.

```yaml
materials:
  air_far:
    rgb: [255, 255, 255]
    coarsen: true
    coarsen_ratio: 4  # Aggressive coarsening far from boundaries
  air_near:
    rgb: [254, 254, 254]  # Slightly different color for transition zone
    coarsen: true
    coarsen_ratio: 2  # Gentler coarsening near boundaries
  iron:
    coarsen: false  # No coarsening for nonlinear material
```

**Pros:**
- Reduces coefficient imbalance at transitions
- User-controllable via YAML
- No code changes needed

**Cons:**
- Requires user to manually define transition zones
- Less automatic than other approaches

**Expected improvement:** Moderate (addresses condition number issue)

### 4.4 Improved Interpolation

**Concept:** Use higher-order or physics-aware interpolation for inactive cells.

**Option A: Bicubic interpolation**
```cpp
double bicubicInterpolate(int i, int j, const VectorXd& Az_coarse);
```

**Option B: Least-squares fitting**
```cpp
// Fit quadratic surface to nearby active cells
double leastSquaresInterpolate(int i, int j, const VectorXd& Az_coarse);
```

**Option C: Flux-preserving interpolation**
```cpp
// Ensure ∇·B = 0 is maintained across interpolation
double fluxPreservingInterpolate(int i, int j, const VectorXd& Az_coarse);
```

**Pros:**
- Higher accuracy at interpolated points
- Smoother gradients across boundaries

**Cons:**
- More computational cost per interpolation
- Bicubic needs more neighbors (may not be available at all points)

**Expected improvement:** Low to moderate (reduces interpolation error)

### 4.5 Relaxation Parameter Tuning

**Concept:** Use more conservative damping for coarsened iterations.

```cpp
double omega = using_coarsening ? 0.5 : 0.7;  // More damping with coarsening
mu_map = omega * mu_new + (1.0 - omega) * mu_old;
```

**Pros:**
- Trivial to implement
- Can stabilize convergence

**Cons:**
- Slower convergence by design
- Doesn't address root cause

**Expected improvement:** Low (stabilization, not acceleration)

### 4.6 Preconditioner Improvement

**Concept:** Use specialized preconditioner for non-uniform mesh.

**Option A: Diagonal scaling**
```cpp
// Scale rows/columns by local mesh spacing
DiagonalMatrix<double> D = computeMeshScaling();
A_scaled = D * A * D;
```

**Option B: Incomplete LU with modified ordering**
```cpp
// Reorder unknowns to reduce fill-in for non-uniform mesh
ILU0Preconditioner<SparseMatrix> precond(A, meshAwareOrdering());
```

**Pros:**
- Can significantly improve condition number
- Standard technique for iterative solvers

**Cons:**
- Requires switching from SparseLU to iterative solver (GMRES/BiCGSTAB)
- Additional implementation complexity

**Expected improvement:** High (directly addresses condition number)

---

## 5. Recommended Approach

### Phase 1: Quick Wins (Low effort, immediate testing)

1. **4.5 Relaxation tuning** - Test with ω=0.5 for coarsened iterations
2. **4.3 Adaptive ratio** - Test with `coarsen_ratio: 2` instead of 4

### Phase 2: Moderate Effort

3. **4.1 Hybrid iteration** - Implement full-grid for first N iterations

### Phase 3: Comprehensive Solution

4. **4.2 Two-grid correction** - Implement proper defect correction
5. **4.6 Preconditioner** - Switch to GMRES with mesh-aware preconditioner

---

## 6. Questions for Discussion

1. **Trade-off priority:** Is faster convergence or simpler implementation more important?

2. **Accuracy requirements:** What is acceptable error between coarsened and full-grid solutions?

3. **Use case:** Is coarsening primarily for:
   - Transient simulations (many time steps, each needs fast solve)?
   - Single steady-state solve (total time matters)?
   - Memory reduction (large grids that don't fit otherwise)?

4. **Alternative approaches:** Should we consider:
   - AMG (Algebraic Multigrid) instead of geometric coarsening?
   - GPU acceleration instead of mesh reduction?
   - Domain decomposition with local refinement?

---

## 7. References

1. Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial*. SIAM.
2. Trottenberg, U., Oosterlee, C. W., & Schuller, A. (2001). *Multigrid*. Academic Press.
3. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*. SIAM.

---

## Appendix: Current Implementation Details

### A.1 Interpolation Function

```cpp
void MagneticFieldAnalyzer::interpolateInactiveCells(const Eigen::VectorXd& Az_coarse) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (!active_cells(j, i)) {
                Az(j, i) = bilinearInterpolateFromCoarse(i, j, Az_coarse);
            }
        }
    }
}
```

### A.2 Non-uniform Stencil Coefficients

```cpp
// Find neighbors
int i_west = findNextActiveX(i, j, -1);
int i_east = findNextActiveX(i, j, +1);

double h_minus = (i - i_west) * dx;
double h_plus = (i_east - i) * dx;

// Coefficients
double coeff_west = 2.0 / (mu_west * h_minus * (h_minus + h_plus));
double coeff_east = 2.0 / (mu_east * h_plus * (h_minus + h_plus));
double coeff_center = -(coeff_west + coeff_east);
```

### A.3 Coarsening Mask Generation

```cpp
void MagneticFieldAnalyzer::generateCoarseningMask() {
    cv::Mat boundaries = detectMaterialBoundaries();  // Canny edge detection

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            std::string mat = getMaterialAt(i, j);
            auto& cc = material_coarsen[mat];

            // Keep cell active if on boundary or coarsening disabled
            if (boundaries.at<uchar>(j, i) > 0 || !cc.enabled) {
                active_cells(j, i) = true;
                continue;
            }

            // Decimate by coarsen_ratio
            active_cells(j, i) = (i % cc.ratio == 0 && j % cc.ratio == 0);
        }
    }
}
```
