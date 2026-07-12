# OpenMagFDM 非線形ソルバー高速化 最終レポート (Phase BK / BL)

> **v1.6.1 status update (2026-07-11):** This document records the BK/BL
> experiments at that time. Subsequent v1.6 work shipped Eisenstat-Walker as
> the recommended template setting, chunk-parallel transient sweeps, DD, and an
> opt-in assembled tangent path whose measured inner-solve cost did not improve
> wall time. Use `docs/CONFIGURATION.md` for supported production settings.

**作成日**: 2026-06-15
**対象**: OpenMagFDM 非線形 Newton-Krylov ソルバー (2D 磁界 FDM, C++17 + Eigen + AMGCL)
**ブランチ**: `feature/superlinear-newton-v1.6` (調査後 `e05512d` = クリーン v1.5.1 に revert、コード変更 ship なし)
**目的（このドキュメント）**: 高速化調査の全結果を自己完結的にまとめ、次の打開策検討（Fable 5 との議論）の入力とする。

---

## 0. エグゼクティブサマリ

OpenMagFDM の非線形磁界解析は、IEEJ-D IPMSM ベンチ（1,339,200 自由度の polar メッシュ、3 transient step）で
**約 124 Newton 反復・約 150 秒**を要する。これを縮めるべく 2 系統の高速化（Phase BK: 行列サイズ削減 / Phase BL:
真の Jacobian と line search）を、有限差分 Jacobian-vector 積（FD-Jv）を「真の Jacobian の oracle」とする
診断プローブで定量検証した。

**結論は包括的に NEGATIVE。** 反復回数 ~124 は **B-H 曲線の鋭い knee に由来する intrinsic な壁**であり、
(a) より正確な Jacobian（consistent tangent, CTSM）でも (b) より賢い line search でも反復回数は減らない。
真の Newton step を matrix-free GMRES で実際に計算しても full step は overshoot し、α≈1 の superlinear 収束は
得られない。v1.5.1（Phase BC の Eisenstat-Walker + 単純 Armijo line search）が既に実用的最適点である。

未検証で有望な打開候補（本レポート §6）: **pseudo-transient continuation (擬似時間発展)**、**B-H 非線形性の
homotopy/continuation**、**workflow 並列化**。これらが Fable 5 との議論の主題となる。

---

## 1. 目的

### 1.1 直接の目的
IEEJ-D ベンチ 1 評価あたりの wall time（≈150 s）を短縮する。

### 1.2 上位の目的（ユーザーの最適化ワークフロー）
ユーザーはモータ形状最適化を行っており、本質的な KPI は **「評価する形状の数 × 評価あたり総時間」** の最小化。
すなわち per-solve 高速化と評価スループット（並列度）の双方が効く。「正確性は速度とのトレードオフで許容」
「要は計算が速くなれば何でもよい」という方針。

---

## 2. 背景

### 2.1 ソルバー構成
- **離散化**: vector potential $A_z$ に対する 2D 静磁場方程式 $\nabla\cdot(\nu \nabla A_z) = -J_z$、ここで
  $\nu = 1/\mu$（reluctivity）、$\mathbf{B} = \nabla\times A_z$、$\mu = \mu(|\mathbf{B}|)$（非線形材料）。
- **座標系**: polar（極座標）divergence-form FDM。
- **非線形解法**: Newton-Krylov (NK) 外反復 + Armijo backtracking line search + （任意で）Anderson 加速。
  各外反復で μ を更新し、frozen-μ 行列 $A(\mu)$ を組み、線形系を AMGCL（smoothed aggregation AMG + SPAI(0)
  smoother + CG）で解く。
- **ベンチ**: IEEJ-D IPMSM 断面、`nr=450`, `ntheta=2976` → $n = 450\times2976 = 1{,}339{,}200$ DOF。
  3 step（cold start + warm-start 2 step、回転スライド）。鉄 μr≈9967、NdFeB μr=1.05、空気 μr=1。

### 2.2 高速化の履歴（このレポート以前）
| Phase | 内容 | 結果 |
|---|---|---|
| Phase BC | **Eisenstat-Walker (EW) 不正確 Newton** — 内部 AMGCL の停止許容を外側残差に応じて緩める | **2.5× 高速化 (377→153 s)**、v1.5.1 に採用 |
| Phase BG | AMG hierarchy 再利用 | NEGATIVE (+153%)、revert |
| Phase BI | line search の cheap trial | NEGATIVE（飽和 knee で frozen-Jacobian が破綻）、revert |
| Phase BJ-8 | カスタム Galerkin coarsening を全削除し、AMGCL native multigrid に一本化 | v1.5.1（`e05512d`） |
| **Phase BK** | **FAS 非線形 multigrid（operator-aware coarsening で行列サイズ削減）** | **NEGATIVE**（§2.3） |
| **Phase BL** | **真の Jacobian (CTSM) と line search（本レポート主題）** | **NEGATIVE**（§4-5） |

### 2.3 Phase BK の結論（前提として重要）
operator-aware coarsening（AMGCL `smoothed_aggregation::transfer_operators`）は旧 bilinear coarsening の
「flux が真値の 1/10」問題を**科学的に解決**した（aggregate の 98% が材料純粋、収束解の L2 射影で flux を
厳密保持）。しかし**高速化は実証的に否定**された:
1. 線形 solve は既に AMGCL multigrid + EW で 2-4 CG 反復に最適化済 → 外側 coarse 補正は冗長。
2. 非線形 $\mu(|\nabla\times A_z|)$ は**解の微分**に依存。coarse 空間は解 $A_z$ を 1.25% で表現できるが、
   微分 $B$ は高周波増幅で 327% 誤差 → cheap な coarse 非線形 solve は不可能。
3. FAS-correction を実装すると **wall +66%** かつ warm step の収束破壊。

そしてこの過程で得られた**根本原因診断**が Phase BL の出発点である:
NK が 48-54 反復かかるのは **α-damping**（line search の step 長 α が iter 5-37 で 0.10 に固定、
残差が 0.88/iter で線形減衰 = 0.1-relaxed Picard）。当時の仮説は「近似 Jacobian（diagonal correction）が
剛性を過大予測 → full step が真残差を増やす → 10% しか取れない」であり、**真の Jacobian で α≈1 superlinear に
すれば iter 48→~5-10 になる**と期待された。これが Phase BL の検証対象。

---

## 3. 理論

### 3.1 離散方程式（polar divergence form）
コード `buildMatrixPolar` (`MagneticFieldAnalyzer.cpp:10893`) の実装。連続式（同 :10987 のコメント）:
$$\frac{1}{r}\frac{\partial}{\partial r}\!\left(r\,\frac{1}{\mu}\,\frac{\partial A_z}{\partial r}\right)
+ \frac{1}{r^2}\frac{\partial}{\partial\theta}\!\left(\frac{1}{\mu}\,\frac{\partial A_z}{\partial\theta}\right) = -J_z .$$
両辺に $r$ を掛けて対称化し（同 :11004）、内部ノード $i$（$r$ 方向）, $j$（$\theta$ 方向）に対し:
- 半径方向係数（同 :11009-11010）: $a_{i+} = \dfrac{r_{i+1/2}}{\mu_{i+1/2}\,\Delta r^2}$,
  $a_{i-} = \dfrac{r_{i-1/2}}{\mu_{i-1/2}\,\Delta r^2}$
- 角度方向係数: $a_\theta = \dfrac{1}{r\,\mu_\theta\,\Delta\theta^2}$
- 面の透磁率 $\mu_{i\pm1/2}, \mu_\theta$ は**調和平均** `getMuAtInterfacePolar` (`MagneticFieldAnalyzer.cpp:10851`)
- 右辺 $= -r\,J_z$、DOF index $= i\cdot n_\theta + j$（row-major）

行列 $A(\mu)$ は対称・負定値（離散 reluctivity ラプラシアン）。空気では $1/\mu \sim 8\times10^5$、
鉄では $\sim 1\times10^2$ なので、$A$ の成分は $10^8$–$10^{12}$ のオーダーで大きく変動する。

### 3.2 非線形性と B-H モデル
材料 `pure_iron_model` の B-H 式（YAML、tinyexpr が parse、`bench_results/.../IEEJ-D-template.yaml`）:
```
B(H) = ( H/(40 + 0.52*H) + 4*pi*1e-7*H ) * ( 1 / (1 + exp(-0.1*(H - 40))) )
```
- core 項 $H/(40+0.52H)$ は Frohlich 型で $H\to\infty$ で $\approx 1.92$ T 飽和。
- **sigmoid 項 $1/(1+e^{-0.1(H-40)})$** は $H\approx40$ A/m を中心に幅 ~10-20 で 0→1 に立ち上がり、
  低磁場の透磁率を抑制する。これにより $\mu(H)$（および $\mu(|B|)$）は $H\approx40$–$80$ 近傍で
  **急峻な knee（高い $d^2B/dH^2$）** を持つ。この knee の鋭さ（係数 0.1）が後述の curvature 律速の元凶。
- `evaluateMu` (`MagneticFieldAnalyzer_nonlinear.cpp:243`), `evaluateMuDerivative` (同 :365)。
- μ 更新 `updateMuDistribution` (同 :919)、H 場 `calculateHField` (同 :855)、
  B 場 `calculateMagneticFieldPolar` (`MagneticFieldAnalyzer.cpp:11775`)。

### 3.3 Newton vs Picard と consistent tangent (CTSM)
非線形残差 $R(A_z) = A(\mu(A_z))\,A_z - b(\mu(A_z))$ に対し:
- **Picard（frozen-μ）**: $\delta = A(\mu)^{-1}(-R)$。$\mu$ を固定して線形化。線形収束。
- **真の Newton**: $\delta = J^{-1}(-R)$、$J = \partial R/\partial A_z = A(\mu) + \dfrac{\partial A}{\partial\mu}
  \dfrac{\partial\mu}{\partial A_z}A_z - \dfrac{\partial b}{\partial\mu}\dfrac{\partial\mu}{\partial A_z}$。
  consistent tangent は reluctivity 形で $J = A(\nu) + \text{（接線項）}$、接線項は
  $+2\,\dfrac{d\nu}{d(B^2)}\,(\mathbf{B}\otimes\mathbf{B})$ という対称・正定値項（飽和材料で剛性を増す）。
  これが FEMM 等の Newton-Raphson が少反復で収束する理由。
- 現行コードの「Jacobian」は $J = A + \text{r-weighted diagonal correction}$
  (`MagneticFieldAnalyzer_nonlinear_newton.cpp:248-323`)。この diagonal correction は CTSM の
  off-diagonal（近傍 $\partial\mu/\partial A_{z,\text{nbr}}$）結合を欠く partial term。

### 3.4 globalization（line search）と EW
- **line search** (`..._newton.cpp:372-539`): Armijo backtracking。$\alpha$ を `alpha_init` から
  $\rho$ 倍で縮め、$\|R(A_z+\alpha\delta)\| \le \|R\|(1-c\alpha)$ で受理。
  config 既定 (`MagneticFieldAnalyzer.cpp:159-164`): `alpha_init=1.0, rho=0.65, c=1e-4, max_trials=50`。
  `alpha_init` は適応ロジック（同 :375-415）で前反復の α から決まる。
- **Eisenstat-Walker (EW)** (`..._newton.cpp:331-369`, parse `MagneticFieldAnalyzer.cpp:188-200`):
  内部線形 solve の相対許容 $\eta_k = \gamma(\|R_k\|/\|R_{k-1}\|)^\alpha$ を $[\eta_{\min},\eta_{\max}]$ に
  clip。既定 $\gamma=0.9,\ \alpha=2,\ \eta_{\min}=10^{-6},\ \eta_{\max}=0.1$。
  内部 solve を緩めることで AMGCL CG を 2-4 反復に抑え高速化（Phase BC）。
- 線形 solve `solveLinearSystem` (`MagneticFieldAnalyzer.cpp:6590`): $n>10000$ で AMGCL（同 :6603-）、
  既定許容 `SOLVER_TOLERANCE=1e-6` (`MagneticFieldAnalyzer.h:41`)。EW は正の tol を渡して上書き（同 :6597-6601）。

### 3.5 評価量: flux linkage
`calculateFluxLinkage` (`MagneticFieldAnalyzer.cpp:2364`): 材料ペアで $\Phi = \langle A_z\rangle_{+} -
\langle A_z\rangle_{-}$（polar は r 重み平均）。BC reference（真値）: step2 で
$\Phi_A=-2.476\times10^{-3},\ \Phi_B=-1.459\times10^{-2},\ \Phi_C=1.903\times10^{-2}$。

---

## 4. 実験内容と結果

### 4.1 診断ツール（FD-Jv プローブ）
`probeJacobianAccuracy`（調査用、revert で working tree から削除済、transcript に保存）を NK の Step 5
（`..._newton.cpp:222` 付近）に env-gated で挿入。中心差分で真の Jacobian-vector 積を計算:
$$J_{\text{true}}\,v \approx \frac{R(A_z + h v) - R(A_z - h v)}{2h},\qquad
h = \frac{\sqrt{\epsilon_{\text{mach}}}\,(1+\|A_z\|)}{\|v\|}.$$
これを oracle として、(a) frozen-μ 行列 $A$ と diagonal-correction Jacobian を比較、(b) 実 step 方向の
tangent 誤差を分解、(c) matrix-free GMRES（precond=$A$）で**真の Newton step $\delta_N=J^{-1}(-R)$ を実際に
計算**し α-sweep（JFNK ceiling test）。検証はすべて WSL2 Linux ビルド（clean PATH）、IEEJ-D 実問題、OMP=4。

### 4.2 Finding 1 — ランダムベクトル probe は誤誘導（dilution）
収束点でランダム単位ベクトル $v$ に対し $\|A v - J_{\text{true}}v\| / \|J_{\text{true}}v\| = 0.36\%$、
$\cos\angle = 1.0$、diagonal correction の寄与 $3\times10^{-19}\%$。一見「$A$ は既に真の Jacobian」。
**誤り**: ランダム $v$ は ~線形の空気領域（DOF の大半）を平均化し、非線形 tangent が効く飽和鉄
（DOF の ~10%）を希釈する。**正しいテストベクトルは実 step 方向**である。

### 4.3 Finding 2 — 実 step 方向では tangent が支配的
実際の Picard step $\delta = A^{-1}(-R)$ 方向で、NK iter 10 における分解:
| step | tangent 誤差 $\|J_{\text{true}}\delta - A\delta\|/\|A\delta\|$ | Picard full-step $\rho(1)$ | Picard 最適 α |
|---|---|---|---|
| step1 (cold) | **0.884 (88%)** | 0.49 | α=0.70 → ρ 0.27 |
| step2 (warm) | **2.587 (259%)** | 4.91（**発散**） | α=0.30 → ρ 0.59 |

ここで $\rho(\alpha) = \|R(A_z+\alpha\delta)\|/\|R(A_z)\|$。frozen-μ の $A$ は剛性を過小評価し
（CTSM の正定値接線項を欠く）、full step が overshoot。**この段階では「CTSM に価値あり」に見えた**。

### 4.4 Finding 3 — JFNK ceiling test（決定打）
真の Newton step $\delta_N = J^{-1}(-R)$ を FD-Jv GMRES（precond=$A$, 20 iters）で実際に計算し α-sweep:
| step | $\|\delta_N\|/\|\delta_P\|$ | $\cos(\delta_N,\delta_P)$ | 真 Newton $\rho_N(1)$ | $\delta_N$ 最適 α | GMRES rel res |
|---|---|---|---|---|---|
| step1 (cold) | 0.71/1.58 | 0.93 | **0.41**（要 damping） | α=0.70 → ρ 0.23 | 7.3e-3 |
| step2 (warm) | 0.33/0.86 | 0.875 | **4.01**（**まだ発散**） | α=0.30 → ρ 0.69 | 2.9e-2 |

**真の Newton step でも full step は overshoot する**（warm は ρ=4.0 で発散）。最適 α でも Picard と大差ない
（step1: 0.27→0.23 の微改善、step2: 0.59→0.69 で**改善せず**）。
→ **CTSM を実装しても α≈1 superlinear にはならない。律速は Jacobian 精度ではなく B-H knee の curvature。**
（注: warm の GMRES は rel res 2.9e-2 で未収束 = やや inconclusive。cold は 7.3e-3 で decisive。
両者とも CTSM が反復を劇的に減らす証拠は得られず。）

### 4.5 Finding 4 — line search の α=0.1 は ratchet で stuck（だが §4.6 で否定される）
α-damping の機構: 適応 `alpha_init` ロジック (`..._newton.cpp:375-415`) は、α が一度 0.1 に落ちると
`alpha_init = max(0.1, 0.8*alpha_prev) = 0.1` となり、以降は下方向 backtrack のみで**より大きい α を
再探索しない**。iter 10 の snapshot では同じ Picard 方向で α=0.7（ρ 0.27）が取れたのに、solver は
α=0.1（ρ 0.87）を選んでいた。「ray 上で残差最小の α を探せば ρ~0.3/iter → ~10 iter で収束」という仮説。

### 4.6 Finding 5 — line search の改善も実測 NEGATIVE
ray-minimizing line search（毎反復 α=1.0 から再スタートし、ray 上で残差最小の α を min-tracking で探す）を
実装し、3-step IEEJ-D で計測（同一マシン）:
| 構成 | wall | NK 反復合計 (step1+2+3) |
|---|---|---|
| **v1.5.1 baseline (EW on, 旧 LS)** | **~153 s** | 124 (48+37+39) |
| EW on + ray-min LS | 409 s | 124 (42+32+50) |
| EW off + ray-min LS (tight 1e-6 solve) | 691 s | 128 (45+40+43) |

**反復回数は減らず、wall だけ悪化。** iter 10 の ρ~0.27 は**旧軌道の snapshot** に過ぎず、line search を
変えると軌道全体が変わり、新軌道は ρ~0.9 の stiff 点を通る。そこで ray-min LS が thrash する
（5-13 trials/iter、各 trial が 1.34M セルの行列再構築）。さらに **EW（緩い内部 solve）は step 方向の
品質も劣化させる**（少数 CG 反復は smooth mode のみ捕捉し、飽和鉄の detail を取り逃す）ため、tight solve
（EW off）でも snapshot 通りにならず、かつ tight は wall 4.5×。

### 4.7 Finding 6 — 「no-op に見えた diagonal correction の除去」も NEGATIVE
Finding 1 で diagonal correction の寄与は $3\times10^{-19}\%$ = 実質 no-op と測定された
（`*= dr^2` のスケーリングで $A$ 対角の ~19 桁下）。「$J=A$ 直接にすれば per-iter コスト節約」を試行
（`..._newton.cpp:248` の copy + 同 :260-324 のセル sweep を削除）。
**実測: cold step1 が 50 反復で収束失敗**（補正ありは @48 収束）、かつ per-iter コストは不変
（OpenMP loop は ~0.05 s で、profiling の "Jbuild 12%" は別物）。tiny でも plateau 収束判定
（`..._newton.cpp:176-185`）を跨ぐのに寄与していた。**net 損失、残置。**

---

## 5. 考察

### 5.1 なぜ curvature 律速なのか
B-H モデルの sigmoid 項（係数 0.1）は $H\approx40$ 近傍に鋭い knee を作り、$\mu(|B|)$ の 2 階微分が大きい。
Newton 法の収束半径はこの曲率に反比例するため、解から離れた反復（cold start や大きな回転 step 直後）では
full step が knee を飛び越えて overshoot する。これは **Jacobian の線形化精度とは独立**の性質であり、
Finding 3（真の Newton でも overshoot）が直接の証拠。FEMM が少反復で収束するのは、おそらく (i) FEM 離散
（本実装は FDM + 調和平均面 μ で stencil がより stiff）、(ii) より滑らかな B-H データ、(iii) 強力な
globalization（後述 PTC）の組合せによる。

### 5.2 EW の方向品質トレードオフ（重要な副産物）
EW は内部 solve を $\eta\le0.1$ までしか解かないため、step **方向**が劣化する（Finding 6 の EW-off 比較で
顕在化）。それでも Phase BC で 2.5× 速かったのは、線形 solve コストの削減が方向品質の損失を上回るため。
逆に「tight solve で良い方向 → 少反復」を狙っても、tight solve コスト（4.5×）が反復削減を食い潰す。
**方向品質 ↔ 内部 solve コストのトレードオフは EW の現設定がほぼ最適点。**

### 5.3 line search の局所性
ある反復で測った ρ(α) ランドスケープは、その反復に至る**軌道に依存**する。line search を変えると軌道が
変わり、過去の snapshot は予測力を持たない。これは「局所的な line search の賢さ」では大域収束を改善できない
ことを意味する（大域的な continuation/homotopy が必要、§6）。

### 5.4 確定した「やっても無駄」リスト（次の議論で再試行しないこと）
1. consistent tangent (CTSM) / 真の Jacobian → α≈1 superlinear（Finding 3 で否定）
2. ray-min / backtrack-from-1.0 等の line search 改良単独（Finding 5 で否定）
3. diagonal correction の除去（Finding 6 で否定）
4. coarsening / FAS による行列サイズ削減（Phase BK で否定）
5. AMG hierarchy 再利用、cheap line-search trial（Phase BG/BI で否定）

---

## 6. 打開策候補（未検証・Fable 5 議論用）

以下は**未検証**。curvature 律速という診断と整合する方向に絞った。

### 6.1 Pseudo-transient continuation (PTC / 擬似時間発展) ★最有望
定常 $R(A_z)=0$ を解く代わりに $\frac{1}{\Delta\tau}(A_z^{k+1}-A_z^k) + R(A_z^{k+1}) = 0$ を解き、
$\Delta\tau$ を SER (switched evolution relaxation) で徐々に増やす。線形系は $(\frac{1}{\Delta\tau}I + J)\delta = -R$
となり、対角に正の項が入って**自動的に step を安定化**（overshoot を抑制）。stiff な定常問題の標準解法で、
line search より大域的。**本調査では未試行。** 既存コードへは Step 5 の行列に $\frac{1}{\Delta\tau}$ を足し、
$\Delta\tau$ スケジュールを足すだけで実装可能（CTSM 不要）。論点: $\Delta\tau$ スケジュール、初期 $\Delta\tau$、
反復削減が PTC オーバーヘッドを上回るか。

### 6.2 B-H 非線形性の continuation / homotopy
sigmoid 係数 0.1 を小さく（滑らかに）した緩い B-H で解き、解を初期値に係数を真値へ段階的に戻す
（continuation in nonlinearity sharpness）。または load（磁石 $B_r$ / コイル電流）の continuation。
knee の曲率を「徐々に」導入することで各段階の Newton 収束半径内に留める。論点: 物理が段階的に変わるが
最終段は真値なので最終解は正しい。段数 × 各段反復が現状を下回るか。

### 6.3 Workflow 並列化（per-solve でなく throughput）
形状評価は embarrassingly parallel。複数ソルバーインスタンスを同時実行すれば「#形状 × 総時間」は
ほぼ線形にスケール（per-solve は不変）。最も確実な実利得。論点: メモリ（1.34M DOF × インスタンス数）、
スケジューリング、WebUI/CLI からの起動設計。

### 6.4 その他（優先度低〜中）
- **ハイブリッド EW**: 序盤 loose・終盤 tight に加え、line search が thrash する反復だけ tight にする適応。
- **Anderson 加速の再調整**: 現在 enabled だが depth/β が最適か未検証（軌道安定化に効く可能性）。
- **nonlinear preconditioning (NEPIN / nonlinear elimination)**: 飽和鉄の局所非線形を先に潰す。
- **メッシュ continuation**: 粗メッシュで収束 → 補間 → 細メッシュ polish（ただし Phase BK で coarse μ は
  327% 誤差なので素朴な補間は注意）。
- **B-H データ平滑化**: knee を物理的に妥当な範囲で滑らかにする（ユーザー判断、物理が変わる）。

---

## 7. まとめ

- OpenMagFDM 非線形解析の ~124 反復 / ~150 s は **B-H knee の curvature に由来する intrinsic な壁**。
- **真の Jacobian（CTSM）も line search も反復回数を減らせない**ことを FD-Jv プローブと JFNK ceiling test で
  実証的に確定（Phase BL）。行列サイズ削減（Phase BK）も既に否定済。
- v1.5.1（Phase BC EW + 単純 LS）が実用的最適点。本調査はコード変更を ship せず、ブランチは `e05512d` に revert。
- **次の一手**は per-solve の局所最適化ではなく、**(1) PTC による大域的安定化、(2) 非線形性 continuation、
  (3) workflow 並列化** のいずれか。これらを Fable 5 と議論する。

---

## 補足 A: コード参照（ファイル:行、すべて `e05512d` 時点）

| 項目 | 場所 | 備考 |
|---|---|---|
| NK 外反復ループ | `MagneticFieldAnalyzer_nonlinear_newton.cpp:30` (`solveNonlinearNewtonKrylov`) | 宣言 `MagneticFieldAnalyzer.h:941` |
| Step 1 場/μ 更新 | `..._newton.cpp:89-96` | calculateMagneticFieldPolar→calculateHField→updateMuDistribution |
| Step 2 行列構築 | `..._newton.cpp:98-105` | buildMatrixPolar(A,b) |
| 残差 $R=A\cdot A_z-b$ | `..._newton.cpp:135` | `residual_coarse` |
| Step 4 収束判定（plateau 含む） | `..._newton.cpp:165-217` | plateau: `residual_rel<TOL*10 && reduction<0.05` (:176-185) |
| Step 5 Jacobian（diagonal correction） | `..._newton.cpp:239-370`、ループ本体 `:248-324` | `J_matrix=A_matrix`(:248) + r 重み対角補正、寄与は実質 no-op だが収束には微寄与 |
| Step 5 EW forcing | `..._newton.cpp:331-369`、solve 呼出し `:368` | `solveLinearSystem(J_matrix,-residual_coarse,{},inner_tol)` |
| Step 6 line search（Armijo + 適応 α_init） | `..._newton.cpp:372-539`、α_init ratchet `:375-415` | 既定 `alpha_init=1.0, rho=0.65, c=1e-4` |
| Step 7 Anderson 加速 | `..._newton.cpp:544-` | enabled 時のみ |
| Step 8 反復統計 | `..._newton.cpp:617-` | |
| polar 離散化 stencil | `MagneticFieldAnalyzer.cpp:10893` (`buildMatrixPolar`) | 連続式コメント :10987、面μ調和平均 :10998-10999、r 重み :11004、$a_{i\pm}$ :11009-11010 |
| 面の透磁率（調和平均） | `MagneticFieldAnalyzer.cpp:10851` (`getMuAtInterfacePolar`) | |
| 線形 solve（AMGCL/SparseLU 分岐 + EW tol 上書き） | `MagneticFieldAnalyzer.cpp:6590`、tol 上書き `:6597-6601`、AMGCL 分岐 `:6603` | `AMGCL_THRESHOLD=10000` (`.h:813`), `SOLVER_TOLERANCE=1e-6` (`.h:41`) |
| AMGCL 型（SA-AMG + SPAI0 + CG） | `MagneticFieldAnalyzer.cpp:6621-6625` | builtin backend |
| B 場（curl）計算 | `MagneticFieldAnalyzer.cpp:11775` (`calculateMagneticFieldPolar`) | $B_r=(1/r)\partial_\theta A_z,\ B_\theta=-\partial_r A_z$ |
| μ / dμ 評価 | `MagneticFieldAnalyzer_nonlinear.cpp:243` (`evaluateMu`), `:365` (`evaluateMuDerivative`) | |
| H 場 / μ 分布更新 | `MagneticFieldAnalyzer_nonlinear.cpp:855` (`calculateHField`), `:919` (`updateMuDistribution`) | |
| flux linkage | `MagneticFieldAnalyzer.cpp:2364` (`calculateFluxLinkage`) | 真値: 上記 §3.5 |
| config parse（NK / LS / EW） | `MagneticFieldAnalyzer.cpp:111-200` | max_iter :111, tol :112, LS :159-164, EW :188-200 |

## 補足 B: git / コミット情報

- **現在ブランチ**: `feature/superlinear-newton-v1.6`、HEAD `e05512d`（= クリーン v1.5.1、Phase BL のコード変更は
  revert 済みで working tree に残っていない）。
- **v1.5.1 の系譜**（`git log --oneline`）:
  - `e05512d` webui+docs: schema cleanup + AMGCL native migration notes (Phase BJ-8 stage 5)
  - `edf0712` nonlinear: delete NK loop dead code from custom Galerkin coarsening (Phase BJ-8 stage 4a)
  - `9586a8b` nonlinear: route NK loop through Standard direct solve only (Phase BJ-8 stage 2)
  - `edc9ddc` nonlinear: Eisenstat-Walker forcing for inner AMGCL tolerance (**Phase BC**, 2.5× の出所)
- **Phase BL の diff**: ship なし（プローブ + ray-min LS + diagonal 除去はすべて検証後に
  `git checkout --` で破棄）。プローブのソースは本セッションの transcript に保存。
- **Phase BK**（前段、別ブランチ `feature/fas-multigrid-v1.6`）: research 記録として残置、main へ merge せず。

## 補足 C: 検証環境とベンチ資産

- **ビルド**: WSL2 Ubuntu、`OpenMagFDM/build_linux/MagFDMsolver`。cmake 再構成は **clean PATH** 必須
  （conda の Windows `tiff.lib` 混入で link 失敗するため: `export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`、
  必要時 `cmake -U tiff_DIR -U TIFF_DIR -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=OFF .`）。実行 `OMP_NUM_THREADS=4`。
- **起動**: `MagFDMsolver <config.yaml> <image.png> [out_dir]`（image は別引数、必須）。
- **ベンチ YAML / 画像**: `bench_results/ipmsm_ieej_d_BC_eisenstat_walker/`（BC reference, run.log が真値）。
  Phase BL の検証ログ・findings: `bench_results/superlinear_poc/`
  （`stage0_jacobian_vs_linesearch.md` = 全 raw 結果、`probe_iter10.log` `probe_jfnk.log` `ls_bench.log`
  `ew_off.log` `noJdiag.log`）。
- **関連メモリ**（Claude 永続メモリ、`~/.claude/projects/.../memory/`）:
  `phase_bl_superlinear_newton.md`（本件）、`phase_bk_fas_multigrid.md`（coarsening）、
  `v1_5_1_convergence_plan.md`、`femm_advantages.md`、`solver_architecture.md`、`performance_issues.md`。

## 補足 D: 主要コードコメント（原文、文脈確認用）

- diagonal correction の意図（`..._newton.cpp:250` 付近）: "Build r-weighted diagonal Jacobian correction
  for polar + nonlinear. ... hoist the flip, replace YAML iteration with the rgb_to_material LUT, and parallelise."
- EW の根拠（`..._newton.cpp:331` 付近 / `MagneticFieldAnalyzer.cpp:6597`）: "the inner AMGCL CG doesn't need
  to converge to 1e-6 when the outer Newton residual is still at 1e+1 ... Tighten the inner tolerance only as
  the outer residual decreases."
- 過去の line search 反省（`..._newton.cpp:405` 付近）: "Previous conservative damping for polar coordinates
  was too aggressive and caused extremely slow convergence (α=0.05-0.07). ... If divergence occurs, consider
  improving the Jacobian approximation instead." ← 本調査はこの "improving the Jacobian" を検証し否定した。
- BJ-8 の一本化（`..._newton.cpp:31` 付近）: custom Galerkin coarsening 全削除、AMGCL native multigrid に一本化。
