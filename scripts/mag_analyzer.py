#!/usr/bin/env python3
"""
2次元磁場解析プログラム（ベクトルポテンシャル法）
極座標系対応版 - 改良版
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import yaml
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Tuple, List, Any
import warnings
import readline

warnings.filterwarnings('ignore')

class MagneticFieldAnalyzer:
    """ベクトルポテンシャル法による2次元磁場解析クラス（極座標対応）"""
    
    def __init__(self, config_path: str, image_path: str):
        """
        初期化
        
        Args:
            config_path: YAMLファイルのパス
            image_path: 媒質画像ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.image = self._load_image(image_path)
        
        # 座標系の判定
        self.coordinate_system = self.config.get('coordinate_system', 'cartesian')
        print(f"座標系: {self.coordinate_system}")
        
        if self.coordinate_system == 'polar':
            self._setup_polar_system()
        else:
            self._setup_cartesian_system()
        
        # 結果格納用
        self.Az = None  # ベクトルポテンシャル
        self.Br = None  # 極座標での半径方向磁束密度
        self.Btheta = None  # 極座標での角度方向磁束密度
        self.Bx = None  # 直交座標でのx方向磁束密度
        self.By = None  # 直交座標でのy方向磁束密度
        
    def _setup_cartesian_system(self):
        """直交座標系のセットアップ"""
        self.ny, self.nx = self.image.shape[:2]
        
        # 物理量の初期化
        self.dx = self.config.get('mesh', {}).get('dx', 1.0)
        self.dy = self.config.get('mesh', {}).get('dy', 1.0)
        
        # 媒質プロパティの設定
        self.mu_map = np.ones((self.ny, self.nx))  # 透磁率分布
        self.jz_map = np.zeros((self.ny, self.nx))  # 電流密度分布
        self._setup_material_properties()
        
    def _setup_polar_system(self):
        """極座標系のセットアップ"""
        # 画像は (nr, ntheta) の形式
        # 行が半径方向、列が角度方向（横軸が角度）
        self.nr, self.ntheta = self.image.shape[:2]
        print(f"極座標メッシュ: nr={self.nr}, ntheta={self.ntheta}")
        
        # 極座標メッシュパラメータ
        polar_mesh = self.config.get('polar_mesh', {})
        self.r_min = polar_mesh.get('r_min', 0.01)  # 最小半径（0を避ける）
        self.r_max = polar_mesh.get('r_max', 1.0)
        
        # メッシュ間隔
        self.dr = (self.r_max - self.r_min) / (self.nr - 1)
        self.dtheta = 2 * np.pi / self.ntheta  # 周期境界条件を仮定
        
        # 半径座標の配列
        self.r = np.linspace(self.r_min, self.r_max, self.nr)
        self.theta = np.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)
        
        # 媒質プロパティの設定（nr x ntheta）
        self.mu_map = np.ones((self.nr, self.ntheta))
        self.jz_map = np.zeros((self.nr, self.ntheta))
        self._setup_material_properties()
        
    def _load_config(self, config_path: str) -> Dict:
        """YAMLファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """画像ファイルの読み込み"""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"画像ファイルを読み込めません: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _setup_material_properties(self):
        """RGB値から媒質プロパティへのマッピング"""
        materials = self.config.get('materials', {})
        
        for material_name, properties in materials.items():
            rgb = properties.get('rgb', [255, 255, 255])
            mu_r = properties.get('mu_r', 1.0)  # 比透磁率
            jz = properties.get('jz', 0.0)  # 電流密度
            
            # RGB値が一致する画素を検出
            mask = np.all(self.image == rgb, axis=2)
            
            # 透磁率と電流密度を設定
            mu_0 = 4 * np.pi * 1e-7  # 真空の透磁率
            self.mu_map[mask] = mu_r * mu_0
            self.jz_map[mask] = jz
            
            print(f"材料 '{material_name}': RGB{rgb} → μr={mu_r}, Jz={jz} A/m²")
            print(f"  該当画素数: {np.sum(mask)}")
    
    def _get_mu_at_interface(self, i: float, j: int, direction: str = None) -> float:
        """
        格子点間の界面における透磁率を計算（調和平均）
        極座標系と直交座標系の両方に対応
        """
        if self.coordinate_system == 'polar':
            return self._get_mu_at_interface_polar(i, j, direction)
        else:
            return self._get_mu_at_interface_cartesian(int(i), j, direction)
    
    def _get_mu_at_interface_cartesian(self, i: int, j: int, direction: str) -> float:
        """直交座標系での界面透磁率（元の実装）"""
        if direction == 'x+':
            if i < self.nx - 1:
                return 2.0 / (1.0/self.mu_map[j, i] + 1.0/self.mu_map[j, i+1])
            else:
                return self.mu_map[j, i]
        elif direction == 'x-':
            if i > 0:
                return 2.0 / (1.0/self.mu_map[j, i] + 1.0/self.mu_map[j, i-1])
            else:
                return self.mu_map[j, i]
        elif direction == 'y+':
            if j < self.ny - 1:
                return 2.0 / (1.0/self.mu_map[j, i] + 1.0/self.mu_map[j+1, i])
            else:
                return self.mu_map[j, i]
        elif direction == 'y-':
            if j > 0:
                return 2.0 / (1.0/self.mu_map[j, i] + 1.0/self.mu_map[j-1, i])
            else:
                return self.mu_map[j, i]
    
    def _get_mu_at_interface_polar(self, r_idx: float, theta_idx: int, direction: str) -> float:
        """極座標系での界面透磁率"""
        r_idx_int = int(r_idx)
        
        if direction == 'r':
            # 半径方向の界面
            if r_idx == r_idx_int:
                # 整数インデックスの場合
                if r_idx_int > 0 and r_idx_int < self.nr - 1:
                    return 2.0 / (1.0/self.mu_map[r_idx_int, theta_idx] + 
                                  1.0/self.mu_map[r_idx_int + 1, theta_idx])
                else:
                    return self.mu_map[min(max(r_idx_int, 0), self.nr-1), theta_idx]
            else:
                # r_idx - 0.5 の位置（格子間）
                r_idx_low = int(np.floor(r_idx))
                if r_idx_low >= 0 and r_idx_low < self.nr - 1:
                    return 2.0 / (1.0/self.mu_map[r_idx_low, theta_idx] + 
                                  1.0/self.mu_map[r_idx_low + 1, theta_idx])
                else:
                    return self.mu_map[min(max(r_idx_low, 0), self.nr-1), theta_idx]
        else:
            # 角度方向は周期境界を考慮
            theta_next = (theta_idx + 1) % self.ntheta
            return 2.0 / (1.0/self.mu_map[r_idx_int, theta_idx] + 
                          1.0/self.mu_map[r_idx_int, theta_next])
    
    def solve(self):
        """座標系に応じた求解"""
        if self.coordinate_system == 'polar':
            self.solve_polar()
        else:
            self.solve_cartesian()
    
    def solve_cartesian(self):
        """直交座標系での有限差分法による方程式の求解（元の実装）"""
        print("\n=== 直交座標系での連立方程式の構築 ===")
        
        bc_left, bc_right, bc_bottom, bc_top = self._validate_boundary_conditions()
        
        print(f"境界条件:")
        print(f"  左: {bc_left['type']}")
        print(f"  右: {bc_right['type']}")
        print(f"  下: {bc_bottom['type']}")
        print(f"  上: {bc_top['type']}")
        
        n = self.nx * self.ny
        row_indices = []
        col_indices = []
        data = []
        rhs = np.zeros(n)
        
        # 各格子点について方程式を構築
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                
                is_left = (i == 0)
                is_right = (i == self.nx - 1)
                is_bottom = (j == 0)
                is_top = (j == self.ny - 1)
                
                # ディリクレ境界条件の場合
                if is_left and bc_left['type'] == 'dirichlet':
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    rhs[idx] = bc_left.get('value', 0.0)
                    continue
                elif is_right and bc_right['type'] == 'dirichlet':
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    rhs[idx] = bc_right.get('value', 0.0)
                    continue
                elif is_bottom and bc_bottom['type'] == 'dirichlet':
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    rhs[idx] = bc_bottom.get('value', 0.0)
                    continue
                elif is_top and bc_top['type'] == 'dirichlet':
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    rhs[idx] = bc_top.get('value', 0.0)
                    continue
                
                # 内部点の処理
                coeff_center = 0.0
                
                # X方向の差分項
                if not is_left:
                    mu_west = self._get_mu_at_interface(i, j, 'x-')
                    coeff_west = 1.0 / (mu_west * self.dx**2)
                    row_indices.append(idx)
                    col_indices.append(idx - 1)
                    data.append(coeff_west)
                    coeff_center -= coeff_west
                
                if not is_right:
                    mu_east = self._get_mu_at_interface(i, j, 'x+')
                    coeff_east = 1.0 / (mu_east * self.dx**2)
                    row_indices.append(idx)
                    col_indices.append(idx + 1)
                    data.append(coeff_east)
                    coeff_center -= coeff_east
                
                # Y方向の差分項
                if not is_bottom:
                    mu_south = self._get_mu_at_interface(i, j, 'y-')
                    coeff_south = 1.0 / (mu_south * self.dy**2)
                    row_indices.append(idx)
                    col_indices.append(idx - self.nx)
                    data.append(coeff_south)
                    coeff_center -= coeff_south
                
                if not is_top:
                    mu_north = self._get_mu_at_interface(i, j, 'y+')
                    coeff_north = 1.0 / (mu_north * self.dy**2)
                    row_indices.append(idx)
                    col_indices.append(idx + self.nx)
                    data.append(coeff_north)
                    coeff_center -= coeff_north
                
                # 中心点の係数
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(coeff_center)
                
                # 右辺（電流密度項）
                rhs[idx] = -self.jz_map[j, i]
        
        # 連立方程式の求解
        A_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        print(f"行列サイズ: {A_matrix.shape}, 非ゼロ要素数: {A_matrix.nnz}")
        
        print("\n=== 方程式の求解中... ===")
        Az_flat = spsolve(A_matrix, rhs)
        self.Az = Az_flat.reshape((self.ny, self.nx))
        
        print("求解完了！")
        self._calculate_magnetic_field_cartesian()
    
    def solve_polar(self):
        """極座標系での有限差分法による方程式の求解"""
        print("\n=== 極座標系での連立方程式の構築 ===")
        print(f"メッシュ: r方向 {self.nr}点, θ方向 {self.ntheta}点")
        print(f"半径範囲: {self.r_min:.3f} ~ {self.r_max:.3f} m")
        
        n = self.nr * self.ntheta
        row_indices = []
        col_indices = []
        data = []
        rhs = np.zeros(n)
        
        # 境界条件の取得
        bc = self.config.get('polar_boundary_conditions', {})
        bc_inner = bc.get('inner', {'type': 'neumann'})  # 内側（r=r_min）
        bc_outer = bc.get('outer', {'type': 'dirichlet', 'value': 0.0})  # 外側（r=r_max）
        
        print(f"境界条件: 内側={bc_inner['type']}, 外側={bc_outer['type']}")
        
        # 各格子点について方程式を構築
        for i in range(self.nr):  # 半径方向
            for j in range(self.ntheta):  # 角度方向
                idx = i * self.ntheta + j
                r = self.r[i]
                
                # 内側境界（r = r_min）
                if i == 0:
                    if bc_inner['type'] == 'dirichlet':
                        row_indices.append(idx)
                        col_indices.append(idx)
                        data.append(1.0)
                        rhs[idx] = bc_inner.get('value', 0.0)
                        continue
                    elif bc_inner['type'] == 'neumann':
                        # ノイマン境界条件: ∂Az/∂r = 0
                        # 前進差分を使用
                        coeff_center = -1.0 / self.dr
                        coeff_outer = 1.0 / self.dr
                        
                        row_indices.append(idx)
                        col_indices.append(idx)
                        data.append(coeff_center)
                        
                        row_indices.append(idx)
                        col_indices.append((i + 1) * self.ntheta + j)
                        data.append(coeff_outer)
                        
                        rhs[idx] = 0.0
                        continue
                
                # 外側境界（r = r_max）
                if i == self.nr - 1:
                    if bc_outer['type'] == 'dirichlet':
                        row_indices.append(idx)
                        col_indices.append(idx)
                        data.append(1.0)
                        rhs[idx] = bc_outer.get('value', 0.0)
                        continue
                    elif bc_outer['type'] == 'neumann':
                        # 後退差分
                        coeff_center = 1.0 / self.dr
                        coeff_inner = -1.0 / self.dr
                        
                        row_indices.append(idx)
                        col_indices.append(idx)
                        data.append(coeff_center)
                        
                        row_indices.append(idx)
                        col_indices.append((i - 1) * self.ntheta + j)
                        data.append(coeff_inner)
                        
                        rhs[idx] = 0.0
                        continue
                
                # 内部点の処理
                coeff_center = 0.0
                
                # 半径方向の項
                mu_inner = self._get_mu_at_interface(i - 0.5, j, 'r')
                mu_outer = self._get_mu_at_interface(i + 0.5, j, 'r')
                
                coeff_r_inner = 1.0 / (mu_inner * self.dr**2)
                coeff_r_outer = 1.0 / (mu_outer * self.dr**2)
                
                # 第2項: (1/r)∂Az/∂r の寄与
                coeff_r_inner *= (1.0 - 0.5 * self.dr / r)
                coeff_r_outer *= (1.0 + 0.5 * self.dr / r)
                
                row_indices.append(idx)
                col_indices.append((i - 1) * self.ntheta + j)
                data.append(coeff_r_inner)
                
                row_indices.append(idx)
                col_indices.append((i + 1) * self.ntheta + j)
                data.append(coeff_r_outer)
                
                coeff_center -= (coeff_r_inner + coeff_r_outer)
                
                # 角度方向の項（周期境界条件）
                mu_current = self.mu_map[i, j]
                coeff_theta = 1.0 / (r**2 * mu_current * self.dtheta**2)
                
                # θ-1の点（周期境界を考慮）
                j_prev = (j - 1) % self.ntheta
                idx_prev = i * self.ntheta + j_prev
                row_indices.append(idx)
                col_indices.append(idx_prev)
                data.append(coeff_theta)
                
                # θ+1の点（周期境界を考慮）
                j_next = (j + 1) % self.ntheta
                idx_next = i * self.ntheta + j_next
                row_indices.append(idx)
                col_indices.append(idx_next)
                data.append(coeff_theta)
                
                coeff_center -= 2 * coeff_theta
                
                # 中心点の係数
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(coeff_center)
                
                # 右辺（電流密度項）
                rhs[idx] = -self.jz_map[i, j]
        
        # 連立方程式の求解
        A_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        print(f"行列サイズ: {A_matrix.shape}")
        print(f"非ゼロ要素数: {A_matrix.nnz}")
        
        print("\n=== 方程式の求解中... ===")
        Az_flat = spsolve(A_matrix, rhs)
        self.Az = Az_flat.reshape((self.nr, self.ntheta))
        
        print("求解完了！")
        self._calculate_magnetic_field_polar()
    
    def _calculate_magnetic_field_cartesian(self):
        """直交座標系での磁束密度B = rot(A)の計算"""
        # Bx = ∂Az/∂y
        self.Bx = np.zeros_like(self.Az)
        self.Bx[1:-1, :] = (self.Az[2:, :] - self.Az[:-2, :]) / (2 * self.dy)
        self.Bx[0, :] = (self.Az[1, :] - self.Az[0, :]) / self.dy
        self.Bx[-1, :] = (self.Az[-1, :] - self.Az[-2, :]) / self.dy
        
        # By = -∂Az/∂x
        self.By = np.zeros_like(self.Az)
        self.By[:, 1:-1] = -(self.Az[:, 2:] - self.Az[:, :-2]) / (2 * self.dx)
        self.By[:, 0] = -(self.Az[:, 1] - self.Az[:, 0]) / self.dx
        self.By[:, -1] = -(self.Az[:, -1] - self.Az[:, -2]) / self.dx
        
        self.B_magnitude = np.sqrt(self.Bx**2 + self.By**2)
    
    def _calculate_magnetic_field_polar(self):
        """極座標系での磁束密度の計算"""
        # Br = (1/r)∂Az/∂θ
        self.Br = np.zeros_like(self.Az)
        for i in range(self.nr):
            r = self.r[i]
            # 周期境界条件を利用した中心差分
            for j in range(self.ntheta):
                j_next = (j + 1) % self.ntheta
                j_prev = (j - 1) % self.ntheta
                self.Br[i, j] = (self.Az[i, j_next] - self.Az[i, j_prev]) / (2 * r * self.dtheta)
        
        # Bθ = -∂Az/∂r
        self.Btheta = np.zeros_like(self.Az)
        self.Btheta[1:-1, :] = -(self.Az[2:, :] - self.Az[:-2, :]) / (2 * self.dr)
        self.Btheta[0, :] = -(self.Az[1, :] - self.Az[0, :]) / self.dr
        self.Btheta[-1, :] = -(self.Az[-1, :] - self.Az[-2, :]) / self.dr
        
        self.B_magnitude = np.sqrt(self.Br**2 + self.Btheta**2)
    
    def _validate_boundary_conditions(self):
        """直交座標系の境界条件の妥当性をチェック"""
        bc = self.config.get('boundary_conditions', {})
        
        bc_left = bc.get('left', {'type': 'dirichlet', 'value': 0.0})
        bc_right = bc.get('right', {'type': 'dirichlet', 'value': 0.0})
        bc_bottom = bc.get('bottom', {'type': 'dirichlet', 'value': 0.0})
        bc_top = bc.get('top', {'type': 'dirichlet', 'value': 0.0})
        
        return bc_left, bc_right, bc_bottom, bc_top
    
    def visualize_results(self):
        """座標系に応じた結果の可視化"""
        if self.coordinate_system == 'polar':
            self._visualize_polar()
        else:
            self._visualize_cartesian()
    
    def _visualize_cartesian(self):
        """直交座標系での可視化（元の実装と同様）"""
        output_config = self.config.get('output', {})
        quantities = output_config.get('quantities', ['Az', 'B'])
        
        if isinstance(quantities[0], list):
            n_rows = len(quantities)
            n_cols = max(len(row) for row in quantities)
            quantities_2d = quantities
        else:
            n_rows = 1
            n_cols = len(quantities)
            quantities_2d = [quantities]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                if i < len(quantities_2d) and j < len(quantities_2d[i]):
                    quantity = quantities_2d[i][j]
                    if quantity:
                        self._plot_quantity(ax, quantity, 'cartesian')
                else:
                    ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_polar(self):
        """極座標系での可視化（YAMLの出力設定に従う）"""
        output_config = self.config.get('output', {})
        quantities = output_config.get('quantities', [['Az', 'B'], ['input', 'polar_input']])
        
        # 2次元配列形式の処理
        if isinstance(quantities[0], list):
            n_rows = len(quantities)
            n_cols = max(len(row) for row in quantities)
            quantities_2d = quantities
        else:
            n_rows = 1
            n_cols = len(quantities)
            quantities_2d = [quantities]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                if i < len(quantities_2d) and j < len(quantities_2d[i]):
                    quantity = quantities_2d[i][j]
                    if quantity:
                        self._plot_quantity(ax, quantity, 'polar')
                else:
                    ax.axis('off')
        
        plt.tight_layout()
        
        # ファイル出力
        if output_config.get('save_figures', False):
            output_dir = Path(output_config.get('output_dir', './output'))
            output_dir.mkdir(exist_ok=True)
            filename = output_dir / 'magnetic_field_analysis_polar.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\n図を保存しました: {filename}")
        
        plt.show()
    
    def _polar_to_cartesian(self, data_polar):
        """極座標データを直交座標系に変換"""
        # 出力画像サイズ
        size = 2 * self.nr
        data_cart = np.zeros((size, size))
        
        for i in range(self.nr):
            r = self.r[i]
            r_norm = (r - self.r_min) / (self.r_max - self.r_min)
            for j in range(self.ntheta):
                theta = self.theta[j]
                
                # 直交座標での位置
                x = int(size/2 + r_norm * self.nr * np.cos(theta))
                y = int(size/2 + r_norm * self.nr * np.sin(theta))
                
                if 0 <= x < size and 0 <= y < size:
                    data_cart[y, x] = data_polar[i, j]
        
        return data_cart
    
    def _plot_quantity(self, ax, quantity, coord_system):
        """物理量のプロット（極座標と直交座標の両方に対応）"""
        colormap_config = self.config.get('output', {}).get('colormaps', {})
        quantity_config = colormap_config.get(quantity, {})
        
        # デフォルトカラーマップ
        default_cmaps = {
            'Az': 'viridis',
            'B': 'hot',
            'Br': 'RdBu_r',
            'Btheta': 'RdBu_r',
            'Bx': 'RdBu_r',
            'By': 'RdBu_r',
            'H': 'plasma',
            'mu': 'Blues',
            'jz': 'RdBu_r',
            'input': 'gray',
            'polar_input': 'gray'
        }
        
        cmap = quantity_config.get('cmap', default_cmaps.get(quantity, 'viridis'))
        vmin = quantity_config.get('vmin', None)
        vmax = quantity_config.get('vmax', None)
        
        if coord_system == 'polar':
            # 極座標系の場合
            if quantity == 'input':
                # 極座標画像を直交座標に変換して表示
                # RGB画像の変換
                img_cart = np.zeros((2*self.nr, 2*self.nr, 3), dtype=np.uint8)
                for c in range(3):
                    img_cart[:, :, c] = self._polar_to_cartesian(self.image[:, :, c])
                ax.imshow(img_cart)
                ax.set_title('Input (Cartesian View)')
                ax.set_aspect('equal')
                
            elif quantity == 'polar_input':
                # 極座標画像をそのまま表示
                ax.imshow(self.image)
                ax.set_title('Input (Polar Coordinates)')
                ax.set_xlabel('θ [mesh]')
                ax.set_ylabel('r [mesh]')
                
            elif quantity == 'Az':
                # ベクトルポテンシャルを直交座標で表示
                Az_cart = self._polar_to_cartesian(self.Az)
                im = ax.contourf(Az_cart, levels=20, cmap=cmap)
                ax.set_title('Vector Potential Az [Wb/m]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
                
            elif quantity == 'B':
                # 磁束密度の大きさを直交座標で表示
                B_cart = self._polar_to_cartesian(self.B_magnitude)
                im = ax.imshow(B_cart, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('Magnetic Flux Density |B| [T]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
                
                # 磁力線の追加
                contour_config = quantity_config.get('contour', {})
                if contour_config.get('show', False):
                    Az_cart = self._polar_to_cartesian(self.Az)
                    levels = contour_config.get('levels', 15)
                    color = contour_config.get('color', 'white')
                    alpha = contour_config.get('alpha', 0.3)
                    ax.contour(Az_cart, levels=levels, colors=color, 
                              alpha=alpha, linewidths=0.5)
                    
            elif quantity == 'Br':
                # 半径方向磁束密度
                Br_cart = self._polar_to_cartesian(self.Br)
                im = ax.imshow(Br_cart, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('Radial Flux Density Br [T]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
                
            elif quantity == 'Btheta':
                # 角度方向磁束密度
                Btheta_cart = self._polar_to_cartesian(self.Btheta)
                im = ax.imshow(Btheta_cart, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('Tangential Flux Density Bθ [T]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
                
            elif quantity == 'mu':
                # 透磁率分布
                mu_cart = self._polar_to_cartesian(self.mu_map)
                im = ax.imshow(mu_cart, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('Permeability μ [H/m]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
                
            elif quantity == 'jz':
                # 電流密度分布
                jz_cart = self._polar_to_cartesian(self.jz_map)
                im = ax.imshow(jz_cart, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('Current Density Jz [A/m²]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
                
            elif quantity == 'H':
                # 磁場 H = B/μ
                H_magnitude = self.B_magnitude / self.mu_map
                H_cart = self._polar_to_cartesian(H_magnitude)
                im = ax.imshow(H_cart, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('Magnetic Field |H| [A/m]')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'Unknown: {quantity}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Unknown: {quantity}')
                
        else:
            # 直交座標系の場合（元の実装）
            if quantity == 'Az':
                im = ax.contourf(self.Az, levels=20, cmap=cmap)
                ax.set_title('Vector Potential Az [Wb/m]')
                plt.colorbar(im, ax=ax)
            elif quantity == 'B':
                im = ax.imshow(self.B_magnitude, origin='lower', cmap=cmap)
                ax.set_title('Magnetic Flux Density |B| [T]')
                plt.colorbar(im, ax=ax)
                
                # 磁力線の追加
                contour_config = quantity_config.get('contour', {})
                if contour_config.get('show', True):
                    levels = contour_config.get('levels', 15)
                    color = contour_config.get('color', 'white')
                    alpha = contour_config.get('alpha', 0.3)
                    ax.contour(self.Az, levels=levels, colors=color, 
                              alpha=alpha, linewidths=0.5)
            elif quantity == 'mu':
                im = ax.imshow(self.mu_map, origin='lower', cmap=cmap)
                ax.set_title('Permeability μ [H/m]')
                plt.colorbar(im, ax=ax)
            elif quantity == 'jz':
                im = ax.imshow(self.jz_map, origin='lower', cmap=cmap)
                ax.set_title('Current Density Jz [A/m²]')
                plt.colorbar(im, ax=ax)
            elif quantity == 'H':
                H_magnitude = self.B_magnitude / self.mu_map
                im = ax.imshow(H_magnitude, origin='lower', cmap=cmap)
                ax.set_title('Magnetic Field |H| [A/m]')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'Unknown: {quantity}', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('X [mesh]')
            ax.set_ylabel('Y [mesh]')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("2次元磁場解析プログラム（ベクトルポテンシャル法）")
    print("極座標系対応版")
    print("=" * 60)
    
    # YAMLファイルの入力
    while True:
        yaml_path = input("\nYAMLファイルのパスを入力してください: ").strip()
        if os.path.exists(yaml_path):
            break
        print(f"エラー: ファイルが見つかりません: {yaml_path}")
    
    # 画像ファイルの入力
    while True:
        image_path = input("媒質画像ファイルのパスを入力してください: ").strip()
        if os.path.exists(image_path):
            break
        print(f"エラー: ファイルが見つかりません: {image_path}")
    
    try:
        # 解析器の初期化
        analyzer = MagneticFieldAnalyzer(yaml_path, image_path)
        
        # 方程式の求解
        analyzer.solve()
        
        # 結果の可視化
        analyzer.visualize_results()
        
        print("\n解析が正常に完了しました！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()