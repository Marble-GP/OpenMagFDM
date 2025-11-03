#!/usr/bin/env python3
"""
磁界解析結果の可視化ツール
C++版MagFDMsolverの出力CSVファイルを読み込んで可視化します
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys


def load_csv(filepath):
    """CSVファイルを読み込む"""
    try:
        data = np.loadtxt(filepath, delimiter=',')
        print(f"Loaded: {filepath} (shape: {data.shape})")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def calculate_magnetic_field(Az, dx=1e-3, dy=1e-3):
    """
    ベクトルポテンシャルから磁束密度を計算
    Bx = ∂Az/∂y
    By = -∂Az/∂x
    """
    ny, nx = Az.shape

    # Bx = ∂Az/∂y
    Bx = np.zeros_like(Az)
    Bx[1:-1, :] = (Az[2:, :] - Az[:-2, :]) / (2 * dy)
    Bx[0, :] = (Az[1, :] - Az[0, :]) / dy
    Bx[-1, :] = (Az[-1, :] - Az[-2, :]) / dy

    # By = -∂Az/∂x
    By = np.zeros_like(Az)
    By[:, 1:-1] = -(Az[:, 2:] - Az[:, :-2]) / (2 * dx)
    By[:, 0] = -(Az[:, 1] - Az[:, 0]) / dx
    By[:, -1] = -(Az[:, -1] - Az[:, -2]) / dx

    # 磁束密度の大きさ
    B_magnitude = np.sqrt(Bx**2 + By**2)

    return Bx, By, B_magnitude


def visualize_basic(Az, Mu, dx=1e-3, dy=1e-3):
    """基本的な可視化（Az, Mu, B）"""
    # 磁束密度の計算
    Bx, By, B_magnitude = calculate_magnetic_field(Az, dx, dy)

    # 磁界強度 H = B/μ
    H_magnitude = B_magnitude / Mu

    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Magnetic Field Analysis Results', fontsize=16, fontweight='bold')

    # 1. ベクトルポテンシャル Az
    ax = axes[0, 0]
    im1 = ax.contourf(Az, levels=20, cmap='viridis')
    ax.contour(Az, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_title('Vector Potential Az [Wb/m]')
    ax.set_xlabel('X [mesh]')
    ax.set_ylabel('Y [mesh]')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax)

    # 2. 透磁率分布 μ
    ax = axes[0, 1]
    im2 = ax.imshow(Mu, origin='lower', cmap='Blues')
    ax.set_title('Permeability μ [H/m]')
    ax.set_xlabel('X [mesh]')
    ax.set_ylabel('Y [mesh]')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax)

    # 3. 磁束密度 B
    ax = axes[1, 0]
    im3 = ax.imshow(B_magnitude, origin='lower', cmap='hot')
    ax.contour(Az, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_title('Magnetic Flux Density |B| [T]')
    ax.set_xlabel('X [mesh]')
    ax.set_ylabel('Y [mesh]')
    ax.set_aspect('equal')
    plt.colorbar(im3, ax=ax, label='Tesla')

    # 4. 磁界強度 H
    ax = axes[1, 1]
    im4 = ax.imshow(H_magnitude, origin='lower', cmap='plasma')
    ax.set_title('Magnetic Field |H| [A/m]')
    ax.set_xlabel('X [mesh]')
    ax.set_ylabel('Y [mesh]')
    ax.set_aspect('equal')
    plt.colorbar(im4, ax=ax, label='A/m')

    plt.tight_layout()
    return fig


def visualize_vector_field(Az, Mu, dx=1e-3, dy=1e-3, skip=10):
    """ベクトル場の可視化"""
    # 磁束密度の計算
    Bx, By, B_magnitude = calculate_magnetic_field(Az, dx, dy)

    ny, nx = Az.shape
    Y, X = np.meshgrid(range(ny), range(nx), indexing='ij')

    fig, ax = plt.subplots(figsize=(10, 8))

    # 背景に磁束密度の大きさを表示
    im = ax.imshow(B_magnitude, origin='lower', cmap='hot', alpha=0.6)

    # ベクトル場を矢印で表示（間引いて表示）
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              By[::skip, ::skip], Bx[::skip, ::skip],
              color='cyan', alpha=0.8, scale=None, width=0.003)

    # 磁力線（等ポテンシャル線）
    ax.contour(Az, levels=20, colors='white', alpha=0.4, linewidths=1)

    ax.set_title('Magnetic Field Vector Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('X [mesh]')
    ax.set_ylabel('Y [mesh]')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|B| [T]')

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize magnetic field analysis results')
    parser.add_argument('az_file', help='Az CSV file path')
    parser.add_argument('mu_file', nargs='?', help='Mu CSV file path (optional, auto-detected)')
    parser.add_argument('--dx', type=float, default=1e-3, help='Mesh spacing in x direction [m]')
    parser.add_argument('--dy', type=float, default=1e-3, help='Mesh spacing in y direction [m]')
    parser.add_argument('--skip', type=int, default=10, help='Vector field skip interval')
    parser.add_argument('--save', action='store_true', help='Save figures instead of showing')
    parser.add_argument('--output-dir', default='.', help='Output directory for saved figures')

    args = parser.parse_args()

    # Az ファイルの読み込み
    az_path = Path(args.az_file)
    if not az_path.exists():
        print(f"Error: File not found: {az_path}")
        sys.exit(1)

    Az = load_csv(az_path)

    # Mu ファイルの自動検出または読み込み
    if args.mu_file:
        mu_path = Path(args.mu_file)
    else:
        # Az_*.csv から Mu_*.csv を推測
        mu_filename = az_path.name.replace('Az_', 'Mu_')
        mu_path = az_path.parent / mu_filename

    if not mu_path.exists():
        print(f"Error: Mu file not found: {mu_path}")
        print("Please specify Mu file explicitly with second argument")
        sys.exit(1)

    Mu = load_csv(mu_path)

    # 配列サイズの確認
    if Az.shape != Mu.shape:
        print(f"Error: Shape mismatch - Az: {Az.shape}, Mu: {Mu.shape}")
        sys.exit(1)

    print(f"\nMesh spacing: dx={args.dx} m, dy={args.dy} m")
    print(f"Array size: {Az.shape[0]} rows x {Az.shape[1]} columns")
    print(f"\nStatistics:")
    print(f"  Az: min={Az.min():.6e}, max={Az.max():.6e}")
    print(f"  Mu: min={Mu.min():.6e}, max={Mu.max():.6e}")

    # 可視化
    print("\n=== Generating visualizations ===")

    # 基本プロット
    print("Creating basic plots...")
    fig1 = visualize_basic(Az, Mu, args.dx, args.dy)

    # ベクトル場プロット
    print("Creating vector field plot...")
    fig2 = visualize_vector_field(Az, Mu, args.dx, args.dy, args.skip)

    # 保存または表示
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        base_name = az_path.stem

        fig1_path = output_dir / f"{base_name}_basic.png"
        fig2_path = output_dir / f"{base_name}_vector.png"

        print(f"\nSaving figures...")
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig1_path}")

        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig2_path}")
    else:
        print("\nShowing plots... (close windows to exit)")
        plt.show()

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
