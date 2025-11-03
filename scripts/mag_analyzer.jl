#!/usr/bin/env julia
"""
2次元磁場解析プログラム(ベクトルポテンシャル法)
極座標系対応版 - Julia実装
"""

using LinearAlgebra
using SparseArrays
using YAML
using Images
using FileIO
using Plots
using Printf

"""
ベクトルポテンシャル法による2次元磁場解析クラス(極座標対応)
"""
mutable struct MagneticFieldAnalyzer
    # 設定
    config::Dict
    image::Array{RGB{N0f8}, 2}
    
    # 座標系
    coordinate_system::String
    
    # 直交座標系パラメータ
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    
    # 極座標系パラメータ
    nr::Int
    ntheta::Int
    r_min::Float64
    r_max::Float64
    dr::Float64
    dtheta::Float64
    r::Vector{Float64}
    theta::Vector{Float64}
    
    # 媒質プロパティ
    mu_map::Matrix{Float64}
    jz_map::Matrix{Float64}
    
    # 結果
    Az::Union{Matrix{Float64}, Nothing}
    Br::Union{Matrix{Float64}, Nothing}
    Btheta::Union{Matrix{Float64}, Nothing}
    Bx::Union{Matrix{Float64}, Nothing}
    By::Union{Matrix{Float64}, Nothing}
    B_magnitude::Union{Matrix{Float64}, Nothing}
    
    function MagneticFieldAnalyzer(config_path::String, image_path::String)
        # 設定と画像の読み込み
        config = YAML.load_file(config_path)
        image = load(image_path)
        
        # 座標系の判定
        coordinate_system = get(config, "coordinate_system", "cartesian")
        println("座標系: ", coordinate_system)
        
        # 仮初期化
        new_obj = new()
        new_obj.config = config
        new_obj.image = image
        new_obj.coordinate_system = coordinate_system
        
        # 初期化値
        new_obj.nx = 0
        new_obj.ny = 0
        new_obj.dx = 0.0
        new_obj.dy = 0.0
        new_obj.nr = 0
        new_obj.ntheta = 0
        new_obj.r_min = 0.0
        new_obj.r_max = 0.0
        new_obj.dr = 0.0
        new_obj.dtheta = 0.0
        new_obj.r = Float64[]
        new_obj.theta = Float64[]
        new_obj.mu_map = zeros(1, 1)
        new_obj.jz_map = zeros(1, 1)
        new_obj.Az = nothing
        new_obj.Br = nothing
        new_obj.Btheta = nothing
        new_obj.Bx = nothing
        new_obj.By = nothing
        new_obj.B_magnitude = nothing
        
        if coordinate_system == "polar"
            setup_polar_system!(new_obj)
        else
            setup_cartesian_system!(new_obj)
        end
        
        return new_obj
    end
end

"""
直交座標系のセットアップ
"""
function setup_cartesian_system!(analyzer::MagneticFieldAnalyzer)
    analyzer.ny, analyzer.nx = size(analyzer.image)
    
    mesh = get(analyzer.config, "mesh", Dict())
    analyzer.dx = get(mesh, "dx", 1.0)
    analyzer.dy = get(mesh, "dy", 1.0)
    
    analyzer.mu_map = ones(analyzer.ny, analyzer.nx)
    analyzer.jz_map = zeros(analyzer.ny, analyzer.nx)
    
    setup_material_properties!(analyzer)
end

"""
極座標系のセットアップ
"""
function setup_polar_system!(analyzer::MagneticFieldAnalyzer)
    analyzer.nr, analyzer.ntheta = size(analyzer.image)
    println("極座標メッシュ: nr=$(analyzer.nr), ntheta=$(analyzer.ntheta)")
    
    polar_mesh = get(analyzer.config, "polar_mesh", Dict())
    analyzer.r_min = get(polar_mesh, "r_min", 0.01)
    analyzer.r_max = get(polar_mesh, "r_max", 1.0)
    
    analyzer.dr = (analyzer.r_max - analyzer.r_min) / (analyzer.nr - 1)
    analyzer.dtheta = 2π / analyzer.ntheta
    
    analyzer.r = range(analyzer.r_min, analyzer.r_max, length=analyzer.nr) |> collect
    analyzer.theta = range(0, 2π, length=analyzer.ntheta+1)[1:end-1] |> collect
    
    analyzer.mu_map = ones(analyzer.nr, analyzer.ntheta)
    analyzer.jz_map = zeros(analyzer.nr, analyzer.ntheta)
    
    setup_material_properties!(analyzer)
end

"""
RGB値から媒質プロパティへのマッピング
"""
function setup_material_properties!(analyzer::MagneticFieldAnalyzer)
    materials = get(analyzer.config, "materials", Dict())
    mu_0 = 4π * 1e-7  # 真空の透磁率
    
    for (material_name, properties) in materials
        rgb_vec = get(properties, "rgb", [255, 255, 255])
        mu_r = get(properties, "mu_r", 1.0)
        jz = get(properties, "jz", 0.0)
        
        # RGB値が一致する画素を検出
        target_color = RGB{N0f8}(rgb_vec[1]/255, rgb_vec[2]/255, rgb_vec[3]/255)
        mask = analyzer.image .== target_color
        
        # 透磁率と電流密度を設定
        analyzer.mu_map[mask] .= mu_r * mu_0
        analyzer.jz_map[mask] .= jz
        
        println("材料 '$material_name': RGB$rgb_vec → μr=$mu_r, Jz=$jz A/m²")
        println("  該当画素数: $(sum(mask))")
    end
end

"""
界面における透磁率を計算(調和平均)
"""
function get_mu_at_interface(analyzer::MagneticFieldAnalyzer, i, j::Int, direction::String)
    if analyzer.coordinate_system == "polar"
        return get_mu_at_interface_polar(analyzer, i, j, direction)
    else
        return get_mu_at_interface_cartesian(analyzer, Int(i), j, direction)
    end
end

"""
直交座標系での界面透磁率
"""
function get_mu_at_interface_cartesian(analyzer::MagneticFieldAnalyzer, i::Int, j::Int, direction::String)
    if direction == "x+"
        if i < analyzer.nx
            return 2.0 / (1.0/analyzer.mu_map[j, i] + 1.0/analyzer.mu_map[j, i+1])
        else
            return analyzer.mu_map[j, i]
        end
    elseif direction == "x-"
        if i > 1
            return 2.0 / (1.0/analyzer.mu_map[j, i] + 1.0/analyzer.mu_map[j, i-1])
        else
            return analyzer.mu_map[j, i]
        end
    elseif direction == "y+"
        if j < analyzer.ny
            return 2.0 / (1.0/analyzer.mu_map[j, i] + 1.0/analyzer.mu_map[j+1, i])
        else
            return analyzer.mu_map[j, i]
        end
    elseif direction == "y-"
        if j > 1
            return 2.0 / (1.0/analyzer.mu_map[j, i] + 1.0/analyzer.mu_map[j-1, i])
        else
            return analyzer.mu_map[j, i]
        end
    end
end

"""
極座標系での界面透磁率
"""
function get_mu_at_interface_polar(analyzer::MagneticFieldAnalyzer, r_idx, theta_idx::Int, direction::String)
    r_idx_int = Int(floor(r_idx))
    
    if direction == "r"
        if r_idx == r_idx_int
            if r_idx_int > 1 && r_idx_int < analyzer.nr
                return 2.0 / (1.0/analyzer.mu_map[r_idx_int, theta_idx] + 
                              1.0/analyzer.mu_map[r_idx_int + 1, theta_idx])
            else
                return analyzer.mu_map[clamp(r_idx_int, 1, analyzer.nr), theta_idx]
            end
        else
            r_idx_low = Int(floor(r_idx))
            if r_idx_low >= 1 && r_idx_low < analyzer.nr
                return 2.0 / (1.0/analyzer.mu_map[r_idx_low, theta_idx] + 
                              1.0/analyzer.mu_map[r_idx_low + 1, theta_idx])
            else
                return analyzer.mu_map[clamp(r_idx_low, 1, analyzer.nr), theta_idx]
            end
        end
    else
        # 角度方向は周期境界
        theta_next = mod1(theta_idx + 1, analyzer.ntheta)
        return 2.0 / (1.0/analyzer.mu_map[r_idx_int, theta_idx] + 
                      1.0/analyzer.mu_map[r_idx_int, theta_next])
    end
end

"""
境界条件の検証
"""
function validate_boundary_conditions(analyzer::MagneticFieldAnalyzer)
    bc = get(analyzer.config, "boundary_conditions", Dict())
    
    bc_left = get(bc, "left", Dict("type" => "dirichlet", "value" => 0.0))
    bc_right = get(bc, "right", Dict("type" => "dirichlet", "value" => 0.0))
    bc_bottom = get(bc, "bottom", Dict("type" => "dirichlet", "value" => 0.0))
    bc_top = get(bc, "top", Dict("type" => "dirichlet", "value" => 0.0))
    
    return bc_left, bc_right, bc_bottom, bc_top
end

"""
直交座標系での有限差分法による求解
"""
function solve_cartesian!(analyzer::MagneticFieldAnalyzer)
    println("\n=== 直交座標系での連立方程式の構築 ===")
    
    bc_left, bc_right, bc_bottom, bc_top = validate_boundary_conditions(analyzer)
    
    println("境界条件:")
    println("  左: $(bc_left["type"])")
    println("  右: $(bc_right["type"])")
    println("  下: $(bc_bottom["type"])")
    println("  上: $(bc_top["type"])")
    
    n = analyzer.nx * analyzer.ny
    I_indices = Int[]
    J_indices = Int[]
    V_values = Float64[]
    rhs = zeros(n)
    
    # 各格子点について方程式を構築
    for j in 1:analyzer.ny
        for i in 1:analyzer.nx
            idx = (j - 1) * analyzer.nx + i
            
            is_left = (i == 1)
            is_right = (i == analyzer.nx)
            is_bottom = (j == 1)
            is_top = (j == analyzer.ny)
            
            # ディリクレ境界条件
            if is_left && bc_left["type"] == "dirichlet"
                push!(I_indices, idx)
                push!(J_indices, idx)
                push!(V_values, 1.0)
                rhs[idx] = get(bc_left, "value", 0.0)
                continue
            elseif is_right && bc_right["type"] == "dirichlet"
                push!(I_indices, idx)
                push!(J_indices, idx)
                push!(V_values, 1.0)
                rhs[idx] = get(bc_right, "value", 0.0)
                continue
            elseif is_bottom && bc_bottom["type"] == "dirichlet"
                push!(I_indices, idx)
                push!(J_indices, idx)
                push!(V_values, 1.0)
                rhs[idx] = get(bc_bottom, "value", 0.0)
                continue
            elseif is_top && bc_top["type"] == "dirichlet"
                push!(I_indices, idx)
                push!(J_indices, idx)
                push!(V_values, 1.0)
                rhs[idx] = get(bc_top, "value", 0.0)
                continue
            end
            
            # 内部点の処理
            coeff_center = 0.0
            
            # X方向の差分項
            if !is_left
                mu_west = get_mu_at_interface(analyzer, i, j, "x-")
                coeff_west = 1.0 / (mu_west * analyzer.dx^2)
                push!(I_indices, idx)
                push!(J_indices, idx - 1)
                push!(V_values, coeff_west)
                coeff_center -= coeff_west
            end
            
            if !is_right
                mu_east = get_mu_at_interface(analyzer, i, j, "x+")
                coeff_east = 1.0 / (mu_east * analyzer.dx^2)
                push!(I_indices, idx)
                push!(J_indices, idx + 1)
                push!(V_values, coeff_east)
                coeff_center -= coeff_east
            end
            
            # Y方向の差分項
            if !is_bottom
                mu_south = get_mu_at_interface(analyzer, i, j, "y-")
                coeff_south = 1.0 / (mu_south * analyzer.dy^2)
                push!(I_indices, idx)
                push!(J_indices, idx - analyzer.nx)
                push!(V_values, coeff_south)
                coeff_center -= coeff_south
            end
            
            if !is_top
                mu_north = get_mu_at_interface(analyzer, i, j, "y+")
                coeff_north = 1.0 / (mu_north * analyzer.dy^2)
                push!(I_indices, idx)
                push!(J_indices, idx + analyzer.nx)
                push!(V_values, coeff_north)
                coeff_center -= coeff_north
            end
            
            # 中心点の係数
            push!(I_indices, idx)
            push!(J_indices, idx)
            push!(V_values, coeff_center)
            
            # 右辺(電流密度項)
            rhs[idx] = -analyzer.jz_map[j, i]
        end
    end
    
    # 連立方程式の求解
    A_matrix = sparse(I_indices, J_indices, V_values, n, n)
    println("行列サイズ: $(size(A_matrix)), 非ゼロ要素数: $(nnz(A_matrix))")
    
    println("\n=== 方程式の求解中... ===")
    Az_flat = A_matrix \ rhs
    analyzer.Az = reshape(Az_flat, analyzer.nx, analyzer.ny)'
    
    println("求解完了!")
    calculate_magnetic_field_cartesian!(analyzer)
end

"""
極座標系での有限差分法による求解
"""
function solve_polar!(analyzer::MagneticFieldAnalyzer)
    println("\n=== 極座標系での連立方程式の構築 ===")
    println("メッシュ: r方向 $(analyzer.nr)点, θ方向 $(analyzer.ntheta)点")
    @printf("半径範囲: %.3f ~ %.3f m\n", analyzer.r_min, analyzer.r_max)
    
    n = analyzer.nr * analyzer.ntheta
    I_indices = Int[]
    J_indices = Int[]
    V_values = Float64[]
    rhs = zeros(n)
    
    # 境界条件の取得
    bc = get(analyzer.config, "polar_boundary_conditions", Dict())
    bc_inner = get(bc, "inner", Dict("type" => "neumann"))
    bc_outer = get(bc, "outer", Dict("type" => "dirichlet", "value" => 0.0))
    
    println("境界条件: 内側=$(bc_inner["type"]), 外側=$(bc_outer["type"])")
    
    # 各格子点について方程式を構築
    for i in 1:analyzer.nr
        for j in 1:analyzer.ntheta
            idx = (i - 1) * analyzer.ntheta + j
            r = analyzer.r[i]
            
            # 内側境界
            if i == 1
                if bc_inner["type"] == "dirichlet"
                    push!(I_indices, idx)
                    push!(J_indices, idx)
                    push!(V_values, 1.0)
                    rhs[idx] = get(bc_inner, "value", 0.0)
                    continue
                elseif bc_inner["type"] == "neumann"
                    # ノイマン境界条件: ∂Az/∂r = 0
                    coeff_center = -1.0 / analyzer.dr
                    coeff_outer = 1.0 / analyzer.dr
                    
                    push!(I_indices, idx)
                    push!(J_indices, idx)
                    push!(V_values, coeff_center)
                    
                    push!(I_indices, idx)
                    push!(J_indices, i * analyzer.ntheta + j)
                    push!(V_values, coeff_outer)
                    
                    rhs[idx] = 0.0
                    continue
                end
            end
            
            # 外側境界
            if i == analyzer.nr
                if bc_outer["type"] == "dirichlet"
                    push!(I_indices, idx)
                    push!(J_indices, idx)
                    push!(V_values, 1.0)
                    rhs[idx] = get(bc_outer, "value", 0.0)
                    continue
                elseif bc_outer["type"] == "neumann"
                    coeff_center = 1.0 / analyzer.dr
                    coeff_inner = -1.0 / analyzer.dr
                    
                    push!(I_indices, idx)
                    push!(J_indices, idx)
                    push!(V_values, coeff_center)
                    
                    push!(I_indices, idx)
                    push!(J_indices, (i - 2) * analyzer.ntheta + j)
                    push!(V_values, coeff_inner)
                    
                    rhs[idx] = 0.0
                    continue
                end
            end
            
            # 内部点の処理
            coeff_center = 0.0
            
            # 半径方向の項
            mu_inner = get_mu_at_interface(analyzer, i - 0.5, j, "r")
            mu_outer = get_mu_at_interface(analyzer, i + 0.5, j, "r")
            
            coeff_r_inner = 1.0 / (mu_inner * analyzer.dr^2)
            coeff_r_outer = 1.0 / (mu_outer * analyzer.dr^2)
            
            # (1/r)∂Az/∂r の寄与
            coeff_r_inner *= (1.0 - 0.5 * analyzer.dr / r)
            coeff_r_outer *= (1.0 + 0.5 * analyzer.dr / r)
            
            push!(I_indices, idx)
            push!(J_indices, (i - 2) * analyzer.ntheta + j)
            push!(V_values, coeff_r_inner)
            
            push!(I_indices, idx)
            push!(J_indices, i * analyzer.ntheta + j)
            push!(V_values, coeff_r_outer)
            
            coeff_center -= (coeff_r_inner + coeff_r_outer)
            
            # 角度方向の項(周期境界条件)
            mu_current = analyzer.mu_map[i, j]
            coeff_theta = 1.0 / (r^2 * mu_current * analyzer.dtheta^2)
            
            # θ-1の点
            j_prev = mod1(j - 1, analyzer.ntheta)
            idx_prev = (i - 1) * analyzer.ntheta + j_prev
            push!(I_indices, idx)
            push!(J_indices, idx_prev)
            push!(V_values, coeff_theta)
            
            # θ+1の点
            j_next = mod1(j + 1, analyzer.ntheta)
            idx_next = (i - 1) * analyzer.ntheta + j_next
            push!(I_indices, idx)
            push!(J_indices, idx_next)
            push!(V_values, coeff_theta)
            
            coeff_center -= 2 * coeff_theta
            
            # 中心点の係数
            push!(I_indices, idx)
            push!(J_indices, idx)
            push!(V_values, coeff_center)
            
            # 右辺(電流密度項)
            rhs[idx] = -analyzer.jz_map[i, j]
        end
    end
    
    # 連立方程式の求解
    A_matrix = sparse(I_indices, J_indices, V_values, n, n)
    println("行列サイズ: $(size(A_matrix))")
    println("非ゼロ要素数: $(nnz(A_matrix))")
    
    println("\n=== 方程式の求解中... ===")
    Az_flat = A_matrix \ rhs
    analyzer.Az = reshape(Az_flat, analyzer.ntheta, analyzer.nr)'
    
    println("求解完了!")
    calculate_magnetic_field_polar!(analyzer)
end

"""
直交座標系での磁束密度の計算
"""
function calculate_magnetic_field_cartesian!(analyzer::MagneticFieldAnalyzer)
    # Bx = ∂Az/∂y
    analyzer.Bx = zeros(size(analyzer.Az))
    analyzer.Bx[2:end-1, :] = (analyzer.Az[3:end, :] - analyzer.Az[1:end-2, :]) / (2 * analyzer.dy)
    analyzer.Bx[1, :] = (analyzer.Az[2, :] - analyzer.Az[1, :]) / analyzer.dy
    analyzer.Bx[end, :] = (analyzer.Az[end, :] - analyzer.Az[end-1, :]) / analyzer.dy
    
    # By = -∂Az/∂x
    analyzer.By = zeros(size(analyzer.Az))
    analyzer.By[:, 2:end-1] = -(analyzer.Az[:, 3:end] - analyzer.Az[:, 1:end-2]) / (2 * analyzer.dx)
    analyzer.By[:, 1] = -(analyzer.Az[:, 2] - analyzer.Az[:, 1]) / analyzer.dx
    analyzer.By[:, end] = -(analyzer.Az[:, end] - analyzer.Az[:, end-1]) / analyzer.dx
    
    analyzer.B_magnitude = sqrt.(analyzer.Bx.^2 + analyzer.By.^2)
end

"""
極座標系での磁束密度の計算
"""
function calculate_magnetic_field_polar!(analyzer::MagneticFieldAnalyzer)
    # Br = (1/r)∂Az/∂θ
    analyzer.Br = zeros(size(analyzer.Az))
    for i in 1:analyzer.nr
        r = analyzer.r[i]
        for j in 1:analyzer.ntheta
            j_next = mod1(j + 1, analyzer.ntheta)
            j_prev = mod1(j - 1, analyzer.ntheta)
            analyzer.Br[i, j] = (analyzer.Az[i, j_next] - analyzer.Az[i, j_prev]) / (2 * r * analyzer.dtheta)
        end
    end
    
    # Bθ = -∂Az/∂r
    analyzer.Btheta = zeros(size(analyzer.Az))
    analyzer.Btheta[2:end-1, :] = -(analyzer.Az[3:end, :] - analyzer.Az[1:end-2, :]) / (2 * analyzer.dr)
    analyzer.Btheta[1, :] = -(analyzer.Az[2, :] - analyzer.Az[1, :]) / analyzer.dr
    analyzer.Btheta[end, :] = -(analyzer.Az[end, :] - analyzer.Az[end-1, :]) / analyzer.dr
    
    analyzer.B_magnitude = sqrt.(analyzer.Br.^2 + analyzer.Btheta.^2)
end

"""
座標系に応じた求解
"""
function solve!(analyzer::MagneticFieldAnalyzer)
    if analyzer.coordinate_system == "polar"
        solve_polar!(analyzer)
    else
        solve_cartesian!(analyzer)
    end
end

"""
極座標データを直交座標系に変換
"""
function polar_to_cartesian(analyzer::MagneticFieldAnalyzer, data_polar::Matrix{Float64})
    size_out = 2 * analyzer.nr
    data_cart = zeros(size_out, size_out)
    
    for i in 1:analyzer.nr
        r = analyzer.r[i]
        r_norm = (r - analyzer.r_min) / (analyzer.r_max - analyzer.r_min)
        for j in 1:analyzer.ntheta
            theta = analyzer.theta[j]
            
            x = Int(round(size_out/2 + r_norm * analyzer.nr * cos(theta)))
            y = Int(round(size_out/2 + r_norm * analyzer.nr * sin(theta)))
            
            if 1 <= x <= size_out && 1 <= y <= size_out
                data_cart[y, x] = data_polar[i, j]
            end
        end
    end
    
    return data_cart
end

"""
結果の可視化
"""
function visualize_results(analyzer::MagneticFieldAnalyzer)
    output_config = get(analyzer.config, "output", Dict())
    quantities = get(output_config, "quantities", ["Az", "B"])
    
    if analyzer.coordinate_system == "polar"
        visualize_polar(analyzer, quantities)
    else
        visualize_cartesian(analyzer, quantities)
    end
end

"""
直交座標系での可視化
"""
function visualize_cartesian(analyzer::MagneticFieldAnalyzer, quantities)
    n_plots = length(quantities)
    p = plot(layout=(1, n_plots), size=(400*n_plots, 400))
    
    for (idx, quantity) in enumerate(quantities)
        if quantity == "Az"
            contourf!(p[idx], analyzer.Az', 
                     title="Vector Potential Az [Wb/m]",
                     xlabel="X [mesh]", ylabel="Y [mesh]",
                     color=:viridis, aspect_ratio=:equal)
        elseif quantity == "B"
            heatmap!(p[idx], analyzer.B_magnitude',
                    title="Magnetic Flux Density |B| [T]",
                    xlabel="X [mesh]", ylabel="Y [mesh]",
                    color=:hot, aspect_ratio=:equal)
            contour!(p[idx], analyzer.Az', levels=15, 
                    color=:white, alpha=0.3, linewidth=0.5)
        end
    end
    
    display(p)
end

"""
極座標系での可視化
"""
function visualize_polar(analyzer::MagneticFieldAnalyzer, quantities)
    n_plots = length(quantities)
    p = plot(layout=(1, n_plots), size=(400*n_plots, 400))
    
    for (idx, quantity) in enumerate(quantities)
        if quantity == "Az"
            Az_cart = polar_to_cartesian(analyzer, analyzer.Az)
            contourf!(p[idx], Az_cart,
                     title="Vector Potential Az [Wb/m]",
                     color=:viridis, aspect_ratio=:equal)
        elseif quantity == "B"
            B_cart = polar_to_cartesian(analyzer, analyzer.B_magnitude)
            heatmap!(p[idx], B_cart,
                    title="Magnetic Flux Density |B| [T]",
                    color=:hot, aspect_ratio=:equal)
        elseif quantity == "Br"
            Br_cart = polar_to_cartesian(analyzer, analyzer.Br)
            heatmap!(p[idx], Br_cart,
                    title="Radial Flux Density Br [T]",
                    color=:RdBu, aspect_ratio=:equal)
        elseif quantity == "Btheta"
            Btheta_cart = polar_to_cartesian(analyzer, analyzer.Btheta)
            heatmap!(p[idx], Btheta_cart,
                    title="Tangential Flux Density Bθ [T]",
                    color=:RdBu, aspect_ratio=:equal)
        end
    end
    
    display(p)
end

"""
メイン実行関数
"""
function main()
    println("="^60)
    println("2次元磁場解析プログラム(ベクトルポテンシャル法)")
    println("極座標系対応版 - Julia実装")
    println("="^60)
    
    # ファイルパスの入力
    print("\nYAMLファイルのパスを入力してください: ")
    yaml_path = readline()
    
    if !isfile(yaml_path)
        println("エラー: ファイルが見つかりません: $yaml_path")
        return
    end
    
    print("媒質画像ファイルのパスを入力してください: ")
    image_path = readline()
    
    if !isfile(image_path)
        println("エラー: ファイルが見つかりません: $image_path")
        return
    end
    
    try
        # 解析器の初期化
        analyzer = MagneticFieldAnalyzer(yaml_path, image_path)
        
        # 方程式の求解
        solve!(analyzer)
        
        # 結果の可視化
        visualize_results(analyzer)
        
        println("\n解析が正常に完了しました!")
        
    catch e
        println("\nエラーが発生しました: $e")
        println(stacktrace(catch_backtrace()))
    end
end

# スクリプトとして実行された場合
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end