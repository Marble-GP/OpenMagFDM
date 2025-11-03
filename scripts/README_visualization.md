# 磁界解析結果の可視化ツール

C++版MagFDMsolverの出力CSVファイルを読み込んで可視化するPythonスクリプトです。

## 必要な環境

- Python 3.x
- numpy
- matplotlib

## インストール

```bash
pip install numpy matplotlib
```

## 使用方法

### 基本的な使い方

```bash
# 画面に表示
python3 scripts/visualize_results.py Az_20251023_231632.csv

# ファイルに保存
python3 scripts/visualize_results.py Az_20251023_231632.csv --save
```

### オプション

```
positional arguments:
  az_file              Az CSV file path
  mu_file              Mu CSV file path (optional, auto-detected)

optional arguments:
  -h, --help           show this help message and exit
  --dx DX              Mesh spacing in x direction [m] (default: 1e-3)
  --dy DY              Mesh spacing in y direction [m] (default: 1e-3)
  --skip SKIP          Vector field skip interval (default: 10)
  --save               Save figures instead of showing
  --output-dir DIR     Output directory for saved figures (default: .)
```

### 使用例

```bash
# Muファイルを明示的に指定
python3 scripts/visualize_results.py Az_20251023_231632.csv Mu_20251023_231632.csv

# メッシュ間隔を指定
python3 scripts/visualize_results.py Az_20251023_231632.csv --dx 0.001 --dy 0.001

# ベクトル場の矢印の間隔を変更
python3 scripts/visualize_results.py Az_20251023_231632.csv --skip 5

# 画像を保存（特定のディレクトリに）
python3 scripts/visualize_results.py Az_20251023_231632.csv --save --output-dir ./output
```

## 出力内容

### 基本プロット (`*_basic.png`)

4つのサブプロットを含む図：
1. **ベクトルポテンシャル Az** - 等高線表示（viridisカラーマップ）
2. **透磁率分布 μ** - 材質分布を表示（Bluesカラーマップ）
3. **磁束密度 |B|** - 磁束密度の大きさ（hotカラーマップ）+ 磁力線
4. **磁界強度 |H|** - 磁界強度の大きさ（plasmaカラーマップ）

### ベクトル場プロット (`*_vector.png`)

- 背景：磁束密度の大きさ（hotカラーマップ）
- 矢印：磁束密度ベクトル場
- 白線：磁力線（Azの等高線）

## 計算される物理量

スクリプト内部で以下の計算が行われます：

1. **磁束密度 B**
   - Bx = ∂Az/∂y
   - By = -∂Az/∂x
   - |B| = √(Bx² + By²)

2. **磁界強度 H**
   - H = B / μ
   - |H| = |B| / μ

## トラブルシューティング

### Muファイルが見つからない

```
Error: Mu file not found: Mu_20251023_231632.csv
```

→ Muファイルを明示的に指定してください：
```bash
python3 scripts/visualize_results.py Az_file.csv Mu_file.csv
```

### 配列サイズが一致しない

```
Error: Shape mismatch - Az: (200, 200), Mu: (100, 100)
```

→ AzとMuのCSVファイルが同じ解析結果のものか確認してください

## Python仮想環境の使用

CLAUDE.mdに記載の仮想環境を使用する場合：

```bash
/home/marble/myPyEnv/py3123/bin/python3 scripts/visualize_results.py Az_20251023_231632.csv --save
```
