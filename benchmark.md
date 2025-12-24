ンチマーク画像構成（Markdown記録）
# Benchmark Image: output_20251214_114950 / output_20251214_122655

## 画像サイズ
- 500 x 500 pixels
- dx = dy = 0.2mm

## 境界条件
- Left/Right: Periodic (周期境界)
- Top/Bottom: Dirichlet (A=0)

## 領域構成
- 灰色帯(上下): y=0〜109, y=391〜499 (固定、スライド外)
- 白色領域: 空気 (mu_r=1)
- 灰色長方形コア: (400,110)-(500,390) → iron_rotor (評価対象)
- 黄色縦線: 電流源 (スライド領域内にある！)

## スライド設定
- 方向: horizontal (x方向)
- 領域: y=110〜390
- 移動量: 4px/step × 100 steps = 400px