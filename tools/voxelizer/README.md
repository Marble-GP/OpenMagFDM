# MagVoxelizer — STEPから3Dボクセルへ

v2.0.0に向けた独立C++17ツール（ツール版0.1.0）。
Open CASCADE Technology（OCCT）でSTEPのBRepソリッドを読み、
セル中心の内外判定から材料ID付き直交格子を生成する。
2Dソルバーと独立してビルド・実行する。現在のMagFDMsolver/WebUIへ3D求解や3D表示を追加するものではない。

ブラウザーでSTEP/ボクセルを回転・拡大し、粒度を変えて生成/描画負荷を比較するには
[Voxel Lab（HTML/JS）](browserlab/README.md)を使用する。`lab.ps1`でローカル起動できる。

## 出力形式

| ファイル | 内容 | 主な用途 |
|---|---|---|
| `model.vti` | VTK ImageData。空気を含む全セルに`material_id`と`solid_id`（UInt32） | 3D格子データ、断面表示、将来のソルバー入力 |
| `surface.vtp` | VTK PolyData。空気に接するボクセル面を四角形で出力。外向き頂点順 | 3D表面表示、矩形メッシュの確認 |
| `manifest.json` | 格子寸法・原点・間隔・単位・ID対応・CAD/ボクセル体積・警告 | データの解釈、再現条件、形状誤差の確認 |

APNGのフレーム列に相当する内容を、**z方向も空間座標として持つ3D配列**で保存する。
VTIは原点・各軸の間隔・格子数から直方体セルを定義するため、全セルの頂点座標を重複保存しない。
VTPは表面だけで、内部の材料配置はVTIに残す。部品同士の内部界面はVTPに出力しない。
STL変換や三角形近似を介した内外判定は行わない。

形式の仕様：[VTK XML file formats](https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtkxml_file_format.html)。
読込の基盤：[OCCT STEPControl_Reader](https://www.occt3d.com/dev/doc/refman/html/class_s_t_e_p_control___reader.html)。

## この作業環境で試す

リポジトリ直下（`OpenMagFDM`）から：

```powershell
# まず部品のID、寸法、体積を確認
.\tools\voxelizer\run.ps1 --input .\tools\voxelizer\examples\rotor_demo.step --inspect

# 出力先は存在しないディレクトリを指定する
.\tools\voxelizer\run.ps1 `
  --input .\tools\voxelizer\examples\rotor_demo.step `
  --output .\build_voxelizer\my_rotor `
  --pitch-mm 0.5 `
  --material-map .\tools\voxelizer\examples\materials.json

# 自分のモデル：x/y方向0.25 mm、z方向0.5 mm、周囲に空気2セル
.\tools\voxelizer\run.ps1 `
  --input C:\models\assembly.step `
  --output .\build_voxelizer\my_assembly `
  --spacing-mm 0.25 0.25 0.5 --padding 2
```

`run.ps1`は`build_voxelizer/Release/MagVoxelizer.exe`を実行する。
今回用意した`build_voxel_deps/Library/bin`を子プロセスのDLL検索パスへ追加し、
終了時にPATHを戻す。グローバル環境の変更やサーバー起動は行わない。
入力の日本語パス・空白入りパスもテスト対象。

サンプルは外周リング・内周リング・軸・4個の矩形部品の7ソリッド。
材料ラベルはリング=1、軸=2、矩形部品=3。磁気特性や磁化方向は設定していない。
モーターの成立性や電磁性能を検証するモデルではなく、CAD→ボクセル変換の例である。
生成コードは`tests/geometry_test.cpp`にある。

この作業で生成済みの結果は`build_voxelizer/rotor_demo/`：
50×50×22セル、材料セル12,880、露出四角形9,560枚、`preview.png`付き。
これはビルド成果物としてGit対象外。元STEPと材料対応JSONは`examples/`に保存する。

## ParaView等での表示

- 表面：`surface.vtp`を開き、Apply。Coloringを`material_id`または`solid_id`、
  Representationを`Surface With Edges`にする。四角形面をそのまま確認できる。
- 体積：`model.vti`を開き、Apply。`Threshold`で`material_id >= 1`を抽出する。
  `Slice`や`Clip`で内部材料・空洞も確認できる。
- 単位はメートル。24 mmのモデル幅は`0.024`になる。

実際のVTK 9.3.1でVTI/VTPを読み、全セルの材料/部品ID、座標、セル数、
全表面セルが`VTK_QUAD`であることを検証した。既存WebUIの読込対応は別途必要。

## 空気の充填と透明度を調整するビューアー

`model.vti`には、モデルの周囲・内部の空隙・padding部分の**空気セルが材料ID 0として既に入っている**。
サンプルでは全55,000セルのうち42,120セルが空気、12,880セルが部品。
表示の透明度を0にしても、解析用データから空気セルを消したり、部品へ変えたりしない。
将来の3DソルバーはこのIDを空気の物性へ対応付けて利用できる。
空気の厚さや外側の境界条件の妥当性は、対象とする解析に応じて別途決める。

追加した`viewer.py`はPython/VTKによる独立したデスクトップビューアー。
`surface.vtp`ではなく**`model.vti`から材料ごとの領域を取り出す**ので、
空気・内部空洞・他材料と接する内部界面も表示対象になる。
空気は実セル領域の境界面を半透明描画する。霧のような体積吸収モデルではなく、
全空気ボクセルを表示用に別々の半透明立方体へ展開するものでもない。

この作業環境ではPython/VTKを準備済み。`OpenMagFDM`直下から：

```powershell
# ウィンドウを開いて回転・拡大・透明度調整
.\tools\voxelizer\view.ps1 .\build_voxelizer\rotor_demo

# 空気の不透明度8%、材料1の不透明度35%で開始
.\tools\voxelizer\view.ps1 .\build_voxelizer\rotor_demo `
  --air-opacity 0.08 --material-opacity 1=0.35

# 空気を完全に透明にし、全部品を不透明で表示
.\tools\voxelizer\view.ps1 .\build_voxelizer\rotor_demo `
  --air-opacity 0 --solid-opacity 1

# ウィンドウを開かず、新しいPNGへ出力
.\tools\voxelizer\view.ps1 .\build_voxelizer\rotor_demo `
  --air-opacity 0.08 --material-opacity 1=0.35 `
  --screenshot .\build_voxelizer\rotor_demo\my_transparency.png
```

操作：

- 左のスライダー：空気の不透明度。**0=完全透明、1=不透明**。既定0.06。
- 右のスライダー：選択した材料の不透明度。左右矢印キーで材料IDを切り替える。
  同じ材料IDを持つ部品はまとめて変わる。
- マウス：回転・拡大・移動。`E`：部品の格子線を切替。`R`：カメラを戻す。
- 薄青の枠線：全格子領域の外周。空気が透明でも解析領域を確認できる。

`--solid-opacity`は全部品の初期値、`--material-opacity ID=値`は材料別の初期値。
後者は複数指定できる。ウィンドウ内の変更はその表示中だけ有効で、元データへの書込はしない。
PNG出力も既存ファイルを上書きしない。
今回の描画確認は`build_voxelizer/rotor_demo/transparency.png`に保存済み。

周囲の空気を増やすには、変換時に`--padding`を指定する。例えば0.5 mm間隔で
`--padding 10`なら、CAD境界箱の両側へ各5 mm分のセルを追加する（端数調整分を除く）。
ビューアーの透明度はこの範囲や格子数を変えない。

表示にはPython 3、VTK、NumPyとOpenGLが必要（このPCではVTK 9.3.1で確認）。
別環境では`python viewer.py OUTPUT_DIRECTORY`で起動できる。
透明面の前後関係は[VTKのdepth peeling](https://vtk.org/doc/nightly/release/9.3/html/classvtkRenderer.html)で処理し、
このPCの描画でも使用されたことを確認した。GPU/ドライバーによっては通常の合成へフォールバックする。
表示処理の既定上限は200万セル・合計200万表面セル・64材料領域。
`--max-cells`、`--max-faces`で前2つを明示変更できるが、
表示用の形状展開にはC++変換器の格子配列より多いメモリを使う。

## ビルド

必要：CMake 3.16以上、C++17コンパイラー、OCCT 7.6以降の7.x開発用ヘッダー/ライブラリ。
テストにはPython 3。VTKライブラリはC++ツールのビルド・実行に不要。
動的リンク版OCCTを使う場合、実行時に同じ版・構成のDLL/共有ライブラリが必要。

Windowsで今回検証した組合せはMSVC 19.50（VS 2026）、OCCT 7.8.1、Release/x64。
既存FreeCADのアプリ配布には開発用ヘッダー/リンクライブラリが揃っていないため、
リポジトリ内の独立Conda環境を使用した。

```powershell
# リポジトリ直下。既に今回の環境があればcreateは不要
conda create --prefix .\build_voxel_deps --override-channels -c conda-forge occt=7.8.1 --yes
$voxelPrefix = (Resolve-Path .\build_voxel_deps).Path
cmake -S tools/voxelizer -B build_voxelizer -A x64 `
  "-DOpenCASCADE_DIR=$voxelPrefix/Library/lib/cmake/opencascade" `
  "-DPython3_EXECUTABLE=$voxelPrefix/python.exe"
cmake --build build_voxelizer --config Release --parallel 4
```

`cmake`がPATHにない場合、このPCでは次の実行ファイルを使用できる：
`C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`。
`ctest.exe`も同じフォルダーにある。

vcpkg用にはこのディレクトリ専用の`vcpkg.json`も用意している。
2D用のmanifestへOCCTを必須依存として追加していない。

```powershell
cmake -S tools/voxelizer -B build_voxelizer_vcpkg `
  "-DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build_voxelizer_vcpkg --config Release --parallel 4
```

Linux等ではOCCTの開発パッケージを入れた環境で：

```sh
cmake -S tools/voxelizer -B build_voxelizer -DCMAKE_BUILD_TYPE=Release
cmake --build build_voxelizer --parallel 4
ctest --test-dir build_voxelizer --output-on-failure
./build_voxelizer/MagVoxelizer --input model.step --output new_output --pitch-mm 0.5
```

vcpkg経路・Linux/macOS・OCCT 7.6/7.7/8.xでは今回ビルド実行していない。
Windowsのテスト環境で取得したConda依存一式は開発用で、配布パッケージではない。
将来のバイナリ配布時はOCCT等のライセンスと同梱ライブラリを整理する。

## CLIとデータの契約

- 入力：`.step` / `.stp`。閉じた有限体積ソリッドのみ。複数部品の位置・回転を反映する。
  ソリッド外の面・シェル・ワイヤーを含む入力は、部分的に捨てずエラーにする。
  BRepの妥当性・閉鎖性・正の体積・無限遠が外側であることを検査する。
- STEP長さ単位を読み、OCCT内部ではmmに変換。CLIの間隔もmm。
  **VTI・VTP・manifestの座標はm、体積はm³**。材料IDは寸法を持たない。
- 間隔は`--pitch-mm N`か`--spacing-mm DX DY DZ`のどちらかを明示する。
  各方向の間隔は有限正数で、内外判定許容値`1e-7 mm`の2倍より大きくする。
- 格子の原点はCAD境界箱の最小値から`padding×spacing`を引いたもの。
  CAD幅をセル数へ切り上げ、両側へpaddingを追加する。
  単位変換などの丸め誤差で余分な1層が増えないよう、整数セル数から
  判定許容値以内の差は整数へ丸める。最大側は最大1セル未満の余剰空間を持つ。
- `material_id=0`、`solid_id=0`は空気。既定では各ソリッドを1始まりの別材料IDにする。
  部品番号は今回のSTEP転送順であり、元CADの永続的な部品名/IDではない。
- 材料対応例：`{"solid_material_ids":[1,1,2,3,3,3,3]}`。
  配列は全ソリッド数と一致させる。正整数UInt32のみ、0は予約。
  異なる部品へ同じ材料IDを割り当てても`solid_id`は別に保持する。
- `index = x + nx * (y + ny * z)`。xが最速、zが最遅。
  VTIの点のExtentは`0 nx 0 ny 0 nz`、CellData数は`nx*ny*nz`。
- 表面は外向き四角形。空洞面も含む。各面が4頂点を独立に持ち、頂点は共有しない。
  隣接する非空気セル同士の内部面は出力しない。
- VTK XMLはraw-appended binary、LittleEndian、UInt64のブロック長。
  `model.vti`の配列はUInt32、`surface.vtp`の座標はFloat64、接続とオフセットはInt64。
  バイナリを含むため汎用XMLパーサーへファイル全体を直接渡さずVTKリーダーを使う。
- 出力ディレクトリは新規限定。親ディレクトリはあらかじめ存在させる。
  検証・分類が完了してから作成し、最後に`manifest.json`を確定する。
  書込失敗時の途中出力は残る場合があるが、完成したmanifestは残さない。
  再実行には新しい出力先を使う。
- `--inspect`はJSONを標準出力へ、進行とエラーは標準エラーへ出す。
  正常終了コード0、失敗1。

## 精度と資源の範囲

**初版はセル中心方式**。セル中心がソリッド内部または境界上なら、そのセル全体を割り当てる。
厳密な体積交差、占有率、境界の部分セル、適応格子は実装していない。
細い空隙・薄板・小部品は消える場合がある。各部品のCAD体積とボクセル体積、
相対誤差、セル数0の警告をmanifestで確認し、格子を細かくして比較する。
体積誤差が小さいことだけで局所形状や将来の磁界精度を保証しない。

セル中心が2ソリッドの内部に入ると、材料が同じでも重複として失敗する。
内部と境界の競合は内部を優先、境界同士は小さいソリッドIDを優先する。
サンプル点間の幾何学的重複までは検出しない。
CAD色・部品名・材料特性の自動取得、IGES等の別形式、画像積層は未対応。

既定上限は2,000万セル。分類中の格子配列は約9バイト/セル
（材料ID4、部品ID4、内外判定状態1）、これにOCCTの形状/分類器や入出力バッファが加わる。
出力はストリーム書込で、VTIの配列本体は8バイト/セル。
表面は既定200万面まで、配列本体144バイト/面。面の全リストはメモリに保持しない。
上限を超えた場合は間隔を広げるか、明示的な上限変更・`--no-surface`を使う。
上限値はRAM空き量の保証ではない。

まずは単一スレッドで、各ソリッドの境界箱内にあるセルだけを判定する。
大規模CADでの時間・メモリや並列化は今後の測定対象。

## 検証

```powershell
conda run --prefix .\build_voxel_deps --no-capture-output `
  ctest --test-dir build_voxelizer -C Release --output-on-failure
```

`ctest`がPATHにない場合は前述の絶対パスを使う。
今回のWindows Releaseで2つのCTestスイートを実行し、両方合格：

1. `voxelizer_geometry`：直方体、異方性間隔、mm/inch/mのSTEP、
   空洞、円柱の格子細分化、回転配置、境界上の競合、
   開いた面/混在形状/重複の拒否、割当上限、数値オーバーフロー、
   7ソリッドのサンプル。
2. `voxelizer_cli`：CLIを別プロセスで実行し、独立のバイナリ解析で
   VTIの全セル値/順序、VTPの接続/面の向き、UInt32最大ID、
   日本語パス、不正入力、上書き防止を検証。

任意の追加検証（Pythonの`vtk`がある環境）：

```powershell
conda run --prefix .\build_voxel_deps --no-capture-output `
  python tools/voxelizer/tests/vtk_reader_smoke.py `
  build_voxelizer/rotor_demo build_voxelizer/rotor_demo/preview.png
```

今回の変更は独立した前処理ツールと文書のみ。既存2Dソルバーの回帰試験は今回再実行していない。

ビューアーの追加回帰（VTK/NumPy/OpenGLが使えるPythonをCMakeに指定）：

```powershell
cmake -S tools/voxelizer -B build_voxelizer -DOMFDM_TEST_VOXEL_VIEWER=ON
conda run --prefix .\build_voxel_deps --no-capture-output `
  ctest --test-dir build_voxelizer -C Release -R voxelizer_viewer --output-on-failure
```

Windowsで合格。26空気セルが中心の材料セルを囲むモデルで、空気領域の外面と内面を検証。
スライダーイベントによる0/中間値/1への変更、左右キーでの材料切替、
UInt32最大ID、空気なしのモデル、PNG出力、入力ファイルのハッシュ不変も確認する。
