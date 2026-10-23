# REST APIマニュアル — v1.7.2

対象はWebUIと同じサーバーのHTTP API。既定ベースURLは`http://localhost:3000`、
共有モードでは設定したHTTPSアドレスとなる。CLI操作は[CLI.md](CLI.md)、
アカウント・保存先・更新は[OPERATIONS.md](OPERATIONS.md)を参照。

## 1. 共通規則

- JSON更新は`Content-Type: application/json`。アップロードは`multipart/form-data`。
- `userId`は作業領域ID。省略時に`default`を使うAPIもあるが、クライアントは明示する。
- 共有モードの所有権はログインセッションから判定する。送信した`userId`は権限を与えない。
- `configs/<name>`、`images/<name>`、`libraries/<name>`、`results/<folder>`は領域内の論理ファイルID。
- 結果取得の`result`は一覧APIが返す`outputs/<workspace>/<folder>`。OS絶対パスを渡さない。
- `file`は結果内の相対パス（例：`Az/step_0001`）。URLのパラメータはエンコードする。
- ファイルの`revision`はSHA-256の64桁16進文字列。更新・削除前に取得する。
- レスポンスはJSON、YAMLテキスト、画像、TIFF、ZIPなどAPIごとに異なる。
- 多くのJSON APIは`success`を返す。一部の旧取得APIはエラーでもHTTP 200と
  `success:false`を返すため、ステータスと本文・Content-Typeの両方を確認する。

MCPのBearerトークンはMCP専用。一般REST APIでは共有Webセッションを使用する。
ローカルモードはループバック・Host・Originを検証する。
外部クライアントも接続先と一致するHostを使い、Originを送る場合は接続先と一致させる。

## 2. 認証とCSRF

### ローカルモード

WebUIログインは不要。`GET /api/auth/me`は`{"success":true,"mode":"local"}`を返す。
MCP管理ペアリングは別のセッションである（後述）。

### 共有HTTPSモード

1. `POST /api/auth/login`に`{name,password,remember?}`を送る。
2. 返されたHttpOnly/Secure/SameSite Cookie（`omfdm_session`）を保存する。
3. `GET /api/auth/me`で`name`、`role`、`workspaces`、`csrf`を取得する。
4. 更新リクエストにはCookie、接続先と一致する`Origin`、`X-CSRF-Token: <csrf>`を送る。

ログインにも一致する`Origin`が必要。ログイン成功の本文は`{success:true}`で、
CSRFは`/api/auth/me`から取得する。ログイン試行制限に達した場合は429。
TLS証明書を信頼したクライアントで接続する。

| Method / path | リクエスト | 応答・効果 |
|---|---|---|
| `POST /api/auth/login` | JSON `name`, `password`, 任意の`remember:boolean` | セッションCookieを発行。共有モードのみ |
| `GET /api/auth/me` | Cookie | 利用者、役割、割当領域、CSRF。ローカル時は`mode`のみ |
| `POST /api/auth/logout` | Cookie、Origin、CSRF | 当該セッションを失効 |
| `POST /api/auth/password` | JSON `currentPassword`, `password`、Cookie、Origin、CSRF | 12文字以上の新パスワードへ変更。既存セッションを失効 |

通常は無操作8時間・最長24時間、`remember:true`は最長30日。
以下のcurl例はHTTPSの準備済みサーバーと専用アカウントを想定する。
Windows PowerShellではOS標準のcurlを`curl.exe`で呼ぶ。

```bash
# login.json: {"name":"alice","password":"<your password>","remember":false}
curl --fail-with-body -c cookies.txt \
  -H 'Origin: https://server.example:3000' -H 'Content-Type: application/json' \
  --data-binary @login.json https://server.example:3000/api/auth/login
curl --fail-with-body -b cookies.txt https://server.example:3000/api/auth/me
# 更新では下記に加え、OriginとX-CSRF-Tokenを付ける。
```

`login.json`とCookieはローカルの認証情報として扱い、公開・コミットしない。
一般ユーザーは割当領域、管理者は全領域を操作できる。
`/api/health`、`/api/solver/info`、MCP管理APIは共有時に管理者が必要。

## 3. 最小の解析ワークフロー

以下はローカルモードのcurl例。共有モードでは前節のCookie・Origin・CSRFを追加する。

```bash
# 1. 画像を新規保存する。同名の既存画像は置換しない。
curl --fail-with-body -F 'userId=demo' -F 'image=@model.png' \
  http://localhost:3000/api/upload-image

# 2. 設定を取得する。本文はYAML、X-File-Revisionに現在のリビジョン。
curl --fail-with-body -D config-headers.txt -o model.yaml \
  'http://localhost:3000/api/config?userId=demo&file=model.yaml'

# 3. 保存する。config-save.jsonは後述のJSON形式。
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @config-save.json http://localhost:3000/api/config

# 4. ジョブを受付する。冪等キーは新しい解析ごとに生成し、再試行では同じ値を使う。
curl --fail-with-body -H 'Content-Type: application/json' \
  --data '{"userId":"demo","configFile":"model.yaml","imageFile":"model.png","idempotencyKey":"demo-run-001"}' \
  http://localhost:3000/api/jobs

# 5. 応答のjobIdで進捗・ログを取得する。
curl --fail-with-body 'http://localhost:3000/api/jobs/JOB_ID?userId=demo'
```

新規保存の`config-save.json`：

```json
{
  "userId": "demo",
  "file": "model.yaml",
  "content": "coordinate_system: cartesian\nmesh: {dx: 0.001, dy: 0.001}\nmaterials:\n  air: {rgb: [255, 255, 255], mu_r: 1, jz: 0}\nboundary_conditions:\n  left: {type: dirichlet, value: 0}\n  right: {type: dirichlet, value: 0}\n  top: {type: dirichlet, value: 0}\n  bottom: {type: dirichlet, value: 0}\n",
  "expectedRevision": null
}
```

実際の画像に含まれる材料色を全て定義する。上記は白色だけの最小設定例。
既存ファイルの更新時は`expectedRevision`を取得済みの値に置き換える。

## 4. 設定・入力・材料ライブラリ

| Method / path | 主な入力 | 応答 |
|---|---|---|
| `GET /api/config/list` | query `userId` | `{success,files:[名前]}` |
| `GET /api/config` | query `userId`, `file`（既定`sample_config.yaml`） | YAMLテキスト、`X-File-Revision` |
| `POST /api/config` | JSON `userId,file,content,expectedRevision` | `{success,revision}` |
| `POST /api/config/inspect` | JSON `content`（1 MiBまで） | `{version:1,valid,diagnostics,summary,scope}` |
| `POST /api/validate-config` | JSON `config`（YAML文字列） | 旧互換の`{valid,errors}` |
| `POST /api/upload-image` | multipart `userId`, `image` | `{success,filename,path,originalName}` |
| `GET /api/images` | query `userId` | `{success,images:[名前]}` |
| `GET /uploads/:user/:filename` | URLの作業領域と画像名 | 所有権を確認した画像バイナリ |
| `GET /api/material-libraries` | query `userId` | `{success,libraries:[{filename,size,modified}]}` |
| `POST /api/material-libraries` | multipart `userId`, `library` | 新規YAMLの`filename,revision` |
| `GET /api/material-libraries/:filename` | query `userId` | YAMLテキスト、`X-File-Revision` |
| `PUT /api/material-libraries/:filename` | JSON `userId,content,expectedRevision` | `{success,filename,revision}` |
| `GET /api/user-files` | query `userId` | 全分類の`{success,files}`。分類、名前、サイズ、更新日時等 |
| `GET /api/workspace/file` | query `userId,id` | `{success,file}`。存在しなければ`file:null` |

設定取得時、指定ファイルがない場合はサーバー既定YAMLを返し、リビジョンは空となる。
実在確認が必要なら`/api/workspace/file`を使用する。
新規作成は`expectedRevision:null`、既存置換は現在のSHA-256リビジョンを指定する。
欠落値を新規扱いにする旧経路でも、既存ファイルへの無条件上書きにはならない。
設定・材料YAML保存は100 KiBまで。画像はPNG/JPEG/BMP、10 MiBまでで、デコード・画素数の制限も適用する。
ファイル数制限はCLIの`limits`と共通。

`inspect`の診断位置は0始まりの`row,column`。`valid`はエディター検査でエラーがない意味で、
物理的妥当性・外部ファイルの解決・式の実評価・収束を保証しない。

## 5. 永続ジョブ

| Method / path | 主な入力 | 応答 |
|---|---|---|
| `POST /api/jobs` | JSON `userId,configFile,imageFile,idempotencyKey`、任意`materialLibraryFile` | 初回202、再送200。`jobId,status,resultId`等 |
| `GET /api/jobs` | query `userId` | `{success,jobs}`。新しい順 |
| `GET /api/jobs/:jobId` | query `userId` | 状態、進捗、完了・収束・警告、`logTail`等 |
| `DELETE /api/jobs/:jobId` | query `userId` | 202。取消要求後の状態 |
| `POST /api/jobs/resume` | JSON `userId` | 保留解除後の`{success,jobs}` |

冪等キーは空でない128文字以下。同じ領域・キー・入力名を再送すると同じジョブを返す。
同じキーを異なる入力名に使うと409。入力ファイル名は領域内の保存名を指定する。
YAML内の参照ファイルも`configs/...`等の領域内論理IDを使用する。
受付時の画像・実効設定・材料・参照ファイルをスナップショットし、その入力で実行する。

状態は`preparing`、`queued`、`held`、`running`、`cancelling`、
`completed`、`failed`、`cancelled`、`interrupted`。
`completed`と`converged`は区別する。未収束警告を伴う完了もあるため、
`warning`、`exitCode`、`message`、ログを合わせて確認する。
結果の論理IDは`resultId`、結果取得API用の識別子は`resultPath`（未準備時はnullの場合あり）。

接続を切っても解析は継続する。画面復帰時は一覧を再取得し、必要なジョブをポーリングする。
実行中取消は即時の終了成功を意味せず、プロセス終了後の状態まで追跡する。
サーバー再起動後、実行途中は`interrupted`、待機中は`held`。自動の途中再開は行わない。
新しい解析としてやり直す場合は新しいキーを使う。

## 6. 結果・分布データ・画像

| Method / path | 入力 | 応答・注意 |
|---|---|---|
| `GET /api/results` | `userId` | `{success,results}`。`name,path,steps`等 |
| `GET /api/user-outputs` | `userId` | `{success,outputs}`。結果サイズ等の詳細一覧 |
| `GET /api/detect-steps` | `result` | `{success,steps,hasTiff,hasCsv}`。CSV/TIFF同一ステップを重複計数しない |
| `GET /api/load-field` | `result,file,prefer?` | TIFFバイナリ、またはCSVをデコードしたJSON |
| `GET /api/load-csv-raw` | `result,file` | 生のCSVテキスト。時系列のヘッダーを保持 |
| `GET /api/load-conditions` | `result` | 条件JSON |
| `GET /api/get-conditions` | `result` | 条件JSONの別名経路 |
| `GET /api/get-log` | `result` | `log.txt`テキスト |
| `GET /api/get-flux-linkage` | `result` | `{success,steps,series,step_column}` |
| `GET /api/get-original-input` | `result` | 受付時の保存入力画像。旧結果等でなければ404 |
| `GET /api/get-step-input-image` | `result,step`（既定0） | 当該ステップの`InputImg`。未出力時は404 |
| `GET /api/get-boundary-image` | `result,step`（既定0） | `BoundaryImg`のPNG |
| `GET /api/get-coarsening-mask` | `result,step`（省略または0は1） | 旧粗大化マスク。未出力時は404 |
| `GET /api/results/:resultFolder/field-at-point` | `userId,x,y,step?` | 点の磁場値。直交はx/y[m]、極座標はx=r[m]、y=θ[rad]。既定step=0 |

表の入力はquery。`resultFolder`には結果フォルダー名だけを指定する。
旧結果や任意出力でデータが存在しない場合を扱う。

```bash
curl --get --data-urlencode 'result=outputs/demo/output_example' \
  --data-urlencode 'file=Az/step_0001' \
  --output field.tiff http://localhost:3000/api/load-field
```

`load-field`は拡張子省略可で、既定はTIFF優先・CSVへフォールバック。
`prefer=csv`はCSV優先。TIFFは`Content-Type: image/tiff`と`X-Field-Format: tiff`、
CSVは`{success:true,data:[[...]],format:"csv",precision:"double"}`。
CSV JSONの行は上下反転され、欠損値はnull。TIFFは元バイナリのため同じ行変換は行わない。
拡張子だけで形式を決めず、Content-Typeでデコードを選ぶ。

元入力と移動・回転後のステップ画像は別物。磁力線表示にはAzがあればよく、
画像がない場合に元入力を現在の材料配置とみなすことはない。

## 7. description・名前変更・削除

| Method / path | JSON入力（GETはquery） | 応答 |
|---|---|---|
| `GET /api/user-outputs/:folderName/description` | `userId` | `{success,description,revision}`。未作成は空文字・null |
| `PUT /api/user-outputs/:folderName/description` | `userId,description,expectedRevision` | 保存後のファイルメタデータ。descriptionは32 KiBまで |
| `PUT /api/user-outputs/:folderName/rename` | `userId,newName,revision` | `{success,newName,revision}` |
| `POST /api/file-operations` | `userId,files:[{id,revision}]` | 200、`{success,operation}`。非同期削除を受付 |
| `GET /api/file-operations` | `userId` | 最近100件までの`operations` |
| `GET /api/file-operations/:id` | `userId` | `operation`の現在状態 |

削除は`/api/workspace/file`で各対象の現在のリビジョンを取得してから要求する。
1操作あたり1〜100対象。同じ作業領域の重複実行は409。

```json
{
  "userId": "demo",
  "files": [{"id": "results/output_example", "revision": "<current 64-character SHA-256>"}]
}
```

`operation`は`id,status,total,current,results,created,finished?`等を持つ。
`status`は`running`、`complete`、`interrupted`。
成功・失敗件数は`results`各項目の`success`から集計する。
`complete`は処理終了であり、全対象の削除成功とは限らない。
再接続後も操作IDから確認できる。サーバー再起動で中断した対象は、現状とリビジョンを再取得してから再要求する。
通信タイムアウトだけで失敗と判断しない。実行中・待機中の結果は削除・名前変更から保護する。

旧互換の同期削除経路は、`DELETE /api/config/:filename`、`/api/images/:filename`、
`/api/material-libraries/:filename`、`/api/user-outputs/:folderName`。
bodyに`userId,revision`、またはqueryの`userId`と`If-Match`ヘッダーで対象を指定する。
`DELETE /api/user-files`はbodyに`userId,category,name,revision`、
`DELETE /api/user-outputs/bulk`は`userId,files:[{name,revision}]`。
大きな結果の一括削除には非同期の`/api/file-operations`を使用する。

## 8. ZIPバックアップ

| Method / path | 入力 | 応答 |
|---|---|---|
| `GET /api/backup-manifest` | query `userId` | `categories,totalSize` |
| `GET /api/export` | query `userId,include?` | ZIP。include既定は`configs,images,libraries` |
| `POST /api/export` | JSON `userId,entries:[論理ファイルID]` | 選択した1〜10,000ファイルのZIP |
| `POST /api/import` | multipart `userId,backup` | 検査結果、`transferId,files:[{id,size,expectedRevision}]`等 |
| `POST /api/import/commit` | JSON `userId,transferId,files:[{id,revision}]` | `success,written,skipped` |
| `DELETE /api/import/:transferId` | query `userId` | 検査用一時データを解放 |

`include`は分類名のカンマ区切り。結果も含める場合は`results`を明示する。
インポートの最初のPOSTは検査と一時保存のみで、既存ファイルを書き換えない。
検査で返された全対象をcommitする。各項目の`expectedRevision`を、commitでは
`revision`というキーに入れる（新規対象はnull）。対象の部分選択はこの経路では行わない。
競合時は409。途中で失敗した場合は`applied`に適用済みIDが返ることがある。
ZIPアップロード上限1 GiB、展開後サイズ等の追加制限あり。
旧JSONバックアップは16 MiBまで。リンク・暗号化・領域外パスを拒否する。

## 9. 画像前処理

保存済みの画像を使う経路はJSON `userId,filename`を共通入力にする。

| Method / path | 追加の主な入力 | 内容 |
|---|---|---|
| `POST /api/materials/detect` | multipart `image`。query `rareThreshold,blendTolerance,minColorDist,blendBaseMinRatio`は任意 | 色、AA候補、材料YAML生成 |
| `POST /api/preprocess-polar/detect` | 任意`hint,color_groups` | 中心、半径、周期性等を推定 |
| `POST /api/preprocess-polar/warp` | `center_x,center_y,r_start_px,r_end_px,theta_start,theta_end,nr,ntheta`、任意`r_orientation,output_filename,r_outer_physical,preview` | 極座標画像と`polar_domain`を生成 |
| `POST /api/preprocess-polar/cleanup-preview` | 共通入力 | 対象画像の一時warpプレビューを削除 |
| `POST /api/preprocess-filter/quick-stats` | 共通入力 | `unique_colors,image_width,image_height,looks_noisy` |
| `POST /api/preprocess-filter/auto-tune` | 任意`nTargets,maxEvals,subsampleSize,initial,rareThreshold` | 量子化のノイズ抑制設定を探索 |
| `POST /api/preprocess-filter/quantize` | 任意`rareThreshold,nTargets,minTargetDist,despeckleRadius,bilateralSigmaSpatial,bilateralSigmaColor,minIslandSize,preview,output_filename` | 均一色画像を生成 |

warpの角度はrad、画像上の中心・半径はpixel、`r_outer_physical`はm。
`nr`と`ntheta`は2以上の整数で積は16×1,024×1,024以下。
`r_orientation`は`horizontal`（既定）または`vertical`。
出力は`filename,path`等を返す。`preview:true`は元画像ごとの固定プレビュー名を使う。
画像出力も共通の作業領域・ファイル数制限で管理する。
これらはWebUIの前処理機能に対応する補助APIで、形状・材料の検出結果は解析前に確認する。

## 10. 診断・MCP管理・旧API

| Method / path | 入力・用途 |
|---|---|
| `GET /api/health` | ソルバーの存在とデータディレクトリの書込可否。200 `ok`、503 `degraded` |
| `GET /api/solver/info` | ソルバーバージョン、検出パス、機能、Node/OS情報 |
| `GET /api/get-transient-config` | サーバー既定YAMLの移動設定。結果固有の条件には`load-conditions`を使用 |
| `GET /api/automation/activity?userId=...` | ループバック用のジョブ・操作履歴 |
| `POST /api/automation/jobs/:id/cancel` | 旧ローカル管理用取消。新規クライアントは`DELETE /api/jobs/:jobId` |

MCP管理APIは一般REST認証に加え、ループバック検証・ブラウザーペアリングを行う。
`POST /api/mcp/pair`へ`{code,remember?}`を送ると`omfdm_admin` Cookieと`csrf`が返る。
変更には`X-OMFDM-CSRF`を使用する（共有Webセッションの`X-CSRF-Token`とは別）。

| Method / path | 入力・内容 |
|---|---|
| `GET /api/mcp/status` | 待受・接続状態と管理CSRF |
| `POST /api/mcp/forget-browser` | 管理ブラウザーの信頼を解除 |
| `GET /api/mcp/solver` | 保存済みソルバーパスと検出状態 |
| `PUT /api/mcp/solver` | `{path}`。絶対パスまたは自動検出へ戻す空文字。再起動が必要 |
| `PUT /api/mcp/settings` | `{enabled:boolean,port:1024..65535}` |
| `POST /api/mcp/connections` | `{name,user,permissions}`。トークンは作成時だけ表示 |
| `PATCH /api/mcp/connections/:id` | 接続の権限・失効等を更新。詳細はMCPマニュアル |
| `POST /api/mcp/connections/:id/test` | サーバー側の接続試験 |
| `GET /api/mcp/setup/:id` | HTTP URLとstdioのcommand/args。トークン本体は返さない |

共有HTTPSでのMCP管理UIには制約がある。[MCP接続手順](MCP_INTEGRATION.md)を参照。
MCPのJSON-RPCツール呼出し先は別待受の`/mcp`で、上記RESTとは別プロトコル。

旧`POST /api/solve`、`/api/solve-stream`、`/api/stop-solver`と`/data`は410。
`GET /api/load-csv`はCSV限定の非推奨互換経路。新規取得は`load-field`を使用する。
旧`GET /api/get-material-image`は共有モードでは410。新規クライアントは領域付き画像経路を使用する。

## 11. エラー時の扱い

| HTTP | 代表的な意味 | クライアントの処理 |
|---|---|---|
| 400 | 入力・パス・形式の不正 | 本文を確認して修正 |
| 401 | ログインなし／セッション失効 | 再ログインして領域・CSRFを再取得 |
| 403 | 領域権限、Origin、CSRF | 割当と送信元・Cookie・ヘッダーを確認 |
| 404 | 対象・出力がない | 任意出力・旧結果・削除を区別 |
| 409 | 更新競合、冪等キー不一致、制限、処理中 | メッセージに応じ再取得／既存操作の照会 |
| 410 | 廃止経路 | 共通ジョブ・領域付きAPIへ移行 |
| 413 | サイズ上限 | 送信サイズ・API別上限を確認 |
| 428 | 削除・更新の前提リビジョンなし | 先にメタデータを取得 |
| 429 | 試行・リソース制限 | 待機して再試行 |
| 500 / 503 | サーバーエラー／初期化・容量等 | 状態とログを確認。受付済み操作を照会して重複を防ぐ |

エラー本文は経路により`{success:false,error,code?}`、`{error,code}`、テキスト等となる。
通信断時、ジョブは同じ冪等キーで再照会・再送し、削除は操作IDで追跡する。
