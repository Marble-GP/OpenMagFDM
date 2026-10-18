# OpenCodeによる解析受入試験

## 準備（利用者）

1. 修正後のNodeサーバーを再起動する。実行中の解析がないことを先に確認する。
2. 対象ユーザーを選び、File Manager → Import backup から
   `openmagfdm-mcp-acceptance.zip` を取り込む。
   内容は設定2件・画像1件・材料ライブラリ1件。同名ファイルがあれば上書きせず、専用の試験ユーザーを使う。
3. MCPの割当ユーザーがそのユーザーであることを確認する。
   試験に必要な権限はreadとrun。既存の全権限接続でもよいが、以下の試験では変更・削除を行わない。
4. OpenCodeを再接続し、以下の指示を渡す。

この素材は32×32の小さな静磁界モデルで、性能ベンチマークや物理精度の参照解ではない。
通常ケースと意図的な未収束ケースで、解析ジョブ・CSV・警告の取り扱いを確認する。
材料ライブラリは検査専用で、解析設定は材料値を直接定義している。
warning設定の許容値0・反復上限2は診断専用であり、実用設定へ流用しない。

## OpenCodeへの指示

```text
登録済みopenmagfdmのMCPツールだけを使い、次の受入試験を実施してください。
シェル・直接ファイルアクセス・REST・ブラウザーで代用しないでください。
トークンや認証情報は表示しないでください。
今回は下記2件の解析開始のみ許可します。既存ファイルの変更・削除、他の解析の開始・停止は禁止します。

1. workspace_statusで割当と権限を確認する。
2. files_listで設定・画像・材料を確認する。不足していれば迂回せず報告して止める。
3. files_read(text:true)でlibraries/mcp_acceptance_materials.yamlを取得してyaml_inspectする。
   summary.kind=materials、materialCount=2で、座標系・ソルバーの既定値を表示しないことを確認する。
4. image_infoでimages/mcp_acceptance_32.pngの32×32、白・赤・灰の材料色を確認する。
5. configs/mcp_acceptance_linear.yamlを読みyaml_inspectする。
   検査合格は物理的正しさの保証ではないと区別する。
6. files_statで設定と画像の現在リビジョンを取得し、analysis_startで通常解析を開始する。
   この試験実行用の一意なidempotencyKeyを生成する。
   同じ引数・同じキーで一度だけ再送し、同じjobIdが返ることを確認する。
7. analysis_getで完了を確認する。1件につき60秒を超えたら追加起動せず状態とログを報告して止める。
   自動停止はせず、利用者にGUIでの確認を依頼する。
8. result_summary、conditions.json、結果フォルダーのfiles_list(prefix)を確認する。
   見つかったAz CSVをresults_tableで2行×3列だけ取得する。
   通常ケースはcompleted=true、converged=trueを期待する。unknownを合格にしない。
9. configs/mcp_acceptance_warning.yamlについて、別の一意なキーで同様に解析する。
   completed=true、warning=true、converged=falseを期待する。
   未収束をクラッシュや収束成功と混同せず、警告と診断用CSVが取得できるか確認する。
10. analysis_listと結果IDの対応を確認する。

呼び出したツール、jobId/resultId、所要時間、収束状態、警告、少量のCSV取得結果、
成功／失敗／未確認を報告してください。予想と実測を区別し、失敗時に設定を勝手に修正しないでください。
```

この試験だけでは、解析中のMCP OFF／切断後の継続、GUIからの停止、長時間のメモリ保持、
Linux/macOS配布物の動作は確認できない。これらは別のリリース確認として残す。

## 開発側の確認（2026-10-19）

- 軽量回帰86/86成功（材料ライブラリの要約回帰を追加）。
- ZIPの4ファイルを隔離したワークスペースへpreview/importする試験に成功。
- 再ビルドしたWindows配布exeでHTTP・stdio双方から同じ素材の実解析を実行し、
  正常収束・未収束警告・CSV取得・材料要約・権限・分離・冪等起動の試験に成功。
- パッケージャーの動的require等の警告は残る。上記経路の実行成功と区別する。
- 試験ハーネスの失敗時終了処理を改善し、途中終了を終了コード0で誤判定しないようにした。
- OpenCode自身による今回の解析試験は利用者への引き渡し待ち。正式公開はまだ行っていない。

素材の再生成：`cd webui` の後 `node tools/create-mcp-benchmark.cjs <新しいZIP出力先>`。
既存ZIPを誤って上書きしないよう、出力先は未作成のファイル名を指定する。
