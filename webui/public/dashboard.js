// ===== グローバル変数 =====
let grid = null;
let currentResultPath = null;
let currentStep = 1;
let totalSteps = 0;
let animationTimer = null;
let isAnimating = false;

// 解析条件情報（conditions.jsonから読み込み）
let analysisConditions = null;

// キャッシュ
let cachedData = {
    az: {},
    mu: {},
    forces: {},
    stressVectors: {}
};

// プロット定義
const plotDefinitions = {
    az_contour: { name: 'Az等高線', icon: '📊', render: renderAzContour },
    az_heatmap: { name: 'Azヒートマップ', icon: '🔥', render: renderAzHeatmap },
    b_magnitude: { name: '|B|分布', icon: '🧲', render: renderBMagnitude },
    h_magnitude: { name: '|H|分布', icon: '⚡', render: renderHMagnitude },
    mu_distribution: { name: '透磁率分布', icon: '🎨', render: renderMuDistribution },
    az_boundary: { name: 'Az+境界', icon: '📐', render: renderAzBoundary },
    material_image: { name: '材質画像', icon: '🖼️', render: renderMaterialImage },
    step_input_image: { name: 'ステップ入力画像', icon: '🎞️', render: renderStepInputImage },
    boundary_only: { name: '境界のみ', icon: '⬜', render: renderBoundaryOnly },
    b_vectors: { name: 'Bベクトル', icon: '➡️', render: renderBVectors },
    h_vectors: { name: 'Hベクトル', icon: '↗️', render: renderHVectors },
    stress_vectors: { name: '応力ベクトル', icon: '⚡', render: renderStressVectors },
    force_x_time: { name: '力X時系列', icon: '📈', render: renderForceXTime },
    force_y_time: { name: '力Y時系列', icon: '📈', render: renderForceYTime },
    torque_time: { name: 'トルク時系列', icon: '📈', render: renderTorqueTime },
    energy_time: { name: 'エネルギー時系列', icon: '⚡', render: renderEnergyTime }
};

// ===== 初期化 =====
window.addEventListener('DOMContentLoaded', async () => {
    // 古いレイアウトに無効なプロットタイプが含まれている場合はクリア
    try {
        const saved = localStorage.getItem('dashboard-layout');
        if (saved) {
            const layout = JSON.parse(saved);
            const hasInvalidPlot = layout.some(item => {
                const plotType = item.content?.match(/data-plot-type="([^"]+)"/)?.[1];
                return plotType && !plotDefinitions[plotType];
            });
            if (hasInvalidPlot) {
                console.log('Clearing old layout with invalid plot types');
                localStorage.removeItem('dashboard-layout');
            }
        }
    } catch (error) {
        console.error('Error checking saved layout:', error);
    }

    // GridStack初期化
    grid = GridStack.init({
        cellHeight: 150,
        minRow: 2,
        column: 12,
        acceptWidgets: true,
        removable: false,
        float: true
    });

    // リサイズイベントでPlotlyをリサイズ
    grid.on('resizestop', (_event, element) => {
        const container = element.querySelector('.plot-container');
        if (container) {
            // Plotlyのデータオブジェクトが存在するか確認
            if (container.data && container.layout) {
                // コンテナの新しいサイズを取得
                const rect = container.getBoundingClientRect();
                console.log(`Resizing plot to: ${rect.width}x${rect.height}`);

                // Plotlyのレイアウトを更新（アスペクト比を維持）
                Plotly.relayout(container, {
                    width: rect.width,
                    height: rect.height
                }).catch(err => {
                    console.error('Plotly relayout error:', err);
                });
            } else {
                console.warn('Container does not have Plotly data');
            }
        }
    });

    // 解析結果リストを読み込み
    await loadResultsList();

    // パレットアイテムにドラッグイベントを設定
    setupPaletteDragDrop();

    // ステップスライダーのイベント
    document.getElementById('stepSlider').addEventListener('input', onStepChange);
    document.getElementById('resultSelect').addEventListener('change', onResultSelect);

    console.log('Dashboard initialized');
});

// ===== 解析結果の読み込み =====
async function loadResultsList() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();

        const select = document.getElementById('resultSelect');
        select.innerHTML = '<option value="">解析結果を選択...</option>';

        if (data.success && data.results.length > 0) {
            data.results.forEach(result => {
                const option = document.createElement('option');
                option.value = result.path;
                const stepInfo = result.steps > 1 ? ` (${result.steps}ステップ)` : '';
                option.textContent = `${result.name}${stepInfo}`;
                option.dataset.steps = result.steps;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('結果リスト読み込みエラー:', error);
    }
}

// ===== 解析結果選択 =====
async function onResultSelect() {
    const select = document.getElementById('resultSelect');
    const path = select.value;

    if (!path) return;

    currentResultPath = path;
    const selectedOption = select.options[select.selectedIndex];
    totalSteps = parseInt(selectedOption.dataset.steps) || 1;

    // 解析条件情報を読み込み
    try {
        const conditionsResponse = await fetch(`/api/load-conditions?result=${encodeURIComponent(path)}`);
        if (conditionsResponse.ok) {
            analysisConditions = await conditionsResponse.json();
            console.log('Analysis conditions loaded:', analysisConditions);
        } else {
            console.warn('conditions.json not found, assuming default (cartesian)');
            analysisConditions = { coordinate_system: 'cartesian', dx: 0.001, dy: 0.001 };
        }
    } catch (error) {
        console.warn('Failed to load conditions.json:', error);
        analysisConditions = { coordinate_system: 'cartesian', dx: 0.001, dy: 0.001 };
    }

    // ステップコントロール表示
    if (totalSteps > 1) {
        document.getElementById('stepControls').classList.remove('hidden');
        document.getElementById('stepSlider').max = totalSteps;
        document.getElementById('totalSteps').textContent = totalSteps;
    } else {
        document.getElementById('stepControls').classList.add('hidden');
    }

    currentStep = 1;
    document.getElementById('currentStep').textContent = currentStep;
    document.getElementById('stepSlider').value = currentStep;

    // 既存のプロットを更新
    await updateAllPlots();

    console.log(`Result selected: ${path}, ${totalSteps} steps`);
}

// ===== ステップ変更 =====
async function onStepChange() {
    currentStep = parseInt(document.getElementById('stepSlider').value);
    document.getElementById('currentStep').textContent = currentStep;

    // 全プロットを更新
    await updateAllPlots();
}

// ===== 全プロット更新 =====
async function updateAllPlots() {
    if (!currentResultPath) {
        console.log('updateAllPlots: No result path');
        return;
    }

    // グリッド内のすべてのプロットコンテナを直接取得
    const contentElements = document.querySelectorAll('.grid-stack-item-content[data-plot-type]');
    console.log(`updateAllPlots: Found ${contentElements.length} plots, currentStep=${currentStep}`);

    for (const contentElement of contentElements) {
        const plotType = contentElement.dataset.plotType;
        const containerId = contentElement.dataset.containerId;
        const container = document.getElementById(containerId);

        // 無効なプロットタイプをスキップ
        if (!plotDefinitions[plotType]) {
            console.warn(`updateAllPlots: Skipping invalid plot type: ${plotType}`);
            continue;
        }

        console.log(`updateAllPlots: Updating ${plotType} in ${containerId}`);

        if (container) {
            try {
                await plotDefinitions[plotType].render(container, currentStep);
                console.log(`updateAllPlots: Successfully updated ${plotType}`);
            } catch (error) {
                console.error(`updateAllPlots: Error updating ${plotType}:`, error);
            }
        } else {
            console.log(`updateAllPlots: Container not found for ${plotType}`);
        }
    }
    console.log('updateAllPlots: Complete');
}

// ===== ドラッグ＆ドロップ設定 =====
function setupPaletteDragDrop() {
    const paletteItems = document.querySelectorAll('.palette-item');
    const canvas = document.querySelector('.grid-stack');

    paletteItems.forEach(item => {
        item.addEventListener('dragstart', (e) => {
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('text/plain', item.dataset.plotType);
            e.dataTransfer.setData('plot-type', item.dataset.plotType);
            item.classList.add('dragging');
        });

        item.addEventListener('dragend', () => {
            item.classList.remove('dragging');
        });
    });

    // キャンバスにドロップイベントを設定
    canvas.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    canvas.addEventListener('drop', (e) => {
        e.preventDefault();
        const plotType = e.dataTransfer.getData('plot-type') || e.dataTransfer.getData('text/plain');

        console.log(`Drop event: plotType=${plotType}`);

        if (plotType && plotDefinitions[plotType]) {
            // ドロップ位置を計算（グリッド座標に変換）
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / (rect.width / 12));
            const y = Math.floor((e.clientY - rect.top) / 150);

            console.log(`Calculated position: x=${x}, y=${y}`);
            addPlot(plotType, x, y, 4, 3);
        } else {
            console.error(`Invalid plot type: ${plotType}`);
        }
    });
}

// ===== プロット追加 =====
let plotIdCounter = 0;

async function addPlot(plotType, x = 0, y = 0, w = 4, h = 3) {
    if (!currentResultPath) {
        alert('まず解析結果を選択してください');
        return;
    }

    const plotDef = plotDefinitions[plotType];
    if (!plotDef) return;

    // 空のキャンバス表示を非表示
    document.getElementById('emptyCanvas').classList.add('hidden');

    // ユニークID
    const plotId = `plot-${plotIdCounter++}`;
    const containerId = `container-${plotId}`;

    // プロットウィジェット作成
    const content = `
        <div class="grid-stack-item-content" data-plot-type="${plotType}" data-container-id="${containerId}">
            <div class="plot-header">
                <span>${plotDef.icon} ${plotDef.name}</span>
                <div class="plot-controls">
                    <button class="interaction-mode-btn" data-plot-id="${plotId}" data-mode="disabled" title="操作モード: タイル移動">📊</button>
                    <button class="reset-zoom-btn" data-plot-id="${plotId}" title="モードを変更">⟲</button>
                    <button class="remove-plot-btn" data-plot-id="${plotId}" title="削除">✕</button>
                </div>
            </div>
            <div class="plot-container" id="${containerId}">
                <div style="text-align: center; padding: 20px;">読み込み中...</div>
            </div>
        </div>
    `;

    // GridStackに追加
    const widgetEl = grid.addWidget({
        x: x,
        y: y,
        w: w,
        h: h,
        content: content,
        id: plotId
    });

    console.log(`Widget added with ID: ${plotId}, element:`, widgetEl);
    console.log(`Widget ID attribute:`, widgetEl.id);

    // 操作モードボタンのイベントリスナーを追加
    const interactionModeBtn = widgetEl.querySelector('.interaction-mode-btn');
    if (interactionModeBtn) {
        interactionModeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleInteractionMode(containerId, interactionModeBtn, widgetEl);
        });
    }

    // 初期状態でタイルの移動を無効化（グラフ操作優先）
    // grid.movable(widgetEl, false);

    // リセットズームボタンのイベントリスナーを追加
    const resetZoomBtn = widgetEl.querySelector('.reset-zoom-btn');
    if (resetZoomBtn) {
        resetZoomBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            console.log(`Reset zoom clicked for ${plotId}`);
            resetPlotZoom(containerId);
        });
    }

    // ×ボタンのイベントリスナーを追加
    const removeBtn = widgetEl.querySelector('.remove-plot-btn');
    if (removeBtn) {
        console.log(`Registering remove button for ${plotId}`);
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // イベントの伝播を防ぐ
            console.log(`Remove button clicked for ${plotId}`);
            removePlot(plotId);
        });
    } else {
        console.error(`Remove button not found in widget for ${plotId}`);
    }

    // プロット描画（非同期で実行）
    const container = widgetEl.querySelector(`#${containerId}`);
    if (!container) {
        console.error(`Container not found: ${containerId}`);
        return;
    }

    console.log(`Rendering plot: ${plotType} in container: ${containerId}`);

    // GridStackのレイアウトが完了するのを待つイベントハンドラ
    const renderPlot = async () => {
        const rect = container.getBoundingClientRect();
        console.log(`Container size at render: ${rect.width}x${rect.height}`);

        if (rect.width < 50 || rect.height < 50) {
            console.warn(`Container size too small: ${rect.width}x${rect.height}, retrying...`);
            // サイズが小さすぎる場合はリトライ
            setTimeout(renderPlot, 200);
            return;
        }

        try {
            await plotDef.render(container, currentStep);
            console.log(`Successfully rendered: ${plotType} at ${rect.width}x${rect.height}`);
        } catch (error) {
            console.error(`Plot render error for ${plotType}:`, error);
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
        }
    };

    // GridStackの'added'イベントを待つ（レイアウト完了後）
    setTimeout(() => {
        // さらにDOM更新を待つ
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                setTimeout(renderPlot, 150);
            });
        });
    }, 100);
}

// ===== 操作モード切り替え =====
function toggleInteractionMode(containerId, button, widgetEl) {
    const container = document.getElementById(containerId);
    if (!container || !container.data || !container.layout) {
        console.warn('No Plotly plot found');
        return;
    }

    const currentMode = button.dataset.mode;
    let newMode, newIcon, newTitle, dragmode, tileMovable;

    if (currentMode === 'zoom') {
        // ズーム → パン
        newMode = 'pan';
        newIcon = '✋';
        newTitle = '操作モード: パン';
        dragmode = 'pan';
        tileMovable = false; // タイル移動不可
    } else if (currentMode === 'pan') {
        // パン → ドラッグ無効
        newMode = 'disabled';
        newIcon = '📊';
        newTitle = '操作モード: タイル移動';
        dragmode = false;
        tileMovable = true; // タイル移動可能
    } else {
        // 無効 → ズーム
        newMode = 'zoom';
        newIcon = '🔍';
        newTitle = '操作モード: ズーム';
        dragmode = 'zoom';
        tileMovable = false; // タイル移動不可
    }

    button.dataset.mode = newMode;
    button.textContent = newIcon;
    button.title = newTitle;

    // Plotlyのドラッグモードを更新
    Plotly.relayout(container, {
        dragmode: dragmode
    }).catch(err => {
        console.error('Failed to update drag mode:', err);
    });

    // GridStackのタイル移動可否を更新
    if (widgetEl) {
        grid.movable(widgetEl, tileMovable);
        console.log(`Tile movable: ${tileMovable}`);
    }

    console.log(`Interaction mode changed to: ${newMode}`);
}

// ===== プロットズームリセット =====
function resetPlotZoom(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container not found: ${containerId}`);
        return;
    }

    // Plotlyのデータオブジェクトが存在するか確認
    if (container.data && container.layout) {
        console.log(`Resetting zoom for: ${containerId}`);

        // Plotlyのautoscaleを使ってズームをリセット
        Plotly.relayout(container, {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        }).catch(err => {
            console.error('Plotly reset zoom error:', err);
        });
    } else {
        console.warn(`No Plotly plot found in: ${containerId}`);
    }
}

// ===== プロット削除 =====
function removePlot(plotId) {
    console.log(`Removing plot: ${plotId}`);

    // GridStackはgs-id属性を使うので、それで要素を探す
    const element = document.querySelector(`[gs-id="${plotId}"]`);

    if (!element) {
        console.error(`Element not found with gs-id: ${plotId}`);
        return;
    }

    console.log(`Found element, removing from grid`);
    grid.removeWidget(element);

    // すべてのプロットが削除されたら空のキャンバスを表示
    if (grid.getGridItems().length === 0) {
        const emptyCanvas = document.getElementById('emptyCanvas');
        if (emptyCanvas) {
            emptyCanvas.classList.remove('hidden');
        }
    }

    console.log(`Plot removed successfully`);
}

// ===== ダッシュボードクリア =====
function clearDashboard() {
    if (confirm('すべてのプロットを削除しますか？')) {
        grid.removeAll();
        document.getElementById('emptyCanvas').classList.remove('hidden');
    }
}

// ===== レイアウト保存 =====
function saveLayout() {
    const layout = grid.save(false);
    localStorage.setItem('dashboard-layout', JSON.stringify(layout));
    alert('レイアウトを保存しました');
}

// ===== レイアウト復元 =====
function loadLayout() {
    const saved = localStorage.getItem('dashboard-layout');
    if (!saved) {
        alert('保存されたレイアウトがありません');
        return;
    }

    try {
        const layout = JSON.parse(saved);

        // 無効なプロットタイプをフィルタリング
        const validLayout = layout.filter(item => {
            const plotType = item.content?.match(/data-plot-type="([^"]+)"/)?.[1];
            if (plotType && !plotDefinitions[plotType]) {
                console.warn(`Skipping invalid plot type: ${plotType}`);
                return false;
            }
            return true;
        });

        if (validLayout.length < layout.length) {
            console.log(`Filtered out ${layout.length - validLayout.length} invalid plot(s)`);
        }

        grid.load(validLayout);
        document.getElementById('emptyCanvas').classList.add('hidden');
        alert('レイアウトを復元しました');

        // プロットを再描画
        updateAllPlots();
    } catch (error) {
        console.error('Layout load error:', error);
        alert('レイアウトの復元に失敗しました');
    }
}

// ===== アニメーション =====
function playAnimation() {
    if (isAnimating || totalSteps <= 1) return;

    isAnimating = true;
    document.getElementById('playBtn').classList.add('hidden');
    document.getElementById('pauseBtn').classList.remove('hidden');

    const speed = parseInt(document.getElementById('animSpeed').value);

    animationTimer = setInterval(async () => {
        currentStep++;
        if (currentStep > totalSteps) {
            currentStep = 1; // ループ
        }

        document.getElementById('stepSlider').value = currentStep;
        document.getElementById('currentStep').textContent = currentStep;

        await updateAllPlots();
    }, speed);
}

function pauseAnimation() {
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }

    isAnimating = false;
    document.getElementById('playBtn').classList.remove('hidden');
    document.getElementById('pauseBtn').classList.add('hidden');
}

function resetAnimation() {
    pauseAnimation();
    currentStep = 1;
    document.getElementById('stepSlider').value = currentStep;
    document.getElementById('currentStep').textContent = currentStep;
    updateAllPlots();
}

// ===== データ読み込みヘルパー =====
async function loadStepData(step) {
    const cacheKey = `${currentResultPath}_${step}`;

    if (!cachedData.az[cacheKey]) {
        try {
            console.log(`Loading step data: ${step}`);
            const azResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(currentResultPath)}&file=Az/step_${String(step).padStart(4, '0')}.csv`);
            const azData = await azResponse.json();

            const muResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(currentResultPath)}&file=Mu/step_${String(step).padStart(4, '0')}.csv`);
            const muData = await muResponse.json();

            if (azData.success && muData.success) {
                cachedData.az[cacheKey] = azData.data;
                cachedData.mu[cacheKey] = muData.data;
                console.log(`Step ${step} data loaded: Az size ${azData.data.length}x${azData.data[0]?.length}`);
            } else {
                console.error(`Failed to load step ${step}: Az success=${azData.success}, Mu success=${muData.success}`);
            }
        } catch (error) {
            console.error(`Data load error for step ${step}:`, error);
        }
    }

    return {
        az: cachedData.az[cacheKey],
        mu: cachedData.mu[cacheKey]
    };
}

async function loadForceData(step) {
    const cacheKey = `${currentResultPath}_${step}`;

    if (!cachedData.forces[cacheKey]) {
        try {
            // 生のテキストCSVを取得するために /api/load-csv-raw を使用
            const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(currentResultPath)}&file=Forces/step_${String(step).padStart(4, '0')}.csv`);

            if (!response.ok) {
                console.warn(`Forces data not found for step ${step}`);
                cachedData.forces[cacheKey] = null;
                return null;
            }

            const textData = await response.text();

            // 空のレスポンスチェック
            if (!textData || textData.trim().length === 0) {
                console.warn(`Empty forces data for step ${step}`);
                cachedData.forces[cacheKey] = null;
                return null;
            }

            // Forces CSVをパース
            // フォーマット: Material,RGB_R,RGB_G,RGB_B,Force_X[N/m],Force_Y[N/m],Force_Magnitude[N/m],Torque[N],Boundary_Pixels
            const lines = textData.trim().split('\n');

            // ヘッダー行を探す
            let headerIdx = -1;
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].startsWith('Material,')) {
                    headerIdx = i;
                    break;
                }
            }

            if (headerIdx === -1) {
                console.error(`No header line found in forces file for step ${step}`);
                console.error(`Total lines: ${lines.length}`);
                if (lines.length > 0) {
                    console.error(`First line starts with: "${lines[0].substring(0, 20)}..."`);
                }
                cachedData.forces[cacheKey] = null;
                return null;
            }

            const headers = lines[headerIdx].split(',');

            // 各カラムのインデックスを取得
            const materialIdx = headers.findIndex(h => h && h.trim() === 'Material');
            const rgbRIdx = headers.findIndex(h => h && h.includes('RGB_R'));
            const rgbGIdx = headers.findIndex(h => h && h.includes('RGB_G'));
            const rgbBIdx = headers.findIndex(h => h && h.includes('RGB_B'));
            const forceXIdx = headers.findIndex(h => h && h.includes('Force_X'));
            const forceYIdx = headers.findIndex(h => h && h.trim().startsWith('Force_Y'));
            const torqueOriginIdx = headers.findIndex(h => h && h.includes('Torque_Origin'));
            const torqueCenterIdx = headers.findIndex(h => h && h.includes('Torque_Center'));
            const energyIdx = headers.findIndex(h => h && h.includes('Magnetic_Energy'));

            // フォールバック: 古い形式（Torqueのみ）の場合
            const torqueIdx = torqueOriginIdx !== -1 ? torqueOriginIdx :
                             headers.findIndex(h => h && h.includes('Torque'));

            if (forceXIdx === -1 || forceYIdx === -1 || torqueIdx === -1) {
                console.error(`Missing force columns in step ${step}`);
                console.error(`Header line was: "${lines[headerIdx]}"`);
                console.error(`Headers array:`, headers);
                console.error(`Found indices: forceX=${forceXIdx}, forceY=${forceYIdx}, torque=${torqueIdx}`);
                cachedData.forces[cacheKey] = null;
                return null;
            }

            // 材料ごとのデータと全体の合計
            const materials = [];
            let totalForceX = 0;
            let totalForceY = 0;
            let totalTorque = 0;
            let dataRowCount = 0;

            for (let i = headerIdx + 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line.startsWith('#') || line.length === 0) continue;

                const values = line.split(',');
                if (values.length > Math.max(forceXIdx, forceYIdx, torqueIdx)) {
                    const materialName = materialIdx !== -1 ? values[materialIdx].trim() : `Material_${dataRowCount}`;
                    const forceX = parseFloat(values[forceXIdx]) || 0;
                    const forceY = parseFloat(values[forceYIdx]) || 0;
                    const torque = parseFloat(values[torqueIdx]) || 0;
                    const energy = energyIdx !== -1 ? (parseFloat(values[energyIdx]) || 0) : 0;

                    // RGB値を取得（カラーコード作成用）
                    const r = rgbRIdx !== -1 ? parseInt(values[rgbRIdx]) || 0 : 0;
                    const g = rgbGIdx !== -1 ? parseInt(values[rgbGIdx]) || 0 : 0;
                    const b = rgbBIdx !== -1 ? parseInt(values[rgbBIdx]) || 0 : 0;
                    const color = `rgb(${r}, ${g}, ${b})`;

                    materials.push({
                        name: materialName,
                        color: color,
                        force_x: forceX,
                        force_y: forceY,
                        torque: torque,
                        energy: energy
                    });

                    totalForceX += forceX;
                    totalForceY += forceY;
                    totalTorque += torque;
                    dataRowCount++;
                }
            }

            // データ行が実際にあるかチェック
            if (dataRowCount === 0) {
                console.log(`No valid data rows found in forces file for step ${step} (only header/comments)`);
            }

            cachedData.forces[cacheKey] = {
                total: {
                    force_x: totalForceX,
                    force_y: totalForceY,
                    torque: totalTorque
                },
                materials: materials
            };
        } catch (error) {
            console.error(`Force data load error for step ${step}:`, error);
            cachedData.forces[cacheKey] = null;
        }
    }

    return cachedData.forces[cacheKey];
}

async function loadStressVectorData(step) {
    const cacheKey = `${currentResultPath}_${step}`;

    if (!cachedData.stressVectors[cacheKey]) {
        try {
            // 生のテキストCSVを取得
            const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(currentResultPath)}&file=StressVectors/step_${String(step).padStart(4, '0')}.csv`);

            if (!response.ok) {
                console.warn(`Stress vectors data not found for step ${step}`);
                cachedData.stressVectors[cacheKey] = null;
                return null;
            }

            const textData = await response.text();

            if (!textData || textData.trim().length === 0) {
                console.warn(`Empty stress vectors data for step ${step}`);
                cachedData.stressVectors[cacheKey] = null;
                return null;
            }

            // CSVをパース
            // フォーマット: i_pixel,j_pixel,x[m],y[m],fx[N/m],fy[N/m],ds[m],nx,ny,Bx[T],By[T],B_mag[T],Material
            const lines = textData.trim().split('\n');

            // ヘッダー行を探す
            let headerIdx = -1;
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].startsWith('i_pixel,')) {
                    headerIdx = i;
                    break;
                }
            }

            if (headerIdx === -1) {
                console.error(`No header line found in stress vectors file for step ${step}`);
                cachedData.stressVectors[cacheKey] = null;
                return null;
            }

            const headers = lines[headerIdx].split(',');

            // カラムインデックスを取得
            const iPixelIdx = headers.findIndex(h => h && h.trim() === 'i_pixel');
            const jPixelIdx = headers.findIndex(h => h && h.trim() === 'j_pixel');
            const xIdx = headers.findIndex(h => h && h.includes('x[m]'));
            const yIdx = headers.findIndex(h => h && h.includes('y[m]'));
            const fxIdx = headers.findIndex(h => h && h.includes('fx[N/m]'));
            const fyIdx = headers.findIndex(h => h && h.includes('fy[N/m]'));
            const dsIdx = headers.findIndex(h => h && h.includes('ds[m]'));
            const nxIdx = headers.findIndex(h => h && h.trim() === 'nx');
            const nyIdx = headers.findIndex(h => h && h.trim() === 'ny');
            const bxIdx = headers.findIndex(h => h && h.includes('Bx[T]'));
            const byIdx = headers.findIndex(h => h && h.includes('By[T]'));
            const bMagIdx = headers.findIndex(h => h && h.includes('B_mag[T]'));
            const materialIdx = headers.findIndex(h => h && h.trim() === 'Material');

            if (iPixelIdx === -1 || jPixelIdx === -1 || xIdx === -1 || yIdx === -1 ||
                fxIdx === -1 || fyIdx === -1) {
                console.error(`Missing required columns in stress vectors file for step ${step}`);
                cachedData.stressVectors[cacheKey] = null;
                return null;
            }

            // データポイントを読み込み（境界ピクセルのみ、fx=0かつfy=0以外）
            const stressPoints = [];

            for (let i = headerIdx + 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line.startsWith('#') || line.length === 0) continue;

                const values = line.split(',');
                if (values.length > Math.max(fxIdx, fyIdx, xIdx, yIdx)) {
                    const fx = parseFloat(values[fxIdx]) || 0;
                    const fy = parseFloat(values[fyIdx]) || 0;

                    // ゼロベクトルはスキップ（非境界ピクセル）
                    if (Math.abs(fx) < 1e-15 && Math.abs(fy) < 1e-15) continue;

                    stressPoints.push({
                        i_pixel: parseInt(values[iPixelIdx]) || 0,
                        j_pixel: parseInt(values[jPixelIdx]) || 0,
                        x: parseFloat(values[xIdx]) || 0,
                        y: parseFloat(values[yIdx]) || 0,
                        fx: fx,
                        fy: fy,
                        ds: dsIdx !== -1 ? (parseFloat(values[dsIdx]) || 0) : 0,
                        nx: nxIdx !== -1 ? (parseFloat(values[nxIdx]) || 0) : 0,
                        ny: nyIdx !== -1 ? (parseFloat(values[nyIdx]) || 0) : 0,
                        bx: bxIdx !== -1 ? (parseFloat(values[bxIdx]) || 0) : 0,
                        by: byIdx !== -1 ? (parseFloat(values[byIdx]) || 0) : 0,
                        b_mag: bMagIdx !== -1 ? (parseFloat(values[bMagIdx]) || 0) : 0,
                        material: materialIdx !== -1 ? values[materialIdx].trim() : ''
                    });
                }
            }

            console.log(`Loaded ${stressPoints.length} stress vectors for step ${step}`);
            cachedData.stressVectors[cacheKey] = stressPoints;
        } catch (error) {
            console.error(`Stress vector data load error for step ${step}:`, error);
            cachedData.stressVectors[cacheKey] = null;
        }
    }

    return cachedData.stressVectors[cacheKey];
}

// ===== 磁場計算ヘルパー =====
function calculateMagneticField(Az, Mu, dx = 0.001, dy = 0.001) {
    const rows = Az.length;
    const cols = Az[0].length;

    const Bx = Array(rows).fill(0).map(() => Array(cols).fill(0));
    const By = Array(rows).fill(0).map(() => Array(cols).fill(0));

    // 座標系を判定（analysisConditionsがロードされている場合）
    const coordSystem = analysisConditions ? analysisConditions.coordinate_system : 'cartesian';

    if (coordSystem === 'polar') {
        // 極座標系での磁場計算
        const polar = analysisConditions.polar;
        const r_start = polar.r_start;
        const r_end = polar.r_end;
        const nr = cols;
        const ntheta = rows;

        // dr, dthetaを計算（conditions.jsonに含まれていれば使用、なければ計算）
        const dr = analysisConditions.dr || (r_end - r_start) / (nr - 1);
        const dtheta = analysisConditions.dtheta || polar.theta_range / (ntheta - 1);

        // r座標配列を生成
        const r_coords = Array(nr).fill(0).map((_, ir) => r_start + ir * dr);

        // 極座標での磁場を計算: Br, Bθ
        const Br = Array(rows).fill(0).map(() => Array(cols).fill(0));
        const Btheta = Array(rows).fill(0).map(() => Array(cols).fill(0));

        for (let jt = 0; jt < ntheta; jt++) {
            for (let ir = 0; ir < nr; ir++) {
                const r = r_coords[ir];
                const safe_r = Math.max(r, 1e-15);

                // Br = (1/r) * ∂Az/∂θ
                let dAz_dtheta = 0;
                if (jt === 0) {
                    dAz_dtheta = (Az[1][ir] - Az[0][ir]) / dtheta;
                } else if (jt === ntheta - 1) {
                    dAz_dtheta = (Az[ntheta-1][ir] - Az[ntheta-2][ir]) / dtheta;
                } else {
                    dAz_dtheta = (Az[jt+1][ir] - Az[jt-1][ir]) / (2 * dtheta);
                }
                Br[jt][ir] = dAz_dtheta / safe_r;

                // Bθ = -∂Az/∂r
                let dAz_dr = 0;
                if (ir === 0) {
                    dAz_dr = (Az[jt][1] - Az[jt][0]) / dr;
                } else if (ir === nr - 1) {
                    dAz_dr = (Az[jt][nr-1] - Az[jt][nr-2]) / dr;
                } else {
                    dAz_dr = (Az[jt][ir+1] - Az[jt][ir-1]) / (2 * dr);
                }
                Btheta[jt][ir] = -dAz_dr;
            }
        }

        // 極座標 → 直交座標変換（可視化用）
        // r_orientation が horizontal の場合: i = r方向、j = θ方向
        // 物理座標: x = r*cos(θ), y = r*sin(θ)
        // 磁場の変換: Bx = Br*cos(θ) - Bθ*sin(θ), By = Br*sin(θ) + Bθ*cos(θ)
        for (let jt = 0; jt < ntheta; jt++) {
            const theta = jt * dtheta;
            const cos_theta = Math.cos(theta);
            const sin_theta = Math.sin(theta);

            for (let ir = 0; ir < nr; ir++) {
                Bx[jt][ir] = Br[jt][ir] * cos_theta - Btheta[jt][ir] * sin_theta;
                By[jt][ir] = Br[jt][ir] * sin_theta + Btheta[jt][ir] * cos_theta;
            }
        }
    } else {
        // 直交座標系での磁場計算
        // 周期境界条件の判定
        const bc = analysisConditions ? analysisConditions.boundary_conditions : null;
        const x_periodic = bc && bc.left && bc.right &&
                          bc.left.type === 'periodic' && bc.right.type === 'periodic';
        const y_periodic = bc && bc.bottom && bc.top &&
                          bc.bottom.type === 'periodic' && bc.top.type === 'periodic';

        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                // Bx = ∂Az/∂y
                if (j === 0) {
                    if (y_periodic) {
                        // Periodic boundary: use central difference with wrap
                        Bx[j][i] = (Az[1][i] - Az[rows-1][i]) / (2 * dy);
                    } else {
                        // Forward difference
                        Bx[j][i] = (Az[1][i] - Az[0][i]) / dy;
                    }
                } else if (j === rows - 1) {
                    if (y_periodic) {
                        // Periodic boundary: use central difference with wrap
                        Bx[j][i] = (Az[0][i] - Az[rows-2][i]) / (2 * dy);
                    } else {
                        // Backward difference
                        Bx[j][i] = (Az[rows-1][i] - Az[rows-2][i]) / dy;
                    }
                } else {
                    // Central difference
                    Bx[j][i] = (Az[j+1][i] - Az[j-1][i]) / (2 * dy);
                }

                // By = -∂Az/∂x
                if (i === 0) {
                    if (x_periodic) {
                        // Periodic boundary: use central difference with wrap
                        By[j][i] = -(Az[j][1] - Az[j][cols-1]) / (2 * dx);
                    } else {
                        // Forward difference
                        By[j][i] = -(Az[j][1] - Az[j][0]) / dx;
                    }
                } else if (i === cols - 1) {
                    if (x_periodic) {
                        // Periodic boundary: use central difference with wrap
                        By[j][i] = -(Az[j][0] - Az[j][cols-2]) / (2 * dx);
                    } else {
                        // Backward difference
                        By[j][i] = -(Az[j][cols-1] - Az[j][cols-2]) / dx;
                    }
                } else {
                    // Central difference
                    By[j][i] = -(Az[j][i+1] - Az[j][i-1]) / (2 * dx);
                }
            }
        }
    }

    // H = B / μ
    const Hx = Bx.map((row, j) => row.map((val, i) => val / Mu[j][i]));
    const Hy = By.map((row, j) => row.map((val, i) => val / Mu[j][i]));

    // 大きさ
    const B = Bx.map((row, j) => row.map((val, i) => Math.sqrt(val**2 + By[j][i]**2)));
    const H = Hx.map((row, j) => row.map((val, i) => Math.sqrt(val**2 + Hy[j][i]**2)));

    return { Bx, By, B, Hx, Hy, H };
}

// ===== プロット描画ヘルパー =====
function getContainerSize(container) {
    const rect = container.getBoundingClientRect();
    console.log(`Container rect: ${rect.width}x${rect.height}`);

    // パディングなし、コンテナいっぱいに表示
    // 高さが小さすぎる場合（初期化前）はデフォルト値を使わない
    return {
        width: rect.width > 50 ? rect.width : 400,
        height: rect.height > 50 ? rect.height : 400
    };
}

// ===== プロット描画関数 =====

async function renderAzContour(container, step) {
    const data = await loadStepData(step);
    if (!data.az) return;

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, [{
        z: data.az,
        type: 'contour',
        colorscale: 'Viridis',
        contours: { coloring: 'lines' }
    }], {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

async function renderAzHeatmap(container, step) {
    const data = await loadStepData(step);
    if (!data.az) return;

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, [{
        z: data.az,
        type: 'heatmap',
        colorscale: 'Hot'
    }], {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

async function renderBMagnitude(container, step) {
    const data = await loadStepData(step);
    if (!data.az || !data.mu) return;

    // Use correct mesh spacing from analysis conditions
    const dx = analysisConditions ? analysisConditions.dx : 0.001;
    const dy = analysisConditions ? analysisConditions.dy : 0.001;
    const fields = calculateMagneticField(data.az, data.mu, dx, dy);

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, [{
        z: fields.B,
        type: 'heatmap',
        colorscale: 'Hot'
    }], {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

async function renderHMagnitude(container, step) {
    const data = await loadStepData(step);
    if (!data.az || !data.mu) return;

    // Use correct mesh spacing from analysis conditions
    const dx = analysisConditions ? analysisConditions.dx : 0.001;
    const dy = analysisConditions ? analysisConditions.dy : 0.001;
    const fields = calculateMagneticField(data.az, data.mu, dx, dy);

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, [{
        z: fields.H,
        type: 'heatmap',
        colorscale: 'Hot'
    }], {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

async function renderMuDistribution(container, step) {
    const data = await loadStepData(step);
    if (!data.mu) return;

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, [{
        z: data.mu,
        type: 'heatmap',
        colorscale: 'Viridis'
    }], {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

// Az + 境界オーバーレイ
async function renderAzBoundary(container, step) {
    const data = await loadStepData(step);
    if (!data.az) return;

    try {
        // 境界画像URLを取得
        const boundaryImgUrl = `/api/get-boundary-image?result=${encodeURIComponent(currentResultPath)}&step=${step}&t=${Date.now()}`;

        // 境界画像を読み込んで黒色を透明に変換
        const transparentBoundaryUrl = await makeBlackTransparent(boundaryImgUrl);

        container.innerHTML = '';
        const size = getContainerSize(container);

        // Az等高線のトレース
        const traces = [
            {
                z: data.az,
                type: 'contour',
                colorscale: 'Viridis',
                contours: { coloring: 'lines' },
                showscale: false,
                name: 'Az'
            }
        ];

        // 境界画像を画像レイヤーとして追加（黒色を透明として扱う）
        const rows = data.az.length;
        const cols = data.az[0].length;

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 35, r: 10, t: 10, b: 35 },
            xaxis: {
                title: 'X',
                range: [0, cols]
            },
            yaxis: {
                title: 'Y',
                range: [0, rows]
            },
            images: [
                {
                    source: transparentBoundaryUrl,
                    xref: 'x',
                    yref: 'y',
                    x: 0,
                    y: rows,
                    sizex: cols,
                    sizey: rows,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'above'
                }
            ],
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Boundary image load error:', error);
        // フォールバック: muデータから境界を検出
        await renderAzBoundaryFallback(container, step, data);
    }
}

// 境界画像が利用できない場合のフォールバック
async function renderAzBoundaryFallback(container, step, data) {
    if (!data.mu) return;

    // Laplacianフィルタで境界検出（透磁率の変化から）
    const rows = data.mu.length;
    const cols = data.mu[0].length;
    const boundary = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let j = 1; j < rows - 1; j++) {
        for (let i = 1; i < cols - 1; i++) {
            const laplacian =
                Math.abs(data.mu[j-1][i] - data.mu[j][i]) +
                Math.abs(data.mu[j+1][i] - data.mu[j][i]) +
                Math.abs(data.mu[j][i-1] - data.mu[j][i]) +
                Math.abs(data.mu[j][i+1] - data.mu[j][i]);
            boundary[j][i] = laplacian > 1e-9 ? 1 : 0;
        }
    }

    // Az等高線と境界を重ねて表示
    const traces = [
        {
            z: data.az,
            type: 'contour',
            colorscale: 'Viridis',
            contours: { coloring: 'lines' },
            showscale: false,
            name: 'Az'
        },
        {
            z: boundary,
            type: 'heatmap',
            colorscale: [[0, 'rgba(0,0,0,0)'], [1, 'rgba(255,0,0,0.5)']],
            showscale: false,
            name: '境界'
        }
    ];

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, traces, {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

// 境界画像を読み込んで黒色を透明に変換
async function makeBlackTransparent(url, threshold = 30) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            // 画像を描画
            ctx.drawImage(img, 0, 0);

            // ピクセルデータを取得
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;

            // 黒色のピクセル（RGB値が閾値以下）を透明に変換
            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];

                // RGB値の合計が閾値以下なら透明にする
                if (r + g + b <= threshold * 3) {
                    pixels[i + 3] = 0;  // アルファチャンネルを0（透明）に設定
                }
            }

            // 変更されたピクセルデータを戻す
            ctx.putImageData(imageData, 0, 0);

            // Data URLとして返す
            resolve(canvas.toDataURL('image/png'));
        };

        img.onerror = function() {
            reject(new Error('Failed to load boundary image for transparency conversion'));
        };

        img.src = url;
    });
}

// 境界画像を読み込んでCSV形式の2D配列に変換
async function loadBoundaryImage(url, targetRows, targetCols) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = targetCols;
            canvas.height = targetRows;
            const ctx = canvas.getContext('2d');

            // 画像をターゲットサイズにリサイズして描画
            ctx.drawImage(img, 0, 0, targetCols, targetRows);

            // ピクセルデータを取得
            const imageData = ctx.getImageData(0, 0, targetCols, targetRows);
            const pixels = imageData.data;

            // 2D配列に変換（グレースケール値を使用）
            const boundary = Array(targetRows).fill(0).map(() => Array(targetCols).fill(0));
            for (let j = 0; j < targetRows; j++) {
                for (let i = 0; i < targetCols; i++) {
                    const idx = (j * targetCols + i) * 4;
                    const gray = (pixels[idx] + pixels[idx+1] + pixels[idx+2]) / 3;
                    boundary[j][i] = gray > 128 ? 1 : 0;  // 2値化
                }
            }

            resolve(boundary);
        };

        img.onerror = function() {
            reject(new Error('Failed to load boundary image'));
        };

        img.src = url;
    });
}

// ステップ入力画像表示（画像のみ）
async function renderStepInputImage(container, step) {
    try {
        // ステップごとの入力画像を取得
        const imgUrl = `/api/get-step-input-image?result=${encodeURIComponent(currentResultPath)}&step=${step}&t=${Date.now()}`;

        container.innerHTML = '';
        const size = getContainerSize(container);

        // Plotlyで画像を表示
        await Plotly.newPlot(container, [], {
            width: size.width,
            height: size.height,
            margin: { l: 0, r: 0, t: 0, b: 0 },
            xaxis: {
                visible: false,
                range: [0, 1]
            },
            yaxis: {
                visible: false,
                range: [0, 1]
            },
            images: [
                {
                    source: imgUrl,
                    xref: 'paper',
                    yref: 'paper',
                    x: 0,
                    y: 1,
                    sizex: 1,
                    sizey: 1,
                    sizing: 'contain',
                    opacity: 1.0,
                    layer: 'below'
                }
            ],
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Step input image load error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">画像読み込みエラー</div>';
    }
}

// 材質画像表示（ステップごとのInputImageを表示）
async function renderMaterialImage(container, step) {
    try {
        // ステップごとの入力画像を取得（InputImageフォルダから）
        const imgUrl = `/api/get-step-input-image?result=${encodeURIComponent(currentResultPath)}&step=${step}&t=${Date.now()}`;

        // Canvas取得または作成（既存のcanvasを保持してチラツキ防止）
        let canvas = container.querySelector('canvas');
        if (!canvas) {
            container.innerHTML = '<canvas style="width: 100%; height: 100%;"></canvas>';
            canvas = container.querySelector('canvas');
        }
        const ctx = canvas.getContext('2d');

        // 画像をプリロード（完全に読み込んでから描画することでチラツキ防止）
        const img = new Image();
        img.onload = function() {
            // Canvasサイズを設定
            const containerRect = container.getBoundingClientRect();
            const scale = Math.min(containerRect.width / img.width, containerRect.height / img.height);

            canvas.width = img.width * scale;
            canvas.height = img.height * scale;

            // 画像描画（この時点で画像は完全に読み込まれている）
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };

        img.onerror = function() {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">材質画像の読み込みに失敗しました</div>';
        };

        img.src = imgUrl;
    } catch (error) {
        console.error('Material image load error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">材質画像読み込みエラー</div>';
    }
}

// 境界のみ表示
async function renderBoundaryOnly(container, step) {
    const data = await loadStepData(step);
    if (!data.mu) return;

    // Laplacianフィルタで境界検出
    const rows = data.mu.length;
    const cols = data.mu[0].length;
    const boundary = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let j = 1; j < rows - 1; j++) {
        for (let i = 1; i < cols - 1; i++) {
            const laplacian =
                Math.abs(data.mu[j-1][i] - data.mu[j][i]) +
                Math.abs(data.mu[j+1][i] - data.mu[j][i]) +
                Math.abs(data.mu[j][i-1] - data.mu[j][i]) +
                Math.abs(data.mu[j][i+1] - data.mu[j][i]);
            boundary[j][i] = laplacian > 1e-9 ? 1 : 0;
        }
    }

    container.innerHTML = '';
    const size = getContainerSize(container);

    await Plotly.newPlot(container, [{
        z: boundary,
        type: 'heatmap',
        colorscale: [[0, 'white'], [1, 'black']],
        showscale: false
    }], {
        width: size.width,
        height: size.height,
        margin: { l: 35, r: 10, t: 10, b: 35 },
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        dragmode: false
    }, { responsive: true, displayModeBar: false });
}

// Bベクトル場
async function renderBVectors(container, step) {
    try {
        const data = await loadStepData(step);
        if (!data.az || !data.mu) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">データがありません</div>';
            return;
        }

        // Use correct mesh spacing from analysis conditions
        const dx = analysisConditions ? analysisConditions.dx : 0.001;
        const dy = analysisConditions ? analysisConditions.dy : 0.001;
        const fields = calculateMagneticField(data.az, data.mu, dx, dy);

        // サブサンプリング（間引き）
        const subsample = 5;
        const rows = fields.Bx.length;
        const cols = fields.Bx[0].length;

        if (rows === 0 || cols === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">データが空です</div>';
            return;
        }

        const x = [], y = [], u = [], v = [];

        for (let j = 0; j < rows; j += subsample) {
            for (let i = 0; i < cols; i += subsample) {
                x.push(i);
                y.push(j);
                u.push(fields.Bx[j][i] || 0);
                v.push(fields.By[j][i] || 0);
            }
        }

        if (x.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">ベクトルデータがありません</div>';
            return;
        }

        // ベクトル場の大きさ（色付け用）
        const magnitude = u.map((ux, idx) => Math.sqrt(ux**2 + v[idx]**2));

        // Plotlyのquiver的な表現
        await Plotly.newPlot(container, [{
            type: 'scatter',
            mode: 'markers',
            x: x,
            y: y,
            marker: {
                size: 5,
                color: magnitude,
                colorscale: 'Hot',
                showscale: true,
                colorbar: { title: '|B|', len: 0.7 }
            },
            hoverinfo: 'text',
            text: magnitude.map((m, idx) => `B: ${m.toExponential(2)}<br>Bx: ${u[idx].toExponential(2)}<br>By: ${v[idx].toExponential(2)}`)
        }], {
            margin: { l: 35, r: 50, t: 10, b: 35 },
            xaxis: {
                title: 'X',
                range: [0, cols]
            },
            yaxis: {
                title: 'Y',
                range: [0, rows]
            },
            dragmode: false
        }, { responsive: true });

        // 矢印を追加（Plotly annotationsとして）- 数を制限
        const annotations = [];
        const maxArrows = 100;  // 最大矢印数
        const step_size = Math.ceil(x.length / maxArrows);

        for (let idx = 0; idx < x.length; idx += step_size) {
            const mag = magnitude[idx];
            if (mag < 1e-10) continue;  // ゼロベクトルは描画しない

            const scale = 3;  // 矢印スケール
            annotations.push({
                x: x[idx] + scale * u[idx] / mag,
                y: y[idx] + scale * v[idx] / mag,
                ax: x[idx],
                ay: y[idx],
                xref: 'x',
                yref: 'y',
                axref: 'x',
                ayref: 'y',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 1.5,
                arrowcolor: 'rgba(0,0,255,0.6)'
            });
        }

        if (annotations.length > 0) {
            await Plotly.relayout(container, { annotations: annotations });
        }
    } catch (error) {
        console.error('B vector plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}

// Hベクトル場
async function renderHVectors(container, step) {
    try {
        const data = await loadStepData(step);
        if (!data.az || !data.mu) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">データがありません</div>';
            return;
        }

        // Use correct mesh spacing from analysis conditions
        const dx = analysisConditions ? analysisConditions.dx : 0.001;
        const dy = analysisConditions ? analysisConditions.dy : 0.001;
        const fields = calculateMagneticField(data.az, data.mu, dx, dy);

        // サブサンプリング
        const subsample = 5;
        const rows = fields.Hx.length;
        const cols = fields.Hx[0].length;

        if (rows === 0 || cols === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">データが空です</div>';
            return;
        }

        const x = [], y = [], u = [], v = [];

        for (let j = 0; j < rows; j += subsample) {
            for (let i = 0; i < cols; i += subsample) {
                x.push(i);
                y.push(j);
                u.push(fields.Hx[j][i] || 0);
                v.push(fields.Hy[j][i] || 0);
            }
        }

        if (x.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">ベクトルデータがありません</div>';
            return;
        }

        const magnitude = u.map((ux, idx) => Math.sqrt(ux**2 + v[idx]**2));

        await Plotly.newPlot(container, [{
            type: 'scatter',
            mode: 'markers',
            x: x,
            y: y,
            marker: {
                size: 5,
                color: magnitude,
                colorscale: 'Hot',
                showscale: true,
                colorbar: { title: '|H|', len: 0.7 }
            },
            hoverinfo: 'text',
            text: magnitude.map((m, idx) => `H: ${m.toExponential(2)}<br>Hx: ${u[idx].toExponential(2)}<br>Hy: ${v[idx].toExponential(2)}`)
        }], {
            margin: { l: 35, r: 50, t: 10, b: 35 },
            xaxis: {
                title: 'X',
                range: [0, cols]
            },
            yaxis: {
                title: 'Y',
                range: [0, rows]
            },
            dragmode: false
        }, { responsive: true });

        const annotations = [];
        const maxArrows = 100;
        const step_size = Math.ceil(x.length / maxArrows);

        for (let idx = 0; idx < x.length; idx += step_size) {
            const mag = magnitude[idx];
            if (mag < 1e-10) continue;

            const scale = 3;
            annotations.push({
                x: x[idx] + scale * u[idx] / mag,
                y: y[idx] + scale * v[idx] / mag,
                ax: x[idx],
                ay: y[idx],
                xref: 'x',
                yref: 'y',
                axref: 'x',
                ayref: 'y',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 1.5,
                arrowcolor: 'rgba(255,0,0,0.6)'
            });
        }

        if (annotations.length > 0) {
            await Plotly.relayout(container, { annotations: annotations });
        }
    } catch (error) {
        console.error('H vector plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}

async function renderForceXTime(container) {
    try {
        // 全ステップのデータを読み込み
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i+1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">Forcesデータがありません</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        // x 軸の値を 1..totalSteps の配列で明示的に作る（1-based）
        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);


        // 材料名のリストを取得（最初のステップから）
        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        // 材料ごとのトレースを作成
        const traces = [];

        // マーカーサイズを計算（常に配列を返すようにする）
        const getMarkerSizes = (baseSize, highlightSize) => {
            // 常に長さ totalSteps の配列を返す
            return Array.from({ length: totalSteps }, (_, i) => {
                // currentStep が 1-based と仮定（i は 0-based）:
                return (isAnimating && (i + 1 === currentStep)) ? highlightSize : baseSize;
            });
        };


        // 材料ごとのトレース
        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        forceData.push(mat.force_x);
                        if (!matColor) matColor = mat.color;
                    } else {
                        forceData.push(0);
                    }
                } else {
                    forceData.push(0);
                }
            }

            traces.push({
                x: xSteps,
                y: forceData,
                type: 'scatter',
                mode: 'lines+markers',
                name: matName,
                line: { color: matColor, width: 2 },
                marker: { color: matColor, size: getMarkerSizes(6, 14) }
            });
        });

        // 合計のトレースを追加
        // const totalForces = allStepsData.map(data =>
        //     data && data.total ? data.total.force_x : 0
        // );
        // traces.push({
        //     x: [...Array(totalSteps).keys()],
        //     y: totalForces,
        //     type: 'scatter',
        //     mode: 'lines+markers',
        //     name: '合計',
        //     line: { color: 'black', width: 3, dash: 'dash' },
        //     marker: { color: 'black', size: getMarkerSizes(8, 16) }
        // });

        // データ範囲を取得
        const allForces = traces.flatMap(t => t.y);
        const maxForce = Math.max(...allForces.map(Math.abs));
        const yrange = maxForce > 1e-10 ? undefined : [-0.1, 0.1];

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'ステップ', range: [1, totalSteps] },
            yaxis: { title: '力X [N/m]', range: yrange },
            showlegend: true,
            legend: { x: 1.02, y: 1, xanchor: 'left' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Force X time plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}

async function renderForceYTime(container) {
    try {
        // 全ステップのデータを読み込み
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i+1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">Forcesデータがありません</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        // x 軸の値を 1..totalSteps の配列で明示的に作る（1-based）
        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);


        // 材料名のリストを取得
        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        // 材料ごとのトレースを作成
        const traces = [];

        // マーカーサイズを計算（常に配列を返すようにする）
        const getMarkerSizes = (baseSize, highlightSize) => {
            // 常に長さ totalSteps の配列を返す
            return Array.from({ length: totalSteps }, (_, i) => {
                // currentStep が 1-based と仮定（i は 0-based）:
                return (isAnimating && (i + 1 === currentStep)) ? highlightSize : baseSize;
            });
        };


        // 材料ごとのトレース
        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        forceData.push(mat.force_y);
                        if (!matColor) matColor = mat.color;
                    } else {
                        forceData.push(0);
                    }
                } else {
                    forceData.push(0);
                }
            }

            traces.push({
                x: xSteps,
                y: forceData,
                type: 'scatter',
                mode: 'lines+markers',
                name: matName,
                line: { color: matColor, width: 2 },
                marker: { color: matColor, size: getMarkerSizes(6, 14) }
            });
        });

        // 合計のトレースを追加
        // const totalForces = allStepsData.map(data =>
        //     data && data.total ? data.total.force_y : 0
        // );
        // traces.push({
        //     x: [...Array(totalSteps).keys()],
        //     y: totalForces,
        //     type: 'scatter',
        //     mode: 'lines+markers',
        //     name: '合計',
        //     line: { color: 'black', width: 3, dash: 'dash' },
        //     marker: { color: 'black', size: getMarkerSizes(8, 16) }
        // });

        // データ範囲を取得
        const allForces = traces.flatMap(t => t.y);
        const maxForce = Math.max(...allForces.map(Math.abs));
        const yrange = maxForce > 1e-10 ? undefined : [-0.1, 0.1];

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'ステップ', range: [1, totalSteps] },
            yaxis: { title: '力Y [N/m]', range: yrange },
            showlegend: true,
            legend: { x: 1.02, y: 1, xanchor: 'left' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Force Y time plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}

async function renderTorqueTime(container) {
    try {
        // 全ステップのデータを読み込み
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i+1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">Forcesデータがありません</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        // x 軸の値を 1..totalSteps の配列で明示的に作る（1-based）
        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);


        // 材料名のリストを取得
        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        // 材料ごとのトレースを作成
        const traces = [];

        // マーカーサイズを計算（常に配列を返すようにする）
        const getMarkerSizes = (baseSize, highlightSize) => {
            // 常に長さ totalSteps の配列を返す
            return Array.from({ length: totalSteps }, (_, i) => {
                // currentStep が 1-based と仮定（i は 0-based）:
                return (isAnimating && (i + 1 === currentStep)) ? highlightSize : baseSize;
            });
        };


        // 材料ごとのトレース
        materialNames.forEach(matName => {
            const torqueData = [];
            let matColor = null;

            for (let i = 0; i < totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        torqueData.push(mat.torque);
                        if (!matColor) matColor = mat.color;
                    } else {
                        torqueData.push(0);
                    }
                } else {
                    torqueData.push(0);
                }
            }

            traces.push({
                x: xSteps,
                y: torqueData,
                type: 'scatter',
                mode: 'lines+markers',
                name: matName,
                line: { color: matColor, width: 2 },
                marker: { color: matColor, size: getMarkerSizes(6, 14) }
            });
        });

        // 合計のトレースを追加
        // const totalTorques = allStepsData.map(data =>
        //     data && data.total ? data.total.torque : 0
        // );
        // traces.push({
        //     x: [...Array(totalSteps).keys()],
        //     y: totalTorques,
        //     type: 'scatter',
        //     mode: 'lines+markers',
        //     name: '合計',
        //     line: { color: 'black', width: 3, dash: 'dash' },
        //     marker: { color: 'black', size: getMarkerSizes(8, 16) }
        // });

        // データ範囲を取得
        const allTorques = traces.flatMap(t => t.y);
        const maxTorque = Math.max(...allTorques.map(Math.abs));
        const yrange = maxTorque > 1e-10 ? undefined : [-0.1, 0.1];

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'ステップ', range: [1, totalSteps] },
            yaxis: { title: 'トルク [N]', range: yrange },
            showlegend: true,
            legend: { x: 1.02, y: 1, xanchor: 'left' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Torque time plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}

// マクスウェル応力ベクトルプロット
async function renderStressVectors(container, step) {
    try {
        const stressData = await loadStressVectorData(step);
        if (!stressData || stressData.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">応力ベクトルデータがありません</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        // x 軸の値を 1..totalSteps の配列で明示的に作る（1-based）
        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);


        // ピクセル座標と物理座標を取得
        const iPixels = stressData.map(p => p.i_pixel);
        const jPixels = stressData.map(p => p.j_pixel);
        const fx = stressData.map(p => p.fx);
        const fy = stressData.map(p => p.fy);
        const fMagnitude = fx.map((f, idx) => Math.sqrt(f**2 + fy[idx]**2));

        // 最大応力ベクトルの大きさ（スケーリング用）
        const maxMag = Math.max(...fMagnitude);
        const scale = maxMag > 1e-10 ? 5.0 / maxMag : 1.0;

        // ベクトル場をscatterで表示（カラーは大きさ）
        const traces = [{
            type: 'scatter',
            mode: 'markers',
            x: iPixels,
            y: jPixels,
            marker: {
                size: 4,
                color: fMagnitude,
                colorscale: 'Hot',
                showscale: true,
                colorbar: { title: '応力 [N/m]', len: 0.7 }
            },
            hoverinfo: 'text',
            text: stressData.map((p, idx) =>
                `Material: ${p.material}<br>` +
                `位置: (${p.i_pixel}, ${p.j_pixel})<br>` +
                `fx: ${p.fx.toExponential(2)} N/m<br>` +
                `fy: ${p.fy.toExponential(2)} N/m<br>` +
                `|f|: ${fMagnitude[idx].toExponential(2)} N/m<br>` +
                `B: ${p.b_mag.toExponential(2)} T`
            ),
            name: '応力点'
        }];

        // グリッドサイズを推定（最大ピクセル座標から）
        const maxI = Math.max(...iPixels);
        const maxJ = Math.max(...jPixels);

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 35, r: 50, t: 10, b: 35 },
            xaxis: {
                title: 'i (pixel)',
                range: [0, maxI + 10]
            },
            yaxis: {
                title: 'j (pixel)',
                range: [0, maxJ + 10],
                scaleanchor: 'x',
                scaleratio: 1
            },
            dragmode: false
        }, { responsive: true, displayModeBar: false });

        // 応力ベクトルを矢印として追加（サブサンプリング）
        const annotations = [];
        const maxArrows = 200;  // 最大矢印数
        const stepSize = Math.max(1, Math.ceil(stressData.length / maxArrows));

        for (let idx = 0; idx < stressData.length; idx += stepSize) {
            const mag = fMagnitude[idx];
            if (mag < maxMag * 0.01) continue;  // 小さすぎるベクトルはスキップ

            const arrowScale = scale * mag;
            annotations.push({
                x: iPixels[idx] + arrowScale * fx[idx] / mag,
                y: jPixels[idx] + arrowScale * fy[idx] / mag,
                ax: iPixels[idx],
                ay: jPixels[idx],
                xref: 'x',
                yref: 'y',
                axref: 'x',
                ayref: 'y',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 1.5,
                arrowcolor: 'rgba(255,0,0,0.6)'
            });
        }

        if (annotations.length > 0) {
            await Plotly.relayout(container, { annotations: annotations });
        }

        console.log(`Rendered ${stressData.length} stress vectors (${annotations.length} arrows shown)`);
    } catch (error) {
        console.error('Stress vector plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}

// 磁気エネルギー時系列プロット
async function renderEnergyTime(container) {
    try {
        // 全ステップのデータを読み込み
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i+1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">Forcesデータがありません</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        // x 軸の値を 1..totalSteps の配列で明示的に作る（1-based）
        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

        // 材料名のリストを取得
        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        // 材料ごとのトレースを作成
        const traces = [];

        // マーカーサイズを計算（常に配列を返すようにする）
        const getMarkerSizes = (baseSize, highlightSize) => {
            // 常に長さ totalSteps の配列を返す
            return Array.from({ length: totalSteps }, (_, i) => {
                // currentStep が 1-based と仮定（i は 0-based）:
                return (isAnimating && (i + 1 === currentStep)) ? highlightSize : baseSize;
            });
        };

        // 材料ごとのトレース
        materialNames.forEach(matName => {
            const energyData = [];
            let matColor = null;

            for (let i = 0; i < totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        energyData.push(mat.energy);
                        if (!matColor) matColor = mat.color;
                    } else {
                        energyData.push(null);
                    }
                } else {
                    energyData.push(null);
                }
            }

            traces.push({
                x: xSteps,
                y: energyData,
                type: 'scatter',
                mode: 'lines+markers',
                name: matName,
                line: { color: matColor, width: 2 },
                marker: { color: matColor, size: getMarkerSizes(6, 14) }
            });
        });

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 50, r: 10, t: 10, b: 35 },
            xaxis: { title: 'ステップ', range: [1, totalSteps] },
            yaxis: { title: '磁気エネルギー [J/m]' },
            showlegend: true,
            legend: { x: 1.02, y: 1, xanchor: 'left' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Energy time plot error:', error);
        container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">描画エラー: ${error.message}</div>`;
    }
}
