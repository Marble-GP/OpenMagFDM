// グローバル変数
let currentAzData = null;
let currentMuData = null;
let currentResultPath = null;
let totalStepsCount = 0;
let animationTimer = null;
let isAnimating = false;

// ===== タブ切り替え =====
function switchTab(tabName) {
    // すべてのタブとコンテンツを非アクティブ化
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    // 選択されたタブとコンテンツをアクティブ化
    event.target.classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

// ===== アラート表示 =====
function showAlert(elementId, message, type = 'info') {
    const alert = document.getElementById(elementId);
    alert.className = `alert ${type}`;
    alert.textContent = message;
    alert.style.display = 'block';

    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            alert.style.display = 'none';
        }, 5000);
    }
}

// ===== Tab 1: 設定 =====

// YAML設定ファイルの読み込み
async function loadConfig() {
    try {
        showAlert('configAlert', '読み込み中...', 'info');

        const response = await fetch('/api/config');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.yaml) {
            document.getElementById('configEditor').value = data.yaml;
            showAlert('configAlert', '設定ファイルを読み込みました', 'success');
        } else {
            showAlert('configAlert', `エラー: ${data.error || '不明なエラー'}`, 'error');
        }
    } catch (error) {
        showAlert('configAlert', `読み込みエラー: ${error.message}`, 'error');
        console.error('Config load error:', error);
    }
}

// YAML設定ファイルの保存
async function saveConfig() {
    const yamlContent = document.getElementById('configEditor').value;

    if (!yamlContent || yamlContent.trim() === '') {
        showAlert('configAlert', 'YAML設定が空です', 'error');
        return;
    }

    try {
        showAlert('configAlert', '保存中...', 'info');

        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ yaml: yamlContent })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            showAlert('configAlert', '設定ファイルを保存しました', 'success');
        } else {
            showAlert('configAlert', `保存エラー: ${data.error || 'YAML形式が不正です'}`, 'error');
        }
    } catch (error) {
        showAlert('configAlert', `保存エラー: ${error.message}`, 'error');
        console.error('Config save error:', error);
    }
}

// 画像のアップロード
async function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    try {
        showAlert('uploadAlert', 'アップロード中...', 'info');

        const response = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showAlert('uploadAlert', `アップロード完了: ${data.filename}`, 'success');
            refreshImageList();
            fileInput.value = '';
        } else {
            showAlert('uploadAlert', `アップロードエラー: ${data.error}`, 'error');
        }
    } catch (error) {
        showAlert('uploadAlert', `アップロードエラー: ${error.message}`, 'error');
    }
}

// 画像リストの更新
async function refreshImageList() {
    try {
        const response = await fetch('/api/images');
        const data = await response.json();

        const imageList = document.getElementById('imageList');
        imageList.innerHTML = '';

        if (data.success && data.images.length > 0) {
            data.images.forEach(img => {
                const option = document.createElement('option');
                option.value = img;
                option.textContent = img;
                imageList.appendChild(option);
            });
        } else {
            imageList.innerHTML = '<option>画像がありません</option>';
        }

        // ソルバータブの画像リストも更新
        refreshSolveImages();
    } catch (error) {
        console.error('画像リスト更新エラー:', error);
    }
}

// ===== Tab 2: 解析実行 =====

// ソルバー用画像リストの更新
async function refreshSolveImages() {
    try {
        const response = await fetch('/api/images');
        const data = await response.json();

        const select = document.getElementById('solveImageSelect');
        select.innerHTML = '<option value="">選択してください</option>';

        if (data.success) {
            data.images.forEach(img => {
                const option = document.createElement('option');
                option.value = img;
                option.textContent = img;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('画像リスト更新エラー:', error);
    }
}

// ソルバーの実行
async function runSolver() {
    const imageFile = document.getElementById('solveImageSelect').value;

    if (!imageFile) {
        showAlert('solverAlert', '材質画像を選択してください', 'error');
        return;
    }

    const runBtn = document.getElementById('runSolverBtn');
    runBtn.disabled = true;

    document.getElementById('solverProgress').classList.remove('hidden');
    document.getElementById('solverOutput').classList.add('hidden');

    try {
        showAlert('solverAlert', 'ソルバーを実行中...', 'info');

        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imageFile: imageFile })
        });

        const data = await response.json();

        document.getElementById('solverProgress').classList.add('hidden');

        if (data.success) {
            showAlert('solverAlert', '解析が完了しました！', 'success');
            document.getElementById('solverOutput').classList.remove('hidden');
            document.getElementById('solverStdout').textContent = data.stdout;

            // 結果リストを更新
            setTimeout(() => {
                refreshResults();
                refreshVisualizationResults();
            }, 500);
        } else {
            showAlert('solverAlert', `ソルバーエラー: ${data.error}`, 'error');
            if (data.stderr) {
                document.getElementById('solverOutput').classList.remove('hidden');
                document.getElementById('solverStdout').textContent = data.stderr;
            }
        }
    } catch (error) {
        document.getElementById('solverProgress').classList.add('hidden');
        showAlert('solverAlert', `実行エラー: ${error.message}`, 'error');
    } finally {
        runBtn.disabled = false;
    }
}

// 結果リストの更新
async function refreshResults() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();

        const resultsList = document.getElementById('resultsList');
        resultsList.innerHTML = '';

        if (data.success && data.results.length > 0) {
            data.results.forEach(result => {
                const li = document.createElement('li');
                li.className = 'result-item';
                const stepInfo = result.steps > 1 ? `${result.steps}ステップ` : '静的解析';
                li.innerHTML = `
                    <div class="result-info">
                        <div class="result-timestamp">${result.name}</div>
                        <div class="result-files">
                            ${stepInfo} | フォルダ: ${result.path}
                        </div>
                    </div>
                `;
                resultsList.appendChild(li);
            });
        } else {
            resultsList.innerHTML = '<li>解析結果がありません</li>';
        }
    } catch (error) {
        console.error('結果リスト更新エラー:', error);
    }
}

// ===== Tab 3: 可視化 =====

// 可視化用結果リストの更新
async function refreshVisualizationResults() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();

        const select = document.getElementById('resultSelect');
        select.innerHTML = '<option value="">選択してください</option>';

        if (data.success && data.results.length > 0) {
            data.results.forEach(result => {
                const option = document.createElement('option');
                option.value = result.path;
                const stepInfo = result.steps > 1 ? ` (${result.steps}ステップ)` : '';
                option.textContent = `${result.name}${stepInfo}`;
                select.appendChild(option);
            });
        }

        document.getElementById('visualizeBtn').disabled = false;
    } catch (error) {
        console.error('結果リスト更新エラー:', error);
    }
}

// 選択された結果を読み込み
function loadSelectedResult() {
    const select = document.getElementById('resultSelect');
    const value = select.value;

    if (value) {
        document.getElementById('visualizeBtn').disabled = false;
    } else {
        document.getElementById('visualizeBtn').disabled = true;
    }
}

// 結果の読み込みと可視化
async function loadResult(azFile, muFile) {
    try {
        // Azファイルの読み込み
        const azResponse = await fetch(`/data/${azFile}`);
        const azText = await azResponse.text();
        currentAzData = parseCSV(azText);

        // Muファイルの読み込み
        const muResponse = await fetch(`/data/${muFile}`);
        const muText = await muResponse.text();
        currentMuData = parseCSV(muText);

        // 可視化ボタンを有効化
        document.getElementById('visualizeBtn').disabled = false;

        // 自動的に可視化
        visualizeResult();

    } catch (error) {
        showAlert('visualizeAlert', `データ読み込みエラー: ${error.message}`, 'error');
    }
}

// CSV解析
function parseCSV(text) {
    const lines = text.trim().split('\n');
    return lines.map(line =>
        line.split(',').map(val => parseFloat(val.trim()))
    );
}

// 磁束密度の計算
function calculateMagneticField(Az, dx, dy) {
    const rows = Az.length;
    const cols = Az[0].length;

    const Bx = Array(rows).fill(0).map(() => Array(cols).fill(0));
    const By = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            if (j === 0) {
                Bx[j][i] = (Az[1][i] - Az[0][i]) / dy;
            } else if (j === rows - 1) {
                Bx[j][i] = (Az[rows-1][i] - Az[rows-2][i]) / dy;
            } else {
                Bx[j][i] = (Az[j+1][i] - Az[j-1][i]) / (2 * dy);
            }

            if (i === 0) {
                By[j][i] = -(Az[j][1] - Az[j][0]) / dx;
            } else if (i === cols - 1) {
                By[j][i] = -(Az[j][cols-1] - Az[j][cols-2]) / dx;
            } else {
                By[j][i] = -(Az[j][i+1] - Az[j][i-1]) / (2 * dx);
            }
        }
    }

    const B = Array(rows).fill(0).map(() => Array(cols).fill(0));
    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            B[j][i] = Math.sqrt(Bx[j][i]**2 + By[j][i]**2);
        }
    }

    return { Bx, By, B };
}

// 可視化実行（過渡解析対応）
async function visualizeResult() {
    const resultPath = document.getElementById('resultSelect').value;
    if (!resultPath) {
        showAlert('visualizeAlert', '解析結果を選択してください', 'error');
        return;
    }

    currentResultPath = resultPath;

    // 材質画像を表示
    if (typeof loadAndDisplayMaterialImage === 'function') {
        await loadAndDisplayMaterialImage();
    }

    // ステップ数を検出
    const steps = await detectSteps(resultPath);
    totalStepsCount = steps;

    if (steps > 1) {
        // 過渡解析：スライダーを表示
        document.getElementById('stepControls').classList.remove('hidden');
        document.getElementById('stepSlider').max = steps;
        document.getElementById('stepSlider').value = 1;
        document.getElementById('totalSteps').textContent = steps;
        document.getElementById('stepValue').textContent = '1';

        // 最初のステップを可視化
        await loadAndVisualizeStep(resultPath, 0);
        showAlert('visualizeAlert', `過渡解析結果を読み込みました（${steps}ステップ）`, 'success');
    } else {
        // 静的解析：従来通り
        document.getElementById('stepControls').classList.add('hidden');
        await loadAndVisualizeStep(resultPath, 0);
        showAlert('visualizeAlert', '可視化完了！', 'success');
    }
}

// プロット関数
function plotHeatmap(divId, data, title) {
    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: 'Hot',
        colorbar: { thickness: 15 }
    };

    const layout = {
        xaxis: { title: 'X' },
        yaxis: { title: 'Y', autorange: 'reversed' },
        margin: { l: 50, r: 50, t: 20, b: 50 }
    };

    Plotly.newPlot(divId, [trace], layout, { responsive: true });
}

function plotContour(divId, data, title) {
    const trace = {
        z: data,
        type: 'contour',
        colorscale: 'Viridis',
        colorbar: { thickness: 15 }
    };

    const layout = {
        xaxis: { title: 'X' },
        yaxis: { title: 'Y', autorange: 'reversed' },
        margin: { l: 50, r: 50, t: 20, b: 50 }
    };

    Plotly.newPlot(divId, [trace], layout, { responsive: true });
}

// 磁界計算
function calculateFields(Az, Mu, dx, dy) {
    const rows = Az.length;
    const cols = Az[0].length;

    const Bx = Array(rows).fill(0).map(() => Array(cols).fill(0));
    const By = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            // Bx = ∂Az/∂y
            if (j === 0) {
                Bx[j][i] = (Az[1][i] - Az[0][i]) / dy;
            } else if (j === rows - 1) {
                Bx[j][i] = (Az[rows-1][i] - Az[rows-2][i]) / dy;
            } else {
                Bx[j][i] = (Az[j+1][i] - Az[j-1][i]) / (2 * dy);
            }

            // By = -∂Az/∂x
            if (i === 0) {
                By[j][i] = -(Az[j][1] - Az[j][0]) / dx;
            } else if (i === cols - 1) {
                By[j][i] = -(Az[j][cols-1] - Az[j][cols-2]) / dx;
            } else {
                By[j][i] = -(Az[j][i+1] - Az[j][i-1]) / (2 * dx);
            }
        }
    }

    // H = B / μ
    const Hx = Array(rows).fill(0).map(() => Array(cols).fill(0));
    const Hy = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            Hx[j][i] = Bx[j][i] / Mu[j][i];
            Hy[j][i] = By[j][i] / Mu[j][i];
        }
    }

    return { Bx, By, Hx, Hy };
}

// ベクトル成分から大きさを計算
function calculateMagnitude(Fx, Fy) {
    const rows = Fx.length;
    const cols = Fx[0].length;
    const magnitude = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            magnitude[j][i] = Math.sqrt(Fx[j][i]**2 + Fy[j][i]**2);
        }
    }

    return magnitude;
}

// ===== 過渡解析対応 =====

// ステップ数を検出
async function detectSteps(resultPath) {
    try {
        const response = await fetch(`/api/detect-steps?result=${encodeURIComponent(resultPath)}`);
        const data = await response.json();

        if (data.success) {
            return data.steps;
        }
        return 1; // デフォルトは1ステップ（静的解析）
    } catch (error) {
        console.error('Step detection error:', error);
        return 1;
    }
}

// ステップ表示を更新
function updateStepDisplay() {
    const stepSlider = document.getElementById('stepSlider');
    const stepValue = document.getElementById('stepValue');
    stepValue.textContent = stepSlider.value;
}

// 特定のステップを可視化
async function visualizeStep() {
    const step = parseInt(document.getElementById('stepSlider').value);
    await loadAndVisualizeStep(currentResultPath, step);
}

// ステップデータを読み込んで可視化
async function loadAndVisualizeStep(resultPath, step) {
    const dx = parseFloat(document.getElementById('dx').value);
    const dy = parseFloat(document.getElementById('dy').value);

    try {
        // Az, Mu データを読み込み
        const azResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Az/step_${String(step).padStart(4, '0')}.csv`);
        const azData = await azResponse.json();

        const muResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Mu/step_${String(step).padStart(4, '0')}.csv`);
        const muData = await muResponse.json();

        if (azData.success && muData.success) {
            currentAzData = azData.data;
            currentMuData = muData.data;

            // 磁束密度と磁界強度を計算
            const { Bx, By, Hx, Hy } = calculateFields(currentAzData, currentMuData, dx, dy);
            const B_magnitude = calculateMagnitude(Bx, By);
            const H_magnitude = calculateMagnitude(Hx, Hy);

            // プロット
            plotHeatmap('plotAz', currentAzData, 'Az');
            plotHeatmap('plotMu', currentMuData, 'μ');
            plotHeatmap('plotB', B_magnitude, '|B|');
            plotHeatmap('plotH', H_magnitude, '|H|');

            document.getElementById('plots').classList.remove('hidden');
        }
    } catch (error) {
        console.error('Visualization error:', error);
    }
}

// アニメーション再生
function playAnimation() {
    if (isAnimating) return;

    isAnimating = true;
    document.getElementById('playBtn').classList.add('hidden');
    document.getElementById('pauseBtn').classList.remove('hidden');

    const stepSlider = document.getElementById('stepSlider');
    const speed = parseInt(document.getElementById('animSpeed').value);

    animationTimer = setInterval(() => {
        let currentStep = parseInt(stepSlider.value);
        currentStep++;

        if (currentStep > totalStepsCount) {
            currentStep = 1; // ループ
        }

        stepSlider.value = currentStep;
        updateStepDisplay();
        visualizeStep();
    }, speed);
}

// アニメーション一時停止
function pauseAnimation() {
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }

    isAnimating = false;
    document.getElementById('playBtn').classList.remove('hidden');
    document.getElementById('pauseBtn').classList.add('hidden');
}

// アニメーションリセット
function resetAnimation() {
    pauseAnimation();

    const stepSlider = document.getElementById('stepSlider');
    stepSlider.value = 1;
    updateStepDisplay();
    visualizeStep();
}

// 初期化
window.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    refreshImageList();
    refreshSolveImages();
    refreshResults();
    refreshVisualizationResults();
});
