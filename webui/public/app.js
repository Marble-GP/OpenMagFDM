// グローバル変数
let azData = null;
let muData = null;
let conditionsData = null;  // conditions.json from analysis results

// DOM要素の取得
const azFileInput = document.getElementById('azFile');
const muFileInput = document.getElementById('muFile');
const conditionsFileInput = document.getElementById('conditionsFile');
const azStatus = document.getElementById('azStatus');
const muStatus = document.getElementById('muStatus');
const conditionsStatus = document.getElementById('conditionsStatus');
const visualizeBtn = document.getElementById('visualizeBtn');
const loading = document.getElementById('loading');
const infoPanel = document.getElementById('infoPanel');
const infoGrid = document.getElementById('infoGrid');
const visualizationArea = document.getElementById('visualizationArea');

// ファイル選択イベント
azFileInput.addEventListener('change', (e) => handleFileUpload(e, 'az'));
muFileInput.addEventListener('change', (e) => handleFileUpload(e, 'mu'));
conditionsFileInput.addEventListener('change', (e) => handleFileUpload(e, 'conditions'));
visualizeBtn.addEventListener('click', visualize);

/**
 * CSVファイルを読み込んで2D配列に変換
 */
async function parseCSV(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.trim().split('\n');
                const data = lines.map(line =>
                    line.split(',').map(val => parseFloat(val.trim()))
                );

                // データの検証
                const rows = data.length;
                const cols = data[0].length;

                // すべての行が同じ列数を持つか確認
                const validData = data.every(row => row.length === cols);
                if (!validData) {
                    throw new Error('不正なCSV形式: 行ごとの列数が異なります');
                }

                // NaN値のチェック
                const hasNaN = data.some(row => row.some(val => isNaN(val)));
                if (hasNaN) {
                    throw new Error('不正なCSV形式: 数値に変換できないデータが含まれています');
                }

                resolve(data);
            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = () => reject(new Error('ファイルの読み込みに失敗しました'));
        reader.readAsText(file);
    });
}

/**
 * JSONファイルを読み込んで解析
 */
async function parseJSON(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const json = JSON.parse(e.target.result);
                resolve(json);
            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = () => reject(new Error('JSONファイルの読み込みに失敗しました'));
        reader.readAsText(file);
    });
}

/**
 * ファイルアップロード処理
 */
async function handleFileUpload(event, type) {
    const file = event.target.files[0];
    if (!file) return;

    const statusElement = type === 'az' ? azStatus : (type === 'mu' ? muStatus : conditionsStatus);
    statusElement.className = 'file-status';
    statusElement.textContent = '読み込み中...';
    statusElement.style.display = 'block';

    try {
        let data;
        if (type === 'conditions') {
            data = await parseJSON(file);
        } else {
            data = await parseCSV(file);
        }

        if (type === 'az') {
            azData = data;
            azStatus.className = 'file-status success';
            azStatus.textContent = `✓ 読み込み完了: ${data.length} x ${data[0].length}`;
        } else if (type === 'mu') {
            muData = data;
            muStatus.className = 'file-status success';
            muStatus.textContent = `✓ 読み込み完了: ${data.length} x ${data[0].length}`;
        } else if (type === 'conditions') {
            conditionsData = data;
            conditionsStatus.className = 'file-status success';
            conditionsStatus.textContent = `✓ 読み込み完了: ${data.coordinate_system || 'cartesian'} 座標系`;
        }

        // 両方のファイルが読み込まれたらボタンを有効化
        if (azData && muData) {
            visualizeBtn.disabled = false;
        }
    } catch (error) {
        statusElement.className = 'file-status error';
        statusElement.textContent = `✗ エラー: ${error.message}`;

        if (type === 'az') {
            azData = null;
        } else if (type === 'mu') {
            muData = null;
        } else if (type === 'conditions') {
            conditionsData = null;
        } else {
            muData = null;
        }
        visualizeBtn.disabled = true;
    }
}

/**
 * 磁束密度の計算 (座標系自動判定)
 * Cartesian: Bx = ∂Az/∂y, By = -∂Az/∂x
 * Polar: Br = (1/r)·∂Az/∂θ, Bθ = -∂Az/∂r
 */
function calculateMagneticField(Az, dx, dy) {
    const rows = Az.length;
    const cols = Az[0].length;

    // Check coordinate system from conditions.json
    const isPolar = conditionsData && conditionsData.coordinate_system === 'polar';

    if (isPolar) {
        // Polar coordinates calculation
        const r_orientation = conditionsData.polar.r_orientation;
        const r_start = conditionsData.polar.r_start;
        const r_end = conditionsData.polar.r_end;
        const dr = conditionsData.dr;
        const dtheta = conditionsData.dtheta;

        // Generate r coordinate array
        const nr = (r_orientation === 'horizontal') ? cols : rows;
        const ntheta = (r_orientation === 'horizontal') ? rows : cols;
        const r_coords = Array(nr).fill(0).map((_, i) => r_start + i * dr);

        const Br = Array(rows).fill(0).map(() => Array(cols).fill(0));
        const Btheta = Array(rows).fill(0).map(() => Array(cols).fill(0));

        // Polar coordinates are ALWAYS periodic in theta direction
        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                // Determine r and theta indices based on orientation
                const ir = (r_orientation === 'horizontal') ? i : j;
                const itheta = (r_orientation === 'horizontal') ? j : i;

                const r = r_coords[ir];

                // Br = (1/r) * ∂Az/∂θ (periodic in theta)
                const j_next = (itheta + 1) % ntheta;
                const j_prev = (itheta - 1 + ntheta) % ntheta;

                let dAz_dtheta;
                if (r_orientation === 'horizontal') {
                    dAz_dtheta = (Az[j_next][i] - Az[j_prev][i]) / (2 * dtheta);
                } else {
                    dAz_dtheta = (Az[j][j_next] - Az[j][j_prev]) / (2 * dtheta);
                }

                Br[j][i] = dAz_dtheta / (r > 1e-15 ? r : 1e-15);

                // Bθ = -∂Az/∂r
                let dAz_dr;
                if (r_orientation === 'horizontal') {
                    if (i === 0) {
                        dAz_dr = (Az[j][1] - Az[j][0]) / dr;
                    } else if (i === cols - 1) {
                        dAz_dr = (Az[j][cols-1] - Az[j][cols-2]) / dr;
                    } else {
                        dAz_dr = (Az[j][i+1] - Az[j][i-1]) / (2 * dr);
                    }
                } else {
                    if (j === 0) {
                        dAz_dr = (Az[1][i] - Az[0][i]) / dr;
                    } else if (j === rows - 1) {
                        dAz_dr = (Az[rows-1][i] - Az[rows-2][i]) / dr;
                    } else {
                        dAz_dr = (Az[j+1][i] - Az[j-1][i]) / (2 * dr);
                    }
                }

                Btheta[j][i] = -dAz_dr;
            }
        }

        // |B| = sqrt(Br^2 + Bθ^2)
        const B = Array(rows).fill(0).map(() => Array(cols).fill(0));
        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                B[j][i] = Math.sqrt(Br[j][i]**2 + Btheta[j][i]**2);
            }
        }

        return { Bx: Br, By: Btheta, B };

    } else {
        // Cartesian coordinates calculation (original code)
        const Bx = Array(rows).fill(0).map(() => Array(cols).fill(0));
        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                if (j === 0) {
                    Bx[j][i] = (Az[1][i] - Az[0][i]) / dy;
                } else if (j === rows - 1) {
                    Bx[j][i] = (Az[rows-1][i] - Az[rows-2][i]) / dy;
                } else {
                    Bx[j][i] = (Az[j+1][i] - Az[j-1][i]) / (2 * dy);
                }
            }
        }

        const By = Array(rows).fill(0).map(() => Array(cols).fill(0));
        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
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
}

/**
 * 統計情報の表示
 */
function displayInfo(Az, Mu, B, H) {
    const azStats = getArrayStats(Az);
    const muStats = getArrayStats(Mu);
    const bStats = getArrayStats(B);
    const hStats = getArrayStats(H);

    infoGrid.innerHTML = `
        <div class="info-item">
            <strong>配列サイズ</strong>
            ${Az.length} × ${Az[0].length}
        </div>
        <div class="info-item">
            <strong>Az範囲</strong>
            ${azStats.min.toExponential(3)} ~ ${azStats.max.toExponential(3)} Wb/m
        </div>
        <div class="info-item">
            <strong>μ範囲</strong>
            ${muStats.min.toExponential(3)} ~ ${muStats.max.toExponential(3)} H/m
        </div>
        <div class="info-item">
            <strong>|B|最大値</strong>
            ${bStats.max.toExponential(3)} T
        </div>
        <div class="info-item">
            <strong>|H|最大値</strong>
            ${hStats.max.toExponential(3)} A/m
        </div>
        <div class="info-item">
            <strong>メッシュ間隔</strong>
            dx=${document.getElementById('dx').value} m, dy=${document.getElementById('dy').value} m
        </div>
    `;

    infoPanel.style.display = 'block';
}

/**
 * 配列の統計情報を計算
 */
function getArrayStats(arr) {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let count = 0;

    for (let row of arr) {
        for (let val of row) {
            if (val < min) min = val;
            if (val > max) max = val;
            sum += val;
            count++;
        }
    }

    return {
        min,
        max,
        mean: sum / count,
        count
    };
}

/**
 * Plotlyでヒートマップをプロット
 */
function plotHeatmap(divId, data, title, colorscale) {
    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: colorscale,
        colorbar: {
            thickness: 20,
            len: 0.7
        }
    };

    const layout = {
        xaxis: { title: 'X [mesh]' },
        yaxis: { title: 'Y [mesh]', autorange: 'reversed' },
        margin: { l: 60, r: 60, t: 30, b: 60 }
    };

    Plotly.newPlot(divId, [trace], layout, { responsive: true });
}

/**
 * Azを等高線プロットで表示
 */
function plotContour(divId, data, title, colorscale) {
    const trace = {
        z: data,
        type: 'contour',
        colorscale: colorscale,
        contours: {
            coloring: 'heatmap',
            showlabels: true
        },
        colorbar: {
            thickness: 20,
            len: 0.7
        }
    };

    const layout = {
        xaxis: { title: 'X [mesh]' },
        yaxis: { title: 'Y [mesh]', autorange: 'reversed' },
        margin: { l: 60, r: 60, t: 30, b: 60 }
    };

    Plotly.newPlot(divId, [trace], layout, { responsive: true });
}

/**
 * 可視化実行
 */
async function visualize() {
    if (!azData || !muData) {
        alert('両方のCSVファイルを読み込んでください');
        return;
    }

    // サイズの確認
    if (azData.length !== muData.length || azData[0].length !== muData[0].length) {
        alert('AzとMuのサイズが一致しません');
        return;
    }

    // ローディング表示
    loading.classList.add('active');
    visualizationArea.style.display = 'none';

    // 少し遅延させて画面更新
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
        // パラメータ取得
        const dx = parseFloat(document.getElementById('dx').value);
        const dy = parseFloat(document.getElementById('dy').value);
        const colormap = document.getElementById('colormap').value;

        // 磁束密度の計算
        const { B } = calculateMagneticField(azData, dx, dy);

        // 磁界強度 H = B / μ
        const H = Array(azData.length).fill(0).map(() => Array(azData[0].length).fill(0));
        for (let j = 0; j < azData.length; j++) {
            for (let i = 0; i < azData[0].length; i++) {
                H[j][i] = B[j][i] / muData[j][i];
            }
        }

        // 情報パネルの更新
        displayInfo(azData, muData, B, H);

        // プロット作成
        plotContour('plotAz', azData, 'Vector Potential Az', colormap);
        plotHeatmap('plotMu', muData, 'Permeability μ', 'Blues');
        plotHeatmap('plotB', B, 'Magnetic Flux Density |B|', 'Hot');
        plotHeatmap('plotH', H, 'Magnetic Field |H|', 'Portland');

        // 可視化エリアを表示
        visualizationArea.style.display = 'block';

        // スクロール
        visualizationArea.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        alert(`エラーが発生しました: ${error.message}`);
        console.error(error);
    } finally {
        loading.classList.remove('active');
    }
}
