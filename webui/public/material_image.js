// ===== 材質画像表示とスライド領域描画 =====

let materialImage = null;
let transientConfig = null;

// 材質画像を読み込んでスライド領域を描画
async function loadAndDisplayMaterialImage() {
    try {
        // 過渡解析設定を取得
        const configResponse = await fetch('/api/get-transient-config');
        const configData = await configResponse.json();

        if (configData.success) {
            transientConfig = configData;
        }

        // 材質画像を取得
        const img = new Image();
        img.onload = function() {
            materialImage = img;
            drawMaterialImage();
            document.getElementById('materialImageContainer').classList.remove('hidden');
        };
        img.onerror = function() {
            console.error('Failed to load material image');
        };
        img.src = '/api/get-material-image?' + Date.now(); // キャッシュ回避
    } catch (error) {
        console.error('Error loading material image:', error);
    }
}

// Canvasに材質画像とスライド領域を描画
function drawMaterialImage() {
    if (!materialImage) return;

    const canvas = document.getElementById('materialImageCanvas');
    const ctx = canvas.getContext('2d');

    // 画像サイズを取得
    const imgWidth = materialImage.width;
    const imgHeight = materialImage.height;

    // ブラウザウィンドウに収まるようにスケーリング
    const maxWidth = 800;
    const maxHeight = 600;
    let scale = 1.0;

    if (imgWidth > maxWidth || imgHeight > maxHeight) {
        scale = Math.min(maxWidth / imgWidth, maxHeight / imgHeight);
    }

    const displayWidth = imgWidth * scale;
    const displayHeight = imgHeight * scale;

    // Canvasサイズを設定
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // 画像を描画
    ctx.drawImage(materialImage, 0, 0, displayWidth, displayHeight);

    // スライド領域を点線で描画
    if (transientConfig && transientConfig.enable_sliding) {
        ctx.strokeStyle = '#FF0000';  // 赤色
        ctx.setLineDash([5, 5]);  // 点線
        ctx.lineWidth = 2;

        const start = transientConfig.slide_region_start * scale;
        const end = transientConfig.slide_region_end * scale;

        if (transientConfig.slide_direction === 'vertical') {
            // vertical: x範囲をy方向にスライド
            ctx.beginPath();
            ctx.moveTo(start, 0);
            ctx.lineTo(start, displayHeight);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(end, 0);
            ctx.lineTo(end, displayHeight);
            ctx.stroke();
        } else {
            // horizontal: y範囲をx方向にスライド
            ctx.beginPath();
            ctx.moveTo(0, start);
            ctx.lineTo(displayWidth, start);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, end);
            ctx.lineTo(displayWidth, end);
            ctx.stroke();
        }

        ctx.setLineDash([]);  // 点線解除
    }

    // 画像情報を表示
    const scalePercent = (scale * 100).toFixed(1);
    const roundedWidth = Math.round(displayWidth);
    const roundedHeight = Math.round(displayHeight);
    document.getElementById('imageInfo').textContent =
        `原寸: ${imgWidth}×${imgHeight} px, 表示: ${scalePercent}% (${roundedWidth}×${roundedHeight} px)`;
}
