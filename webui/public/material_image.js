// ===== Material image and sliding-region preview =====

let materialImage = null;
let transientConfig = null;

// Load the material image and draw the sliding region.
async function loadAndDisplayMaterialImage() {
    try {
        // Read transient settings.
        const configResponse = await fetch('/api/get-transient-config');
        const configData = await configResponse.json();

        if (configData.success) {
            transientConfig = configData;
        }

        // Read the material image.
        const img = new Image();
        img.onload = function() {
            materialImage = img;
            drawMaterialImage();
            document.getElementById('materialImageContainer').classList.remove('hidden');
        };
        img.onerror = function() {
            console.error('Failed to load material image');
        };
        img.src = '/api/get-material-image?' + Date.now(); // Bypass the cache.
    } catch (error) {
        console.error('Error loading material image:', error);
    }
}

// Draw the material image and sliding region on the canvas.
function drawMaterialImage() {
    if (!materialImage) return;

    const canvas = document.getElementById('materialImageCanvas');
    const ctx = canvas.getContext('2d');

    // Read the image dimensions.
    const imgWidth = materialImage.width;
    const imgHeight = materialImage.height;

    // Scale to fit the preview.
    const maxWidth = 800;
    const maxHeight = 600;
    let scale = 1.0;

    if (imgWidth > maxWidth || imgHeight > maxHeight) {
        scale = Math.min(maxWidth / imgWidth, maxHeight / imgHeight);
    }

    const displayWidth = imgWidth * scale;
    const displayHeight = imgHeight * scale;

    // Set canvas dimensions.
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Draw the image.
    ctx.drawImage(materialImage, 0, 0, displayWidth, displayHeight);

    // Mark the sliding region with dashed lines.
    if (transientConfig && transientConfig.enable_sliding) {
        ctx.strokeStyle = '#FF0000';
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 2;

        const start = transientConfig.slide_region_start * scale;
        const end = transientConfig.slide_region_end * scale;

        if (transientConfig.slide_direction === 'vertical') {
            // Vertical motion: slide the selected x range along y.
            ctx.beginPath();
            ctx.moveTo(start, 0);
            ctx.lineTo(start, displayHeight);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(end, 0);
            ctx.lineTo(end, displayHeight);
            ctx.stroke();
        } else {
            // Horizontal motion: slide the selected y range along x.
            ctx.beginPath();
            ctx.moveTo(0, start);
            ctx.lineTo(displayWidth, start);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, end);
            ctx.lineTo(displayWidth, end);
            ctx.stroke();
        }

        ctx.setLineDash([]);
    }

    // Display image dimensions and scale.
    const scalePercent = (scale * 100).toFixed(1);
    const roundedWidth = Math.round(displayWidth);
    const roundedHeight = Math.round(displayHeight);
    document.getElementById('imageInfo').textContent =
        `Original: ${imgWidth}×${imgHeight} px, display: ${scalePercent}% (${roundedWidth}×${roundedHeight} px)`;
}
