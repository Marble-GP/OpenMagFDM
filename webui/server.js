const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const { exec, spawn } = require('child_process');
const multer = require('multer');
const yaml = require('js-yaml');
// NOTE: TIFF decoding is intentionally NOT done on the server. geotiff's CJS
// entry pulls in ESM-only quick-lru/web-worker, which breaks when this file
// is bundled by `pkg` (Node 18 has no CJS->ESM bridge for require()). Instead
// the server streams raw TIFF bytes and the browser decodes them via the
// vendored UMD bundle at /lib/geotiff.js.

// Count transient analysis step files in a directory listing. Accepts both
// CSV (legacy) and TIFF (v1.4 default) outputs. Same step number written in
// both formats is counted once. Used by /api/detect-steps, /api/results,
// and the analysis-output listing.
function countTransientSteps(files) {
    const ids = new Set();
    for (const f of files) {
        const m = f.match(/^step_(\d{4})\.(csv|tiff)$/);
        if (m) ids.add(m[1]);
    }
    return ids.size;
}

const app = express();
const PORT = process.env.PORT || 3000;

// pkg実行時は実行ファイルのディレクトリを使用、通常時は__dirnameの親を使用
const isPkg = typeof process.pkg !== 'undefined';
const BASE_DIR = isPkg ? path.dirname(process.execPath) : path.join(__dirname, '..');

// アップロードディレクトリの設定
const UPLOAD_DIR = path.join(BASE_DIR, 'uploads');
const SOLVER_PATH = isPkg
    ? path.join(BASE_DIR, 'MagFDMsolver.exe')  // pkg版ではexeが同じディレクトリにある
    : path.join(BASE_DIR, 'build', 'MagFDMsolver');
const CONFIG_PATH = path.join(BASE_DIR, 'sample_config.yaml');
const USER_CONFIGS_DIR = path.join(BASE_DIR, 'configs');
const OUTPUTS_DIR = path.join(BASE_DIR, 'outputs');
const USER_LIBS_DIR = path.join(BASE_DIR, 'user-libs');

// 基本ディレクトリを同期的に作成（Multerが使用する前に必要）
try {
    if (!fsSync.existsSync(UPLOAD_DIR)) fsSync.mkdirSync(UPLOAD_DIR, { recursive: true });
    if (!fsSync.existsSync(USER_CONFIGS_DIR)) fsSync.mkdirSync(USER_CONFIGS_DIR, { recursive: true });
    if (!fsSync.existsSync(OUTPUTS_DIR)) fsSync.mkdirSync(OUTPUTS_DIR, { recursive: true });
    if (!fsSync.existsSync(USER_LIBS_DIR)) fsSync.mkdirSync(USER_LIBS_DIR, { recursive: true });
} catch (error) {
    console.error('Error creating base directories:', error);
}

// ディレクトリの作成（非同期初期化関数で確実に作成）
async function initializeDirectories() {
    try {
        await fs.mkdir(UPLOAD_DIR, { recursive: true });
        await fs.mkdir(USER_CONFIGS_DIR, { recursive: true });
        await fs.mkdir(OUTPUTS_DIR, { recursive: true });
        await fs.mkdir(USER_LIBS_DIR, { recursive: true });
        console.log('Directories initialized successfully');
    } catch (error) {
        console.error('Error creating directories:', error);
    }
}

// Image management constants
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_IMAGES_PER_USER = 20;

// Running solver processes (userId -> process)
const runningProcesses = new Map();

// Async job queue (jobId -> job record)
// job: { jobId, userId, status, progress, log, resultPath, process, created, finished }
const jobs = new Map();

/**
 * Generate timestamp-based output folder name
 * @returns {string} Folder name in format: output_YYYYMMDD_HHMMSS
 */
function generateTimestampFolderName() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hour = String(now.getHours()).padStart(2, '0');
    const minute = String(now.getMinutes()).padStart(2, '0');
    const second = String(now.getSeconds()).padStart(2, '0');

    return `output_${year}${month}${day}_${hour}${minute}${second}`;
}

/**
 * Check if a folder is an analysis result
 * @param {string} folderPath - Full path to the folder
 * @returns {Promise<boolean>} True if folder contains Az subfolder and conditions.json
 */
async function isAnalysisResult(folderPath) {
    try {
        // Check for Az subfolder
        await fs.access(path.join(folderPath, 'Az'));

        // Check for conditions.json
        await fs.access(path.join(folderPath, 'conditions.json'));

        return true;
    } catch {
        return false;
    }
}

/**
 * Get or create user-specific output directory
 * @param {string} userId - User ID
 * @returns {Promise<string>} Full path to user's output directory for this run
 */
async function prepareUserOutputDirectory(userId) {
    const userIdKey = userId || 'default';
    const userOutputBase = path.join(OUTPUTS_DIR, `${userIdKey}`);

    // Create user's output base directory if it doesn't exist
    await fs.mkdir(userOutputBase, { recursive: true });

    // Generate timestamped folder for this run
    const timestampFolder = generateTimestampFolderName();
    const fullOutputPath = path.join(userOutputBase, timestampFolder);

    return fullOutputPath;
}

// Multerの設定（ファイルアップロード用）
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOAD_DIR);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// JSONボディパーサー
app.use(express.json());

// 静的ファイルの提供 (use BASE_DIR for pkg compatibility)
const PUBLIC_DIR = isPkg ? path.join(BASE_DIR, 'public') : path.join(__dirname, 'public');
const ICON_DIR = isPkg ? path.join(BASE_DIR, 'icon') : path.join(__dirname, 'icon');
app.use(express.static(PUBLIC_DIR));
app.use('/icon', express.static(ICON_DIR));

// 親ディレクトリのCSVファイルへのアクセス
app.use('/data', express.static(BASE_DIR));

// ユーザーごとのアップロード画像へのアクセス
app.use('/uploads', express.static(UPLOAD_DIR));

// ===== API エンドポイント =====

// Constants for security
const MAX_FILE_SIZE = 100 * 1024; // 100KB
const MAX_FILES_PER_USER = 20;
const USER_EXPIRY_DAYS = 366;

// Helper: Get user config directory
function getUserDir(userId) {
    // Sanitize user ID to prevent directory traversal
    const safeUserId = userId.replace(/[^a-zA-Z0-9_-]/g, '');
    return path.join(USER_CONFIGS_DIR, safeUserId);
}

// Helper: Get user uploads directory
function getUserUploadsDir(userId) {
    const safeUserId = userId.replace(/[^a-zA-Z0-9_-]/g, '');
    return path.join(UPLOAD_DIR, safeUserId);
}

// Helper: Get user material libraries directory
function getUserLibsDir(userId) {
    const safeUserId = userId.replace(/[^a-zA-Z0-9_-]/g, '');
    return path.join(USER_LIBS_DIR, safeUserId);
}

// Helper: Merge material library YAML into analysis config YAML
// Library material_presets and library materials (without rgb) become presets
// Analysis config overrides library presets if same key exists
function mergeLibraryIntoConfig(analysisYamlStr, libraryYamlStr) {
    const analysis = yaml.load(analysisYamlStr);
    const library  = yaml.load(libraryYamlStr);
    const merged   = Object.assign({}, analysis);

    // Collect library presets: explicit material_presets + materials stripped of rgb
    const libPresets = Object.assign({}, library.material_presets || {});
    for (const [name, props] of Object.entries(library.materials || {})) {
        const { rgb, ...rest } = props; // eslint-disable-line no-unused-vars
        libPresets[name] = rest;
    }

    // Analysis material_presets override library presets with the same key
    merged.material_presets = Object.assign({}, libPresets, analysis.material_presets || {});

    return yaml.dump(merged, { lineWidth: -1 });
}

// Helper: Get user-specific config file path
function getUserConfigPath(userId, fileName) {
    // Sanitize filename to prevent directory traversal
    const safeName = path.basename(fileName || 'sample_config.yaml');
    return path.join(getUserDir(userId), safeName);
}

// Helper: Initialize user directory with default config
async function initializeUserDir(userId) {
    const userDir = getUserDir(userId);
    try {
        await fs.access(userDir);
        // Directory exists
    } catch {
        // Directory doesn't exist, create it
        await fs.mkdir(userDir, { recursive: true });

        // Copy default config
        const defaultConfig = await fs.readFile(CONFIG_PATH, 'utf8');
        const defaultConfigPath = path.join(userDir, 'sample_config.yaml');
        await fs.writeFile(defaultConfigPath, defaultConfig, 'utf8');
        console.log(`Initialized new user directory: ${userId}`);
    }
}

// Helper: Enforce file count limit (keep newest 20 files)
async function enforceFileLimit(userId) {
    const userDir = getUserDir(userId);
    try {
        await ensureDir(userDir);
        const files = await fs.readdir(userDir);
        const yamlFiles = files.filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));

        if (yamlFiles.length > MAX_FILES_PER_USER) {
            // Get file stats with modification times
            const fileStats = await Promise.all(
                yamlFiles.map(async (file) => {
                    const filePath = path.join(userDir, file);
                    const stats = await fs.stat(filePath);
                    return { file, mtime: stats.mtime };
                })
            );

            // Sort by modification time (oldest first)
            fileStats.sort((a, b) => a.mtime - b.mtime);

            // Delete oldest files
            const filesToDelete = fileStats.slice(0, yamlFiles.length - MAX_FILES_PER_USER);
            for (const { file } of filesToDelete) {
                await fs.unlink(path.join(userDir, file));
                console.log(`Deleted old file: ${userId}/${file}`);
            }
        }
    } catch (error) {
        console.error(`Error enforcing file limit for ${userId}:`, error);
    }
}

// Helper: Enforce image count limit for user (LRU - delete oldest)
async function enforceImageLimit(userId) {
    const userUploadDir = getUserUploadsDir(userId);
    try {
        await fs.access(userUploadDir);
        const files = await fs.readdir(userUploadDir);
        const imageFiles = files.filter(f => /\.(png|jpg|jpeg|bmp)$/i.test(f));

        if (imageFiles.length > MAX_IMAGES_PER_USER) {
            // Get file stats with modification times
            const fileStats = await Promise.all(
                imageFiles.map(async (file) => {
                    const filePath = path.join(userUploadDir, file);
                    const stats = await fs.stat(filePath);
                    return { file, mtime: stats.mtime };
                })
            );

            // Sort by modification time (oldest first)
            fileStats.sort((a, b) => a.mtime - b.mtime);

            // Delete oldest files
            const filesToDelete = fileStats.slice(0, imageFiles.length - MAX_IMAGES_PER_USER);
            for (const { file } of filesToDelete) {
                await fs.unlink(path.join(userUploadDir, file));
                console.log(`Deleted old image: ${userId}/${file}`);
            }
        }
    } catch (error) {
        // Directory doesn't exist yet, ignore
        if (error.code !== 'ENOENT') {
            console.error(`Error enforcing image limit for ${userId}:`, error);
        }
    }
}

// Helper: Ensure directory exists, create if not
async function ensureDir(dirPath) {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
        if (error.code !== 'EEXIST') {
            console.error(`Error creating directory ${dirPath}:`, error);
        }
    }
}

// Helper: Clean up expired user directories (run on server start)
async function cleanupExpiredUsers() {
    try {
        const expiryTime = Date.now() - (USER_EXPIRY_DAYS * 24 * 60 * 60 * 1000);

        // Ensure directories exist before reading
        await ensureDir(USER_CONFIGS_DIR);
        await ensureDir(UPLOAD_DIR);

        // Clean up config directories
        const configUsers = await fs.readdir(USER_CONFIGS_DIR);
        for (const user of configUsers) {
            const userDir = path.join(USER_CONFIGS_DIR, user);
            const stats = await fs.stat(userDir);

            if (stats.isDirectory() && stats.mtime < expiryTime) {
                await fs.rm(userDir, { recursive: true, force: true });
                console.log(`Cleaned up expired user config directory: ${user}`);
            }
        }

        // Clean up upload directories
        const uploadUsers = await fs.readdir(UPLOAD_DIR);
        for (const user of uploadUsers) {
            const userUploadDir = path.join(UPLOAD_DIR, user);
            const stats = await fs.stat(userUploadDir);

            if (stats.isDirectory() && stats.mtime < expiryTime) {
                await fs.rm(userUploadDir, { recursive: true, force: true });
                console.log(`Cleaned up expired user upload directory: ${user}`);
            }
        }
    } catch (error) {
        console.error('Error cleaning up expired users:', error);
    }
}

// Run initialization and cleanup on server start
initializeDirectories().then(() => cleanupExpiredUsers());

// Get list of YAML files for a user
app.get('/api/config/list', async (req, res) => {
    try {
        const userId = req.query.userId || 'default';

        // Initialize user directory if needed
        await initializeUserDir(userId);

        const userDir = getUserDir(userId);
        const files = await fs.readdir(userDir);
        const yamlFiles = files.filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));

        res.json({
            success: true,
            files: yamlFiles
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// YAML設定ファイルの読み込み
app.get('/api/config', async (req, res) => {
    try {
        const userId = req.query.userId || 'default';
        const fileName = req.query.file || 'sample_config.yaml';

        // Initialize user directory if needed
        await initializeUserDir(userId);

        const userConfigPath = getUserConfigPath(userId, fileName);

        let configData;
        try {
            // Try to load user-specific config
            configData = await fs.readFile(userConfigPath, 'utf8');
        } catch (error) {
            // File doesn't exist, return default config
            configData = await fs.readFile(CONFIG_PATH, 'utf8');
        }

        // Return as plain text (YAML)
        res.type('text/plain').send(configData);
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// YAML設定ファイルの保存
app.post('/api/config', async (req, res) => {
    try {
        const { userId, file, content } = req.body;

        // Validate file size (100KB limit)
        const contentSize = Buffer.byteLength(content, 'utf8');
        if (contentSize > MAX_FILE_SIZE) {
            return res.status(400).json({
                success: false,
                error: `File size (${(contentSize / 1024).toFixed(1)}KB) exceeds maximum allowed size (${MAX_FILE_SIZE / 1024}KB)`
            });
        }

        // YAML validation
        try {
            yaml.load(content);
        } catch (yamlError) {
            return res.status(400).json({
                success: false,
                error: `YAML syntax error: ${yamlError.message}`
            });
        }

        // Initialize user directory if needed
        await initializeUserDir(userId || 'default');

        // Enforce file count limit
        await enforceFileLimit(userId || 'default');

        const userConfigPath = getUserConfigPath(userId || 'default', file);

        // Save to user-specific file
        await fs.writeFile(userConfigPath, content, 'utf8');

        res.json({
            success: true,
            message: 'Configuration saved successfully',
            path: userConfigPath
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Validate YAML configuration without saving
app.post('/api/validate-config', async (req, res) => {
    try {
        const { config } = req.body;
        if (!config || typeof config !== 'string') {
            return res.status(400).json({ valid: false, errors: [{ field: 'config', message: 'config field is required' }] });
        }

        const errors = [];

        // 1. YAML syntax check
        let parsed;
        try {
            parsed = yaml.load(config);
        } catch (yamlError) {
            return res.json({ valid: false, errors: [{ field: 'yaml', message: `YAML syntax error: ${yamlError.message}` }] });
        }

        if (!parsed || typeof parsed !== 'object') {
            return res.json({ valid: false, errors: [{ field: 'yaml', message: 'Config must be a YAML mapping' }] });
        }

        // 2. coordinate_system
        const cs = parsed['coordinate_system'];
        if (!cs) {
            errors.push({ field: 'coordinate_system', message: 'coordinate_system is required (cartesian or polar)' });
        } else if (!['cartesian', 'polar'].includes(cs)) {
            errors.push({ field: 'coordinate_system', message: `Unknown coordinate_system: "${cs}". Must be cartesian or polar` });
        }

        // 3. materials
        if (!parsed['materials'] || typeof parsed['materials'] !== 'object') {
            errors.push({ field: 'materials', message: 'materials section is required' });
        }

        // 4. Coordinate-system specific checks
        if (cs === 'cartesian') {
            if (!parsed['boundary_conditions']) {
                errors.push({ field: 'boundary_conditions', message: 'boundary_conditions is required for cartesian coordinate system' });
            }
            if (!parsed['mesh']) {
                errors.push({ field: 'mesh', message: 'mesh section (dx, dy) is required for cartesian coordinate system' });
            }
        } else if (cs === 'polar') {
            if (!parsed['polar_domain']) {
                errors.push({ field: 'polar_domain', message: 'polar_domain is required for polar coordinate system' });
            } else {
                const pd = parsed['polar_domain'];
                if (pd['r_start'] === undefined) errors.push({ field: 'polar_domain.r_start', message: 'polar_domain.r_start is required' });
                if (pd['r_end'] === undefined) errors.push({ field: 'polar_domain.r_end', message: 'polar_domain.r_end is required' });
                if (pd['theta_range'] === undefined) errors.push({ field: 'polar_domain.theta_range', message: 'polar_domain.theta_range is required' });
            }
        }

        res.json({ valid: errors.length === 0, errors });
    } catch (error) {
        res.status(500).json({ valid: false, errors: [{ field: 'server', message: error.message }] });
    }
});

// 画像ファイルのアップロード（ユーザーごとのディレクトリ）
app.post('/api/upload-image', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'No file uploaded'
            });
        }

        // Check file size
        if (req.file.size > MAX_IMAGE_SIZE) {
            await fs.unlink(req.file.path);
            return res.status(400).json({
                success: false,
                error: `File size (${(req.file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (${MAX_IMAGE_SIZE / 1024 / 1024}MB)`
            });
        }

        const userId = req.body.userId || 'default';
        const userUploadDir = getUserUploadsDir(userId);

        // Create user upload directory if it doesn't exist
        await fs.mkdir(userUploadDir, { recursive: true });

        // Enforce image limit before upload
        await enforceImageLimit(userId);

        // Use original filename
        const filename = req.file.originalname;
        const oldPath = req.file.path;
        const newPath = path.join(userUploadDir, filename);

        // Move file to user directory
        await fs.rename(oldPath, newPath);

        res.json({
            success: true,
            filename: filename,
            path: `/uploads/${userId}/${filename}`,
            originalName: req.file.originalname
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Detect unique colors in an image and generate YAML material template.
// Anti-aliasing detection: rare colors (≤ rareThreshold of total pixels) that can be
// expressed as a linear blend of two dominant colors are identified as AA pixels.
// Their base colors get antialias:true in the YAML template.
app.post('/api/materials/detect', upload.single('image'), async (req, res) => {
    const tmpPath = req.file ? req.file.path : null;
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, error: 'No image file provided' });
        }

        // Configurable thresholds (query params override defaults)
        const RARE_THRESHOLD  = parseFloat(req.query.rareThreshold  ?? '0.05'); // coverage ratio
        const BLEND_TOLERANCE = parseFloat(req.query.blendTolerance  ?? '8');   // per-channel [0-255]
        const MIN_COLOR_DIST  = parseFloat(req.query.minColorDist    ?? '30');  // Euclidean distance
        // Anti-aliasing blends are mixtures of *material* colours, not of other
        // AA fragments. Restricting the candidate base set to colours whose
        // coverage is >= this fraction turns the rare-colour AA check from
        // O(N^3) -- N filter passes through the full sorted list, each scanning
        // an O(N) candidate window pair-wise -- into O(N * K^2) where K is the
        // number of plausible material colours (typically 5-30 on a CAD image
        // even when the image has tens of thousands of unique RGB values from
        // anti-aliasing). On a 1197x896 outer-SPMSM screenshot this cuts the
        // endpoint runtime from ~3.5 min to <1 sec.
        const BLEND_BASE_MIN_RATIO = parseFloat(req.query.blendBaseMinRatio ?? '0.001');

        // Returns t∈[0,1] if C ≈ t·A + (1-t)·B, otherwise null.
        function isLinearBlend(C, A, B) {
            // A and B must be sufficiently different to form a meaningful pair
            const distSq = A.reduce((s, a, k) => s + (a - B[k]) ** 2, 0);
            if (distSq < MIN_COLOR_DIST * MIN_COLOR_DIST) return null;

            // Estimate t from non-degenerate channels (|A[ch]-B[ch]| > 10)
            const tEsts = [];
            for (let ch = 0; ch < 3; ch++) {
                const denom = A[ch] - B[ch];
                if (Math.abs(denom) > 10) {
                    tEsts.push((C[ch] - B[ch]) / denom);
                } else {
                    // Degenerate channel: A≈B, so C must also ≈A
                    if (Math.abs(C[ch] - A[ch]) > BLEND_TOLERANCE) return null;
                }
            }
            if (tEsts.length === 0) return null; // A≈B in all channels

            // All t estimates must be mutually consistent
            const tMin = Math.min(...tEsts), tMax = Math.max(...tEsts);
            if (tMax - tMin > 0.15) return null;

            const t = tEsts.reduce((a, b) => a + b) / tEsts.length;
            if (t < -0.05 || t > 1.05) return null; // outside blend range

            // Final residual check across all channels
            for (let ch = 0; ch < 3; ch++) {
                const expected = t * A[ch] + (1 - t) * B[ch];
                if (Math.abs(C[ch] - expected) > BLEND_TOLERANCE) return null;
            }
            return Math.max(0, Math.min(1, t));
        }

        const Jimp = require('jimp');
        const image = await Jimp.read(tmpPath);
        const totalPixels = image.bitmap.width * image.bitmap.height;

        // Count occurrences of each unique RGB color (ignore fully transparent pixels)
        const colorCounts = new Map();
        image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
            const r = image.bitmap.data[idx];
            const g = image.bitmap.data[idx + 1];
            const b = image.bitmap.data[idx + 2];
            const a = image.bitmap.data[idx + 3];
            if (a > 0) {
                const key = `${r},${g},${b}`;
                colorCounts.set(key, (colorCounts.get(key) || 0) + 1);
            }
        });

        // Sort by count descending, compute coverage ratio
        const sorted = Array.from(colorCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([key, count]) => ({
                rgb:   key.split(',').map(Number),
                count,
                ratio: count / totalPixels
            }));

        // Split: dominant = coverage > RARE_THRESHOLD, rare = candidates for AA detection
        const dominant = sorted.filter(c => c.ratio >  RARE_THRESHOLD);
        const rare     = sorted.filter(c => c.ratio <= RARE_THRESHOLD);

        // Anti-aliasing detection: a rare color is AA only if it is a linear blend
        // of two colors that BOTH have higher coverage than the rare color itself.
        // This prevents AA blends from being mistakenly used as AA bases.
        const antialiasBaseIdx = new Set(); // indices into dominant[]
        const aaBlends = [];               // detected blend records

        // Pre-restrict candidate AA bases to colours whose coverage is at
        // least BLEND_BASE_MIN_RATIO (sorted desc by ratio). On a CAD image
        // with ~40k unique RGB values from antialiasing this is typically
        // 5-30 entries; only these can plausibly be "material" colours, so
        // restricting blends to mixtures of them is both faster *and* more
        // physically meaningful than considering every higher-coverage
        // fragment.
        const blendBases = sorted.filter(c => c.ratio >= BLEND_BASE_MIN_RATIO);
        // O(1) lookup into dominant[] keyed by "r,g,b" so we don't run a
        // linear findIndex per matched blend.
        const dominantIdxByKey = new Map();
        for (let i = 0; i < dominant.length; i++) {
            const [dr, dg, db] = dominant[i].rgb;
            dominantIdxByKey.set(`${dr},${dg},${db}`, i);
        }

        const rareUnique = [];  // rare colors that are NOT AA blends → treated as materials
        let aaNoiseCount = 0;    // singletons that don't blend-match; reported as a count only
        for (const rareColor of rare) {
            let found = false;
            // blendBases is sorted desc, so once a base's ratio drops to <=
            // rareColor.ratio we can stop scanning -- everything after it
            // has equal-or-lower coverage and isn't a valid base.
            for (let i = 0; i < blendBases.length && !found; i++) {
                if (blendBases[i].ratio <= rareColor.ratio) break;
                for (let j = i + 1; j < blendBases.length && !found; j++) {
                    if (blendBases[j].ratio <= rareColor.ratio) break;
                    const t = isLinearBlend(rareColor.rgb, blendBases[i].rgb, blendBases[j].rgb);
                    if (t !== null) {
                        const [ar, ag, ab] = blendBases[i].rgb;
                        const [br, bg, bb] = blendBases[j].rgb;
                        const idxA = dominantIdxByKey.get(`${ar},${ag},${ab}`);
                        const idxB = dominantIdxByKey.get(`${br},${bg},${bb}`);
                        if (idxA != null) antialiasBaseIdx.add(idxA);
                        if (idxB != null) antialiasBaseIdx.add(idxB);
                        aaBlends.push({
                            rgb:   rareColor.rgb,
                            count: rareColor.count,
                            ratio: rareColor.ratio,
                            baseA: blendBases[i].rgb,
                            baseB: blendBases[j].rgb,
                            t:     Math.round(t * 1000) / 1000
                        });
                        found = true;
                    }
                }
            }
            if (!found) {
                // A rare colour that doesn't blend-match any pair of
                // material-class bases. We treat it as a real (minor)
                // material only if its coverage is itself >= the
                // material-class threshold; anything below that is AA
                // noise / dithering artefacts and gets dropped silently
                // (counted, not enumerated) so we don't inflate the
                // materials list with thousands of sub-ppm RGBs.
                if (rareColor.ratio >= BLEND_BASE_MIN_RATIO) {
                    rareUnique.push(rareColor);
                } else {
                    aaNoiseCount++;
                }
            }
        }

        // Build YAML template (dominant + rare-unique colors; AA bases get antialias:true)
        const toHex = ([r, g, b]) =>
            r.toString(16).padStart(2, '0') +
            g.toString(16).padStart(2, '0') +
            b.toString(16).padStart(2, '0');

        const totalMaterials = dominant.length + rareUnique.length;
        const aaTotal = aaBlends.length + aaNoiseCount;
        const lines = [
            `# Auto-generated from ${req.file.originalname}`,
            `# ${totalMaterials} material color(s) detected` +
                (rareUnique.length > 0
                    ? ` (${dominant.length} dominant + ${rareUnique.length} rare-unique)`
                    : '') +
                (aaTotal > 0
                    ? `, ${aaTotal} anti-aliasing blend(s) excluded`
                      + (aaNoiseCount > 0 ? ` (${aaNoiseCount} singleton noise)` : '')
                    : ''),
            `# Fill in mu_r and jz for each material`,
            `coordinate_system: cartesian`,
            ``,
            `materials:`
        ];
        for (let i = 0; i < dominant.length; i++) {
            const { rgb: [r, g, b], ratio } = dominant[i];
            const hex = toHex([r, g, b]);
            lines.push(`  material_${hex}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            lines.push(`    mu_r: 1.0       # Set permeability  (coverage: ${(ratio * 100).toFixed(1)}%)`);
            lines.push(`    jz: 0.0`);
            if (antialiasBaseIdx.has(i)) lines.push(`    antialias: true`);
        }
        // Rare-unique colors: small coverage but not AA blends → genuine materials
        for (const ru of rareUnique) {
            const [r, g, b] = ru.rgb;
            const hex = toHex([r, g, b]);
            lines.push(`  material_${hex}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            lines.push(`    mu_r: 1.0       # Set permeability  (coverage: ${(ru.ratio * 100).toFixed(2)}%)`);
            lines.push(`    jz: 0.0`);
        }
        // Cap the YAML AA-comment to the strongest blends so the template
        // stays readable on antialiased CAD images (full count is in the
        // header). Sort by coverage so the most visible blends come first.
        const AA_YAML_MAX = 50;
        if (aaBlends.length > 0) {
            const aaSortedDesc = aaBlends.slice().sort((a, b) => b.ratio - a.ratio);
            const shown = aaSortedDesc.slice(0, AA_YAML_MAX);
            lines.push(``, `# Anti-aliasing blends detected (excluded from materials, top ${shown.length}/${aaBlends.length}):`);
            for (const blend of shown) {
                const pct = (blend.ratio * 100).toFixed(2);
                lines.push(`#   [${blend.rgb}]  ${pct}%  =  t=${blend.t} · [${blend.baseA}]  +  (1-t) · [${blend.baseB}]`);
            }
        }

        // Clean up temp file
        await fs.unlink(tmpPath);

        // Combine dominant + rare-unique for the colors response
        const allMaterialColors = [
            ...dominant.map((c, i) => ({
                rgb:       c.rgb,
                ratio:     c.ratio,
                antialias: antialiasBaseIdx.has(i)
            })),
            ...rareUnique.map(c => ({
                rgb:       c.rgb,
                ratio:     c.ratio,
                antialias: false
            }))
        ];

        // Cap the wire payload: render-all-DOM-rows on the frontend hangs
        // for many seconds when the list has tens of thousands of entries,
        // and the frontend only displays the top-N anyway. Keep the totals
        // in the response so the UI can show "+X more not shown" if it
        // wants to.
        const AA_RESPONSE_MAX = 200;
        const aaForResponse = aaBlends.length > AA_RESPONSE_MAX
            ? aaBlends.slice().sort((a, b) => b.ratio - a.ratio).slice(0, AA_RESPONSE_MAX)
            : aaBlends;

        res.json({
            success:        true,
            colors:         allMaterialColors,
            aaBlends:       aaForResponse,
            aaBlendsTotal:  aaBlends.length + aaNoiseCount,
            aaNoiseCount,
            uniqueColors:   sorted.length,
            yamlTemplate:   lines.join('\n')
        });
    } catch (error) {
        if (tmpPath) { try { await fs.unlink(tmpPath); } catch { /* ignore */ } }
        res.status(500).json({ success: false, error: error.message });
    }
});

// ============================================================
// Polar pre-processing pipeline (v1.5)
// ============================================================
// Goal: take a Cartesian CAD screenshot of a motor cross-section and
// produce the polar-warped image that OpenMagFDM's polar analysis expects,
// together with the matching polar_domain block.
//
// detectPolarGeometry() inspects the image and returns initial geometry
// (center, inner/outer radii in px, full-circle vs sector, and N-fold
// rotational symmetry). The UI lets the user adjust those values, then
// warpPolar() rebuilds the actual polar image with nearest-neighbour
// sampling (material colours are preserved bit-exact).

const POLAR_FOREGROUND_DISTANCE = 20;   // RGB Euclidean distance > this == foreground
const POLAR_BG_CORNER_PX        = 8;    // size of corner sample for background colour
const POLAR_CIRCULARITY_THRESHOLD = 0.75;

function colorDistance(r1, g1, b1, r2, g2, b2) {
    const dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
    return Math.sqrt(dr * dr + dg * dg + db * db);
}

// Sample the four corners and return the average RGB. CAD screenshots almost
// always have a uniform background so this is enough; the .scan() call only
// touches the 4 * POLAR_BG_CORNER_PX^2 corner pixels.
function estimateBackgroundColor(jimpImage) {
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const s = POLAR_BG_CORNER_PX;
    const corners = [
        [0, 0], [W - s, 0], [0, H - s], [W - s, H - s],
    ];
    let r = 0, g = 0, b = 0, n = 0;
    for (const [x0, y0] of corners) {
        for (let dy = 0; dy < s; dy++) {
            for (let dx = 0; dx < s; dx++) {
                const idx = ((y0 + dy) * W + (x0 + dx)) * 4;
                r += jimpImage.bitmap.data[idx];
                g += jimpImage.bitmap.data[idx + 1];
                b += jimpImage.bitmap.data[idx + 2];
                n++;
            }
        }
    }
    return [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
}

// Stage 1 + 2: scan every pixel, build a foreground bitmask and bbox, then
// classify the outer shape (circular vs rectangular) via circularity.
// Returns { mask, bbox, area, circularity, shape }.
function buildForegroundMaskAndShape(jimpImage, bg) {
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const mask = new Uint8Array(W * H);
    let xmin = W, ymin = H, xmax = -1, ymax = -1;
    let area = 0;
    const data = jimpImage.bitmap.data;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const i4 = (y * W + x) * 4;
            const d = colorDistance(data[i4], data[i4 + 1], data[i4 + 2], bg[0], bg[1], bg[2]);
            if (d > POLAR_FOREGROUND_DISTANCE) {
                mask[y * W + x] = 1;
                area++;
                if (x < xmin) xmin = x; if (x > xmax) xmax = x;
                if (y < ymin) ymin = y; if (y > ymax) ymax = y;
            }
        }
    }
    if (area === 0) {
        return { mask, bbox: null, area: 0, circularity: 0, shape: 'rectangular' };
    }

    // Perimeter via 4-neighbour boundary count (any foreground pixel adjacent
    // to a background pixel or the image edge contributes).
    let perimeter = 0;
    for (let y = ymin; y <= ymax; y++) {
        for (let x = xmin; x <= xmax; x++) {
            if (!mask[y * W + x]) continue;
            const up    = y > 0     ? mask[(y - 1) * W + x] : 0;
            const down  = y < H - 1 ? mask[(y + 1) * W + x] : 0;
            const left  = x > 0     ? mask[y * W + x - 1]   : 0;
            const right = x < W - 1 ? mask[y * W + x + 1]   : 0;
            if (!up || !down || !left || !right) perimeter++;
        }
    }
    const circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;
    const bbox = { x: xmin, y: ymin, w: xmax - xmin + 1, h: ymax - ymin + 1 };
    const aspect = bbox.w / bbox.h;
    const aspectOk = aspect > 0.85 && aspect < 1.18;
    const shape = (circularity > POLAR_CIRCULARITY_THRESHOLD && aspectOk) ? 'circular' : 'rectangular';
    return { mask, bbox, area, circularity, shape };
}

// Build a center-to-foreground-pixel distance histogram (1 px bins).
// Returned shape: { hist: Uint32Array, total, maxR }. Used by both the
// radius estimator and the histogram-based shape refinement (which lets
// us spot ring topologies that fool the perimeter-based circularity).
function computeDistanceHistogram(mask, bbox, W, cx, cy) {
    if (!bbox) return { hist: new Uint32Array(1), total: 0, maxR: 0 };
    const maxR = Math.ceil(Math.hypot(bbox.w, bbox.h)) + 2;
    const hist = new Uint32Array(maxR + 1);
    let total = 0;
    for (let y = bbox.y; y < bbox.y + bbox.h; y++) {
        for (let x = bbox.x; x < bbox.x + bbox.w; x++) {
            if (!mask[y * W + x]) continue;
            const d = Math.round(Math.hypot(x - cx, y - cy));
            if (d <= maxR) { hist[d]++; total++; }
        }
    }
    return { hist, total, maxR };
}

// Locate air-gap dips in the distance histogram and return the top-N
// candidates, sorted best-first.
//
// What is a "dip" here:
//   * deep enough relative to the +/-5 px neighbourhood (excluding the
//     +/-1 px around the dip itself so a thin gap is not self-suppressing):
//     hist[r] < 0.85 of that mean -- i.e. at least a 15 % drop. The
//     threshold is intentionally loose because real CAD / paper-screenshot
//     air gaps are often only 5-8 px wide with anti-aliased edges, so
//     density drops 20-40 % rather than the 80-100 % we see on synthetic
//     fixtures or the IEEJ-D-model. A monotone slope is naturally rejected
//     because for a linear hist[r], the symmetric neighbourhood mean
//     equals hist[r], giving ratio ~= 1.
//   * bidirectional band sanity: both the inner band (r-30..r-5) and the
//     outer band (r+5..r+30) must carry >= 100 mean density. That accepts:
//       * inner rotor (rotor inside, stator outside) -- both sides dense
//       * outer rotor (stator inside, rotor ring outside) -- both sides dense
//     and rejects:
//       * bore-with-sparse-coils synthetic patterns (outer side sparse)
//       * the inside edge of a thin ring (one side empty)
//
// Score = (1 - ratio) * min(innerBand, outerBand). Deeper drop with
// denser surrounding material ranks higher. The IEEJ-D-model and
// synthetic fixtures keep ratio ~0, so the air gap clearly wins on
// those; shallower real-world dips compete only with other dips on
// the same image. After scoring we suppress neighbours within +/-5 px
// of an already-accepted candidate so a single wide gap contributes
// one entry. The caller uses the top entry as r_inner_px and exposes
// the full list as `dip_candidates` so the UI dropdown can switch when
// the highest-scored guess isn't the physical air gap (e.g. when an
// aux mid-yoke gap outscores the rotor/stator gap on a multi-gap
// design, or when two adjacent dips are both physically plausible).
function findAirGapDipCandidates(hist, r_outer, maxCandidates = 5) {
    const minMargin = 30;
    if (r_outer - minMargin < 40) return [];
    const neighWindow = 5;
    function neighbourMean(r) {
        let s = 0, n = 0;
        for (let k = r - neighWindow; k <= r - 2; k++) {
            if (k >= 0) { s += hist[k]; n++; }
        }
        for (let k = r + 2; k <= r + neighWindow; k++) {
            if (k < hist.length) { s += hist[k]; n++; }
        }
        return n > 0 ? s / n : 0;
    }
    function bandDensity(rLo, rHi) {
        let s = 0, n = 0;
        for (let k = rLo; k <= rHi; k++) {
            if (k >= 0 && k < hist.length) { s += hist[k]; n++; }
        }
        return n > 0 ? s / n : 0;
    }
    const raw = [];
    for (let r = r_outer - minMargin; r >= 40; r--) {
        const ne = neighbourMean(r);
        if (ne < 100) continue;
        const ratio = hist[r] / ne;
        if (ratio >= 0.85) continue;
        const innerBand = bandDensity(Math.max(0, r - 30), r - 5);
        const outerBand = bandDensity(r + 5, Math.min(hist.length - 1, r + 30));
        if (innerBand < 100 || outerBand < 100) continue;
        const score = (1 - ratio) * Math.min(innerBand, outerBand);
        raw.push({
            r,
            ratio: Number(ratio.toFixed(3)),
            inner_band: Math.round(innerBand),
            outer_band: Math.round(outerBand),
            score: Math.round(score),
        });
    }
    // Sort best-first, then suppress neighbours within +/- 5 px so a wide
    // gap that produces 4-5 adjacent qualifying bins only contributes one
    // entry. (We sort *before* suppressing so the best bin of each gap wins.)
    raw.sort((a, b) => b.score - a.score);
    const out = [];
    for (const c of raw) {
        if (out.some(o => Math.abs(o.r - c.r) <= 5)) continue;
        out.push(c);
        if (out.length >= maxCandidates) break;
    }
    return out;
}

// Backwards-compatible wrapper: returns the r of the best dip, or -1.
function findAirGapInner(hist, r_outer) {
    const cs = findAirGapDipCandidates(hist, r_outer, 1);
    return cs.length > 0 ? cs[0].r : -1;
}

// Histogram-based shape classifier. The perimeter-based circularity from
// Stage 2 collapses to ~0.2 on ring topologies (inner boundary inflates
// the perimeter), so the cumulative 95th-percentile rim heuristic also
// fails on rotors with heavy interior structure (slot+hole+V-magnets)
// because the cumulative threshold lands just *inside* the actual outer
// edge. Detect the rim directly as "the highest-density bin in the outer
// half of the histogram"; that is robust to internal complexity because
// the outer edge is always a thin annulus of high density regardless of
// what happens at smaller radii. Then promote to circular if either
//   (a) the +/-3 px ring around that peak holds >= 4 % of the foreground
//       (classical "thin ring" case), or
//   (b) the peak density is >= 1.5x the average per-bin density in the
//       30-80 % radius band (the IEEJ-D-model case: dense interior but
//       still a sharp outer circle).
function refineShapeFromHistogram({ hist, total, maxR }, autoShape) {
    if (total === 0 || maxR < 10) return autoShape;

    // The histogram is sized to the bbox diagonal, but the foreground itself
    // tops out at the actual outer radius. Find that extent first, then
    // search for the rim peak in the upper half of [0, outerExtent].
    let outerExtent = 0;
    for (let r = maxR; r >= 0; r--) {
        if (hist[r] > 0) { outerExtent = r; break; }
    }
    if (outerExtent < 10) return autoShape;
    const half = Math.floor(outerExtent / 2);
    let rimR = half, rimV = 0;
    for (let r = half; r <= outerExtent; r++) {
        if (hist[r] > rimV) { rimV = hist[r]; rimR = r; }
    }
    if (rimV === 0) return autoShape;

    const rimLo = Math.max(0, rimR - 3);
    const rimHi = Math.min(maxR, rimR + 3);
    let rim = 0;
    for (let r = rimLo; r <= rimHi; r++) rim += hist[r];
    const rimRatio = rim / total;
    if (rimRatio >= 0.04) return 'circular';

    const r30 = Math.floor(rimR * 0.30);
    const r80 = Math.floor(rimR * 0.80);
    let interior = 0, interiorBins = 0;
    for (let r = r30; r <= r80; r++) {
        if (r >= rimLo && r <= rimHi) continue;
        interior += hist[r];
        interiorBins++;
    }
    if (interiorBins <= 0) return autoShape;
    const interiorDensity = interior / interiorBins;
    if (interiorDensity > 0 && rimV / interiorDensity >= 1.5) return 'circular';

    return autoShape;
}

// Stage 3: center + r_inner_px + r_outer_px from the foreground mask.
// For circular shapes we walk the distance histogram. For rectangular ones
// we fall back to the bbox center and look for an interior hole (rotor air).
function estimateCenterAndRadii(mask, bbox, W, H, shape, precomputedHist) {
    if (!bbox) {
        return { center_x: Math.floor(W / 2), center_y: Math.floor(H / 2),
                 r_inner_px: 0, r_outer_px: Math.floor(Math.min(W, H) / 2 - 5) };
    }
    let cx = Math.round(bbox.x + bbox.w / 2);
    let cy = Math.round(bbox.y + bbox.h / 2);

    if (shape === 'circular') {
        const { hist, total, maxR } =
            precomputedHist || computeDistanceHistogram(mask, bbox, W, cx, cy);
        if (total === 0) return { center_x: cx, center_y: cy, r_inner_px: 0, r_outer_px: 0 };
        // r_outer: highest-density bin in the upper half of the actual
        //          foreground extent. Robust to dense interiors.
        let outerExtent = 0;
        for (let r = maxR; r >= 0; r--) {
            if (hist[r] > 0) { outerExtent = r; break; }
        }
        const half = Math.floor(outerExtent / 2);
        let r_outer = outerExtent, peak = 0;
        for (let r = half; r <= outerExtent; r++) {
            if (hist[r] > peak) { peak = hist[r]; r_outer = r; }
        }
        // r_inner: prefer the air-gap dip (the strongest histogram valley
        //          between rotor and stator -- best score across the
        //          bidirectional candidate list) so the polar warp covers
        //          the actual airgap-to-stator-OD band. Fall back to the 5 %
        //          cumulative percentile if no clear dip is detected; the
        //          UI also exposes the full candidate list so the user can
        //          switch (useful on outer-rotor / multi-airgap topologies
        //          where the highest-scored dip isn't always the physical
        //          air gap).
        const dipCandidates = findAirGapDipCandidates(hist, r_outer, 5);
        let r_inner;
        let dipFallback = false;
        if (dipCandidates.length > 0) {
            r_inner = dipCandidates[0].r;
        } else {
            let cum = 0;
            r_inner = 0;
            for (let r = 0; r <= maxR; r++) {
                cum += hist[r];
                if (cum >= total * 0.05) { r_inner = r; break; }
            }
            dipFallback = true;
        }
        return {
            center_x: cx, center_y: cy,
            r_inner_px: r_inner, r_outer_px: r_outer,
            dip_candidates: dipCandidates,
            dip_fallback: dipFallback,
        };
    }

    // rectangular: a square/rectangular stator outline does not encode the
    // physical rotation axis directly, but on every real motor cross-section
    // (a) the stator is roughly centered in the bbox, and (b) the rotor sits
    // along the rotation axis. We therefore seed with the bbox center; if a
    // rotor exists it shows up as the dominant circular signal in the Hough
    // refinement (Stage 3.5), which will snap the seed to it. A previous
    // "largest interior hole" heuristic mis-identified the bore-shaped
    // connected component on a fixture with a centered rotor, so it's gone.
    const r_outer = Math.max(5, Math.floor(Math.min(bbox.w, bbox.h) / 2 - 5));
    return { center_x: cx, center_y: cy, r_inner_px: 0, r_outer_px: r_outer };
}

// ---- Stage 3.5: Hough circle refinement (Phase 5b.6) ----
// Stage 3 gets the center "in the right neighbourhood" from bbox or hole
// flood-fill. For rectangular outer shapes that can still be 50+ pixels
// off the physical rotation axis (rotor offset inside a square stator).
// We refine by Sobel + a tiny forward-voting Hough that scores integrated
// edge strength along each candidate (cx, cy, r) within a small window
// around the coarse seed. Cost is bounded because both search ranges are
// tight (~21x21 center * ~7 radii * 90 angles = ~130k ops, ~5 ms).

// Sobel-magnitude edge map on the grayscale view of the input. Returned
// as a Float32Array of length W*H; the border row/col stay 0.
function computeEdgeMap(jimpImage) {
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const data = jimpImage.bitmap.data;
    const gray = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) {
        const i4 = i * 4;
        gray[i] = 0.299 * data[i4] + 0.587 * data[i4 + 1] + 0.114 * data[i4 + 2];
    }
    const edge = new Float32Array(W * H);
    for (let y = 1; y < H - 1; y++) {
        for (let x = 1; x < W - 1; x++) {
            const i = y * W + x;
            const gx = -gray[i - W - 1] + gray[i - W + 1]
                     - 2 * gray[i - 1] + 2 * gray[i + 1]
                     - gray[i + W - 1] + gray[i + W + 1];
            const gy = -gray[i - W - 1] - 2 * gray[i - W] - gray[i - W + 1]
                     + gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1];
            edge[i] = Math.sqrt(gx * gx + gy * gy);
        }
    }
    return edge;
}

// Precomputed unit-circle samples used by the Hough scoring loop. 90 angles
// are enough: at the radii we score (100-300 px) the angular resolution
// (4 deg) maps to ~7-21 px arc steps, so we still touch most edge pixels
// while keeping the per-trial cost down.
const HOUGH_ANGLES = 90;
const HOUGH_COS = new Float64Array(HOUGH_ANGLES);
const HOUGH_SIN = new Float64Array(HOUGH_ANGLES);
for (let a = 0; a < HOUGH_ANGLES; a++) {
    HOUGH_COS[a] = Math.cos(2 * Math.PI * a / HOUGH_ANGLES);
    HOUGH_SIN[a] = Math.sin(2 * Math.PI * a / HOUGH_ANGLES);
}

// Forward-voting Hough: scan a small (cx, cy, r) cube around the seed
// and pick the trial whose integrated edge strength (normalised by the
// number of in-bounds samples) is highest. Returns {center_x, center_y,
// r_px, score} or null if the search was no better than ~the seed.
function houghRefineCircle(edge, W, H, cx0, cy0, r0, opts) {
    const centerHalf = opts.centerHalfWindow != null ? opts.centerHalfWindow : 15;
    const radiusHalf = opts.radiusHalfWindow != null ? opts.radiusHalfWindow : 3;
    let bestScore = -1, bestCx = cx0, bestCy = cy0, bestR = r0;
    const cxLo = Math.max(0, cx0 - centerHalf);
    const cxHi = Math.min(W - 1, cx0 + centerHalf);
    const cyLo = Math.max(0, cy0 - centerHalf);
    const cyHi = Math.min(H - 1, cy0 + centerHalf);
    const rLo  = Math.max(5, r0 - radiusHalf);
    const rHi  = Math.max(rLo, r0 + radiusHalf);
    for (let cy = cyLo; cy <= cyHi; cy++) {
        for (let cx = cxLo; cx <= cxHi; cx++) {
            for (let r = rLo; r <= rHi; r++) {
                let s = 0, hits = 0;
                for (let a = 0; a < HOUGH_ANGLES; a++) {
                    const x = (cx + r * HOUGH_COS[a]) | 0;
                    const y = (cy + r * HOUGH_SIN[a]) | 0;
                    if (x < 0 || x >= W || y < 0 || y >= H) continue;
                    s += edge[y * W + x];
                    hits++;
                }
                if (hits < HOUGH_ANGLES * 0.5) continue;  // mostly off-image: skip
                const norm = s / hits;
                if (norm > bestScore) {
                    bestScore = norm; bestCx = cx; bestCy = cy; bestR = r;
                }
            }
        }
    }
    if (bestScore < 0) return null;
    return { center_x: bestCx, center_y: bestCy, r_px: bestR, score: bestScore };
}

// Apply Hough refinement to the appropriate circle for the given shape:
//   circular   -> refine the outer ring  (r_outer_px)
//   rectangular w/ rotor hole -> refine the inner hole (r_inner_px) so the
//                                center jumps to the actual rotation axis
// We only accept the refined values if their Hough score beats the score
// of the seed by a meaningful margin (>= 10 %): otherwise we trust the
// coarse seed (avoids drifting on rectangular/no-circle images).
function refineWithHough(jimpImage, coarse, shape) {
    if (!coarse || coarse.r_outer_px <= 0) return coarse;
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const edge = computeEdgeMap(jimpImage);

    // For a centered circular outline the bbox already pins the center
    // within a few pixels, so a small 20-px window is enough. For a
    // rectangular stator the bbox center is centered relative to the
    // rectangle itself but the rotation axis (= rotor center) can be way
    // off; we widen the search to bracket the likely rotor position. Also
    // sweep a wider band of radii: the bbox-inscribed-circle seed will be
    // a few tens of pixels larger than the actual rotor.
    const isRect = (shape === 'rectangular');
    const centerHalf = isRect ? 80 : 20;
    const radiusHalf = isRect ? 30 : 4;

    // For rectangular the seed radius is "inscribed bbox / 2" which is too
    // large to score the rotor; sweep down from there. We pick the seed
    // radius midway between a plausible rotor (~r_outer/3) and the bbox
    // inscribed value so that radiusHalf brackets both extremes.
    const seedR = isRect
        ? Math.round((coarse.r_outer_px + Math.max(15, coarse.r_outer_px * 0.3)) / 2)
        : coarse.r_outer_px;

    const seed = houghRefineCircle(edge, W, H,
        coarse.center_x, coarse.center_y, seedR,
        { centerHalfWindow: 0, radiusHalfWindow: 0 });
    const refined = houghRefineCircle(edge, W, H,
        coarse.center_x, coarse.center_y, seedR,
        { centerHalfWindow: centerHalf, radiusHalfWindow: radiusHalf });
    if (!seed || !refined) return coarse;
    if (refined.score < seed.score * 1.1) return coarse;

    // Carry the refined values back. For a rectangular outline the Hough
    // search was deliberately tuned to lock on to the rotor surface, so the
    // refined radius becomes r_inner_px; we keep the bbox-inscribed value
    // for r_outer_px. For a circular outline the refined radius is the
    // outer ring itself; if there was an inner radius from Stage 3 we
    // rescale it proportionally to preserve the user-visible ratio.
    const out = { ...coarse, center_x: refined.center_x, center_y: refined.center_y };
    if (isRect) {
        out.r_inner_px = refined.r_px;
        // r_outer is whatever the bbox suggested; the user can adjust.
    } else {
        const scale = refined.r_px / Math.max(1, coarse.r_outer_px);
        out.r_outer_px = refined.r_px;
        out.r_inner_px = coarse.r_inner_px > 0
            ? Math.max(1, Math.round(coarse.r_inner_px * scale))
            : 0;
        // dip candidates were computed in the pre-Hough histogram frame;
        // rescale so the r values stay meaningful in the refined frame.
        if (Array.isArray(coarse.dip_candidates) && coarse.dip_candidates.length > 0) {
            out.dip_candidates = coarse.dip_candidates.map(c => ({
                ...c,
                r: Math.max(1, Math.round(c.r * scale)),
            }));
        }
    }
    out.hough_score = refined.score;
    out.hough_seed_score = seed.score;
    return out;
}

// 360-point DFT computed directly. Picked over a zero-padded radix-2 FFT
// because:
//   * the source signal is angular bins (period == 360), so integer bin
//     indices already line up with integer N-fold symmetries -- no fractional
//     frequencies due to a mismatched FFT length;
//   * with cos/sin tables the O(N^2/2) cost on N=360 is ~5 ms, well below
//     anything user-visible.
const DFT_N = 360;
const DFT_COS = new Float64Array(DFT_N);
const DFT_SIN = new Float64Array(DFT_N);
for (let i = 0; i < DFT_N; i++) {
    DFT_COS[i] = Math.cos(-2 * Math.PI * i / DFT_N);
    DFT_SIN[i] = Math.sin(-2 * Math.PI * i / DFT_N);
}
function fftMagnitude(real360) {
    // Returns magnitude for k in [0, N/2]; the rest is the complex conjugate
    // mirror because the input is real.
    const N = DFT_N;
    const half = N / 2;
    const mag = new Float64Array(N);
    for (let k = 0; k <= half; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < N; n++) {
            const idx = (k * n) % N;
            re += real360[n] * DFT_COS[idx];
            im += real360[n] * DFT_SIN[idx];
        }
        mag[k] = Math.sqrt(re * re + im * im);
    }
    return mag;
}

// Jacobsen estimator: given the magnitude spectrum and the integer bin
// index of the peak, return a refined non-integer peak position. Falls
// back to `kPeak` if neighbouring bins are degenerate.
function jacobsenInterpolate(mag, kPeak) {
    if (kPeak <= 0 || kPeak >= mag.length - 1) return kPeak;
    const xm = mag[kPeak - 1], x0 = mag[kPeak], xp = mag[kPeak + 1];
    const denom = 2 * x0 - xm - xp;
    if (Math.abs(denom) < 1e-12) return kPeak;
    return kPeak + 0.5 * (xm - xp) / denom;
}

// Build a 360-bin angular signature of the [r_inner, r_outer] ring.
//   mode === 'grayscale' -> 1 scalar per bin (luminance Y), keeps
//                           geometric periods (e.g. 24 slots is N=24
//                           even if the slots are coloured U/V/W).
//   mode === 'rgb'       -> the L2 norm of the (R, G, B) bin mean,
//                           which surfaces colour-coded periods like
//                           a 3-phase coil pattern showing up as N=3.
//
// The function returns { values: Float64Array(360), bg_value, has_data:
// boolean } -- `bg_value` is the bin-average of empty bins (mean of
// filled ones) so we don't punch holes in the signal.
function buildAngularSignature(jimpImage, mask, geom, mode) {
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const cx = geom.center_x, cy = geom.center_y;
    // Trim 8 % off each side of the ring: the inside boundary (rotor surface)
    // and the outer boundary (stator OD) are angularly uniform on a typical
    // motor cross-section and only add DC noise. Restricting to the middle
    // 84 % keeps the coil / pole signal dominant in the spectrum.
    const r0 = Math.max(0, geom.r_inner_px);
    const r1 = Math.max(r0 + 1, geom.r_outer_px);
    const margin = (r1 - r0) * 0.08;
    const rMin = r0 + margin;
    const rMax = r1 - margin;
    const N_BIN = 360;
    const sumR = new Float64Array(N_BIN);
    const sumG = new Float64Array(N_BIN);
    const sumB = new Float64Array(N_BIN);
    const cnt  = new Uint32Array(N_BIN);
    const data = jimpImage.bitmap.data;
    const xmin = Math.max(0, Math.floor(cx - rMax - 1));
    const xmax = Math.min(W - 1, Math.ceil(cx + rMax + 1));
    const ymin = Math.max(0, Math.floor(cy - rMax - 1));
    const ymax = Math.min(H - 1, Math.ceil(cy + rMax + 1));
    let filled = 0;
    for (let y = ymin; y <= ymax; y++) {
        for (let x = xmin; x <= xmax; x++) {
            if (!mask[y * W + x]) continue;
            const dx = x - cx, dy = y - cy;
            const r2 = dx * dx + dy * dy;
            if (r2 < rMin * rMin || r2 > rMax * rMax) continue;
            let t = Math.atan2(dy, dx);
            if (t < 0) t += 2 * Math.PI;
            const bin = Math.floor((t / (2 * Math.PI)) * N_BIN) % N_BIN;
            const i4 = (y * W + x) * 4;
            sumR[bin] += data[i4];
            sumG[bin] += data[i4 + 1];
            sumB[bin] += data[i4 + 2];
            cnt[bin]++;
        }
    }
    for (let b = 0; b < N_BIN; b++) if (cnt[b] > 0) filled++;
    const values = new Float64Array(N_BIN);
    let runningSum = 0, runningCnt = 0;
    for (let b = 0; b < N_BIN; b++) {
        if (cnt[b] === 0) { values[b] = NaN; continue; }
        const r = sumR[b] / cnt[b];
        const g = sumG[b] / cnt[b];
        const bl = sumB[b] / cnt[b];
        const v = (mode === 'grayscale')
            ? (0.299 * r + 0.587 * g + 0.114 * bl)
            : Math.sqrt(r * r + g * g + bl * bl);
        values[b] = v;
        runningSum += v;
        runningCnt++;
    }
    // Backfill NaN bins with the overall mean so the FFT sees a smooth signal.
    const mean = runningCnt > 0 ? runningSum / runningCnt : 0;
    for (let b = 0; b < N_BIN; b++) if (!Number.isFinite(values[b])) values[b] = mean;
    // DC-subtract so the period peaks dominate the spectrum.
    for (let b = 0; b < N_BIN; b++) values[b] -= mean;
    return { values, mean, filled, has_data: filled >= N_BIN * 0.5 };
}

// Scan the magnitude spectrum for peaks corresponding to N in [Nmin..Nmax]
// where a "peak" means a strict local max with at least minSNR signal-
// to-baseline ratio. The Jacobsen interpolation gives the non-integer
// k position; non-integer N = k_true (since N_FFT = 512 and the source
// signal is 360 long, the relationship is N = k_true * 360 / N_FFT, but
// because we zero-padded, k_true directly corresponds to "cycles within
// the original 360 bins" which is what we want).
function findPeakCandidates(mag, Nmin, Nmax, minSNR) {
    // baseline = median over bins 1..N/2 (DC and Nyquist excluded)
    const half = mag.length / 2;
    const buf = mag.slice(1, half);
    buf.sort();
    const baseline = buf[Math.floor(buf.length / 2)] || 1e-9;
    const candidates = [];
    // Because N_FFT = 512 and the source has 360 samples, the relevant
    // "N-fold" bin is k = N * 512 / 360. Search a contiguous window for
    // each integer N to find the best local max, then refine.
    // With the direct 360-point DFT, integer bin k corresponds exactly to
    // N=k cycles per signal, so the per-N search window is just the bin
    // itself plus immediate neighbours for parabolic interpolation.
    for (let N = Nmin; N <= Nmax; N++) {
        const k = N;
        if (k < 1 || k >= half) continue;
        if (mag[k] < mag[k - 1] || mag[k] < mag[k + 1]) continue;  // not a local max
        const snr = mag[k] / baseline;
        if (snr < minSNR) continue;
        const kTrue = jacobsenInterpolate(mag, k);
        candidates.push({
            n_fold: Number(kTrue.toFixed(3)),
            n_integer: Math.round(kTrue),
            confidence: Number(Math.min(1, snr / 20).toFixed(3)),
            snr: Number(snr.toFixed(2)),
            peak_bin: k,
        });
    }
    // sort by SNR desc
    candidates.sort((a, b) => b.snr - a.snr);
    return candidates;
}

// Run FFT-based periodicity detection on a single signature mode.
// Grouped signature: per-bin mean of *group IDs* over pixels whose exact
// RGB matches one of the user-supplied colour groups. Bins that contain
// no grouped pixels are backfilled with the overall mean (same backfill
// trick as the grayscale/rgb signature) so the DFT input stays smooth.
// `colorMap` is a Map<rgbKey, groupIndex> (rgbKey = (r<<16)|(g<<8)|b).
function buildGroupedAngularSignature(jimpImage, mask, geom, colorMap) {
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const cx = geom.center_x, cy = geom.center_y;
    const r0 = Math.max(0, geom.r_inner_px);
    const r1 = Math.max(r0 + 1, geom.r_outer_px);
    const margin = (r1 - r0) * 0.08;
    const rMin = r0 + margin;
    const rMax = r1 - margin;
    const N_BIN = 360;
    const sumG = new Float64Array(N_BIN);
    const cntG = new Uint32Array(N_BIN);
    const data = jimpImage.bitmap.data;
    const xmin = Math.max(0, Math.floor(cx - rMax - 1));
    const xmax = Math.min(W - 1, Math.ceil(cx + rMax + 1));
    const ymin = Math.max(0, Math.floor(cy - rMax - 1));
    const ymax = Math.min(H - 1, Math.ceil(cy + rMax + 1));
    for (let y = ymin; y <= ymax; y++) {
        for (let x = xmin; x <= xmax; x++) {
            if (!mask[y * W + x]) continue;
            const dx = x - cx, dy = y - cy;
            const r2 = dx * dx + dy * dy;
            if (r2 < rMin * rMin || r2 > rMax * rMax) continue;
            const i4 = (y * W + x) * 4;
            const key = (data[i4] << 16) | (data[i4 + 1] << 8) | data[i4 + 2];
            const gid = colorMap.get(key);
            if (gid === undefined) continue;
            let t = Math.atan2(dy, dx);
            if (t < 0) t += 2 * Math.PI;
            const bin = Math.floor((t / (2 * Math.PI)) * N_BIN) % N_BIN;
            sumG[bin] += gid;
            cntG[bin]++;
        }
    }
    let filled = 0;
    const values = new Float64Array(N_BIN);
    let runningSum = 0, runningCnt = 0;
    for (let b = 0; b < N_BIN; b++) {
        if (cntG[b] === 0) { values[b] = NaN; continue; }
        filled++;
        const v = sumG[b] / cntG[b];
        values[b] = v;
        runningSum += v;
        runningCnt++;
    }
    const mean = runningCnt > 0 ? runningSum / runningCnt : 0;
    for (let b = 0; b < N_BIN; b++) if (!Number.isFinite(values[b])) values[b] = mean;
    for (let b = 0; b < N_BIN; b++) values[b] -= mean;
    return { values, mean, filled, has_data: filled >= N_BIN * 0.25 };
}

function periodicityForMode(jimpImage, mask, geom, mode, Nmin, Nmax) {
    const sig = buildAngularSignature(jimpImage, mask, geom, mode);
    if (!sig.has_data) return { n_fold: null, confidence: 0, alternatives: [] };
    const mag = fftMagnitude(sig.values);
    const candidates = findPeakCandidates(mag, Nmin, Nmax, 4 /* minSNR */);
    if (candidates.length === 0) return { n_fold: null, confidence: 0, alternatives: [] };
    // Suppress multiples of a stronger lower-N candidate AND suppress
    // candidates that latch on to the same spectral peak (because the
    // per-N search windows overlap, a real peak in between two integer
    // Ns can be picked up by both -- we only want the strongest one).
    const kept = [];
    for (const c of candidates) {
        const sameBin = kept.some(k => Math.abs(k.peak_bin - c.peak_bin) < 1.5);
        if (sameBin) continue;
        const dominated = kept.some(k =>
            k.n_integer > 0 && c.n_integer % k.n_integer === 0 && c.n_integer !== k.n_integer);
        if (!dominated) kept.push(c);
    }
    const top = kept[0];
    return {
        n_fold: top.n_fold,
        n_integer: top.n_integer,
        confidence: top.confidence,
        snr: top.snr,
        sector_theta_range: Number((2 * Math.PI / top.n_fold).toFixed(6)),
        alternatives: kept.slice(1, 3).map(c => ({
            n_fold: c.n_fold, n_integer: c.n_integer,
            confidence: c.confidence, snr: c.snr,
        })),
    };
}

// Stage 4: rotational symmetry detection.
// Runs grayscale + RGB-magnitude signatures in parallel:
//   - grayscale picks up geometric period (slot count even for 3-phase coil
//     colouring)
//   - rgb     picks up colour-coded period (3-phase phases, N/S magnets)
// Also computes a `recommended_ntheta` so the UI can default to a value
// where each detected sector lands on an integer number of output pixels.
function detectPeriodicity(jimpImage, mask, geom, colorGroups) {
    const rMin = Math.max(0, geom.r_inner_px);
    const rMax = Math.max(rMin + 1, geom.r_outer_px);
    if (rMax - rMin < 5) {
        return {
            grayscale: { n_fold: null, confidence: 0, alternatives: [] },
            rgb:       { n_fold: null, confidence: 0, alternatives: [] },
            recommended_ntheta: null,
        };
    }

    // Bound the N search by the geometric "smallest resolvable" period: at
    // r_outer_px we want at least ~5 px per sector, otherwise sampling is
    // hopeless. We still cap at 64 as a sanity ceiling.
    const Nmax = Math.min(64, Math.floor((2 * Math.PI * rMax) / 5));
    const Nmin = 2;

    const grayscale = periodicityForMode(jimpImage, mask, geom, 'grayscale', Nmin, Nmax);
    const rgb       = periodicityForMode(jimpImage, mask, geom, 'rgb',       Nmin, Nmax);

    // Optional "grouped" mode (Phase 5d.3). Each group is an array of
    // RGB triples that share the same group id; the user assigns these
    // through the colour-chip dropdowns in the Polar Preprocess modal.
    // For a 3-phase outer-rotor SPMSM where the user groups all phase
    // colours under one id, the dominant period drops from N=12 (the
    // raw slot count) to N=12/3=4 (the geometric phase symmetry).
    let grouped = null;
    if (Array.isArray(colorGroups) && colorGroups.length > 0) {
        const colorMap = new Map();
        for (let i = 0; i < colorGroups.length; i++) {
            const list = Array.isArray(colorGroups[i].colors) ? colorGroups[i].colors : [];
            for (const rgb of list) {
                if (!Array.isArray(rgb) || rgb.length < 3) continue;
                const r = rgb[0] | 0, g = rgb[1] | 0, b = rgb[2] | 0;
                const key = (r << 16) | (g << 8) | b;
                colorMap.set(key, i);
            }
        }
        if (colorMap.size > 0) {
            const sig = buildGroupedAngularSignature(jimpImage, mask, geom, colorMap);
            if (sig.has_data) {
                const mag = fftMagnitude(sig.values);
                const cands = findPeakCandidates(mag, Nmin, Nmax, 4);
                const kept = [];
                for (const c of cands) {
                    if (kept.some(k => Math.abs(k.peak_bin - c.peak_bin) < 1.5)) continue;
                    if (kept.some(k => k.n_integer > 0
                        && c.n_integer % k.n_integer === 0
                        && c.n_integer !== k.n_integer)) continue;
                    kept.push(c);
                }
                if (kept.length > 0) {
                    const top = kept[0];
                    grouped = {
                        n_fold: top.n_fold,
                        n_integer: top.n_integer,
                        confidence: top.confidence,
                        snr: top.snr,
                        sector_theta_range: Number((2 * Math.PI / top.n_fold).toFixed(6)),
                        group_count: colorGroups.length,
                        matched_colors: colorMap.size,
                        alternatives: kept.slice(1, 3).map(c => ({
                            n_fold: c.n_fold, n_integer: c.n_integer,
                            confidence: c.confidence, snr: c.snr,
                        })),
                    };
                } else {
                    grouped = { n_fold: null, confidence: 0, group_count: colorGroups.length,
                                matched_colors: colorMap.size, alternatives: [] };
                }
            } else {
                grouped = { n_fold: null, confidence: 0, group_count: colorGroups.length,
                            matched_colors: colorMap.size, alternatives: [], reason: 'insufficient grouped pixels in ring' };
            }
        }
    }

    // Recommended ntheta: take the grayscale N (geometric), round it to the
    // nearest integer, then snap a naive 2*PI*r_outer total length so each
    // sector ends up with an integer pixel count. Falls back to RGB if
    // grayscale finds nothing.
    const dominant = grayscale.n_integer || rgb.n_integer;
    let recommended_ntheta = null;
    if (dominant && dominant >= 2) {
        const naive = Math.round(2 * Math.PI * rMax);
        const perSector = Math.max(2, Math.round(naive / dominant));
        recommended_ntheta = perSector * dominant;
    }

    return { grayscale, rgb, grouped, recommended_ntheta };
}

// Composite: stages 1-4.
function detectPolarGeometry(jimpImage, hint, colorGroups) {
    const W = jimpImage.bitmap.width;
    const H = jimpImage.bitmap.height;
    const bg = estimateBackgroundColor(jimpImage);
    const stage12 = buildForegroundMaskAndShape(jimpImage, bg);
    const { mask, bbox, area, circularity, shape: perimeterShape } = stage12;

    // Stage 2.5: histogram-based shape refinement. We need a provisional
    // center for the histogram; bbox center is good enough at this stage.
    let provisionalShape = perimeterShape;
    let provisionalCx = bbox ? Math.round(bbox.x + bbox.w / 2) : Math.floor(W / 2);
    let provisionalCy = bbox ? Math.round(bbox.y + bbox.h / 2) : Math.floor(H / 2);
    const histPre = computeDistanceHistogram(mask, bbox, W, provisionalCx, provisionalCy);
    if (perimeterShape === 'rectangular') {
        provisionalShape = refineShapeFromHistogram(histPre, perimeterShape);
    }
    const autoShape = provisionalShape;
    const shape = (hint && hint.shape && hint.shape !== 'auto') ? hint.shape : autoShape;

    // Stage 3: center + radii. Pass the pre-computed histogram so we don't
    // walk the foreground a second time when shape ends up being circular.
    const coarse = estimateCenterAndRadii(mask, bbox, W, H, shape, histPre);

    // Stage 3.5: Hough refinement (Phase 5b.6). Coarse-to-fine: only a tiny
    // window around the seed gets scored, so the cost stays in the ms range
    // and the result either confirms the seed or snaps it to the actual
    // outer circle / rotor bore.
    const refined = refineWithHough(jimpImage, coarse, shape);
    const { center_x, center_y, r_inner_px, r_outer_px,
            hough_score, hough_seed_score } = refined;
    const dip_candidates = Array.isArray(refined.dip_candidates) ? refined.dip_candidates : [];
    const dip_fallback = !!refined.dip_fallback;

    // Stage 4: rotational symmetry over the [r_inner, r_outer] ring.
    const periodicity = detectPeriodicity(jimpImage, mask, {
        center_x, center_y, r_inner_px, r_outer_px,
    }, colorGroups);

    return {
        shape,
        auto_shape: autoShape,
        perimeter_shape: perimeterShape,
        center_x, center_y,
        r_inner_px, r_outer_px,
        dip_candidates,
        dip_fallback,
        hough: hough_score != null
            ? { score: Number(hough_score.toFixed(2)),
                seed_score: Number(hough_seed_score.toFixed(2)),
                gain: Number((hough_score / Math.max(1e-9, hough_seed_score)).toFixed(2)) }
            : null,
        is_full_circle: true,
        theta_start: 0,
        theta_end: 2 * Math.PI,
        bg_color: bg,
        foreground_area: area,
        circularity: Number(circularity.toFixed(3)),
        image_width: W,
        image_height: H,
        periodicity,
    };
}

// Nearest-neighbour Cartesian -> Polar warp. Output layout:
//   r_orientation == "horizontal" -> (rows=ntheta, cols=nr)
//   r_orientation == "vertical"   -> (rows=nr,     cols=ntheta)
// Pixels falling outside the source remain alpha=0 (transparent).
function warpPolar(srcJimp, opts) {
    const Jimp = require('jimp');
    const {
        center_x, center_y,
        r_start_px, r_end_px,
        theta_start, theta_end,
        nr, ntheta, r_orientation,
    } = opts;

    const theta_range = theta_end - theta_start;
    const dr     = nr > 1 ? (r_end_px - r_start_px) / (nr - 1) : 0;
    const dtheta = ntheta > 0 ? theta_range / ntheta : 0;
    const horizontal = (r_orientation === 'horizontal');
    const outW = horizontal ? nr     : ntheta;
    const outH = horizontal ? ntheta : nr;

    const out = new Jimp(outW, outH, 0x00000000);
    const src = srcJimp.bitmap.data;
    const srcW = srcJimp.bitmap.width;
    const srcH = srcJimp.bitmap.height;
    const dst = out.bitmap.data;

    for (let j = 0; j < outH; j++) {
        for (let i = 0; i < outW; i++) {
            const r_idx = horizontal ? i : j;
            const t_idx = horizontal ? j : i;
            const r = r_start_px + r_idx * dr;
            const theta = theta_start + t_idx * dtheta;
            const sx = Math.round(center_x + r * Math.cos(theta));
            const sy = Math.round(center_y + r * Math.sin(theta));
            if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) continue;
            const sIdx = (sy * srcW + sx) * 4;
            const dIdx = (j * outW + i) * 4;
            dst[dIdx]     = src[sIdx];
            dst[dIdx + 1] = src[sIdx + 1];
            dst[dIdx + 2] = src[sIdx + 2];
            dst[dIdx + 3] = 255;
        }
    }
    return out;
}

// Pick an output filename that does not collide with an existing file in the
// user's uploads directory. Suffix _polar, _polar_2, _polar_3, ...
async function chooseOutputFilename(uploadsDir, originalFilename, requested) {
    if (requested) {
        return requested.replace(/[^\w.\-]/g, '_');
    }
    const ext = path.extname(originalFilename) || '.png';
    const base = path.basename(originalFilename, ext);
    let candidate = `${base}_polar${ext}`;
    let counter = 2;
    while (true) {
        try {
            await fs.access(path.join(uploadsDir, candidate));
            candidate = `${base}_polar_${counter}${ext}`;
            counter++;
            if (counter > 99) throw new Error('Too many polar variants');
        } catch {
            return candidate;
        }
    }
}

// POST /api/preprocess-polar/detect
//   Inspects the image and returns geometry hints for the UI.
//   Body (JSON): { userId, filename, hint?: { shape: "auto"|"circular"|"rectangular" } }
app.post('/api/preprocess-polar/detect', async (req, res) => {
    try {
        const Jimp = require('jimp');
        const { userId = 'default', filename } = req.body || {};
        if (!filename) {
            return res.status(400).json({ success: false, error: 'filename is required' });
        }
        const uploadsDir = getUserUploadsDir(userId);
        const filePath = path.join(uploadsDir, filename);
        const img = await Jimp.read(filePath);
        const hint = (req.body && req.body.hint) || {};
        // Optional colour groupings drive the new periodicity.grouped
        // field. Shape: [{id: 'A', colors: [[r,g,b], ...]}, ...]. Anything
        // missing or malformed just falls through to "no grouped result".
        const colorGroups = Array.isArray(req.body && req.body.color_groups)
            ? req.body.color_groups
            : null;
        const geom = detectPolarGeometry(img, hint, colorGroups);
        res.json({ success: true, ...geom });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST /api/preprocess-polar/warp
//   Generates the polar-warped image and writes it to /uploads/<userId>/.
//   Returns the filename and a ready-to-insert polar_domain block.
app.post('/api/preprocess-polar/warp', async (req, res) => {
    try {
        const Jimp = require('jimp');
        const {
            userId = 'default', filename,
            center_x, center_y,
            r_start_px, r_end_px,
            theta_start, theta_end,
            nr, ntheta, r_orientation = 'horizontal',
            output_filename,
            r_outer_physical = 1.0,
            preview = false,
        } = req.body || {};

        // Minimum input validation; the UI is the primary guardrail.
        for (const [k, v] of Object.entries({ filename, center_x, center_y, r_start_px,
                                              r_end_px, theta_start, theta_end, nr, ntheta })) {
            if (v === undefined || v === null || Number.isNaN(v)) {
                return res.status(400).json({ success: false, error: `Missing or invalid: ${k}` });
            }
        }
        if (nr < 2 || ntheta < 2) {
            return res.status(400).json({ success: false, error: 'nr and ntheta must be >= 2' });
        }

        const uploadsDir = getUserUploadsDir(userId);
        const srcPath = path.join(uploadsDir, filename);
        const src = await Jimp.read(srcPath);

        const warped = warpPolar(src, {
            center_x, center_y, r_start_px, r_end_px,
            theta_start, theta_end, nr, ntheta, r_orientation,
        });

        // Preview mode (default for the modal's live preview) writes to a
        // deterministic filename that gets overwritten on every request,
        // so debounced slider edits don't pile up files. The frontend can
        // clean it up explicitly via /api/preprocess-polar/cleanup-preview.
        let outName;
        if (preview) {
            const stem = filename.replace(/\.[^.]+$/, '');
            outName = `${stem}.__warp_preview__.png`;
        } else {
            outName = await chooseOutputFilename(uploadsDir, filename, output_filename);
        }
        const outPath = path.join(uploadsDir, outName);
        await warped.writeAsync(outPath);

        // Best-effort: enforce per-user image cap only for permanent saves;
        // the preview file is overwritten so it doesn't contribute to the cap.
        if (!preview && typeof enforceImageLimit === 'function') {
            try { await enforceImageLimit(userId); } catch { /* ignore */ }
        }

        // Compute physical polar_domain. r_outer_px maps to r_outer_physical;
        // r_inner is proportional.
        const r_outer = Number(r_outer_physical);
        const r_inner = r_end_px > 0 ? (r_start_px / r_end_px) * r_outer : 0;

        res.json({
            success: true,
            filename: outName,
            path: `/uploads/${userId}/${outName}`,
            preview: !!preview,
            output_width: warped.bitmap.width,
            output_height: warped.bitmap.height,
            polar_domain: {
                r_start: Number(r_inner.toFixed(6)),
                r_end:   Number(r_outer.toFixed(6)),
                r_orientation,
                theta_range: Number((theta_end - theta_start).toFixed(6)),
            },
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Clean up the deterministic preview file. Called from the polar modal's
// close path so we don't leave stale `__warp_preview__.png` files behind
// for sources the user has stopped working on.
app.post('/api/preprocess-polar/cleanup-preview', async (req, res) => {
    try {
        const { userId = 'default', filename } = req.body || {};
        if (!filename) {
            return res.json({ success: true, deleted: false });
        }
        const uploadsDir = getUserUploadsDir(userId);
        const stem = filename.replace(/\.[^.]+$/, '');
        const previewName = `${stem}.__warp_preview__.png`;
        const previewPath = path.join(uploadsDir, previewName);
        try {
            await fs.unlink(previewPath);
            res.json({ success: true, deleted: true, filename: previewName });
        } catch (e) {
            // Missing file is fine; surface other errors only.
            if (e && e.code === 'ENOENT') {
                res.json({ success: true, deleted: false });
            } else {
                res.status(500).json({ success: false, error: e.message });
            }
        }
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// ============================================================
// Uniform-colour quantization filter (Phase 5c.2)
// ============================================================
// Many real CAD/paper screenshots reach us with JPEG-style compression
// noise: the visible structure is a handful of solid colours but the file
// has thousands of one-pixel AA fragments around every boundary. Material
// detection still works (the rare-colour AA detector handles it) but
// downstream pipelines that expect "one RGB == one material" fail badly.
//
// This endpoint snaps every pixel to its nearest dominant colour. The
// caller supplies:
//   - rareThreshold : minimum coverage (0..1) for a source colour to be
//                     eligible as a "target". Anything below this is
//                     treated as AA noise and gets folded into a target.
//   - nTargets      : how many of the eligible dominant colours to keep
//                     as targets (sorted by coverage desc).
//   - preview       : true -> write to a deterministic filename so the UI
//                     can show a live thumbnail without piling up files
//                     (overwritten on each request). false -> create a
//                     normal new file with chooseOutputFilename().
//
// Output: a PNG that contains at most nTargets distinct RGB values plus
// any fully-transparent pixels from the source.
// Lightweight per-image AA-noise probe. Counts the number of distinct
// RGB values (ignoring fully transparent pixels) and reports it so the
// frontend can show a noise warning on image load without paying for
// the full materials/detect (AA blend classification, YAML template,
// dominant/rare partitioning). Typical runtime: ~50-200 ms on a 4 MP
// image vs. ~1 s for materials/detect.
// Bilateral filter on RGB image (in-place). Edge-preserving smoothing:
// each pixel becomes a weighted average of its neighbours where the
// weight is the product of a Gaussian in space (controlled by
// sigmaSpatial) and a Gaussian in colour distance (sigmaColor). Pixels
// across a colour edge contribute very little, so the boundary stays
// crisp while AA fragments get pulled toward the dominant colour on
// their own side of the edge -- exactly what we want before quantize.
//
// The spatial window radius is 2 * sigmaSpatial (truncates the Gaussian
// at the 2-sigma tail where the weight is ~13 % of peak). Spatial
// weights are pre-computed once. Colour weight uses an exp() lookup
// table on the squared-distance / (2 sigma_c^2) ratio mapped to 0..255.
function applyBilateralFilterRGB(data, W, H, sigmaSpatial, sigmaColor) {
    const R = Math.max(1, Math.round(2 * sigmaSpatial));
    const ws = 2 * R + 1;
    // Precompute spatial weights for the (2R+1)x(2R+1) window.
    const sw = new Float32Array(ws * ws);
    const inv2ss = 1 / (2 * sigmaSpatial * sigmaSpatial);
    for (let dy = -R; dy <= R; dy++) {
        for (let dx = -R; dx <= R; dx++) {
            sw[(dy + R) * ws + (dx + R)] = Math.exp(-(dx * dx + dy * dy) * inv2ss);
        }
    }
    // exp() lookup for colour weights. Index in [0..1023] maps to
    // colourDistSq in [0..1024]. Anything larger has weight ~0.
    const colorLUT = new Float32Array(1024);
    const inv2sc = 1 / (2 * sigmaColor * sigmaColor);
    for (let i = 0; i < 1024; i++) {
        colorLUT[i] = Math.exp(-i * inv2sc);
    }
    // Work on a copy so writes don't pollute reads.
    const src = new Uint8Array(data);
    for (let y = 0; y < H; y++) {
        const y0 = Math.max(0, y - R);
        const y1 = Math.min(H - 1, y + R);
        for (let x = 0; x < W; x++) {
            const p4 = (y * W + x) * 4;
            if (src[p4 + 3] === 0) continue;
            const cR = src[p4], cG = src[p4 + 1], cB = src[p4 + 2];
            const x0 = Math.max(0, x - R);
            const x1 = Math.min(W - 1, x + R);
            let sumR = 0, sumG = 0, sumB = 0, sumW = 0;
            for (let yy = y0; yy <= y1; yy++) {
                const dy = yy - y;
                for (let xx = x0; xx <= x1; xx++) {
                    const i4 = (yy * W + xx) * 4;
                    if (src[i4 + 3] === 0) continue;
                    const dx = xx - x;
                    const dr = src[i4]     - cR;
                    const dg = src[i4 + 1] - cG;
                    const db = src[i4 + 2] - cB;
                    const cd = (dr * dr + dg * dg + db * db) >> 2;  // /4
                    const cw = cd >= 1024 ? 0 : colorLUT[cd];
                    if (cw === 0) continue;
                    const w = sw[(dy + R) * ws + (dx + R)] * cw;
                    sumR += w * src[i4];
                    sumG += w * src[i4 + 1];
                    sumB += w * src[i4 + 2];
                    sumW += w;
                }
            }
            if (sumW > 0) {
                data[p4]     = Math.min(255, Math.max(0, Math.round(sumR / sumW)));
                data[p4 + 1] = Math.min(255, Math.max(0, Math.round(sumG / sumW)));
                data[p4 + 2] = Math.min(255, Math.max(0, Math.round(sumB / sumW)));
            }
        }
    }
}

app.post('/api/preprocess-filter/quick-stats', async (req, res) => {
    try {
        const Jimp = require('jimp');
        const { userId = 'default', filename } = req.body || {};
        if (!filename) {
            return res.status(400).json({ success: false, error: 'filename required' });
        }
        const uploadsDir = getUserUploadsDir(userId);
        const srcPath = path.join(uploadsDir, filename);
        const src = await Jimp.read(srcPath);
        const W = src.bitmap.width;
        const H = src.bitmap.height;
        const data = src.bitmap.data;
        const len = W * H * 4;
        const seen = new Set();
        for (let i = 0; i < len; i += 4) {
            if (data[i + 3] === 0) continue;
            seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
        }
        res.json({
            success: true,
            unique_colors: seen.size,
            image_width: W,
            image_height: H,
            // Match the heuristic used inline by the frontend so both sides
            // agree on what "noisy" means.
            looks_noisy: seen.size > 1000,
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// ============================================================
// Auto-tune helpers (Phase 5c.5b)
// ============================================================
// Self-contained re-implementation of the quantize pipeline that runs
// on a small RGBA buffer and returns a single scalar "scattered noise"
// score, so Nelder-Mead can drive the parameters toward a noise minimum
// without bouncing off Jimp / disk I/O on every evaluation.

// Score = number of opaque pixels whose palette index is strictly
// outvoted by another index in the 3x3 neighbourhood (i.e. the pixel is
// a minority island). Lower = cleaner.
function boundaryNoiseScore(idxMap, W, H, K) {
    let score = 0;
    const counts = new Int32Array(K);
    for (let y = 0; y < H; y++) {
        const y0 = y > 0 ? y - 1 : 0;
        const y1 = y < H - 1 ? y + 1 : H - 1;
        for (let x = 0; x < W; x++) {
            const p = y * W + x;
            const my = idxMap[p];
            if (my === 0xFF) continue;
            const x0 = x > 0 ? x - 1 : 0;
            const x1 = x < W - 1 ? x + 1 : W - 1;
            counts.fill(0);
            for (let yy = y0; yy <= y1; yy++) {
                const row = yy * W;
                for (let xx = x0; xx <= x1; xx++) {
                    const v = idxMap[row + xx];
                    if (v !== 0xFF) counts[v]++;
                }
            }
            const myCount = counts[my];
            let bestC = myCount;
            for (let k = 0; k < K; k++) {
                if (counts[k] > bestC) bestC = counts[k];
            }
            if (bestC > myCount) score++;
        }
    }
    return score;
}

// Per-colour connected-component "island erosion". 4-connectivity
// Union-Find labels every same-colour region; components whose pixel
// count is below `minSize` are then dissolved by repainting each of
// their pixels to the most common *different* colour in the 3x3
// neighbourhood. Repeats until no further changes (capped at 4 passes)
// so a fragment that survives one pass partly intact still collapses
// on the next. Cost: O(W*H * passes), milliseconds even on 1 MP.
function shrinkIslands(idxMap, W, H, K, minSize, maxPasses = 4) {
    if (!(minSize >= 2)) return 0;
    const N = W * H;
    const parent = new Int32Array(N);
    const counts = new Int32Array(K);
    let totalChanges = 0;

    for (let pass = 0; pass < maxPasses; pass++) {
        for (let i = 0; i < N; i++) parent[i] = i;
        const find = (x) => {
            let r = x;
            while (parent[r] !== r) r = parent[r];
            while (parent[x] !== r) { const nx = parent[x]; parent[x] = r; x = nx; }
            return r;
        };
        const union = (a, b) => {
            const ra = find(a), rb = find(b);
            if (ra !== rb) parent[ra] = rb;
        };

        for (let y = 0; y < H; y++) {
            const row = y * W;
            for (let x = 0; x < W; x++) {
                const p = row + x;
                const my = idxMap[p];
                if (my === 0xFF) continue;
                if (x > 0 && idxMap[p - 1] === my) union(p, p - 1);
                if (y > 0 && idxMap[p - W] === my) union(p, p - W);
            }
        }
        const sizes = new Int32Array(N);  // sparse: only roots populated
        for (let i = 0; i < N; i++) {
            if (idxMap[i] === 0xFF) continue;
            sizes[find(i)]++;
        }
        let changes = 0;
        for (let y = 0; y < H; y++) {
            const row = y * W;
            const y0 = y > 0 ? y - 1 : 0;
            const y1 = y < H - 1 ? y + 1 : H - 1;
            for (let x = 0; x < W; x++) {
                const p = row + x;
                const my = idxMap[p];
                if (my === 0xFF) continue;
                if (sizes[find(p)] >= minSize) continue;
                const x0 = x > 0 ? x - 1 : 0;
                const x1 = x < W - 1 ? x + 1 : W - 1;
                counts.fill(0);
                for (let yy = y0; yy <= y1; yy++) {
                    const r2 = yy * W;
                    for (let xx = x0; xx <= x1; xx++) {
                        const v = idxMap[r2 + xx];
                        if (v !== 0xFF && v !== my) counts[v]++;
                    }
                }
                let bestK = my, bestC = 0;
                for (let k = 0; k < K; k++) {
                    if (counts[k] > bestC) { bestC = counts[k]; bestK = k; }
                }
                if (bestK !== my) { idxMap[p] = bestK; changes++; }
            }
        }
        totalChanges += changes;
        if (changes === 0) break;
    }
    return totalChanges;
}

// Single evaluation of the quantize pipeline on a small RGBA buffer.
// `params` = [minTargetDist, despeckleRadius, bilateralSigmaSpatial,
//             bilateralSigmaColor, minIslandSize].
// `N` and `rare` are held fixed by the caller -- both define which colours
// count as "materials" and so are user-controlled.
function evaluateQuantizeParams(srcData, W, H, params, N, rare) {
    const [mtd, dspF, bfs, bfc, misF] = params;
    const dsp = Math.max(0, Math.min(10, Math.round(dspF)));
    const mis = Math.max(0, Math.min(64, Math.round(misF)));
    const mtd2 = mtd * mtd;

    const data = new Uint8Array(srcData);
    if (bfs > 0) applyBilateralFilterRGB(data, W, H, bfs, bfc);

    // Count colours.
    const colorCounts = new Map();
    const len = W * H * 4;
    for (let i = 0; i < len; i += 4) {
        if (data[i + 3] === 0) continue;
        const k = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        colorCounts.set(k, (colorCounts.get(k) || 0) + 1);
    }
    if (colorCounts.size === 0) return Infinity;
    const totalPixels = W * H;
    const sorted = Array.from(colorCounts.entries())
        .map(([k, c]) => ({
            r: (k >> 16) & 0xff, g: (k >> 8) & 0xff, b: k & 0xff,
            count: c, ratio: c / totalPixels,
        }))
        .sort((a, b) => b.count - a.count);

    let eligible = sorted.filter(c => c.ratio >= rare);
    if (eligible.length === 0) eligible = sorted;
    const targets = [];
    for (const c of eligible) {
        let ok = true;
        for (const t of targets) {
            const dr = c.r - t.r, dg = c.g - t.g, db = c.b - t.b;
            if (dr * dr + dg * dg + db * db < mtd2) { ok = false; break; }
        }
        if (ok) targets.push(c);
        if (targets.length >= N) break;
    }
    const K = targets.length;
    if (K === 0) return Infinity;

    const tR = new Int32Array(K);
    const tG = new Int32Array(K);
    const tB = new Int32Array(K);
    for (let k = 0; k < K; k++) {
        tR[k] = targets[k].r; tG[k] = targets[k].g; tB[k] = targets[k].b;
    }
    const idxMap = new Uint8Array(W * H);
    for (let i = 0, p = 0; i < len; i += 4, p++) {
        if (data[i + 3] === 0) { idxMap[p] = 0xFF; continue; }
        const r = data[i], g = data[i + 1], b = data[i + 2];
        let bestD = Infinity, bestI = 0;
        for (let k = 0; k < K; k++) {
            const dr = r - tR[k], dg = g - tG[k], db = b - tB[k];
            const d = dr * dr + dg * dg + db * db;
            if (d < bestD) { bestD = d; bestI = k; }
        }
        idxMap[p] = bestI;
    }

    if (dsp > 0) {
        const counts = new Int32Array(K);
        const passes = (dsp === 1) ? 2 : 1;
        for (let pass = 0; pass < passes; pass++) {
            const src2 = new Uint8Array(idxMap);
            for (let y = 0; y < H; y++) {
                const y0 = Math.max(0, y - dsp);
                const y1 = Math.min(H - 1, y + dsp);
                for (let x = 0; x < W; x++) {
                    const p = y * W + x;
                    const my = src2[p];
                    if (my === 0xFF) continue;
                    const x0 = Math.max(0, x - dsp);
                    const x1 = Math.min(W - 1, x + dsp);
                    counts.fill(0);
                    for (let yy = y0; yy <= y1; yy++) {
                        const row = yy * W;
                        for (let xx = x0; xx <= x1; xx++) {
                            const v = src2[row + xx];
                            if (v !== 0xFF) counts[v]++;
                        }
                    }
                    const myCount = counts[my];
                    let bestK = my, bestC = myCount;
                    for (let k = 0; k < K; k++) {
                        if (counts[k] > bestC) { bestC = counts[k]; bestK = k; }
                    }
                    if (bestK !== my && bestC > myCount * 1.5) idxMap[p] = bestK;
                }
            }
        }
    }

    if (mis >= 2) shrinkIslands(idxMap, W, H, K, mis, 4);

    // Normalised noise count so changes in image size don't dominate.
    return boundaryNoiseScore(idxMap, W, H, K) / totalPixels;
}

// Standard Nelder-Mead simplex search. Works on a scaled (~normalised)
// representation internally so each parameter's per-unit step is
// comparable, then maps back to native units before each evaluation.
function nelderMead(fn, x0, bounds, maxEvals = 60) {
    const n = x0.length;
    const lo = bounds.map(b => b[0]);
    const hi = bounds.map(b => b[1]);
    const range = bounds.map((b, i) => Math.max(1e-9, b[1] - b[0]));
    const toNorm = (x) => x.map((v, i) => (v - lo[i]) / range[i]);
    const toReal = (u) => u.map((v, i) => lo[i] + Math.max(0, Math.min(1, v)) * range[i]);

    const x0n = toNorm(x0);
    const simplex = [x0n.slice()];
    for (let i = 0; i < n; i++) {
        const v = x0n.slice();
        v[i] = Math.min(1, v[i] + 0.18);
        simplex.push(v);
    }
    let evals = 0;
    const scores = simplex.map(v => { evals++; return fn(toReal(v)); });
    const order = () => {
        const idx = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
        const ns = idx.map(i => simplex[i]);
        const nss = idx.map(i => scores[i]);
        for (let i = 0; i < simplex.length; i++) { simplex[i] = ns[i]; scores[i] = nss[i]; }
    };
    order();

    while (evals < maxEvals) {
        order();
        // Centroid of best n vertices.
        const centroid = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) centroid[j] += simplex[i][j];
        }
        for (let j = 0; j < n; j++) centroid[j] /= n;
        const worst = simplex[n];
        // Reflect (alpha=1).
        const reflect = centroid.map((c, j) => c + (c - worst[j]));
        const sR = fn(toReal(reflect)); evals++;
        if (sR < scores[n - 1] && sR >= scores[0]) {
            simplex[n] = reflect; scores[n] = sR; continue;
        }
        if (sR < scores[0]) {
            // Expand (gamma=2).
            const expand = centroid.map((c, j) => c + 2 * (reflect[j] - c));
            const sE = fn(toReal(expand)); evals++;
            if (sE < sR) { simplex[n] = expand; scores[n] = sE; }
            else         { simplex[n] = reflect; scores[n] = sR; }
            continue;
        }
        // Contract (rho=0.5).
        const contract = centroid.map((c, j) => c + 0.5 * (worst[j] - c));
        const sC = fn(toReal(contract)); evals++;
        if (sC < scores[n]) { simplex[n] = contract; scores[n] = sC; continue; }
        // Shrink (sigma=0.5).
        for (let i = 1; i < simplex.length; i++) {
            for (let j = 0; j < n; j++) simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
            scores[i] = fn(toReal(simplex[i])); evals++;
            if (evals >= maxEvals) break;
        }
    }
    order();
    return { best: toReal(simplex[0]), bestScore: scores[0], evals };
}

app.post('/api/preprocess-filter/auto-tune', async (req, res) => {
    try {
        const Jimp = require('jimp');
        const {
            userId = 'default', filename,
            nTargets = 8,
            maxEvals = 60,
            subsampleSize = 256,
            initial,            // optional: start from the user's current sliders
        } = req.body || {};
        if (!filename) {
            return res.status(400).json({ success: false, error: 'filename required' });
        }
        const N = Math.max(1, Math.min(64, Math.floor(nTargets)));

        const uploadsDir = getUserUploadsDir(userId);
        const srcPath = path.join(uploadsDir, filename);
        const src = await Jimp.read(srcPath);
        const fullW = src.bitmap.width;
        const fullH = src.bitmap.height;
        // Downsample to keep per-eval cost ~50-150 ms even for 4 MP sources.
        let scaleFactor = 1;
        if (Math.max(fullW, fullH) > subsampleSize) {
            scaleFactor = subsampleSize / Math.max(fullW, fullH);
        }
        const small = scaleFactor < 1
            ? src.clone().scale(scaleFactor)
            : src.clone();
        const W = small.bitmap.width;
        const H = small.bitmap.height;
        const srcData = small.bitmap.data;

        // N and rareThreshold are user-controlled "what counts as a
        // material" decisions, not noise-suppression knobs, so the
        // optimiser leaves them alone. If the user hasn't supplied a
        // rare value we fall back to the same default the manual flow
        // ships with.
        const rare = Number.isFinite(Number(req.body && req.body.rareThreshold))
            ? Math.max(0, Math.min(1, Number(req.body.rareThreshold)))
            : 0.005;

        const bounds = [
            [1,  80],     // minTargetDist
            [0,  5],      // despeckleRadius
            [0,  5],      // bilateralSigmaSpatial
            [5,  60],     // bilateralSigmaColor
            [0,  30],     // minIslandSize (connected-component shrink)
        ];
        const defaults = [10, 1, 0, 20, 4];
        const x0 = Array.isArray(initial) && initial.length === 5
            ? initial.map((v, i) => Math.max(bounds[i][0], Math.min(bounds[i][1], Number(v))))
            : defaults;

        const fn = (p) => evaluateQuantizeParams(srcData, W, H, p, N, rare);
        const t0 = Date.now();
        const initialScore = fn(x0);
        const result = nelderMead(fn, x0, bounds, Math.max(20, Math.min(200, Math.floor(maxEvals))));
        const elapsedMs = Date.now() - t0;

        const [mtd, dspF, bfs, bfc, misF] = result.best;
        res.json({
            success: true,
            params: {
                minTargetDist:         Number(mtd.toFixed(2)),
                despeckleRadius:       Math.max(0, Math.min(10, Math.round(dspF))),
                bilateralSigmaSpatial: Number(bfs.toFixed(2)),
                bilateralSigmaColor:   Number(bfc.toFixed(1)),
                minIslandSize:         Math.max(0, Math.min(64, Math.round(misF))),
            },
            n_targets:        N,
            rare_threshold:   rare,
            score_initial:    Number(initialScore.toFixed(6)),
            score_final:      Number(result.bestScore.toFixed(6)),
            evals:            result.evals,
            elapsed_ms:       elapsedMs,
            subsample_w:      W,
            subsample_h:      H,
            full_w:           fullW,
            full_h:           fullH,
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.post('/api/preprocess-filter/quantize', async (req, res) => {
    try {
        const Jimp = require('jimp');
        const {
            userId = 'default', filename,
            rareThreshold = 0.005,
            nTargets = 8,
            minTargetDist = 30,
            despeckleRadius = 1,
            bilateralSigmaSpatial = 0,
            bilateralSigmaColor   = 20,
            minIslandSize = 0,
            preview = false,
            output_filename,
        } = req.body || {};

        if (!filename) {
            return res.status(400).json({ success: false, error: 'filename required' });
        }
        const N = Math.max(1, Math.min(64, Math.floor(nTargets)));
        const RT = Math.max(0, Math.min(1, Number(rareThreshold)));
        const MTD = Math.max(0, Number(minTargetDist));     // RGB Euclidean
        const MTD2 = MTD * MTD;
        const DSP = Math.max(0, Math.min(10, Math.floor(despeckleRadius)));
        // Bilateral pre-filter. SigmaSpatial=0 disables it. The spatial
        // window radius is 2 * sigmaSpatial (clamped to [0, 8]).
        const BFS = Math.max(0, Math.min(8, Number(bilateralSigmaSpatial)));
        const BFC = Math.max(1, Math.min(150, Number(bilateralSigmaColor)));
        // Connected-component "island shrink" pass after despeckle.
        // 0 or 1 disables it; otherwise components below this pixel
        // count get dissolved into the local majority colour.
        const MIS = Math.max(0, Math.min(1024, Math.floor(minIslandSize)));

        const uploadsDir = getUserUploadsDir(userId);
        const srcPath = path.join(uploadsDir, filename);
        const src = await Jimp.read(srcPath);
        const W = src.bitmap.width;
        const H = src.bitmap.height;
        const totalPixels = W * H;

        // Bilateral pre-filter (edge-preserving smoothing). Removes
        // chromatic noise in flat AA regions so the nearest-target snap
        // doesn't pull JPEG-noised pixels toward unrelated palette colours.
        // sigmaSpatial=0 disables it entirely (no copy, no work).
        if (BFS > 0) {
            applyBilateralFilterRGB(src.bitmap.data, W, H, BFS, BFC);
        }

        // Count occurrences of each unique RGB (skip fully-transparent pixels).
        const colorCounts = new Map();
        src.scan(0, 0, W, H, (x, y, idx) => {
            const a = src.bitmap.data[idx + 3];
            if (a === 0) return;
            const r = src.bitmap.data[idx];
            const g = src.bitmap.data[idx + 1];
            const b = src.bitmap.data[idx + 2];
            const key = (r << 16) | (g << 8) | b;
            colorCounts.set(key, (colorCounts.get(key) || 0) + 1);
        });

        const sorted = Array.from(colorCounts.entries())
            .map(([key, count]) => ({
                r: (key >> 16) & 0xff,
                g: (key >> 8) & 0xff,
                b: key & 0xff,
                count,
                ratio: count / totalPixels,
            }))
            .sort((a, b) => b.count - a.count);

        if (sorted.length === 0) {
            return res.status(400).json({ success: false, error: 'No opaque pixels in source' });
        }

        // Eligible = above threshold; if the threshold is so strict that
        // nothing qualifies, fall back to the full sorted list so we
        // always produce something usable.
        let eligible = sorted.filter(c => c.ratio >= RT);
        if (eligible.length === 0) eligible = sorted;
        // Greedy farthest-point pick from `eligible`, sorted by coverage:
        // walk the list, keep the next colour only if it's at least MTD
        // away (RGB Euclidean) from every already-chosen target. This
        // suppresses anti-aliased neighbours of dominant colours -- e.g.
        // [254,254,254] sits 1.7 away from [255,255,255] so it gets
        // rejected and the AA pixels later snap to the white target.
        // Without this, raw-count-only selection drops genuine minor
        // materials (a red magnet at 0.6 %) in favour of AA-white
        // fragments (a blend at 0.7 %) that aren't physical materials.
        const targets = [];
        for (const c of eligible) {
            let ok = true;
            for (const t of targets) {
                const dr = c.r - t.r, dg = c.g - t.g, db = c.b - t.b;
                if (dr * dr + dg * dg + db * db < MTD2) { ok = false; break; }
            }
            if (ok) targets.push(c);
            if (targets.length >= N) break;
        }

        // Quantize: nearest-target lookup with squared-Euclidean RGB distance.
        // ~1 ns/pixel/target -> well under 100 ms even for 4 MP images.
        const K = targets.length;
        const tR = new Int32Array(K);
        const tG = new Int32Array(K);
        const tB = new Int32Array(K);
        for (let k = 0; k < K; k++) {
            tR[k] = targets[k].r; tG[k] = targets[k].g; tB[k] = targets[k].b;
        }

        const out = src.clone();
        const data = out.bitmap.data;
        const len = W * H * 4;
        // Track each pixel's palette index so we can run a fast majority
        // filter on the result. 0xFF marks "skipped" (fully transparent).
        const idxMap = new Uint8Array(W * H);
        for (let i = 0, p = 0; i < len; i += 4, p++) {
            if (data[i + 3] === 0) { idxMap[p] = 0xFF; continue; }
            const r = data[i], g = data[i + 1], b = data[i + 2];
            let bestD = Infinity, bestI = 0;
            for (let k = 0; k < K; k++) {
                const dr = r - tR[k], dg = g - tG[k], db = b - tB[k];
                const d = dr * dr + dg * dg + db * db;
                if (d < bestD) { bestD = d; bestI = k; }
            }
            idxMap[p] = bestI;
            data[i]     = tR[bestI];
            data[i + 1] = tG[bestI];
            data[i + 2] = tB[bestI];
        }

        // Despeckle pass: a per-pixel majority filter over the palette
        // indices. Suppresses "salt-and-pepper" misclassifications -- e.g.
        // a JPEG-noised grey AA pixel whose RGB happens to land closer to
        // a dark phase colour than to grey/white, but whose neighbours are
        // all grey/white. The pixel is replaced only when the majority
        // colour outvotes the pixel's own colour by > 1.5x (counting the
        // pixel itself), so genuine thin features survive. Two passes are
        // cheap and let radius-1 also clean up 2-3 pixel clusters.
        if (DSP > 0) {
            const ws = 2 * DSP + 1;        // window side
            const counts = new Int32Array(K);
            const passes = (DSP === 1) ? 2 : 1;
            for (let pass = 0; pass < passes; pass++) {
                const src2 = new Uint8Array(idxMap);
                for (let y = 0; y < H; y++) {
                    const y0 = Math.max(0, y - DSP);
                    const y1 = Math.min(H - 1, y + DSP);
                    for (let x = 0; x < W; x++) {
                        const p = y * W + x;
                        const my = src2[p];
                        if (my === 0xFF) continue;
                        const x0 = Math.max(0, x - DSP);
                        const x1 = Math.min(W - 1, x + DSP);
                        counts.fill(0);
                        for (let yy = y0; yy <= y1; yy++) {
                            const row = yy * W;
                            for (let xx = x0; xx <= x1; xx++) {
                                const v = src2[row + xx];
                                if (v !== 0xFF) counts[v]++;
                            }
                        }
                        const myCount = counts[my];
                        let bestK = my, bestC = myCount;
                        for (let k = 0; k < K; k++) {
                            if (counts[k] > bestC) { bestC = counts[k]; bestK = k; }
                        }
                        // Replace only when the alternative is a strict
                        // majority over the pixel's own colour -- keeps
                        // true edges intact.
                        if (bestK !== my && bestC > myCount * 1.5) {
                            idxMap[p] = bestK;
                            const i4 = p * 4;
                            data[i4]     = tR[bestK];
                            data[i4 + 1] = tG[bestK];
                            data[i4 + 2] = tB[bestK];
                        }
                    }
                }
            }
        }

        // Connected-component shrink: drop small same-colour islands
        // (< MIS pixels) into the local majority colour. Runs *after*
        // despeckle so neighbour majorities reflect the cleaned state.
        // Mirrors the changes into the RGBA buffer.
        if (MIS >= 2) {
            const changes = shrinkIslands(idxMap, W, H, K, MIS, 4);
            if (changes > 0) {
                for (let p = 0; p < idxMap.length; p++) {
                    const v = idxMap[p];
                    if (v === 0xFF) continue;
                    const i4 = p * 4;
                    data[i4]     = tR[v];
                    data[i4 + 1] = tG[v];
                    data[i4 + 2] = tB[v];
                }
            }
        }

        // Filename: preview overwrites a deterministic file per source so we
        // don't pile up uploads; final goes through chooseOutputFilename.
        let outName;
        if (preview) {
            const stem = filename.replace(/\.[^.]+$/, '');
            outName = `${stem}.__quantize_preview__.png`;
        } else {
            const suggested = output_filename
                || `${filename.replace(/\.[^.]+$/, '')}_uniformN${K}.png`;
            outName = await chooseOutputFilename(uploadsDir, filename, suggested);
        }
        const outPath = path.join(uploadsDir, outName);
        await out.writeAsync(outPath);

        if (!preview && typeof enforceImageLimit === 'function') {
            try { await enforceImageLimit(userId); } catch { /* ignore */ }
        }

        res.json({
            success: true,
            filename: outName,
            path: `/uploads/${userId}/${outName}`,
            preview: !!preview,
            targets: targets.map(t => ({
                rgb: [t.r, t.g, t.b],
                ratio: Number(t.ratio.toFixed(6)),
                count: t.count,
            })),
            unique_colors_in: sorted.length,
            n_targets_used: K,
            // Heuristic: an image with thousands of unique RGB values and a
            // long ratio tail is the canonical "JPEG-AA noise" case. Surface
            // this so the UI can show a recommendation badge without having
            // to recompute on the client.
            looks_noisy: sorted.length > 1000,
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// アップロードされた画像の一覧（ユーザーごと）
app.get('/api/images', async (req, res) => {
    try {
        const { userId } = req.query;
        const userUploadDir = getUserUploadsDir(userId || 'default');

        // Check if directory exists
        try {
            await fs.access(userUploadDir);
        } catch {
            // Directory doesn't exist, return empty list
            return res.json({
                success: true,
                images: []
            });
        }

        const files = await fs.readdir(userUploadDir);
        // Filter only image files
        const imageFiles = files.filter(f => /\.(png|jpg|jpeg|bmp)$/i.test(f));

        res.json({
            success: true,
            images: imageFiles
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// 画像ファイルの削除
app.delete('/api/images/:filename', async (req, res) => {
    try {
        const { filename } = req.params;
        const { userId } = req.query;

        // Sanitize filename to prevent directory traversal
        const safeFilename = path.basename(filename);
        const userUploadDir = getUserUploadsDir(userId || 'default');
        const filePath = path.join(userUploadDir, safeFilename);

        await fs.unlink(filePath);

        res.json({
            success: true,
            message: 'Image deleted successfully'
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Configファイルの削除
app.delete('/api/config/:filename', async (req, res) => {
    try {
        const { filename } = req.params;
        const { userId } = req.query;

        const userDir = getUserDir(userId || 'default');
        const filePath = path.join(userDir, filename);

        await fs.unlink(filePath);

        // 削除後、ファイルが残っているかチェック
        const files = await fs.readdir(userDir);
        const yamlFiles = files.filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));

        // ファイルがすべて削除された場合、デフォルトのsample_config.yamlをコピー
        if (yamlFiles.length === 0) {
            const defaultConfig = await fs.readFile(CONFIG_PATH, 'utf8');
            const defaultConfigPath = path.join(userDir, 'sample_config.yaml');
            await fs.writeFile(defaultConfigPath, defaultConfig, 'utf8');
        }

        res.json({
            success: true,
            message: 'Config deleted successfully'
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// ソルバーの実行（ストリーミングなし）
app.post('/api/solve', async (req, res) => {
    try {
        const { configFile, imageFile, userId } = req.body;

        // パスの構築
        let configPath;
        if (configFile) {
            // User-specific config file
            const userDir = getUserDir(userId || 'default');
            configPath = path.join(userDir, configFile);
        } else {
            // Default config
            configPath = CONFIG_PATH;
        }

        // Get image path from user upload directory
        const userUploadDir = getUserUploadsDir(userId || 'default');
        const imagePath = path.join(userUploadDir, imageFile);

        // ファイルの存在確認
        await fs.access(configPath);
        await fs.access(imagePath);
        await fs.access(SOLVER_PATH);

        // コマンドの構築
        const command = `"${SOLVER_PATH}" "${configPath}" "${imagePath}"`;

        console.log('Executing:', command);

        // ソルバーの実行
        exec(command, {
            cwd: BASE_DIR,
            maxBuffer: 10 * 1024 * 1024 // 10MB
        }, (error, stdout, stderr) => {
            if (error) {
                console.error('Solver error:', error);
                return res.status(500).json({
                    success: false,
                    error: error.message,
                    stderr: stderr
                });
            }

            // 出力ファイル名の取得（Muファイルも）
            // const azFile = outputPath;
            // const muFile = outputPath.replace('Az_', 'Mu_');

            res.json({
                success: true,
                message: 'Solver completed successfully',
                stdout: stdout,
            });
        });

    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// ソルバーの実行（プログレス付きSSEストリーミング）
app.post('/api/solve-stream', async (req, res) => {
    try {
        const { configFile, imageFile, userId, materialLibraryFile } = req.body;

        // パスの構築
        let configPath;
        if (configFile) {
            // User-specific config file
            const userDir = getUserDir(userId || 'default');
            configPath = path.join(userDir, configFile);
        } else {
            // Default config
            configPath = CONFIG_PATH;
        }

        // Get image path from user upload directory
        const userUploadDir = getUserUploadsDir(userId || 'default');
        const imagePath = path.join(userUploadDir, imageFile);

        // ファイルの存在確認
        await fs.access(configPath);
        await fs.access(imagePath);
        await fs.access(SOLVER_PATH);

        // Merge material library if provided
        const userIdKey = userId || 'default';
        let effectiveConfigPath = configPath;
        let mergedTempPath = null;
        if (materialLibraryFile) {
            const libDir  = getUserLibsDir(userIdKey);
            const libPath = path.join(libDir, path.basename(materialLibraryFile));
            if (!path.resolve(libPath).startsWith(path.resolve(libDir))) {
                throw new Error('Invalid library path');
            }
            const configYaml = await fs.readFile(configPath, 'utf8');
            const libYaml    = await fs.readFile(libPath, 'utf8');
            const mergedYaml = mergeLibraryIntoConfig(configYaml, libYaml);
            mergedTempPath   = path.join(getUserDir(userIdKey), `.merged_${Date.now()}.yaml`);
            await fs.writeFile(mergedTempPath, mergedYaml);
            effectiveConfigPath = mergedTempPath;
        }

        // SSEヘッダーの設定
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        console.log('Executing solver with streaming:', SOLVER_PATH);

        // Prepare user-specific output directory
        const outputPath = await prepareUserOutputDirectory(userId);
        console.log(`Output directory for user ${userIdKey}: ${outputPath}`);

        // spawnを使用してリアルタイムで出力を取得（第3引数に出力パスを追加）
        const solverProcess = spawn(SOLVER_PATH, [effectiveConfigPath, imagePath, outputPath], {
            cwd: BASE_DIR,
        });

        // プロセスをマップに登録
        runningProcesses.set(userIdKey, solverProcess);
        console.log(`Registered solver process for user: ${userIdKey}`);

        let outputBuffer = '';
        let errorBuffer = '';

        // 標準出力の処理
        solverProcess.stdout.on('data', (data) => {
            const text = data.toString();
            outputBuffer += text;

            // 行ごとに処理
            const lines = outputBuffer.split('\n');
            outputBuffer = lines.pop() || ''; // 最後の不完全な行を保持

            for (const line of lines) {
                if (!line.trim()) continue;

                // プログレス情報を抽出
                let progressData = { type: 'log', message: line };

                // "--- Step X / Y ---" の形式をパース
                const stepMatch = line.match(/---\s*Step\s+(\d+)\s*\/\s*(\d+)\s*---/i);
                if (stepMatch) {
                    const current = parseInt(stepMatch[1]);
                    const total = parseInt(stepMatch[2]);
                    // percentage reflects COMPLETED steps, not started steps
                    // Step 1/N -> 0%, Step N/N -> (N-1)/N%, 100% only on complete
                    progressData = {
                        type: 'progress',
                        step: current,
                        total: total,
                        percentage: Math.round(((current - 1) / total) * 100),
                        message: line
                    };
                }

                // 他の重要なメッセージ
                if (line.includes('Solving linear system') ||
                    line.includes('Using AMGCL') ||
                    line.includes('Using direct solver')) {
                    progressData.type = 'status';
                }

                if (line.includes('completed successfully')) {
                    progressData.type = 'complete';
                }

                // SSEフォーマットで送信
                res.write(`data: ${JSON.stringify(progressData)}\n\n`);
            }
        });

        // 標準エラー出力の処理
        solverProcess.stderr.on('data', (data) => {
            const text = data.toString();
            errorBuffer += text;

            const lines = text.split('\n');
            for (const line of lines) {
                if (!line.trim()) continue;

                const errorData = {
                    type: 'error',
                    message: line
                };

                res.write(`data: ${JSON.stringify(errorData)}\n\n`);
            }
        });

        // プロセス終了時の処理
        solverProcess.on('close', (code) => {
            console.log(`Solver process exited with code ${code}`);

            // プロセスをマップから削除
            runningProcesses.delete(userIdKey);
            console.log(`Removed solver process for user: ${userIdKey}`);

            // Clean up merged temp config if used
            if (mergedTempPath) {
                fs.unlink(mergedTempPath).catch(() => {});
                mergedTempPath = null;
            }

            // 最後のバッファを送信
            if (outputBuffer.trim()) {
                res.write(`data: ${JSON.stringify({ type: 'log', message: outputBuffer.trim() })}\n\n`);
            }

            // 完了メッセージ
            const finalData = {
                type: code === 0 ? 'done' : 'error',
                success: code === 0,
                exitCode: code,
                message: code === 0 ? 'Solver completed successfully' : (code === null ? 'Solver was stopped by user' : 'Solver failed')
            };

            res.write(`data: ${JSON.stringify(finalData)}\n\n`);
            res.end();
        });

        // エラー時の処理
        solverProcess.on('error', (error) => {
            console.error('Solver process error:', error);

            // プロセスをマップから削除
            runningProcesses.delete(userIdKey);

            const errorData = {
                type: 'error',
                success: false,
                error: error.message
            };
            res.write(`data: ${JSON.stringify(errorData)}\n\n`);
            res.end();
        });

        // クライアントが接続を切断した場合
        req.on('close', () => {
            console.log('Client disconnected, terminating solver process');
            solverProcess.kill();
        });

    } catch (error) {
        console.error('Error starting solver:', error);
        const errorData = {
            type: 'error',
            success: false,
            error: error.message
        };
        res.write(`data: ${JSON.stringify(errorData)}\n\n`);
        res.end();
    }
});

// ソルバーの停止
app.post('/api/stop-solver', async (req, res) => {
    try {
        const { userId } = req.body;
        const userIdKey = userId || 'default';

        const solverProcess = runningProcesses.get(userIdKey);

        if (!solverProcess) {
            return res.status(404).json({
                success: false,
                error: 'No running solver process found for this user'
            });
        }

        console.log(`Stopping solver process for user: ${userIdKey}`);

        // プロセスを強制終了
        solverProcess.kill('SIGTERM');

        // プロセスがすぐに終了しない場合のタイムアウト
        setTimeout(() => {
            if (!solverProcess.killed) {
                console.log(`Force killing solver process for user: ${userIdKey}`);
                solverProcess.kill('SIGKILL');
            }
        }, 5000);

        res.json({
            success: true,
            message: 'Solver process stopped successfully'
        });

    } catch (error) {
        console.error('Error stopping solver:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// 出力ファイルの一覧
app.get('/api/results', async (req, res) => {
    try {
        const { userId } = req.query;
        const userIdKey = userId || 'default';

        // User-specific output directory
        const userOutputDir = path.join(OUTPUTS_DIR, `${userIdKey}`);

        // Check if user output directory exists
        try {
            await fs.access(userOutputDir);
        } catch {
            // No outputs for this user yet
            return res.json({
                success: true,
                results: []
            });
        }

        const files = await fs.readdir(userOutputDir, { withFileTypes: true });

        // Analysis result folders (detected by Az + conditions.json)
        const resultFolders = [];

        for (const file of files) {
            if (file.isDirectory()) {
                const folderName = file.name;
                const folderPath = path.join(userOutputDir, folderName);

                // Check if this is an analysis result folder
                if (await isAnalysisResult(folderPath)) {
                    try {
                        // Count steps (CSV or TIFF; same step number written
                        // in both formats counts once)
                        const azFolder = path.join(folderPath, 'Az');
                        const azFiles = await fs.readdir(azFolder);
                        const stepCount = countTransientSteps(azFiles);

                        resultFolders.push({
                            name: folderName,
                            path: `outputs/${userIdKey}/${folderName}`,
                            timestamp: folderName.replace('output_', ''), // Extract timestamp if present
                            steps: stepCount
                        });
                    } catch {
                        // Skip if cannot read Az folder
                        continue;
                    }
                }
            }
        }

        // タイムスタンプでソート（新しい順）
        resultFolders.sort((a, b) => b.timestamp.localeCompare(a.timestamp));

        res.json({
            success: true,
            results: resultFolders
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// ===== Output File Management API =====

/**
 * Calculate directory size recursively
 * @param {string} dirPath - Directory path
 * @returns {Promise<number>} Size in bytes
 */
async function getDirectorySize(dirPath) {
    let totalSize = 0;

    try {
        const items = await fs.readdir(dirPath, { withFileTypes: true });

        for (const item of items) {
            const itemPath = path.join(dirPath, item.name);

            if (item.isDirectory()) {
                totalSize += await getDirectorySize(itemPath);
            } else if (item.isFile()) {
                const stats = await fs.stat(itemPath);
                totalSize += stats.size;
            }
        }
    } catch (error) {
        console.error(`Error calculating size for ${dirPath}:`, error);
    }

    return totalSize;
}

/**
 * Format bytes to human readable format
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size string
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Health check endpoint
app.get('/api/health', async (req, res) => {
    try {
        const checks = {};

        // Check solver binary
        try {
            await fs.access(SOLVER_PATH);
            checks.solver = 'ok';
        } catch {
            checks.solver = 'missing';
        }

        // Check directories writable
        for (const [name, dir] of [['uploads', UPLOAD_DIR], ['configs', USER_CONFIGS_DIR], ['outputs', OUTPUTS_DIR]]) {
            try {
                await fs.access(dir, fsSync.constants.W_OK);
                checks[name] = 'ok';
            } catch {
                checks[name] = 'not_writable';
            }
        }

        const allOk = Object.values(checks).every(v => v === 'ok');
        res.status(allOk ? 200 : 503).json({
            status: allOk ? 'ok' : 'degraded',
            checks,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ status: 'error', error: error.message });
    }
});

// Solver info endpoint
app.get('/api/solver/info', async (req, res) => {
    try {
        // Get version from solver binary
        let version = 'unknown';
        try {
            await new Promise((resolve) => {
                const proc = spawn(SOLVER_PATH, ['--version']);
                let output = '';
                proc.stdout.on('data', d => { output += d.toString(); });
                proc.stderr.on('data', d => { output += d.toString(); });
                proc.on('close', () => {
                    const match = output.match(/[\d]+\.[\d]+\.[\d]+/);
                    if (match) version = match[0];
                    resolve();
                });
                proc.on('error', resolve);
                setTimeout(() => { proc.kill(); resolve(); }, 3000);
            });
        } catch { /* ignore */ }

        res.json({
            version,
            solverPath: SOLVER_PATH,
            capabilities: {
                coordinate_systems: ['cartesian', 'polar'],
                nonlinear_solvers: ['picard', 'newton_krylov'],
                features: ['coarsening', 'adaptive_mesh', 'transient', 'sliding_mesh', 'force_calculation']
            },
            build: {
                nodeVersion: process.version,
                platform: process.platform
            }
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get list of user output folders with details
app.get('/api/user-outputs', async (req, res) => {
    try {
        const { userId } = req.query;
        const userIdKey = userId || 'default';

        // Sanitize userId to prevent directory traversal
        const safeUserId = userIdKey.replace(/[^a-zA-Z0-9_-]/g, '');
        const userOutputDir = path.join(OUTPUTS_DIR, `${safeUserId}`);

        // Check if user output directory exists
        try {
            await fs.access(userOutputDir);
        } catch {
            return res.json({
                success: true,
                outputs: []
            });
        }

        const items = await fs.readdir(userOutputDir, { withFileTypes: true });
        const outputFolders = [];

        for (const item of items) {
            if (item.isDirectory()) {
                const folderPath = path.join(userOutputDir, item.name);

                // Check if this is an analysis result folder
                if (await isAnalysisResult(folderPath)) {
                    try {
                        // Get folder stats
                        const stats = await fs.stat(folderPath);

                        // Calculate folder size
                        const size = await getDirectorySize(folderPath);

                        // Count steps (CSV and/or TIFF)
                        let stepCount = 0;
                        try {
                            const azFolder = path.join(folderPath, 'Az');
                            const azFiles = await fs.readdir(azFolder);
                            stepCount = countTransientSteps(azFiles);
                        } catch {
                            // Az folder might not exist
                            stepCount = 0;
                        }

                        outputFolders.push({
                            name: item.name,
                            timestamp: item.name.replace('output_', ''), // Extract timestamp if present
                            created: stats.birthtime.toISOString(),
                            size: size,
                            sizeFormatted: formatBytes(size),
                            steps: stepCount
                        });
                    } catch (error) {
                        console.error(`Error processing folder ${item.name}:`, error);
                    }
                }
            }
        }

        // Sort by timestamp (newest first)
        outputFolders.sort((a, b) => b.timestamp.localeCompare(a.timestamp));

        res.json({
            success: true,
            outputs: outputFolders
        });

    } catch (error) {
        console.error('Error listing user outputs:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Delete multiple output folders (bulk delete)
// NOTE: This route must be defined BEFORE /api/user-outputs/:folderName
// to prevent Express from matching "bulk" as a :folderName parameter.
app.delete('/api/user-outputs/bulk', async (req, res) => {
    try {
        const { userId, folderNames } = req.body;
        const userIdKey = userId || 'default';

        // Validate inputs
        if (!Array.isArray(folderNames) || folderNames.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'folderNames must be a non-empty array'
            });
        }

        if (folderNames.length > 100) {
            return res.status(400).json({
                success: false,
                error: 'Cannot delete more than 100 folders at once'
            });
        }

        const safeUserId = userIdKey.replace(/[^a-zA-Z0-9_-]/g, '');
        const userOutputDir = path.join(OUTPUTS_DIR, safeUserId);
        const resolvedUserOutputDir = path.resolve(userOutputDir);

        const results = [];

        for (const folderName of folderNames) {
            const result = { name: folderName };

            try {
                const safeFolderName = path.basename(folderName);
                const folderPath = path.join(userOutputDir, safeFolderName);
                const resolvedFolderPath = path.resolve(folderPath);

                // Security check
                if (!resolvedFolderPath.startsWith(resolvedUserOutputDir)) {
                    result.success = false;
                    result.error = 'Path traversal detected';
                    results.push(result);
                    continue;
                }

                // Check if exists
                try {
                    await fs.access(folderPath);
                } catch {
                    result.success = false;
                    result.error = 'Folder not found';
                    results.push(result);
                    continue;
                }

                // Verify it's an analysis result
                if (!await isAnalysisResult(folderPath)) {
                    result.success = false;
                    result.error = 'Not an analysis result folder';
                    results.push(result);
                    continue;
                }

                // Delete folder
                await fs.rm(folderPath, { recursive: true, force: true });

                console.log(`Bulk deleted: ${safeUserId}/${safeFolderName}`);

                result.success = true;
                results.push(result);

            } catch (error) {
                result.success = false;
                result.error = error.message;
                results.push(result);
            }
        }

        // Check if any failed
        const failedCount = results.filter(r => !r.success).length;
        const successCount = results.filter(r => r.success).length;

        res.json({
            success: true,
            message: `Deleted ${successCount} folder(s), ${failedCount} failed`,
            results: results
        });

    } catch (error) {
        console.error('Error in bulk delete:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Delete a specific output folder
app.delete('/api/user-outputs/:folderName', async (req, res) => {
    try {
        const { folderName } = req.params;
        const { userId } = req.query;
        const userIdKey = userId || 'default';

        // Sanitize inputs to prevent directory traversal
        const safeUserId = userIdKey.replace(/[^a-zA-Z0-9_-]/g, '');
        const safeFolderName = path.basename(folderName); // Prevent path traversal

        const userOutputDir = path.join(OUTPUTS_DIR, `${safeUserId}`);
        const folderPath = path.join(userOutputDir, safeFolderName);

        // Security check: ensure the resolved path is within user's output directory
        const resolvedFolderPath = path.resolve(folderPath);
        const resolvedUserOutputDir = path.resolve(userOutputDir);

        if (!resolvedFolderPath.startsWith(resolvedUserOutputDir)) {
            return res.status(403).json({
                success: false,
                error: 'Access denied: Path traversal detected'
            });
        }

        // Check if folder exists and is an analysis result
        try {
            await fs.access(folderPath);
        } catch {
            return res.status(404).json({
                success: false,
                error: 'Output folder not found'
            });
        }

        // Verify it's an analysis result folder
        if (!await isAnalysisResult(folderPath)) {
            return res.status(400).json({
                success: false,
                error: 'Not an analysis result folder'
            });
        }

        // Delete the folder recursively
        await fs.rm(folderPath, { recursive: true, force: true });

        console.log(`Deleted output folder: ${safeUserId}/${safeFolderName}`);

        res.json({
            success: true,
            message: 'Output folder deleted successfully'
        });

    } catch (error) {
        console.error('Error deleting output folder:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Rename an output folder
app.put('/api/user-outputs/:folderName/rename', async (req, res) => {
    try {
        const { folderName } = req.params;
        const { userId, newName } = req.body;
        const userIdKey = userId || 'default';

        // Sanitize inputs
        const safeUserId = userIdKey.replace(/[^a-zA-Z0-9_-]/g, '');
        const safeFolderName = path.basename(folderName);
        const safeNewName = path.basename(newName).replace(/[^a-zA-Z0-9_\-\.]/g, '_');

        // Validate new name
        if (!safeNewName || safeNewName.length === 0 || safeNewName.length > 100) {
            return res.status(400).json({
                success: false,
                error: 'Invalid folder name (1-100 characters, alphanumeric and _-. only)'
            });
        }

        const userOutputDir = path.join(OUTPUTS_DIR, safeUserId);
        const oldPath = path.join(userOutputDir, safeFolderName);
        const newPath = path.join(userOutputDir, safeNewName);

        // Security check
        const resolvedOldPath = path.resolve(oldPath);
        const resolvedNewPath = path.resolve(newPath);
        const resolvedUserOutputDir = path.resolve(userOutputDir);

        if (!resolvedOldPath.startsWith(resolvedUserOutputDir) ||
            !resolvedNewPath.startsWith(resolvedUserOutputDir)) {
            return res.status(403).json({
                success: false,
                error: 'Access denied: Path traversal detected'
            });
        }

        // Check if old folder exists and is an analysis result
        try {
            await fs.access(oldPath);
        } catch {
            return res.status(404).json({
                success: false,
                error: 'Output folder not found'
            });
        }

        if (!await isAnalysisResult(oldPath)) {
            return res.status(400).json({
                success: false,
                error: 'Not an analysis result folder'
            });
        }

        // Check if new name already exists
        try {
            await fs.access(newPath);
            return res.status(409).json({
                success: false,
                error: 'Folder with this name already exists'
            });
        } catch {
            // Good - new path doesn't exist
        }

        // Rename folder
        await fs.rename(oldPath, newPath);

        console.log(`Renamed output folder: ${safeUserId}/${safeFolderName} -> ${safeNewName}`);

        res.json({
            success: true,
            message: 'Folder renamed successfully',
            newName: safeNewName
        });

    } catch (error) {
        console.error('Error renaming output folder:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get description for an output folder
app.get('/api/user-outputs/:folderName/description', async (req, res) => {
    try {
        const { folderName } = req.params;
        const { userId } = req.query;
        const userIdKey = userId || 'default';

        // Sanitize inputs
        const safeUserId = userIdKey.replace(/[^a-zA-Z0-9_-]/g, '');
        const safeFolderName = path.basename(folderName);

        const userOutputDir = path.join(OUTPUTS_DIR, safeUserId);
        const folderPath = path.join(userOutputDir, safeFolderName);
        const descPath = path.join(folderPath, 'description.txt');

        // Security check
        const resolvedFolderPath = path.resolve(folderPath);
        const resolvedUserOutputDir = path.resolve(userOutputDir);

        if (!resolvedFolderPath.startsWith(resolvedUserOutputDir)) {
            return res.status(403).json({
                success: false,
                error: 'Access denied: Path traversal detected'
            });
        }

        // Check if folder exists
        try {
            await fs.access(folderPath);
        } catch {
            return res.status(404).json({
                success: false,
                error: 'Output folder not found'
            });
        }

        // Verify it's an analysis result
        if (!await isAnalysisResult(folderPath)) {
            return res.status(400).json({
                success: false,
                error: 'Not an analysis result folder'
            });
        }

        // Read description file (if exists)
        let description = '';
        try {
            description = await fs.readFile(descPath, 'utf8');
        } catch {
            // File doesn't exist - return empty string
            description = '';
        }

        res.json({
            success: true,
            description: description
        });

    } catch (error) {
        console.error('Error reading description:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Update description for an output folder
app.put('/api/user-outputs/:folderName/description', async (req, res) => {
    try {
        const { folderName } = req.params;
        const { userId, description } = req.body;
        const userIdKey = userId || 'default';

        // Sanitize inputs
        const safeUserId = userIdKey.replace(/[^a-zA-Z0-9_-]/g, '');
        const safeFolderName = path.basename(folderName);

        // Validate description length
        if (description && description.length > 10000) {
            return res.status(400).json({
                success: false,
                error: 'Description too long (max 10000 characters)'
            });
        }

        const userOutputDir = path.join(OUTPUTS_DIR, safeUserId);
        const folderPath = path.join(userOutputDir, safeFolderName);
        const descPath = path.join(folderPath, 'description.txt');

        // Security check
        const resolvedFolderPath = path.resolve(folderPath);
        const resolvedUserOutputDir = path.resolve(userOutputDir);

        if (!resolvedFolderPath.startsWith(resolvedUserOutputDir)) {
            return res.status(403).json({
                success: false,
                error: 'Access denied: Path traversal detected'
            });
        }

        // Check if folder exists
        try {
            await fs.access(folderPath);
        } catch {
            return res.status(404).json({
                success: false,
                error: 'Output folder not found'
            });
        }

        // Verify it's an analysis result
        if (!await isAnalysisResult(folderPath)) {
            return res.status(400).json({
                success: false,
                error: 'Not an analysis result folder'
            });
        }

        // Write or delete description file
        if (description && description.trim().length > 0) {
            await fs.writeFile(descPath, description, 'utf8');
            console.log(`Updated description for: ${safeUserId}/${safeFolderName}`);
        } else {
            // Delete file if description is empty
            try {
                await fs.unlink(descPath);
                console.log(`Deleted description for: ${safeUserId}/${safeFolderName}`);
            } catch {
                // File doesn't exist - ignore
            }
        }

        res.json({
            success: true,
            message: 'Description updated successfully'
        });

    } catch (error) {
        console.error('Error updating description:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// ============================================================
// Field Query: Get field values at a physical coordinate
// ============================================================

// GET /api/results/:resultFolder/field-at-point
// Query params:
//   userId - user id
//   x      - physical x [m] (Cartesian) or r [m] (Polar)
//   y      - physical y [m] (Cartesian) or theta [rad] (Polar)
//   step   - step number (0-based, default 0)
app.get('/api/results/:resultFolder/field-at-point', async (req, res) => {
    try {
        const { resultFolder } = req.params;
        const { userId, x, y, step } = req.query;

        const userIdKey = (userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');
        const safeFolderName = path.basename(resultFolder);
        const folderPath = path.join(OUTPUTS_DIR, userIdKey, safeFolderName);

        // Security: verify path stays inside user's output dir
        const resolvedFolder = path.resolve(folderPath);
        const resolvedBase = path.resolve(path.join(OUTPUTS_DIR, userIdKey));
        if (!resolvedFolder.startsWith(resolvedBase)) {
            return res.status(403).json({ success: false, error: 'Access denied' });
        }

        // Load conditions
        const conditionsPath = path.join(folderPath, 'conditions.json');
        const conditions = JSON.parse(await fs.readFile(conditionsPath, 'utf8'));

        const cs = conditions.coordinate_system || 'cartesian';
        const xVal = parseFloat(x);
        const yVal = parseFloat(y);
        const stepNum = parseInt(step || '0');

        if (isNaN(xVal) || isNaN(yVal)) {
            return res.status(400).json({ success: false, error: 'x and y must be numeric' });
        }

        // Determine grid dimensions and convert to fractional indices
        let nx, ny, dx, dy, i_float, j_float;
        if (cs === 'polar') {
            const rStart = conditions.polar?.r_start ?? 0;
            dx = conditions.dr;
            dy = conditions.dtheta;
            nx = conditions.image_width;  // nr = columns
            ny = conditions.image_height; // ntheta = rows
            i_float = (xVal - rStart) / dx; // x = r
            j_float = yVal / dy;            // y = theta
        } else {
            dx = conditions.dx;
            dy = conditions.dy;
            nx = conditions.image_width;
            ny = conditions.image_height;
            i_float = xVal / dx;
            j_float = yVal / dy;
        }

        // Clamp to valid range
        i_float = Math.max(0, Math.min(nx - 1, i_float));
        j_float = Math.max(0, Math.min(ny - 1, j_float));

        // Load Az CSV
        const stepStr = String(stepNum).padStart(4, '0');
        const csvPath = path.join(folderPath, 'Az', `step_${stepStr}.csv`);
        const csvContent = await fs.readFile(csvPath, 'utf8');

        // Parse CSV into 2D array (rows = j, cols = i)
        const rows = csvContent.trim().split('\n').map(r => r.split(',').map(Number));
        const getAz = (i, j) => {
            const ri = Math.max(0, Math.min(ny - 1, j));
            const ci = Math.max(0, Math.min(nx - 1, i));
            return (rows[ri] && rows[ri][ci] !== undefined) ? rows[ri][ci] : 0;
        };

        // Bilinear interpolation for Az
        const i0 = Math.floor(i_float), i1 = Math.min(i0 + 1, nx - 1);
        const j0 = Math.floor(j_float), j1 = Math.min(j0 + 1, ny - 1);
        const fi = i_float - i0, fj = j_float - j0;
        const Az = getAz(i0, j0) * (1 - fi) * (1 - fj)
                 + getAz(i1, j0) * fi * (1 - fj)
                 + getAz(i0, j1) * (1 - fi) * fj
                 + getAz(i1, j1) * fi * fj;

        // Central difference for B = curl A (Cartesian: Bx=dAz/dy, By=-dAz/dx)
        const Bx = (getAz(Math.round(i_float), Math.min(Math.round(j_float) + 1, ny - 1))
                  - getAz(Math.round(i_float), Math.max(Math.round(j_float) - 1, 0))) / (2 * dy);
        const By = -(getAz(Math.min(Math.round(i_float) + 1, nx - 1), Math.round(j_float))
                   - getAz(Math.max(Math.round(i_float) - 1, 0), Math.round(j_float))) / (2 * dx);
        const Babs = Math.sqrt(Bx * Bx + By * By);

        res.json({
            success: true,
            coordinate_system: cs,
            x: xVal, y: yVal,
            step: stepNum,
            Az, Bx, By, Babs
        });
    } catch (error) {
        if (error.code === 'ENOENT') {
            return res.status(404).json({ success: false, error: 'Result folder or step not found' });
        }
        res.status(500).json({ success: false, error: error.message });
    }
});

// ============================================================
// Async Job Queue API
// ============================================================

// Submit async solve job (non-blocking, returns jobId immediately)
app.post('/api/jobs', async (req, res) => {
    try {
        const { configFile, imageFile, userId, materialLibraryFile } = req.body;
        const userIdKey = (userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');

        if (!configFile || !imageFile) {
            return res.status(400).json({ success: false, error: 'configFile and imageFile are required' });
        }

        // Build paths
        const userDir = getUserDir(userIdKey);
        const configPath = path.join(userDir, path.basename(configFile));
        const userUploadDir = getUserUploadsDir(userIdKey);
        const imagePath = path.join(userUploadDir, path.basename(imageFile));

        // Validate files exist
        await fs.access(configPath);
        await fs.access(imagePath);
        await fs.access(SOLVER_PATH);

        // Merge material library if provided
        let effectiveConfigPath = configPath;
        let mergedTempPath = null;
        if (materialLibraryFile) {
            const libDir  = getUserLibsDir(userIdKey);
            const libPath = path.join(libDir, path.basename(materialLibraryFile));
            if (!path.resolve(libPath).startsWith(path.resolve(libDir))) {
                throw new Error('Invalid library path');
            }
            const configYaml = await fs.readFile(configPath, 'utf8');
            const libYaml    = await fs.readFile(libPath, 'utf8');
            const mergedYaml = mergeLibraryIntoConfig(configYaml, libYaml);
            mergedTempPath   = path.join(userDir, `.merged_${Date.now()}.yaml`);
            await fs.writeFile(mergedTempPath, mergedYaml);
            effectiveConfigPath = mergedTempPath;
        }

        const jobId = require('crypto').randomUUID();
        const outputPath = await prepareUserOutputDirectory(userIdKey);

        const job = {
            jobId,
            userId: userIdKey,
            status: 'running',
            progress: 0,
            log: [],
            resultPath: null,
            process: null,
            created: new Date().toISOString(),
            finished: null
        };
        jobs.set(jobId, job);

        // Spawn solver in background
        const solverProcess = spawn(SOLVER_PATH, [effectiveConfigPath, imagePath, outputPath], { cwd: BASE_DIR });
        job.process = solverProcess;

        let outputBuffer = '';
        solverProcess.stdout.on('data', (data) => {
            outputBuffer += data.toString();
            const lines = outputBuffer.split('\n');
            outputBuffer = lines.pop() || '';
            for (const line of lines) {
                if (!line.trim()) continue;
                job.log.push(line);
                if (job.log.length > 500) job.log.shift(); // cap log

                const stepMatch = line.match(/---\s*Step\s+(\d+)\s*\/\s*(\d+)\s*---/i);
                if (stepMatch) {
                    const current = parseInt(stepMatch[1]);
                    const total = parseInt(stepMatch[2]);
                    job.progress = Math.round(((current - 1) / total) * 100);
                }
                if (line.includes('completed successfully')) {
                    job.progress = 100;
                }
            }
        });
        solverProcess.stderr.on('data', (data) => {
            const text = data.toString();
            for (const line of text.split('\n')) {
                if (line.trim()) job.log.push(`[err] ${line}`);
            }
        });
        solverProcess.on('close', (code) => {
            job.status = code === 0 ? 'completed' : 'failed';
            job.finished = new Date().toISOString();
            job.process = null;
            if (code === 0) {
                job.resultPath = outputPath;
                job.progress = 100;
            }
            if (mergedTempPath) {
                fs.unlink(mergedTempPath).catch(() => {});
                mergedTempPath = null;
            }
            console.log(`Job ${jobId} finished with code ${code}`);
        });
        solverProcess.on('error', (err) => {
            job.status = 'failed';
            job.finished = new Date().toISOString();
            job.process = null;
            job.log.push(`[error] ${err.message}`);
            if (mergedTempPath) {
                fs.unlink(mergedTempPath).catch(() => {});
                mergedTempPath = null;
            }
        });

        res.json({ success: true, jobId, status: 'running' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// List jobs for a user
app.get('/api/jobs', (req, res) => {
    try {
        const { userId } = req.query;
        const userIdKey = (userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');

        const userJobs = Array.from(jobs.values())
            .filter(j => j.userId === userIdKey)
            .map(({ jobId, status, progress, resultPath, created, finished }) =>
                ({ jobId, status, progress, resultPath, created, finished }))
            .sort((a, b) => new Date(b.created) - new Date(a.created));

        res.json({ success: true, jobs: userJobs });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get single job status and log tail
app.get('/api/jobs/:jobId', (req, res) => {
    try {
        const job = jobs.get(req.params.jobId);
        if (!job) return res.status(404).json({ success: false, error: 'Job not found' });

        const logTail = job.log.slice(-50);
        res.json({
            success: true,
            jobId: job.jobId,
            status: job.status,
            progress: job.progress,
            resultPath: job.resultPath,
            created: job.created,
            finished: job.finished,
            logTail
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Cancel a running job
app.delete('/api/jobs/:jobId', async (req, res) => {
    try {
        const job = jobs.get(req.params.jobId);
        if (!job) return res.status(404).json({ success: false, error: 'Job not found' });

        if (job.status !== 'running' || !job.process) {
            return res.json({ success: true, message: `Job already ${job.status}` });
        }

        job.process.kill('SIGTERM');
        const killTimer = setTimeout(() => {
            if (job.process) job.process.kill('SIGKILL');
        }, 5000);
        job.process.once('close', () => clearTimeout(killTimer));

        job.status = 'cancelled';
        job.finished = new Date().toISOString();
        job.process = null;

        res.json({ success: true, message: 'Job cancelled' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// =====================================================
// Material Library API
// =====================================================

// List library files for a user
app.get('/api/material-libraries', async (req, res) => {
    try {
        const userId = (req.query.userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');
        const libDir = getUserLibsDir(userId);
        await fs.mkdir(libDir, { recursive: true });

        const files = await fs.readdir(libDir);
        const yamlFiles = files.filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));

        const entries = await Promise.all(yamlFiles.map(async (filename) => {
            const fp = path.join(libDir, filename);
            const stat = await fs.stat(fp);
            return { filename, size: stat.size, modified: stat.mtime.toISOString() };
        }));

        res.json({ success: true, libraries: entries });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Upload a new library file (multipart)
app.post('/api/material-libraries', upload.single('library'), async (req, res) => {
    try {
        const userId = (req.body.userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');

        if (!req.file) {
            return res.status(400).json({ success: false, error: 'No file uploaded' });
        }

        const filename = path.basename(req.file.originalname).replace(/[^a-zA-Z0-9_\-. ]/g, '_');
        if (!filename.endsWith('.yaml') && !filename.endsWith('.yml')) {
            await fs.unlink(req.file.path).catch(() => {});
            return res.status(400).json({ success: false, error: 'Only .yaml/.yml files are allowed' });
        }

        const libDir = getUserLibsDir(userId);
        await fs.mkdir(libDir, { recursive: true });

        const destPath = path.join(libDir, filename);
        if (!path.resolve(destPath).startsWith(path.resolve(libDir))) {
            await fs.unlink(req.file.path).catch(() => {});
            return res.status(400).json({ success: false, error: 'Invalid filename' });
        }

        const content = await fs.readFile(req.file.path, 'utf8');
        await fs.unlink(req.file.path).catch(() => {});

        // Validate YAML syntax
        try {
            yaml.load(content);
        } catch (yamlErr) {
            return res.status(400).json({ success: false, error: `YAML parse error: ${yamlErr.message}` });
        }

        await fs.writeFile(destPath, content, 'utf8');
        res.json({ success: true, filename });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get library file content
app.get('/api/material-libraries/:filename', async (req, res) => {
    try {
        const userId = (req.query.userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');
        const filename = path.basename(req.params.filename);
        const libDir = getUserLibsDir(userId);
        const filePath = path.join(libDir, filename);

        if (!path.resolve(filePath).startsWith(path.resolve(libDir))) {
            return res.status(400).json({ success: false, error: 'Invalid filename' });
        }

        const content = await fs.readFile(filePath, 'utf8');
        res.type('text/plain').send(content);
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ success: false, error: 'Library file not found' });
        } else {
            res.status(500).json({ success: false, error: error.message });
        }
    }
});

// Save (create or update) library file content via JSON body
app.put('/api/material-libraries/:filename', async (req, res) => {
    try {
        const userId = (req.body.userId || req.query.userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');
        const filename = path.basename(req.params.filename).replace(/[^a-zA-Z0-9_\-. ]/g, '_');

        if (!filename.endsWith('.yaml') && !filename.endsWith('.yml')) {
            return res.status(400).json({ success: false, error: 'Only .yaml/.yml files are allowed' });
        }

        const content = req.body.content;
        if (typeof content !== 'string') {
            return res.status(400).json({ success: false, error: 'content field (string) is required' });
        }

        // Validate YAML syntax
        try {
            yaml.load(content);
        } catch (yamlErr) {
            return res.status(400).json({ success: false, error: `YAML parse error: ${yamlErr.message}` });
        }

        const libDir = getUserLibsDir(userId);
        await fs.mkdir(libDir, { recursive: true });

        const filePath = path.join(libDir, filename);
        if (!path.resolve(filePath).startsWith(path.resolve(libDir))) {
            return res.status(400).json({ success: false, error: 'Invalid filename' });
        }

        await fs.writeFile(filePath, content, 'utf8');
        res.json({ success: true, filename });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Delete a library file
app.delete('/api/material-libraries/:filename', async (req, res) => {
    try {
        const userId = (req.query.userId || 'default').replace(/[^a-zA-Z0-9_-]/g, '');
        const filename = path.basename(req.params.filename);
        const libDir = getUserLibsDir(userId);
        const filePath = path.join(libDir, filename);

        if (!path.resolve(filePath).startsWith(path.resolve(libDir))) {
            return res.status(400).json({ success: false, error: 'Invalid filename' });
        }

        await fs.unlink(filePath);
        res.json({ success: true, message: `Deleted ${filename}` });
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ success: false, error: 'Library file not found' });
        } else {
            res.status(500).json({ success: false, error: error.message });
        }
    }
});

// ルートへのアクセス
app.get('/', (req, res) => {
    res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// サーバー起動
app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('MagFDM Visualizer Server (Integrated)');
    console.log('='.repeat(60));
    console.log(`Server running at: http://localhost:${PORT}`);
    console.log(`Serving files from: ${PUBLIC_DIR}`);
    console.log(`CSV data directory: ${BASE_DIR}`);
    console.log(`Upload directory: ${UPLOAD_DIR}`);
    console.log(`Solver path: ${SOLVER_PATH}`);
    console.log(`Config file: ${CONFIG_PATH}`);
    console.log('='.repeat(60));
    console.log('Available APIs:');
    console.log('  GET  /api/config          - Get YAML configuration');
    console.log('  POST /api/config          - Save YAML configuration');
    console.log('  POST /api/upload-image    - Upload material image');
    console.log('  GET  /api/images          - List uploaded images');
    console.log('  POST /api/solve           - Run FDM solver');
    console.log('  GET  /api/results         - List result files');
    console.log('  GET  /api/detect-steps    - Detect number of transient steps');
    console.log('  GET  /api/load-csv        - Load CSV file for specific step');
    console.log('='.repeat(60));
    console.log('Press Ctrl+C to stop the server');
    console.log('');
});

// ===== 過渡解析対応API =====

// 解析に使用された画像ファイルを取得
app.get('/api/get-material-image', async (req, res) => {
    try {
        const parentDir = BASE_DIR;

        const potentialImageNames = [];

        for (const imageName of potentialImageNames) {
            const imagePath = path.join(parentDir, imageName);
            try {
                await fs.access(imagePath);
                return res.sendFile(imagePath);
            } catch {
                continue;
            }
        }

        // uploadsフォルダもチェック
        const uploadFiles = await fs.readdir(UPLOAD_DIR);
        const imageFiles = uploadFiles.filter(f => /\.(png|jpg|jpeg|bmp)$/i.test(f));
        if (imageFiles.length > 0) {
            return res.sendFile(path.join(UPLOAD_DIR, imageFiles[imageFiles.length - 1]));
        }

        res.status(404).json({ success: false, error: 'Material image not found' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// YAMLから過渡解析設定を取得
app.get('/api/get-transient-config', async (req, res) => {
    try {
        const configData = await fs.readFile(CONFIG_PATH, 'utf8');
        const config = yaml.load(configData);

        const transientConfig = config.transient || {};

        res.json({
            success: true,
            enabled: transientConfig.enabled || false,
            enable_sliding: transientConfig.enable_sliding || false,
            slide_direction: transientConfig.slide_direction || 'vertical',
            slide_region_start: transientConfig.slide_region_start || 0,
            slide_region_end: transientConfig.slide_region_end || 0
        });
    } catch (error) {
        res.json({ success: false, error: error.message });
    }
});

// ステップ数の検出
// Counts step_XXXX.{csv,tiff} entries in the Az folder. Same step number with
// both CSV and TIFF is counted once. ".tmp" sentinels written by AsyncWriter
// mid-write are ignored.
app.get('/api/detect-steps', async (req, res) => {
    try {
        const resultPath = req.query.result;
        if (!resultPath) {
            return res.json({ success: false, error: 'Result path required' });
        }

        const azFolder = path.join(BASE_DIR, resultPath, 'Az');
        const files = await fs.readdir(azFolder);

        const steps = countTransientSteps(files);
        const hasTiff = files.some(f => /^step_\d{4}\.tiff$/.test(f));
        const hasCsv  = files.some(f => /^step_\d{4}\.csv$/.test(f));

        res.json({ success: true, steps, hasTiff, hasCsv });
    } catch (error) {
        res.json({ success: false, error: error.message, steps: 1 });
    }
});

// Format-agnostic field loader.
//
// CSV path: decodes server-side into a JSON 2D array (legacy response shape).
// TIFF path: streams the raw TIFF file with Content-Type image/tiff and an
//            X-Field-* header set; the browser handles decoding via the
//            vendored geotiff bundle at /lib/geotiff.js. This avoids pulling
//            geotiff into the Node bundle (it transitively requires ESM-only
//            modules that pkg cannot handle).
//
// Resolution order (when both exist or no extension given):
//   - ?prefer=csv  -> CSV first, TIFF fallback
//   - (default)    -> TIFF first, CSV fallback
async function fileExists(p) {
    try { await fs.access(p); return true; } catch { return false; }
}

async function decodeCsv(filePath) {
    const content = await fs.readFile(filePath, 'utf8');
    const lines = content.trim().split('\n');
    const data = lines.map(line =>
        line.split(',').map(val => {
            if (val.length === 0) return null;  // active-only paths leave blanks for NaN
            const f = parseFloat(val);
            return Number.isNaN(f) ? null : f;
        })
    );
    data.reverse();
    return { data, precision: 'double' };
}

app.get('/api/load-field', async (req, res) => {
    try {
        const resultPath = req.query.result;
        const file = req.query.file;
        const prefer = req.query.prefer;  // "csv" | "tiff" | undefined
        if (!resultPath || !file) {
            return res.json({ success: false, error: 'Missing parameters' });
        }

        const base = file.replace(/\.(csv|tiff)$/i, '');
        const tiffPath = path.join(BASE_DIR, resultPath, base + '.tiff');
        const csvPath  = path.join(BASE_DIR, resultPath, base + '.csv');

        const preferCsv = prefer === 'csv';
        const order = preferCsv ? [csvPath, tiffPath] : [tiffPath, csvPath];

        for (const p of order) {
            if (!await fileExists(p)) continue;
            if (p.endsWith('.tiff')) {
                // Stream raw TIFF; the browser decodes it.
                res.setHeader('Content-Type', 'image/tiff');
                res.setHeader('X-Field-Format', 'tiff');
                return res.sendFile(p);
            } else {
                const { data, precision } = await decodeCsv(p);
                return res.json({ success: true, data, format: 'csv', precision });
            }
        }

        return res.json({
            success: false,
            error: `Neither ${base}.tiff nor ${base}.csv exists under ${resultPath}`
        });
    } catch (error) {
        res.json({ success: false, error: error.message });
    }
});

// Phase B.3: flux-linkage time series. The solver writes one row per
// transient step into `<result>/FluxLinkage/flux_linkage.csv` with a
// "step,<name1>,<name2>,..." header (one column per flux_linkage entry,
// whether path or material-pair). Returns { steps: [...], series: { name:
// [values] } } so a Plotly trace can be drawn straight from the response.
app.get('/api/get-flux-linkage', async (req, res) => {
    try {
        const resultPath = req.query.result;
        if (!resultPath) {
            return res.json({ success: false, error: 'Missing result parameter' });
        }
        const csvPath = path.join(BASE_DIR, resultPath, 'FluxLinkage', 'flux_linkage.csv');
        if (!await fileExists(csvPath)) {
            return res.json({
                success: false,
                error: `No flux_linkage.csv under ${resultPath}/FluxLinkage`,
            });
        }
        const text = await fs.readFile(csvPath, 'utf8');
        const lines = text.trim().split(/\r?\n/);
        if (lines.length < 2) {
            return res.json({
                success: true, steps: [], series: {},
                empty: true,
            });
        }
        const header = lines[0].split(',').map(s => s.trim());
        // Expect first column = "step"; everything after = series names.
        const stepCol = header[0];
        const seriesNames = header.slice(1);
        const steps = [];
        const series = {};
        for (const name of seriesNames) series[name] = [];
        for (let li = 1; li < lines.length; li++) {
            const row = lines[li].split(',');
            if (row.length < 2) continue;
            steps.push(Number(row[0]));
            for (let i = 0; i < seriesNames.length; i++) {
                const v = parseFloat(row[i + 1]);
                series[seriesNames[i]].push(Number.isFinite(v) ? v : null);
            }
        }
        res.json({
            success: true,
            step_column: stepCol,
            steps,
            series,
        });
    } catch (error) {
        res.json({ success: false, error: error.message });
    }
});

// 特定ステップのCSVファイル読み込み (deprecated; use /api/load-field).
// Kept for backwards compatibility with external scripts and older bookmarks.
let _loadCsvDeprecationWarned = false;
app.get('/api/load-csv', async (req, res) => {
    if (!_loadCsvDeprecationWarned) {
        console.warn('[deprecated] /api/load-csv hit; please migrate callers to /api/load-field which also handles TIFF.');
        _loadCsvDeprecationWarned = true;
    }
    try {
        const resultPath = req.query.result;
        const file = req.query.file; // e.g., "Az/step_0000.csv"

        if (!resultPath || !file) {
            return res.json({ success: false, error: 'Missing parameters' });
        }

        const filePath = path.join(BASE_DIR, resultPath, file);
        const content = await fs.readFile(filePath, 'utf8');

        // CSVをパース
        const lines = content.trim().split('\n');
        const data = lines.map(line =>
            line.split(',').map(val => parseFloat(val))
        );

        data.reverse(); // データ座標系から画像座標系に対応、Y軸反転

        res.json({ success: true, data: data });
    } catch (error) {
        res.json({ success: false, error: error.message });
    }
});

// Forces用の生テキストCSV読み込み（ヘッダー行とテキスト列を含む）
app.get('/api/load-csv-raw', async (req, res) => {
    try {
        const resultPath = req.query.result;
        const file = req.query.file; // e.g., "Forces/step_0000.csv"

        if (!resultPath || !file) {
            return res.status(400).send('Missing parameters');
        }

        const filePath = path.join(BASE_DIR, resultPath, file);
        const content = await fs.readFile(filePath, 'utf8');

        // 生のテキストとして返す
        res.type('text/plain').send(content);
    } catch (error) {
        res.status(500).send(`Error: ${error.message}`);
    }
});

// 解析条件情報の取得
app.get('/api/load-conditions', async (req, res) => {
    try {
        const resultPath = req.query.result;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        const conditionsPath = path.join(BASE_DIR, resultPath, 'conditions.json');

        // ファイルが存在するか確認
        await fs.access(conditionsPath);

        // JSONファイルを読み込んで送信
        const content = await fs.readFile(conditionsPath, 'utf8');
        const conditions = JSON.parse(content);
        res.json(conditions);
    } catch (error) {
        res.status(404).send(`Conditions file not found: ${error.message}`);
    }
});

// 境界画像の取得
app.get('/api/get-boundary-image', async (req, res) => {
    try {
        const resultPath = req.query.result;
        const step = parseInt(req.query.step) || 0;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        const stepName = `step_${String(step).padStart(4, '0')}`;
        const imagePath = path.join(BASE_DIR, resultPath, 'BoundaryImg', `${stepName}.png`);

        // ファイルが存在するか確認
        await fs.access(imagePath);

        // 画像ファイルを送信
        res.sendFile(imagePath);
    } catch (error) {
        res.status(404).send(`Boundary image not found: ${error.message}`);
    }
});

// ステップ入力画像の取得
app.get('/api/get-step-input-image', async (req, res) => {
    try {
        const resultPath = req.query.result;
        const step = parseInt(req.query.step) || 0;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        const stepName = `step_${String(step).padStart(4, '0')}`;
        const imagePath = path.join(BASE_DIR, resultPath, 'InputImg', `${stepName}.png`);

        // ファイルが存在するか確認
        await fs.access(imagePath);

        // 画像ファイルを送信
        res.sendFile(imagePath);
    } catch (error) {
        res.status(404).send(`Step input image not found: ${error.message}`);
    }
});

// Get coarsening mask image (if adaptive mesh was used)
// Now supports per-step masks: /api/get-coarsening-mask?result=...&step=1
app.get('/api/get-coarsening-mask', async (req, res) => {
    try {
        const resultPath = req.query.result;
        const step = parseInt(req.query.step) || 1;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        // Format step number with leading zeros (step_0001.png)
        const stepStr = String(step).padStart(4, '0');
        const maskPath = path.join(BASE_DIR, resultPath, 'CoarseningMask', `step_${stepStr}.png`);

        // Check if file exists
        await fs.access(maskPath);

        // Send image file
        res.sendFile(maskPath);
    } catch (error) {
        res.status(404).send(`Coarsening mask not found: ${error.message}`);
    }
});

// Get log.txt from result directory
app.get('/api/get-log', async (req, res) => {
    try {
        const resultPath = req.query.result;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        const logPath = path.join(BASE_DIR, resultPath, 'log.txt');

        // Check if file exists
        await fs.access(logPath);

        // Read and send log file
        const content = await fs.readFile(logPath, 'utf8');
        res.type('text/plain').send(content);
    } catch (error) {
        res.status(404).send(`Log file not found: ${error.message}`);
    }
});

// Get conditions.json for a result
app.get('/api/get-conditions', async (req, res) => {
    try {
        const resultPath = req.query.result;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        const conditionsPath = path.join(BASE_DIR, resultPath, 'conditions.json');

        // Check if file exists
        await fs.access(conditionsPath);

        // Read and send conditions file
        const content = await fs.readFile(conditionsPath, 'utf8');
        res.type('application/json').send(content);
    } catch (error) {
        res.status(404).send(`Conditions file not found: ${error.message}`);
    }
});
