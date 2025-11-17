const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const { exec, spawn } = require('child_process');
const multer = require('multer');
const yaml = require('js-yaml');

const app = express();
const PORT = process.env.PORT || 3000;

// アップロードディレクトリの設定
const UPLOAD_DIR = path.join(__dirname, '..', 'uploads');
const SOLVER_PATH = path.join(__dirname, '..', 'build', 'MagFDMsolver');
const CONFIG_PATH = path.join(__dirname, '..', 'sample_config.yaml');
const USER_CONFIGS_DIR = path.join(__dirname, '..', 'configs');

// ディレクトリの作成
fs.mkdir(UPLOAD_DIR, { recursive: true }).catch(console.error);
fs.mkdir(USER_CONFIGS_DIR, { recursive: true }).catch(console.error);

// Image management constants
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_IMAGES_PER_USER = 20;

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

// 静的ファイルの提供
app.use(express.static('public'));
app.use('/icon', express.static(path.join(__dirname, 'icon')));

// 親ディレクトリのCSVファイルへのアクセス
app.use('/data', express.static(path.join(__dirname, '..')));

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

// Helper: Clean up expired user directories (run on server start)
async function cleanupExpiredUsers() {
    try {
        const expiryTime = Date.now() - (USER_EXPIRY_DAYS * 24 * 60 * 60 * 1000);

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

// Run cleanup on server start
cleanupExpiredUsers();

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

// ソルバーの実行
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
            cwd: path.join(__dirname, '..'),
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

// 出力ファイルの一覧
app.get('/api/results', async (req, res) => {
    try {
        const parentDir = path.join(__dirname, '..');
        const files = await fs.readdir(parentDir, { withFileTypes: true });

        // output_* または transient_output などのフォルダを検出
        const resultFolders = [];

        for (const file of files) {
            if (file.isDirectory()) {
                const folderName = file.name;
                // output_で始まる、またはtransient_outputなど
                if (folderName.startsWith('output_') || folderName === 'transient_output') {
                    const folderPath = path.join(parentDir, folderName);

                    // Azフォルダの存在確認
                    try {
                        const azFolder = path.join(folderPath, 'Az');
                        await fs.access(azFolder);

                        // ステップ数をカウント
                        const azFiles = await fs.readdir(azFolder);
                        const stepFiles = azFiles.filter(f => /^step_\d{4}\.csv$/.test(f));

                        resultFolders.push({
                            name: folderName,
                            path: folderName,
                            timestamp: folderName.replace('output_', ''),
                            steps: stepFiles.length
                        });
                    } catch {
                        // Azフォルダがない場合はスキップ
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

// ルートへのアクセス
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// サーバー起動
app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('MagFDM Visualizer Server (Integrated)');
    console.log('='.repeat(60));
    console.log(`Server running at: http://localhost:${PORT}`);
    console.log(`Serving files from: ${path.join(__dirname, 'public')}`);
    console.log(`CSV data directory: ${path.join(__dirname, '..')}`);
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
        const parentDir = path.join(__dirname, '..');

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
app.get('/api/detect-steps', async (req, res) => {
    try {
        const resultPath = req.query.result;
        if (!resultPath) {
            return res.json({ success: false, error: 'Result path required' });
        }

        const azFolder = path.join(__dirname, '..', resultPath, 'Az');
        const files = await fs.readdir(azFolder);

        // step_XXXX.csv 形式のファイルをカウント
        const stepFiles = files.filter(f => /^step_\d{4}\.csv$/.test(f));

        res.json({
            success: true,
            steps: stepFiles.length
        });
    } catch (error) {
        res.json({ success: false, error: error.message, steps: 1 });
    }
});

// 特定ステップのCSVファイル読み込み
app.get('/api/load-csv', async (req, res) => {
    try {
        const resultPath = req.query.result;
        const file = req.query.file; // e.g., "Az/step_0000.csv"

        if (!resultPath || !file) {
            return res.json({ success: false, error: 'Missing parameters' });
        }

        const filePath = path.join(__dirname, '..', resultPath, file);
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

        const filePath = path.join(__dirname, '..', resultPath, file);
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

        const conditionsPath = path.join(__dirname, '..', resultPath, 'conditions.json');

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
        const imagePath = path.join(__dirname, '..', resultPath, 'BoundaryImg', `${stepName}.png`);

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
        const imagePath = path.join(__dirname, '..', resultPath, 'InputImg', `${stepName}.png`);

        // ファイルが存在するか確認
        await fs.access(imagePath);

        // 画像ファイルを送信
        res.sendFile(imagePath);
    } catch (error) {
        res.status(404).send(`Step input image not found: ${error.message}`);
    }
});

// Get log.txt from result directory
app.get('/api/get-log', async (req, res) => {
    try {
        const resultPath = req.query.result;

        if (!resultPath) {
            return res.status(400).send('Missing result parameter');
        }

        const logPath = path.join(__dirname, '..', resultPath, 'log.txt');

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

        const conditionsPath = path.join(__dirname, '..', resultPath, 'conditions.json');

        // Check if file exists
        await fs.access(conditionsPath);

        // Read and send conditions file
        const content = await fs.readFile(conditionsPath, 'utf8');
        res.type('application/json').send(content);
    } catch (error) {
        res.status(404).send(`Conditions file not found: ${error.message}`);
    }
});
