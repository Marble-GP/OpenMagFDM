const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const { exec } = require('child_process');
const multer = require('multer');
const yaml = require('js-yaml');

const app = express();
const PORT = process.env.PORT || 3000;

// アップロードディレクトリの設定
const UPLOAD_DIR = path.join(__dirname, '..', 'uploads');
const SOLVER_PATH = path.join(__dirname, '..', 'build', 'MagFDMsolver');
const CONFIG_PATH = path.join(__dirname, '..', 'sample_config.yaml');

// アップロードディレクトリの作成
fs.mkdir(UPLOAD_DIR, { recursive: true }).catch(console.error);

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

// 親ディレクトリのCSVファイルへのアクセス
app.use('/data', express.static(path.join(__dirname, '..')));

// ===== API エンドポイント =====

// YAML設定ファイルの読み込み
app.get('/api/config', async (req, res) => {
    try {
        const configData = await fs.readFile(CONFIG_PATH, 'utf8');
        const config = yaml.load(configData);
        res.json({
            success: true,
            config: config,
            yaml: configData
        });
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
        const { yaml: yamlContent } = req.body;

        // YAML検証
        const parsed = yaml.load(yamlContent);

        // ファイルに書き込み
        await fs.writeFile(CONFIG_PATH, yamlContent, 'utf8');

        res.json({
            success: true,
            message: 'Configuration saved successfully',
            config: parsed
        });
    } catch (error) {
        res.status(400).json({
            success: false,
            error: error.message
        });
    }
});

// 画像ファイルのアップロード
app.post('/api/upload-image', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'No file uploaded'
            });
        }

        res.json({
            success: true,
            filename: req.file.filename,
            path: req.file.path,
            originalName: req.file.originalname
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// アップロードされた画像の一覧
app.get('/api/images', async (req, res) => {
    try {
        const files = await fs.readdir(UPLOAD_DIR);
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

// ソルバーの実行
app.post('/api/solve', async (req, res) => {
    try {
        const { configFile, imageFile, outputFile } = req.body;

        // パスの構築
        const configPath = configFile || CONFIG_PATH;
        const imagePath = path.join(UPLOAD_DIR, imageFile);
        // const outputPath = outputFile || `output_${Date.now()}`;

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
    console.log('🧲 MagFDM Visualizer Server (Integrated)');
    console.log('='.repeat(60));
    console.log(`📡 Server running at: http://localhost:${PORT}`);
    console.log(`📁 Serving files from: ${path.join(__dirname, 'public')}`);
    console.log(`📊 CSV data directory: ${path.join(__dirname, '..')}`);
    console.log(`🖼️  Upload directory: ${UPLOAD_DIR}`);
    console.log(`⚙️  Solver path: ${SOLVER_PATH}`);
    console.log(`📝 Config file: ${CONFIG_PATH}`);
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
