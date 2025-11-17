// =====================================================
// OpenMagFDM Integrated Environment - Main Application
// =====================================================

// ===== Global State =====
const AppState = {
    currentTab: 'config',
    configData: null,
    uploadedImage: null,
    uploadedImageFilename: null,  // Uploaded image filename on server
    currentStep: 1,
    totalSteps: 1,
    resultsData: {},
    gridStack: null,
    aceEditor: null,  // Ace Editor instance
    yamlSchema: null,  // YAML schema for autocomplete
    userId: null,  // User identifier (from cookie)
    animationTimer: null,  // Animation interval timer
    isAnimating: false  // Animation state flag
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', async () => {
    initializeUserId();
    initializeTabs();
    await initializeConfigEditor();
    initializeDashboard();
    await refreshConfigList();
    await loadConfig();
    await refreshImageList();
    await refreshResultsList();
});

// ===== Tab Management =====
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update active tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });

    AppState.currentTab = tabName;

    // Tab-specific initialization
    if (tabName === 'dashboard' && !AppState.gridStack) {
        initializeDashboard();
    }
}

// ===== User Management =====
function initializeUserId() {
    // Get or create user ID from cookie
    let userId = getCookie('magfdm_user_id');
    if (!userId) {
        userId = 'user_' + Math.random().toString(36).substr(2, 9);
    }
    // Always update cookie expiration on every visit (365 days from now)
    setCookie('magfdm_user_id', userId, 365);
    AppState.userId = userId;
    console.log('User ID:', userId);

    // Display user ID in header
    const userIdDisplay = document.getElementById('userIdDisplay');
    if (userIdDisplay) {
        userIdDisplay.textContent = userId;
    }
}

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

function setCookie(name, value, days) {
    const expires = new Date();
    expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
}

// ===== Config Editor (Tab 1) =====
async function initializeConfigEditor() {
    // Load YAML schema
    try {
        const response = await fetch('/yaml-schema.json');
        AppState.yamlSchema = await response.json();
    } catch (error) {
        console.error('Failed to load YAML schema:', error);
    }

    // Initialize Ace Editor
    const editor = ace.edit('yamlEditor');
    editor.setTheme('ace/theme/monokai');
    editor.session.setMode('ace/mode/yaml');
    editor.setOptions({
        enableBasicAutocompletion: true,  // Enable autocompletion (we'll use only custom completer)
        enableLiveAutocompletion: true,
        enableSnippets: false,  // Disable snippets to avoid unwanted completions
        showPrintMargin: false,
        fontSize: '14px',
        tabSize: 2,
        useSoftTabs: true
    });

    // Add custom YAML autocompleter with context awareness
    const yamlCompleter = {
        getCompletions: function(editor, session, pos, prefix, callback) {
            if (!AppState.yamlSchema) {
                callback(null, []);
                return;
            }

            const completions = [];
            const keywords = AppState.yamlSchema.keywords;

            // Get parent context (nest recognition)
            const contextPath = getContextPath(editor, session, pos);
            const parentContext = contextPath.length > 0 ? contextPath[contextPath.length - 1] : null;
            const grandparentContext = contextPath.length > 1 ? contextPath[contextPath.length - 2] : null;
            let availableKeywords = [];

            if (parentContext) {
                const parentInfo = keywords[parentContext];

                // Check if parent accepts any child (like materials)
                if (parentInfo && parentInfo.acceptsAnyChild) {
                    // We're inside a container that accepts any child name
                    // Show the properties that can be used inside those children
                    if (parentInfo.childrenProperties) {
                        availableKeywords = parentInfo.childrenProperties;
                    }
                }
                // Check if grandparent accepts any child (we're inside a specific material)
                else if (grandparentContext) {
                    const grandparentInfo = keywords[grandparentContext];
                    if (grandparentInfo && grandparentInfo.acceptsAnyChild && grandparentInfo.childrenProperties) {
                        availableKeywords = grandparentInfo.childrenProperties;
                    }
                }
                // Normal parent with defined children
                else if (parentInfo && parentInfo.children) {
                    availableKeywords = parentInfo.children;
                } else {
                    // Parent doesn't have specific children, show all
                    availableKeywords = Object.keys(keywords);
                }
            } else {
                // Top level - show root-level keywords
                availableKeywords = Object.keys(keywords);
            }

            // Add keyword completions
            for (const keyword of availableKeywords) {
                const info = keywords[keyword];
                if (!info) continue;

                const completion = {
                    caption: keyword,
                    value: keyword + ': ',
                    meta: info.type || 'keyword',
                    score: 1000,
                    docHTML: `<b>${keyword}</b><br>${info.description}<br><code>${info.example || ''}</code>`
                };
                completions.push(completion);

                // Add value suggestions
                if (info.values) {
                    info.values.forEach(val => {
                        completions.push({
                            caption: `${keyword}: ${val}`,
                            value: `${keyword}: ${val}`,
                            meta: 'value',
                            score: 900
                        });
                    });
                }
            }

            // Add snippets based on context
            addContextSnippets(completions, parentContext, grandparentContext);

            callback(null, completions);
        }
    };

    // Remove all default completers and use only our custom completer
    ace.require('ace/ext/language_tools');
    editor.completers = [yamlCompleter];  // Only use our custom completer

    // Enable YAML validation on change
    editor.session.on('change', function() {
        validateYAML(editor);
    });

    AppState.aceEditor = editor;

    // Initial validation
    validateYAML(editor);
}

// Add context-aware snippets
function addContextSnippets(completions, parentContext, grandparentContext) {
    if (!AppState.yamlSchema || !AppState.yamlSchema.snippets) return;

    const snippets = AppState.yamlSchema.snippets;

    // Add boundary condition snippets (for inner, outer, left, right, top, bottom)
    const boundaryContexts = ['inner', 'outer', 'left', 'right', 'top', 'bottom'];
    if (boundaryContexts.includes(parentContext)) {
        // Add boundary type snippets
        if (snippets.boundary_dirichlet) {
            completions.push({
                caption: '[Snippet] Dirichlet (value=0)',
                value: snippets.boundary_dirichlet.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Snippet: Dirichlet Boundary</b><br>Creates type: dirichlet with value: 0.0'
            });
        }
        if (snippets.boundary_neumann) {
            completions.push({
                caption: '[Snippet] Neumann (value=0)',
                value: snippets.boundary_neumann.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Snippet: Neumann Boundary</b><br>Creates type: neumann with value: 0.0'
            });
        }
        if (snippets.boundary_periodic) {
            completions.push({
                caption: '[Snippet] Periodic',
                value: snippets.boundary_periodic.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Snippet: Periodic Boundary</b><br>Creates type: periodic with value: 0.0'
            });
        }
    }

    // Add material template snippet (when inside a specific material under materials)
    if (grandparentContext === 'materials' && parentContext) {
        if (snippets.material_template) {
            completions.push({
                caption: '[Snippet] Material Template',
                value: snippets.material_template.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Snippet: Material Definition</b><br>Creates full material definition with rgb, mu_r, jz, calc_force'
            });
        }
    }

    // Add transient template snippet (when parent is transient)
    if (parentContext === 'transient') {
        if (snippets.transient_template) {
            completions.push({
                caption: '[Snippet] Transient Template',
                value: snippets.transient_template.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Snippet: Transient Analysis</b><br>Creates full transient configuration'
            });
        }
    }
}

// Get full context path (all parent keywords) by analyzing indentation
function getContextPath(editor, session, pos) {
    const currentLine = pos.row;
    const currentIndent = getIndentLevel(session.getLine(currentLine));
    const path = [];

    // Build path by finding all parent keywords
    let searchIndent = currentIndent;
    for (let i = currentLine - 1; i >= 0; i--) {
        const line = session.getLine(i);
        const indent = getIndentLevel(line);

        // Found a line with less indentation
        if (indent < searchIndent) {
            const match = line.match(/^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:/);
            if (match) {
                path.unshift(match[1]); // Add to beginning of path
                searchIndent = indent;

                // Stop when we reach the top level
                if (indent === 0) break;
            }
        }
    }

    return path;
}

// Get parent context by analyzing indentation and previous lines
function getParentContext(editor, session, pos) {
    const path = getContextPath(editor, session, pos);
    return path.length > 0 ? path[path.length - 1] : null;
}

function getIndentLevel(line) {
    const match = line.match(/^(\s*)/);
    return match ? match[1].length : 0;
}

// YAML validation function with linting
function validateYAML(editor) {
    const content = editor.getValue();
    const annotations = [];

    try {
        // Parse YAML for syntax errors
        const yaml = window.jsyaml || jsyaml;
        const parsed = yaml.load(content);

        // Validate against schema
        if (AppState.yamlSchema) {
            validateAgainstSchema(editor, parsed, annotations);
        }
    } catch (error) {
        // YAML syntax error
        const lineMatch = error.message.match(/at line (\d+)/);
        const line = lineMatch ? parseInt(lineMatch[1]) - 1 : 0;

        annotations.push({
            row: line,
            column: 0,
            text: `Syntax error: ${error.message}`,
            type: 'error'
        });
    }

    editor.session.setAnnotations(annotations);
}

function validateAgainstSchema(editor, parsed, annotations) {
    if (!parsed || typeof parsed !== 'object') return;

    const keywords = AppState.yamlSchema.keywords;
    const session = editor.session;
    const lines = session.getDocument().getAllLines();

    // Check each line for unknown keywords
    lines.forEach((line, idx) => {
        const match = line.match(/^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:/);
        if (match) {
            const keyword = match[1];

            // Get full context path for this line
            const contextPath = getContextPathForLine(session, idx);
            const parentContext = contextPath.length > 0 ? contextPath[contextPath.length - 1] : null;
            const grandparentContext = contextPath.length > 1 ? contextPath[contextPath.length - 2] : null;

            if (parentContext) {
                const parentInfo = keywords[parentContext];

                // Check if parent accepts any child (like materials)
                if (parentInfo && parentInfo.acceptsAnyChild) {
                    // Any child name is allowed, don't validate
                    return;
                }

                // Check if grandparent accepts any child (we're inside a specific material)
                if (grandparentContext) {
                    const grandparentInfo = keywords[grandparentContext];
                    if (grandparentInfo && grandparentInfo.acceptsAnyChild) {
                        // Check if keyword is in childrenProperties
                        if (grandparentInfo.childrenProperties &&
                            !grandparentInfo.childrenProperties.includes(keyword)) {
                            annotations.push({
                                row: idx,
                                column: match.index,
                                text: `Unknown property '${keyword}' in material definition. Valid properties: ${grandparentInfo.childrenProperties.join(', ')}`,
                                type: 'warning'
                            });
                        }
                        return;
                    }
                }

                // Normal parent with defined children
                if (parentInfo && parentInfo.children) {
                    // Check if keyword is valid child
                    if (!parentInfo.children.includes(keyword) && !keywords[keyword]) {
                        annotations.push({
                            row: idx,
                            column: match.index,
                            text: `Unknown keyword '${keyword}' in '${parentContext}' context. Valid keywords: ${parentInfo.children.join(', ')}`,
                            type: 'warning'
                        });
                    }
                } else if (!keywords[keyword]) {
                    // Parent doesn't have children list, check if keyword exists globally
                    annotations.push({
                        row: idx,
                        column: match.index,
                        text: `Unknown keyword '${keyword}' in '${parentContext}' context`,
                        type: 'warning'
                    });
                }
            } else {
                // Top level - check if keyword exists in schema
                if (!keywords[keyword]) {
                    annotations.push({
                        row: idx,
                        column: match.index,
                        text: `Unknown top-level keyword '${keyword}'`,
                        type: 'warning'
                    });
                }
            }
        }
    });
}

function getContextPathForLine(session, lineNum) {
    const currentIndent = getIndentLevel(session.getLine(lineNum));
    const path = [];

    // Build path by finding all parent keywords
    let searchIndent = currentIndent;
    for (let i = lineNum - 1; i >= 0; i--) {
        const line = session.getLine(i);
        const indent = getIndentLevel(line);

        // Found a line with less indentation
        if (indent < searchIndent) {
            const match = line.match(/^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:/);
            if (match) {
                path.unshift(match[1]); // Add to beginning of path
                searchIndent = indent;

                // Stop when we reach the top level
                if (indent === 0) break;
            }
        }
    }

    return path;
}

function getParentContextForLine(session, lineNum) {
    const path = getContextPathForLine(session, lineNum);
    return path.length > 0 ? path[path.length - 1] : null;
}

// Refresh the list of config files
async function refreshConfigList() {
    try {
        const response = await fetch(`/api/config/list?userId=${AppState.userId}`);
        if (!response.ok) throw new Error('Failed to load config list');

        const result = await response.json();
        const select = document.getElementById('configFileSelect');
        const currentValue = select.value;

        // Clear existing options
        select.innerHTML = '';

        // Add all files
        result.files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            select.appendChild(option);
        });

        // Restore previous selection if it still exists
        if (result.files.includes(currentValue)) {
            select.value = currentValue;
        } else if (result.files.length > 0) {
            select.value = result.files[0];
        }
    } catch (error) {
        showStatus('configStatus', `Error loading file list: ${error.message}`, 'error');
    }
}

// Load selected config from dropdown
async function loadSelectedConfig() {
    await loadConfig();
}

async function loadConfig() {
    const fileName = document.getElementById('configFileSelect').value || 'sample_config.yaml';

    try {
        const response = await fetch(`/api/config?file=${encodeURIComponent(fileName)}&userId=${AppState.userId}`);
        if (!response.ok) throw new Error('Failed to load configuration');

        const config = await response.text();

        // Set value in Ace Editor
        if (AppState.aceEditor) {
            AppState.aceEditor.setValue(config, -1); // -1 moves cursor to start
        }

        AppState.configData = config;
        showStatus('configStatus', `Configuration loaded: ${fileName}`, 'success');
    } catch (error) {
        showStatus('configStatus', `Error loading config: ${error.message}`, 'error');
    }
}

async function saveConfig() {
    const fileName = document.getElementById('configFileSelect').value || 'sample_config.yaml';

    if (!AppState.aceEditor) {
        showStatus('configStatus', 'Editor not initialized', 'error');
        return;
    }

    const yamlContent = AppState.aceEditor.getValue();

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file: fileName,
                userId: AppState.userId,
                content: yamlContent
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to save configuration');
        }

        AppState.configData = yamlContent;
        showStatus('configStatus', `Configuration saved: ${fileName}`, 'success');
    } catch (error) {
        showStatus('configStatus', `Error saving config: ${error.message}`, 'error');
    }
}

async function saveConfigAs() {
    let newFileName = prompt('Enter new config file name:', 'my_config.yaml');
    if (!newFileName) return;

    // Ensure .yaml extension
    if (!newFileName.endsWith('.yaml') && !newFileName.endsWith('.yml')) {
        newFileName += '.yaml';
    }

    if (!AppState.aceEditor) {
        showStatus('configStatus', 'Editor not initialized', 'error');
        return;
    }

    const yamlContent = AppState.aceEditor.getValue();

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file: newFileName,
                userId: AppState.userId,
                content: yamlContent
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to save configuration');
        }

        showStatus('configStatus', `Configuration saved as: ${newFileName}`, 'success');

        // Refresh file list and select the new file
        await refreshConfigList();
        document.getElementById('configFileSelect').value = newFileName;
    } catch (error) {
        showStatus('configStatus', `Error saving config: ${error.message}`, 'error');
    }
}

// ===== Run & Preview (Tab 2) =====
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById('uploadedImage');
        img.src = e.target.result;
        img.classList.remove('hidden');
        AppState.uploadedImage = file;
    };
    reader.readAsDataURL(file);

    // Upload to server
    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('userId', AppState.userId);

        const response = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to upload image');
        }

        const result = await response.json();
        AppState.uploadedImageFilename = result.filename;
        showStatus('solverStatus', `Image uploaded: ${result.filename}`, 'success');

        // Refresh image list
        await refreshImageList();
    } catch (error) {
        showStatus('solverStatus', `Upload error: ${error.message}`, 'error');
    }
}

async function refreshImageList() {
    try {
        const response = await fetch(`/api/images?userId=${AppState.userId}`);
        if (!response.ok) throw new Error('Failed to load images');

        const result = await response.json();
        const select = document.getElementById('imageSelect');
        const currentValue = select.value;

        select.innerHTML = '<option value="">Select uploaded image...</option>';
        result.images.forEach(img => {
            const option = document.createElement('option');
            option.value = img;
            option.textContent = img;
            select.appendChild(option);
        });

        if (result.images.includes(currentValue)) {
            select.value = currentValue;
        }
    } catch (error) {
        console.error('Error loading image list:', error);
    }
}

function loadSelectedImage() {
    const select = document.getElementById('imageSelect');
    const filename = select.value;
    if (!filename) {
        showStatus('solverStatus', 'Please select an image', 'error');
        return;
    }

    AppState.uploadedImageFilename = filename;
    const img = document.getElementById('uploadedImage');
    img.src = `/uploads/${AppState.userId}/${filename}`;
    img.classList.remove('hidden');
    showStatus('solverStatus', `Image loaded: ${filename}`, 'success');
}

async function deleteSelectedImage() {
    const select = document.getElementById('imageSelect');
    const filename = select.value;
    if (!filename) {
        showStatus('solverStatus', 'Please select an image to delete', 'error');
        return;
    }

    if (!confirm(`Delete ${filename}?`)) return;

    try {
        const response = await fetch(`/api/images/${filename}?userId=${AppState.userId}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete image');

        showStatus('solverStatus', 'Image deleted successfully', 'success');
        await refreshImageList();

        // Clear if this was the selected image
        if (AppState.uploadedImageFilename === filename) {
            AppState.uploadedImageFilename = null;
            document.getElementById('uploadedImage').classList.add('hidden');
        }
    } catch (error) {
        showStatus('solverStatus', `Delete error: ${error.message}`, 'error');
    }
}

async function deleteConfig() {
    const select = document.getElementById('configFileSelect');
    const filename = select.value;
    if (!filename) {
        showStatus('configStatus', 'Please select a config to delete', 'error');
        return;
    }

    if (!confirm(`Delete ${filename}?`)) return;

    try {
        const response = await fetch(`/api/config/${filename}?userId=${AppState.userId}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete config');

        showStatus('configStatus', 'Config deleted successfully', 'success');
        await refreshConfigList();
        await loadConfig();
    } catch (error) {
        showStatus('configStatus', `Delete error: ${error.message}`, 'error');
    }
}

// ===== Result Management =====
async function refreshResultsList() {
    try {
        const response = await fetch('/api/results');
        if (!response.ok) throw new Error('Failed to load results');

        const result = await response.json();
        const select = document.getElementById('resultSelect');
        const currentValue = select.value;

        select.innerHTML = '<option value="">Select result...</option>';
        result.results.forEach(res => {
            const option = document.createElement('option');
            option.value = res.path;
            option.textContent = `${res.name} (${res.steps} steps)`;
            select.appendChild(option);
        });

        if (result.results.map(r => r.path).includes(currentValue)) {
            select.value = currentValue;
        }
    } catch (error) {
        console.error('Error loading results list:', error);
    }
}

async function loadSelectedResult() {
    const select = document.getElementById('resultSelect');
    const resultPath = select.value;

    if (!resultPath) {
        showStatus('solverStatus', 'Please select a result to load', 'error');
        return;
    }

    try {
        const response = await fetch(`/api/detect-steps?result=${encodeURIComponent(resultPath)}`);
        if (!response.ok) throw new Error('Failed to detect steps');

        const data = await response.json();
        AppState.totalSteps = data.steps || 1;
        AppState.currentStep = 1;
        AppState.resultsData.currentResult = resultPath;

        // Update dashboard controls
        const stepSlider = document.getElementById('stepSlider');
        const totalStepsDisplay = document.getElementById('totalStepsDisplay');
        const currentStepDisplay = document.getElementById('currentStep');

        if (stepSlider) {
            stepSlider.max = AppState.totalSteps;
            stepSlider.value = 1;
        }
        if (totalStepsDisplay) {
            totalStepsDisplay.textContent = AppState.totalSteps;
        }
        if (currentStepDisplay) {
            currentStepDisplay.textContent = 1;
        }

        showStatus('solverStatus', `Loaded result: ${resultPath} (${AppState.totalSteps} steps)`, 'success');

        // Load preview
        await loadQuickPreviewFromResult(resultPath);

        // Load and display log.txt
        await loadResultLog(resultPath);
    } catch (error) {
        showStatus('solverStatus', `Error loading result: ${error.message}`, 'error');
    }
}

async function loadResultLog(resultPath) {
    try {
        const response = await fetch(`/api/get-log?result=${encodeURIComponent(resultPath)}`);
        if (response.ok) {
            const logContent = await response.text();
            const logOutput = document.getElementById('logOutput');
            const logPanel = document.getElementById('logPanel');

            logOutput.textContent = logContent;
            logPanel.style.display = 'block';
        } else {
            // Log file not found, hide panel
            const logPanel = document.getElementById('logPanel');
            logPanel.style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading log:', error);
        const logPanel = document.getElementById('logPanel');
        logPanel.style.display = 'none';
    }
}

async function loadQuickPreviewFromResult(resultPath) {
    const step = 1; // Always show first step in preview

    try {
        // Load Input Image (step_0000.png from InputImg folder)
        const inputImgContainer = document.getElementById('previewPlot1');
        inputImgContainer.innerHTML = `
            <h4 style="text-align: center; margin-bottom: 10px;">Input Image (Step 1)</h4>
            <img src="/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=0"
                 style="max-width: 100%; height: auto; display: block; margin: 0 auto;"
                 onerror="this.parentElement.innerHTML='<p style=\\'text-align:center; padding:20px;\\'>Input image not available</p>'">
        `;

        // Load |B| distribution
        const bResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=B/step_0000.csv`);
        if (bResponse.ok) {
            const bData = await bResponse.json();
            plotHeatmap('previewPlot2', bData.data, '|B| Distribution (Step 1)');
        } else {
            document.getElementById('previewPlot2').innerHTML = '<p style="text-align:center; padding:20px;">|B| data not available</p>';
        }

        // Load |H| distribution
        const hResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=H/step_0000.csv`);
        if (hResponse.ok) {
            const hData = await hResponse.json();
            plotHeatmap('previewPlot3', hData.data, '|H| Distribution (Step 1)');
        } else {
            document.getElementById('previewPlot3').innerHTML = '<p style="text-align:center; padding:20px;">|H| data not available</p>';
        }

    } catch (error) {
        console.error('Preview error:', error);
    }
}

async function runSolver() {
    const btn = document.getElementById('solverBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const outputDiv = document.getElementById('solverOutput');
    outputDiv.textContent = 'Starting solver...\n';

    try {
        // Check if image is uploaded
        if (!AppState.uploadedImageFilename) {
            throw new Error('Please upload a material image first');
        }

        // Get config file path
        const configSelect = document.getElementById('configFileSelect');
        const configFile = configSelect ? configSelect.value : 'sample_config.yaml';

        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                configFile: configFile,
                imageFile: AppState.uploadedImageFilename,
                userId: AppState.userId
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Solver execution failed');
        }

        const result = await response.json();
        // Use textContent to preserve line breaks
        outputDiv.textContent += (result.stdout || 'Solver completed successfully') + '\n';

        showStatus('solverStatus', 'Solver completed successfully', 'success');

        // Load results and update dashboard
        await loadResults();

    } catch (error) {
        showStatus('solverStatus', `Solver error: ${error.message}`, 'error');
        outputDiv.textContent += `\nError: ${error.message}`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Solver';
    }
}

async function loadResults() {
    try {
        // Refresh results list to get the latest results
        await refreshResultsList();

        // Get the results list
        const response = await fetch('/api/results');
        if (!response.ok) throw new Error('Failed to load results');

        const result = await response.json();
        if (result.results.length === 0) {
            showStatus('solverStatus', 'No results found', 'error');
            return;
        }

        // Select the newest result (first in the list)
        const newestResult = result.results[0];
        const select = document.getElementById('resultSelect');
        select.value = newestResult.path;

        // Load the selected result
        await loadSelectedResult();
    } catch (error) {
        showStatus('solverStatus', `Error loading results: ${error.message}`, 'error');
    }
}

// Deprecated: This function is replaced by loadQuickPreviewFromResult
// Kept for backward compatibility but not used
async function loadQuickPreview() {
    // This function is no longer used - loadResults() handles preview display
    console.warn('loadQuickPreview() is deprecated, use loadResults() instead');
}

// ===== Dashboard (Tab 3) =====

// Plot definitions
const plotDefinitions = {
    az_heatmap: { name: 'Az Heatmap', render: renderAzHeatmap },
    jz_distribution: { name: 'Jz Distribution', render: renderJzDistribution },
    b_magnitude: { name: '|B| Distribution', render: renderBMagnitude },
    h_magnitude: { name: '|H| Distribution', render: renderHMagnitude },
    mu_distribution: { name: 'Permeability', render: renderMuDistribution },
    energy_density: { name: 'Energy Density', render: renderEnergyDensity },
    az_boundary: { name: 'Field Lines + Boundary', render: renderAzBoundary },
    material_image: { name: 'Material Image', render: renderMaterialImage },
    step_input_image: { name: 'Step Input Image', render: renderStepInputImage },
    force_x_time: { name: 'Force X-axis', render: renderForceXTime },
    force_y_time: { name: 'Force Y-axis', render: renderForceYTime },
    torque_time: { name: 'Torque', render: renderTorqueTime },
    energy_time: { name: 'Magnetic Energy', render: renderEnergyTime }
};

let plotIdCounter = 0;

function initializeDashboard() {
    if (AppState.gridStack) return; // Already initialized

    const grid = GridStack.init({
        cellHeight: 150,
        minRow: 2,
        column: 12,
        acceptWidgets: true,
        removable: false,
        float: true
    }, '#dashboard-grid');

    AppState.gridStack = grid;

    // Setup palette drag and drop
    setupPaletteDragDrop();
}

// Setup drag and drop for palette items
function setupPaletteDragDrop() {
    const paletteItems = document.querySelectorAll('.palette-item');
    const canvas = document.querySelector('#dashboard-grid');

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

    // Canvas drop events
    canvas.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    canvas.addEventListener('drop', (e) => {
        e.preventDefault();
        const plotType = e.dataTransfer.getData('plot-type') || e.dataTransfer.getData('text/plain');

        if (plotType && plotDefinitions[plotType]) {
            // Hide empty canvas message
            const emptyCanvas = document.getElementById('emptyCanvas');
            if (emptyCanvas) emptyCanvas.style.display = 'none';

            // Calculate drop position
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / (rect.width / 12));
            const y = Math.floor((e.clientY - rect.top) / 150);

            addPlotWidget(plotType, x, y, 4, 3);
        }
    });
}

// Add plot widget to dashboard
async function addPlotWidget(plotType, x = 0, y = 0, w = 4, h = 3) {
    const plotDef = plotDefinitions[plotType];
    if (!plotDef) {
        console.error(`Unknown plot type: ${plotType}`);
        return;
    }

    const plotId = `plot-${plotIdCounter++}`;
    const containerId = `container-${plotId}`;

    // Create widget content with control buttons and SVG icons
    const content = `
        <div class="plot-header">
            <span>${plotDef.name}</span>
            <div class="plot-controls">
                <button class="interaction-mode-btn" data-plot-id="${plotId}" data-container-id="${containerId}" data-mode="disabled" title="Mode: Move" onclick="toggleInteractionMode('${plotId}', '${containerId}', this)">
                    <img src="/icon/window.svg" alt="Move" style="width: 14px; height: 14px; vertical-align: middle; filter: brightness(0) invert(1);">
                </button>
                <button class="reset-zoom-btn" title="Reset Zoom" onclick="resetPlotZoom('${containerId}')">
                    <img src="/icon/reset.svg" alt="Reset" style="width: 14px; height: 14px; vertical-align: middle; filter: brightness(0) invert(1);">
                </button>
                <button class="remove-plot-btn" title="Remove" onclick="removePlot('${plotId}')">
                    <img src="/icon/remove.svg" alt="Remove" style="width: 14px; height: 14px; vertical-align: middle; filter: brightness(0) invert(1);">
                </button>
            </div>
        </div>
        <div class="plot-container" id="${containerId}" data-plot-type="${plotType}">
            <div style="text-align: center; padding: 20px; color: #999;">Loading...</div>
        </div>
    `;

    // Add to GridStack
    const widget = AppState.gridStack.addWidget({
        x: x,
        y: y,
        w: w,
        h: h,
        content: content,
        id: plotId
    });

    // Render plot
    setTimeout(async () => {
        try {
            await plotDef.render(containerId, AppState.currentStep);
        } catch (error) {
            console.error(`Error rendering ${plotType}:`, error);
            document.getElementById(containerId).innerHTML = `<p style="color:red; padding:20px;">Error: ${error.message}</p>`;
        }
    }, 100);
}

function removePlot(plotId) {
    const items = AppState.gridStack.engine.nodes;
    const item = items.find(n => n.id === plotId);
    if (item && item.el) {
        AppState.gridStack.removeWidget(item.el);
    }
}

// ===== Interaction Mode Toggle =====
function toggleInteractionMode(plotId, containerId, button) {
    const container = document.getElementById(containerId);
    if (!container || !container.data || !container.layout) {
        console.warn('No Plotly plot found');
        return;
    }

    const currentMode = button.dataset.mode;
    let newMode, newIcon, newTitle, dragmode, tileMovable;

    if (currentMode === 'zoom') {
        // Zoom -> Pan
        newMode = 'pan';
        newIcon = '/icon/pan.svg';
        newTitle = 'Mode: Pan';
        dragmode = 'pan';
        tileMovable = false;
    } else if (currentMode === 'pan') {
        // Pan -> Move (disabled)
        newMode = 'disabled';
        newIcon = '/icon/window.svg';
        newTitle = 'Mode: Move';
        dragmode = false;
        tileMovable = true;
    } else {
        // Move -> Zoom
        newMode = 'zoom';
        newIcon = '/icon/zoom.svg';
        newTitle = 'Mode: Zoom';
        dragmode = 'zoom';
        tileMovable = false;
    }

    button.dataset.mode = newMode;
    button.title = newTitle;

    // Update button icon
    const img = button.querySelector('img');
    if (img) {
        img.src = newIcon;
    }

    // Update Plotly dragmode
    Plotly.relayout(container, { dragmode: dragmode }).catch(err => {
        console.error('Failed to update drag mode:', err);
    });

    // Update GridStack tile movability
    const widgetEl = document.getElementById(plotId);
    if (widgetEl && AppState.gridStack) {
        AppState.gridStack.movable(widgetEl, tileMovable);
    }
}

// ===== Reset Plot Zoom =====
function resetPlotZoom(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (container.data && container.layout) {
        Plotly.relayout(container, {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        }).catch(err => {
            console.error('Plotly reset zoom error:', err);
        });
    }
}

// ===== Reset All Plots =====
function resetAllPlots() {
    const containers = document.querySelectorAll('.plot-container[id^="container-"]');
    containers.forEach(container => {
        if (container.data && container.layout) {
            Plotly.relayout(container, {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            }).catch(err => {
                console.error('Plotly reset zoom error:', err);
            });
        }
    });
}

// ===== Update All Plots =====
async function updateAllPlots() {
    const containers = document.querySelectorAll('.plot-container[data-plot-type]');

    for (const container of containers) {
        const plotType = container.dataset.plotType;
        const plotDef = plotDefinitions[plotType];

        if (plotDef) {
            try {
                await plotDef.render(container.id, AppState.currentStep);
            } catch (error) {
                console.error(`Error updating ${plotType}:`, error);
            }
        }
    }
}

// ===== Animation Controls =====
function playAnimation() {
    if (AppState.isAnimating || AppState.totalSteps <= 1) return;

    AppState.isAnimating = true;
    document.getElementById('playBtn').style.display = 'none';
    document.getElementById('pauseBtn').style.display = 'inline-block';

    const speed = parseInt(document.getElementById('animSpeed').value);

    AppState.animationTimer = setInterval(async () => {
        AppState.currentStep++;
        if (AppState.currentStep > AppState.totalSteps) {
            AppState.currentStep = 1; // Loop
        }

        document.getElementById('stepSlider').value = AppState.currentStep;
        document.getElementById('currentStep').textContent = AppState.currentStep;

        await updateAllPlots();
    }, speed);
}

function pauseAnimation() {
    if (AppState.animationTimer) {
        clearInterval(AppState.animationTimer);
        AppState.animationTimer = null;
    }

    AppState.isAnimating = false;
    document.getElementById('playBtn').style.display = 'inline-block';
    document.getElementById('pauseBtn').style.display = 'none';
}

function setStep(step) {
    pauseAnimation();
    AppState.currentStep = step;
    document.getElementById('currentStep').textContent = step;
    updateAllPlots();
}

function clearDashboard() {
    if (AppState.gridStack) {
        AppState.gridStack.removeAll();
    }
}

function saveLayout() {
    if (!AppState.gridStack) return;

    const layout = AppState.gridStack.save();
    localStorage.setItem('dashboardLayout', JSON.stringify(layout));
    alert('Layout saved successfully');
}

// ===== Plot Rendering Functions =====
// Helper function to get current result path
function getCurrentResultPath() {
    return AppState.resultsData.currentResult || '';
}

// Helper function to format step filename
function formatStepFilename(step) {
    return `step_${String(step - 1).padStart(4, '0')}.csv`;
}

// Placeholder implementations - these will call actual data loading and plotting
async function renderAzContour(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Az/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load Az data');
    const result = await response.json();
    plotContour(containerId, result.data, 'Az (Magnetic Vector Potential)');
}

async function renderAzHeatmap(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Az/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load Az data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Az Heatmap');
}

async function renderJzDistribution(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Jz/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load Jz data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Jz (Current Density)');
}

async function renderBMagnitude(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=B/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load B data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, '|B| Distribution');
}

async function renderHMagnitude(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=H/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load H data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, '|H| Distribution');
}

async function renderMuDistribution(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Mu/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load Mu data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Permeability Distribution');
}

async function renderEnergyDensity(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=EnergyDensity/${formatStepFilename(step)}`);
    if (!response.ok) throw new Error('Failed to load energy data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Energy Density');
}

async function renderAzBoundary(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Field Lines + Boundary visualization coming soon</p>';
}

async function renderMaterialImage(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Material Image visualization coming soon</p>';
}

async function renderStepInputImage(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Step Input Image visualization coming soon</p>';
}

async function renderForceXTime(containerId) {
    try {
        const resultPath = getCurrentResultPath();
        if (!resultPath) throw new Error('No result selected');

        // Load force summary data
        const forceData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepFile = `step_${String(i).padStart(4, '0')}.csv`;
            const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(resultPath)}&file=Forces/${stepFile}`);
            if (response.ok) {
                const text = await response.text();
                const lines = text.trim().split('\n');
                // Parse CSV: assume Force_X is in a specific column
                if (lines.length > 1) {
                    const dataLine = lines[1].split(',');
                    forceData.push(parseFloat(dataLine[1]) || 0); // Assuming Force_X is column 1
                }
            }
        }

        const container = document.getElementById(containerId);
        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const markerSizes = xSteps.map((s) =>
            (AppState.isAnimating && s === AppState.currentStep) ? 12 : 6
        );

        const trace = {
            x: xSteps,
            y: forceData.length > 0 ? forceData : xSteps.map(() => 0),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Force X',
            line: { color: '#667eea', width: 2 },
            marker: { color: '#667eea', size: markerSizes }
        };

        await Plotly.newPlot(container, [trace], {
            margin: { l: 50, r: 10, t: 10, b: 40 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Force X [N/m]' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Force X time plot error:', error);
        document.getElementById(containerId).innerHTML = `<p style="padding:20px; color:red;">Error: ${error.message}</p>`;
    }
}

async function renderForceYTime(containerId) {
    try {
        const resultPath = getCurrentResultPath();
        if (!resultPath) throw new Error('No result selected');

        const forceData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepFile = `step_${String(i).padStart(4, '0')}.csv`;
            const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(resultPath)}&file=Forces/${stepFile}`);
            if (response.ok) {
                const text = await response.text();
                const lines = text.trim().split('\n');
                if (lines.length > 1) {
                    const dataLine = lines[1].split(',');
                    forceData.push(parseFloat(dataLine[2]) || 0); // Assuming Force_Y is column 2
                }
            }
        }

        const container = document.getElementById(containerId);
        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const markerSizes = xSteps.map((s) =>
            (AppState.isAnimating && s === AppState.currentStep) ? 12 : 6
        );

        const trace = {
            x: xSteps,
            y: forceData.length > 0 ? forceData : xSteps.map(() => 0),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Force Y',
            line: { color: '#764ba2', width: 2 },
            marker: { color: '#764ba2', size: markerSizes }
        };

        await Plotly.newPlot(container, [trace], {
            margin: { l: 50, r: 10, t: 10, b: 40 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Force Y [N/m]' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Force Y time plot error:', error);
        document.getElementById(containerId).innerHTML = `<p style="padding:20px; color:red;">Error: ${error.message}</p>`;
    }
}

async function renderTorqueTime(containerId) {
    try {
        const resultPath = getCurrentResultPath();
        if (!resultPath) throw new Error('No result selected');

        const torqueData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepFile = `step_${String(i).padStart(4, '0')}.csv`;
            const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(resultPath)}&file=Forces/${stepFile}`);
            if (response.ok) {
                const text = await response.text();
                const lines = text.trim().split('\n');
                if (lines.length > 1) {
                    const dataLine = lines[1].split(',');
                    torqueData.push(parseFloat(dataLine[3]) || 0); // Assuming Torque is column 3
                }
            }
        }

        const container = document.getElementById(containerId);
        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const markerSizes = xSteps.map((s) =>
            (AppState.isAnimating && s === AppState.currentStep) ? 12 : 6
        );

        const trace = {
            x: xSteps,
            y: torqueData.length > 0 ? torqueData : xSteps.map(() => 0),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Torque',
            line: { color: '#f093fb', width: 2 },
            marker: { color: '#f093fb', size: markerSizes }
        };

        await Plotly.newPlot(container, [trace], {
            margin: { l: 50, r: 10, t: 10, b: 40 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Torque [Nm/m]' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Torque time plot error:', error);
        document.getElementById(containerId).innerHTML = `<p style="padding:20px; color:red;">Error: ${error.message}</p>`;
    }
}

async function renderEnergyTime(containerId) {
    try {
        const resultPath = getCurrentResultPath();
        if (!resultPath) throw new Error('No result selected');

        const energyData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepFile = `step_${String(i).padStart(4, '0')}.csv`;
            const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(resultPath)}&file=Forces/${stepFile}`);
            if (response.ok) {
                const text = await response.text();
                const lines = text.trim().split('\n');
                if (lines.length > 1) {
                    const dataLine = lines[1].split(',');
                    energyData.push(parseFloat(dataLine[4]) || 0); // Assuming Energy is column 4
                }
            }
        }

        const container = document.getElementById(containerId);
        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const markerSizes = xSteps.map((s) =>
            (AppState.isAnimating && s === AppState.currentStep) ? 12 : 6
        );

        const trace = {
            x: xSteps,
            y: energyData.length > 0 ? energyData : xSteps.map(() => 0),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Magnetic Energy',
            line: { color: '#4facfe', width: 2 },
            marker: { color: '#4facfe', size: markerSizes }
        };

        await Plotly.newPlot(container, [trace], {
            margin: { l: 50, r: 10, t: 10, b: 40 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Energy [J/m]' },
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Energy time plot error:', error);
        document.getElementById(containerId).innerHTML = `<p style="padding:20px; color:red;">Error: ${error.message}</p>`;
    }
}

// ===== Plotting Functions =====
function plotContour(elementId, data, title) {
    if (!data || data.length === 0) {
        document.getElementById(elementId).innerHTML = '<p>No data available</p>';
        return;
    }

    const trace = {
        z: data,
        type: 'contour',
        colorscale: 'Viridis',
        contours: {
            coloring: 'heatmap'
        },
        colorbar: {
            title: title
        }
    };

    const layout = {
        title: title,
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        margin: { t: 40, r: 20, b: 40, l: 60 }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

// ===== Plotting Functions =====
function plotHeatmap(elementId, data, title) {
    if (!data || data.length === 0) {
        document.getElementById(elementId).innerHTML = '<p>No data available</p>';
        return;
    }

    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {
            title: title
        }
    };

    const layout = {
        title: title,
        xaxis: { title: 'Columns' },
        yaxis: { title: 'Rows' },
        margin: { t: 40, r: 20, b: 40, l: 60 }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

function plotForceGraph(elementId, data, title) {
    // Parse force data from CSV
    if (!data || !Array.isArray(data) || data.length < 2) {
        document.getElementById(elementId).innerHTML = '<p>No force data available</p>';
        return;
    }

    // Assuming first row is header, rest is data
    const headers = data[0];
    const values = data.slice(1);

    const trace = {
        x: values.map((_, i) => i),
        y: values.map(row => row[4]), // Assuming column 4 is a force value
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Force'
    };

    const layout = {
        title: title,
        xaxis: { title: 'Index' },
        yaxis: { title: 'Force [N/m]' },
        margin: { t: 40, r: 20, b: 40, l: 60 }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

// ===== Utility Functions =====
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.className = `status-message status-${type}`;
    element.textContent = message;
    element.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(() => {
        element.style.display = 'none';
    }, 5000);
}

// ===== Helper Functions for CSV Parsing =====
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

                resolve(data);
            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}
