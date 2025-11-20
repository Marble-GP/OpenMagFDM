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
    isAnimating: false,  // Animation state flag
    analysisConditions: null,  // Analysis conditions from conditions.json
    dataCache: {},  // Data cache for preloaded steps: { 'resultPath:Az:step': data, ... }
    maxCacheEntries: 500  // Maximum cache entries to prevent memory leak (500 * ~2MB = ~1GB max)
};

// ===== Utility Functions =====
/**
 * Flip 2D array vertically (convert from analysis coordinate system y-up to image coordinate system y-down)
 * @param {Array<Array<number>>} data - 2D array
 * @returns {Array<Array<number>>} - Flipped 2D array
 */
function flipVertical(data) {
    if (!data || !Array.isArray(data) || data.length === 0) {
        return data;
    }
    // Clone and reverse array (do not modify original data)
    return data.slice().reverse();
}

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

    // Setup step slider event handler
    const stepSlider = document.getElementById('stepSlider');
    if (stepSlider) {
        stepSlider.addEventListener('input', onStepChange);
    }
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

        // Clear cache when switching to a different result (BEFORE updating currentResult)
        const previousResult = AppState.resultsData.currentResult;
        if (previousResult && previousResult !== resultPath) {
            const cacheSize = Object.keys(AppState.dataCache).length;
            console.log(`Switching from ${previousResult} to ${resultPath}, clearing ${cacheSize} cached entries`);
            AppState.dataCache = {};  // Complete cache clear
        }

        AppState.resultsData.currentResult = resultPath;

        // Load analysis conditions from conditions.json
        try {
            const conditionsResponse = await fetch(`/api/load-conditions?result=${encodeURIComponent(resultPath)}`);
            if (conditionsResponse.ok) {
                AppState.analysisConditions = await conditionsResponse.json();
                console.log('Analysis conditions loaded:', AppState.analysisConditions);
            } else {
                console.warn('conditions.json not found, assuming default (cartesian)');
                AppState.analysisConditions = { coordinate_system: 'cartesian', dx: 0.001, dy: 0.001 };
            }
        } catch (error) {
            console.warn('Failed to load conditions.json:', error);
            AppState.analysisConditions = { coordinate_system: 'cartesian', dx: 0.001, dy: 0.001 };
        }

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

// Helper: Calculate magnetic fields from Az and Mu (supports both polar and Cartesian coordinates)
function calculateMagneticField(Az, Mu, dx = 0.001, dy = 0.001) {
    const rows = Az.length;
    const cols = Az[0].length;

    const Bx = Array(rows).fill(0).map(() => Array(cols).fill(0));
    const By = Array(rows).fill(0).map(() => Array(cols).fill(0));

    // Determine coordinate system (if analysisConditions is loaded)
    const coordSystem = AppState.analysisConditions ? AppState.analysisConditions.coordinate_system : 'cartesian';

    if (coordSystem === 'polar') {
        // Polar coordinate magnetic field calculation
        const polar = AppState.analysisConditions.polar;
        if (!polar || !polar.r_start || !polar.r_end || !polar.theta_range) {
            console.error('Polar coordinate parameters missing in analysisConditions:', AppState.analysisConditions);
            throw new Error('Polar coordinate parameters not found in conditions.json');
        }
        const r_start = polar.r_start;
        const r_end = polar.r_end;
        const nr = cols;
        const ntheta = rows;

        // Calculate dr, dtheta (use from conditions.json if available, otherwise calculate)
        const dr = AppState.analysisConditions.dr || (r_end - r_start) / (nr - 1);
        const dtheta = AppState.analysisConditions.dtheta || polar.theta_range / (ntheta - 1);

        // Generate r coordinate array
        const r_coords = Array(nr).fill(0).map((_, ir) => r_start + ir * dr);

        // Calculate magnetic field in polar coordinates: Br, Bθ
        const Br = Array(rows).fill(0).map(() => Array(cols).fill(0));
        const Btheta = Array(rows).fill(0).map(() => Array(cols).fill(0));

        // Polar coordinates are ALWAYS periodic in theta direction
        for (let jt = 0; jt < ntheta; jt++) {
            for (let ir = 0; ir < nr; ir++) {
                const r = r_coords[ir];
                const safe_r = Math.max(r, 1e-15);

                // Br = (1/r) * ∂Az/∂θ
                // IMPORTANT: Use periodic boundary in theta direction
                const jt_next = (jt + 1) % ntheta;
                const jt_prev = (jt - 1 + ntheta) % ntheta;
                const dAz_dtheta = (Az[jt_next][ir] - Az[jt_prev][ir]) / (2 * dtheta);
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

        // Polar → Cartesian transformation (for visualization)
        // If r_orientation is horizontal: i = r direction, j = θ direction
        // Physical coordinates: x = r*cos(θ), y = r*sin(θ)
        // Field transformation: Bx = Br*cos(θ) - Bθ*sin(θ), By = Br*sin(θ) + Bθ*cos(θ)
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
        // Cartesian coordinate magnetic field calculation
        // Determine periodic boundary conditions
        const bc = AppState.analysisConditions ? AppState.analysisConditions.boundary_conditions : null;
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

    // Magnitude
    const B = Bx.map((row, j) => row.map((val, i) => Math.sqrt(val**2 + By[j][i]**2)));
    const H = Hx.map((row, j) => row.map((val, i) => Math.sqrt(val**2 + Hy[j][i]**2)));

    return { Bx, By, B, Hx, Hy, H };
}

// Helper: Calculate magnitude from vector components
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

async function loadQuickPreviewFromResult(resultPath) {
    const step = 1; // Always show first step in preview

    try {
        // Load Input Image (step_0001.png from InputImg folder)
        const inputImgContainer = document.getElementById('previewPlot1');
        inputImgContainer.innerHTML = `
            <h4 style="text-align: center; margin-bottom: 10px;">Input Image (Step 1)</h4>
            <img src="/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=1"
                 style="max-width: 100%; max-height: calc(100% - 50px); object-fit: contain; display: block; margin: 0 auto;"
                 onerror="this.parentElement.innerHTML='<p style=\\'text-align:center; padding:20px;\\'>Input image not available</p>'">
        `;

        // Use grid spacing from analysis conditions
        // For polar coordinates: use dr/dtheta, for Cartesian: use dx/dy
        const dx = AppState.analysisConditions ?
            (AppState.analysisConditions.dx || AppState.analysisConditions.dr || 0.001) : 0.001;
        const dy = AppState.analysisConditions ?
            (AppState.analysisConditions.dy || AppState.analysisConditions.dtheta || 0.001) : 0.001;

        // Load Az and Mu data
        const azResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Az/step_0001.csv`);
        const muResponse = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=Mu/step_0001.csv`);

        if (azResponse.ok && muResponse.ok) {
            const azData = await azResponse.json();
            const muData = await muResponse.json();

            if (azData.success && muData.success) {
                // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
                const azFlipped = flipVertical(azData.data);
                const muFlipped = flipVertical(muData.data);

                console.log('Az data dimensions:', azFlipped.length, 'x', azFlipped[0]?.length);
                console.log('Mu data dimensions:', muFlipped.length, 'x', muFlipped[0]?.length);
                console.log('Grid spacing: dx =', dx, ', dy =', dy);

                // Calculate B and H fields
                const { Bx, By, B, Hx, Hy, H } = calculateMagneticField(azFlipped, muFlipped, dx, dy);
                console.log('B field calculated, Bx dimensions:', Bx.length, 'x', Bx[0]?.length);
                console.log('B magnitude dimensions:', B.length, 'x', B[0]?.length);
                console.log('H magnitude dimensions:', H.length, 'x', H[0]?.length);
                console.log('B magnitude sample values:', B[0]?.slice(0, 3));

                // Plot |B| and |H|
                plotHeatmap('previewPlot2', B, '|B| [T/m]', true);
                plotHeatmap('previewPlot3', H, '|H| [A/m]', true);
            } else {
                console.error('Az/Mu data loading failed:', { azSuccess: azData.success, muSuccess: muData.success });
                document.getElementById('previewPlot2').innerHTML = '<p style="text-align:center; padding:20px;">Failed to process Az/Mu data</p>';
                document.getElementById('previewPlot3').innerHTML = '<p style="text-align:center; padding:20px;">Failed to process Az/Mu data</p>';
            }
        } else {
            document.getElementById('previewPlot2').innerHTML = '<p style="text-align:center; padding:20px;">Az/Mu data not available</p>';
            document.getElementById('previewPlot3').innerHTML = '<p style="text-align:center; padding:20px;">Az/Mu data not available</p>';
        }

    } catch (error) {
        console.error('Preview error:', error);
        document.getElementById('previewPlot2').innerHTML = '<p style="text-align:center; padding:20px;">Error loading preview</p>';
        document.getElementById('previewPlot3').innerHTML = '<p style="text-align:center; padding:20px;">Error loading preview</p>';
    }
}

async function runSolver() {
    const btn = document.getElementById('solverBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const outputDiv = document.getElementById('solverOutput');
    const progressContainer = document.getElementById('solverProgressContainer');
    const progressBar = document.getElementById('solverProgressBar');
    const progressText = document.getElementById('solverProgressText');
    const progressPercent = document.getElementById('solverProgressPercent');

    outputDiv.textContent = '';
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'Initializing...';
    progressPercent.textContent = '0%';

    // Performance optimization: Buffer output updates
    let outputBuffer = [];
    let lastUpdateTime = 0;
    const UPDATE_INTERVAL = 100; // Update DOM every 100ms
    const MAX_LOG_LINES = 1000; // Keep only last 1000 lines

    function flushOutputBuffer() {
        if (outputBuffer.length === 0) return;

        const lines = outputDiv.textContent.split('\n');
        const newLines = lines.concat(outputBuffer);

        // Keep only last MAX_LOG_LINES
        if (newLines.length > MAX_LOG_LINES) {
            const excess = newLines.length - MAX_LOG_LINES;
            newLines.splice(0, excess);
        }

        outputDiv.textContent = newLines.join('\n');
        outputBuffer = [];
        outputDiv.scrollTop = outputDiv.scrollHeight;
    }

    try {
        // Check if image is uploaded
        if (!AppState.uploadedImageFilename) {
            throw new Error('Please upload a material image first');
        }

        // Get config file path
        const configSelect = document.getElementById('configFileSelect');
        const configFile = configSelect ? configSelect.value : 'sample_config.yaml';

        // Use fetch to POST the request body, then manually handle SSE
        const response = await fetch('/api/solve-stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                configFile: configFile,
                imageFile: AppState.uploadedImageFilename,
                userId: AppState.userId
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Solver execution failed' }));
            throw new Error(errorData.error || 'Solver execution failed');
        }

        // Process SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));

                    // Buffer output log (performance optimization)
                    if (data.message) {
                        outputBuffer.push(data.message);

                        // Throttle DOM updates - flush every UPDATE_INTERVAL ms
                        const now = Date.now();
                        if (now - lastUpdateTime >= UPDATE_INTERVAL) {
                            flushOutputBuffer();
                            lastUpdateTime = now;
                        }
                    }

                    // Update progress bar (lightweight DOM updates)
                    if (data.type === 'progress') {
                        const percentage = data.percentage || 0;
                        progressBar.style.width = percentage + '%';
                        progressPercent.textContent = percentage + '%';
                        progressText.textContent = `Step ${data.step} / ${data.total}`;
                    } else if (data.type === 'status') {
                        progressText.textContent = data.message || 'Processing...';
                    } else if (data.type === 'complete') {
                        progressBar.style.width = '100%';
                        progressPercent.textContent = '100%';
                        progressText.textContent = 'Completed successfully';
                    } else if (data.type === 'done') {
                        // Flush remaining buffer
                        flushOutputBuffer();

                        if (data.success) {
                            showStatus('solverStatus', 'Solver completed successfully', 'success');
                            // Load results and update dashboard
                            await loadResults();
                        } else {
                            throw new Error(data.message || 'Solver failed');
                        }
                    } else if (data.type === 'error') {
                        // Don't throw - stderr messages (including WARNINGs) are just logged
                        // Actual errors are determined by the exit code in 'done' event
                        // Message is already buffered above
                    }
                }
            }
        }

    } catch (error) {
        // Flush buffer before showing error
        flushOutputBuffer();

        showStatus('solverStatus', `Solver error: ${error.message}`, 'error');
        outputBuffer.push(`\nError: ${error.message}`);
        flushOutputBuffer();
        progressBar.style.background = 'linear-gradient(90deg, #dc3545 0%, #c82333 100%)';
        progressText.textContent = 'Error occurred';
    } finally {
        // Ensure all buffered messages are displayed
        flushOutputBuffer();

        btn.disabled = false;
        btn.textContent = 'Run Solver';

        // Hide progress bar after 3 seconds if completed successfully
        setTimeout(() => {
            if (progressText.textContent === 'Completed successfully') {
                progressContainer.style.display = 'none';
            }
        }, 3000);
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
        <div class="grid-stack-item-content" data-plot-type="${plotType}" data-container-id="${containerId}">
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
            <div class="plot-container" id="${containerId}">
                <div style="text-align: center; padding: 20px; color: #999;">Loading...</div>
            </div>
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

    // Set initial tile movability (default is move mode, so movable is true)
    if (widget && AppState.gridStack) {
        AppState.gridStack.movable(widget, true);
        console.log(`Widget ${plotId} added with tile movable: true`);
    }

    // Check if result is selected
    if (!AppState.resultsData.currentResult) {
        console.error('addPlotWidget: No result selected');
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = '<p style="color:red; padding:20px;">Please select a result first</p>';
        }
        return;
    }

    // Render plot after GridStack layout is complete
    setTimeout(async () => {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`addPlotWidget: Container not found: ${containerId}`);
            return;
        }

        const rect = container.getBoundingClientRect();
        console.log(`addPlotWidget: Container size: ${rect.width}x${rect.height}`);

        if (rect.width < 50 || rect.height < 50) {
            console.warn(`addPlotWidget: Container size too small, retrying...`);
            setTimeout(async () => {
                try {
                    console.log(`addPlotWidget: Rendering ${plotType} in ${containerId} for step ${AppState.currentStep}`);
                    await plotDef.render(containerId, AppState.currentStep);
                    console.log(`addPlotWidget: Successfully rendered ${plotType}`);
                } catch (error) {
                    console.error(`Error rendering ${plotType}:`, error);
                    console.error('Error stack:', error.stack);
                    container.innerHTML = `<p style="color:red; padding:20px;">Error: ${error.message}</p>`;
                }
            }, 200);
            return;
        }

        try {
            console.log(`addPlotWidget: Rendering ${plotType} in ${containerId} for step ${AppState.currentStep}`);
            await plotDef.render(containerId, AppState.currentStep);
            console.log(`addPlotWidget: Successfully rendered ${plotType}`);
        } catch (error) {
            console.error(`Error rendering ${plotType}:`, error);
            console.error('Error stack:', error.stack);
            container.innerHTML = `<p style="color:red; padding:20px;">Error: ${error.message}</p>`;
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
    if (!AppState.resultsData.currentResult) {
        console.log('updateAllPlots: No result selected');
        return;
    }

    // Get all plot containers from GridStack items
    const contentElements = document.querySelectorAll('.grid-stack-item-content[data-plot-type]');
    console.log(`updateAllPlots: Found ${contentElements.length} plots, currentStep=${AppState.currentStep}`);

    for (const contentElement of contentElements) {
        const plotType = contentElement.dataset.plotType;
        const containerId = contentElement.dataset.containerId;
        const container = document.getElementById(containerId);

        // Skip invalid plot types
        if (!plotDefinitions[plotType]) {
            console.warn(`updateAllPlots: Skipping invalid plot type: ${plotType}`);
            continue;
        }

        if (!container) {
            console.warn(`updateAllPlots: Container not found: ${containerId}`);
            continue;
        }

        try {
            console.log(`updateAllPlots: Rendering ${plotType} in ${containerId} for step ${AppState.currentStep}`);
            await plotDefinitions[plotType].render(containerId, AppState.currentStep);
            console.log(`updateAllPlots: Successfully rendered ${plotType}`);
        } catch (error) {
            console.error(`Error updating ${plotType}:`, error);
            console.error('Error stack:', error.stack);
            container.innerHTML = `<p style="color:red; padding:20px;">Error: ${error.message}</p>`;
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

async function preloadAllSteps() {
    const currentResult = AppState.resultsData.currentResult;
    if (!currentResult || AppState.totalSteps <= 0) {
        alert('Please select analysis results first');
        return;
    }

    const btn = document.getElementById('preloadBtn');
    if (!btn) return;

    btn.disabled = true;
    const originalText = btn.textContent;

    try {
        // Check if preload would exceed cache limit (Az + Mu = 2 entries per step)
        const estimatedCacheEntries = AppState.totalSteps * 2;
        if (estimatedCacheEntries > AppState.maxCacheEntries) {
            const proceed = confirm(
                `Warning: Preloading ${AppState.totalSteps} steps (${estimatedCacheEntries} cache entries) ` +
                `exceeds the limit of ${AppState.maxCacheEntries}.\n\n` +
                `This may consume ~${Math.round(estimatedCacheEntries * 2)}MB of memory.\n\n` +
                `Continue anyway? (Cache will be cleared if limit is reached)`
            );
            if (!proceed) {
                btn.disabled = false;
                return;
            }
        }

        console.log(`Starting preload of ${AppState.totalSteps} steps (estimated ${estimatedCacheEntries} entries)`);

        for (let step = 1; step <= AppState.totalSteps; step++) {
            btn.textContent = `Loading ${step}/${AppState.totalSteps}`;

            // Preload Az and Mu data
            const azFile = `Az/step_${String(step).padStart(4, '0')}.csv`;
            const muFile = `Mu/step_${String(step).padStart(4, '0')}.csv`;

            const [azResponse, muResponse] = await Promise.all([
                fetch(`/api/load-csv?result=${encodeURIComponent(currentResult)}&file=${azFile}`),
                fetch(`/api/load-csv?result=${encodeURIComponent(currentResult)}&file=${muFile}`)
            ]);

            // Save to cache with size check
            const currentCacheSize = Object.keys(AppState.dataCache).length;
            if (currentCacheSize >= AppState.maxCacheEntries) {
                console.warn(`Cache limit reached during preload at step ${step}/${AppState.totalSteps}, stopping`);
                btn.textContent = `Stopped at ${step}/${AppState.totalSteps}`;
                setTimeout(() => { btn.textContent = originalText; }, 3000);
                break;
            }

            if (azResponse.ok) {
                const azData = await azResponse.json();
                if (azData.success) {
                    const cacheKey = `${currentResult}:Az:${step}`;
                    AppState.dataCache[cacheKey] = azData.data;
                }
            }

            if (muResponse.ok) {
                const muData = await muResponse.json();
                if (muData.success) {
                    const cacheKey = `${currentResult}:Mu:${step}`;
                    AppState.dataCache[cacheKey] = muData.data;
                }
            }

            // Preload force data if available
            const forceData = await loadForceData(step);
            if (forceData) {
                const cacheKey = `${currentResult}:Force:${step}`;
                AppState.dataCache[cacheKey] = forceData;
            }

            // Small delay to prevent overwhelming the server
            if (step < AppState.totalSteps) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        const cacheSize = Object.keys(AppState.dataCache).length;
        console.log(`Preload complete: ${AppState.totalSteps} steps cached (${cacheSize} entries)`);
        btn.textContent = 'Preloaded ✓';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);

    } catch (error) {
        console.error('Preload error:', error);
        alert(`Preload failed: ${error.message}`);
        btn.textContent = originalText;
    } finally {
        btn.disabled = false;
    }
}

async function onStepChange() {
    AppState.currentStep = parseInt(document.getElementById('stepSlider').value);
    document.getElementById('currentStep').textContent = AppState.currentStep;

    // Update all plots
    await updateAllPlots();
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
    return `step_${String(step).padStart(4, '0')}.csv`;
}

// Helper function to load CSV data with caching
async function loadCsvData(dataType, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    // Check cache first
    const cacheKey = `${resultPath}:${dataType}:${step}`;
    if (AppState.dataCache[cacheKey]) {
        console.log(`Cache hit: ${cacheKey}`);
        return AppState.dataCache[cacheKey];
    }

    // Cache miss - fetch from server
    const file = `${dataType}/${formatStepFilename(step)}`;
    const response = await fetch(`/api/load-csv?result=${encodeURIComponent(resultPath)}&file=${file}`);
    if (!response.ok) throw new Error(`Failed to load ${dataType} data`);

    const result = await response.json();
    if (!result.success) throw new Error(`Failed to parse ${dataType} data`);

    // Store in cache with size limit check
    const currentCacheSize = Object.keys(AppState.dataCache).length;
    if (currentCacheSize >= AppState.maxCacheEntries) {
        console.warn(`Cache full (${currentCacheSize}/${AppState.maxCacheEntries}), clearing cache to prevent memory leak`);
        AppState.dataCache = {};
    }
    AppState.dataCache[cacheKey] = result.data;
    return result.data;
}

// Placeholder implementations - these will call actual data loading and plotting
async function renderAzContour(containerId, step) {
    const data = await loadCsvData('Az', step);
    const flipped = flipVertical(data);
    plotContour(containerId, flipped, 'Az [Wb/m]', true);
}

async function renderAzHeatmap(containerId, step) {
    const data = await loadCsvData('Az', step);
    const flipped = flipVertical(data);
    plotHeatmap(containerId, flipped, 'Az [Wb/m]', true);
}

async function renderJzDistribution(containerId, step) {
    const data = await loadCsvData('Jz', step);
    const flipped = flipVertical(data);
    plotHeatmap(containerId, flipped, 'Jz [A/m²]', true);
}

async function renderBMagnitude(containerId, step) {
    // Use grid spacing from analysis conditions
    const dx = AppState.analysisConditions ? AppState.analysisConditions.dx : 0.001;
    const dy = AppState.analysisConditions ? AppState.analysisConditions.dy : 0.001;

    // Load Az and Mu with caching
    const azData = await loadCsvData('Az', step);
    const muData = await loadCsvData('Mu', step);

    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const azFlipped = flipVertical(azData);
    const muFlipped = flipVertical(muData);

    const { B } = calculateMagneticField(azFlipped, muFlipped, dx, dy);

    plotHeatmap(containerId, B, '|B| [T/m]', true);
}

async function renderHMagnitude(containerId, step) {
    // Use grid spacing from analysis conditions
    const dx = AppState.analysisConditions ? AppState.analysisConditions.dx : 0.001;
    const dy = AppState.analysisConditions ? AppState.analysisConditions.dy : 0.001;

    // Load Az and Mu with caching
    const azData = await loadCsvData('Az', step);
    const muData = await loadCsvData('Mu', step);

    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const azFlipped = flipVertical(azData);
    const muFlipped = flipVertical(muData);

    const { H } = calculateMagneticField(azFlipped, muFlipped, dx, dy);

    plotHeatmap(containerId, H, '|H| [A/m]', true);
}

async function renderMuDistribution(containerId, step) {
    const data = await loadCsvData('Mu', step);
    const flipped = flipVertical(data);
    plotHeatmap(containerId, flipped, 'μ [H/m]', true);
}

async function renderEnergyDensity(containerId, step) {
    const data = await loadCsvData('EnergyDensity', step);
    const flipped = flipVertical(data);
    plotHeatmap(containerId, flipped, 'Energy [J/m³]', true);
}

// Helper: Convert black pixels in image to transparent
async function makeBlackTransparent(url, threshold = 30) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            // Draw image
            ctx.drawImage(img, 0, 0);

            // Get pixel data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;

            // Convert black pixels (RGB values below threshold) to transparent
            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];

                // If RGB sum is below threshold, make transparent
                if (r + g + b <= threshold * 3) {
                    pixels[i + 3] = 0;  // Set alpha channel to 0 (transparent)
                }
            }

            // Put modified pixel data back
            ctx.putImageData(imageData, 0, 0);

            // Return as Data URL
            resolve(canvas.toDataURL('image/png'));
        };

        img.onerror = function() {
            reject(new Error('Failed to load boundary image for transparency conversion'));
        };

        img.src = url;
    });
}

async function renderAzBoundary(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Load Az data with caching
        const azData = await loadCsvData('Az', step);
        const azFlipped = flipVertical(azData);

        // Get boundary image URL
        const boundaryImgUrl = `/api/get-boundary-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        // Load boundary image and convert black to transparent
        const transparentBoundaryUrl = await makeBlackTransparent(boundaryImgUrl);

        container.innerHTML = '';
        const size = getContainerSize(container);

        const rows = azFlipped.length;
        const cols = azFlipped[0].length;

        // Generate physical coordinates if available
        let xVals, yVals, xTitle, yTitle, xMin, xMax, yMin, yMax;
        if (AppState.analysisConditions) {
            const coordSys = AppState.analysisConditions.coordinate_system || 'cartesian';
            if (coordSys === 'polar') {
                const theta_start = AppState.analysisConditions.theta_start || 0;
                const dr = AppState.analysisConditions.dr || 0.001;
                const dtheta = AppState.analysisConditions.dtheta || 0.001;
                xVals = Array.from({ length: cols }, (_, i) => i * dr * 1000);
                yVals = Array.from({ length: rows }, (_, i) => theta_start + i * dtheta);
                xTitle = 'r - r_start [mm]';
                yTitle = 'θ [rad]';
                xMin = 0;
                xMax = (cols - 1) * dr * 1000;
                yMin = theta_start;
                yMax = theta_start + (rows - 1) * dtheta;
            } else {
                const dx = AppState.analysisConditions.dx || 0.001;
                const dy = AppState.analysisConditions.dy || 0.001;
                xVals = Array.from({ length: cols }, (_, i) => i * dx * 1000);
                yVals = Array.from({ length: rows }, (_, i) => i * dy * 1000);
                xTitle = 'X [mm]';
                yTitle = 'Y [mm]';
                xMin = 0;
                xMax = (cols - 1) * dx * 1000;
                yMin = 0;
                yMax = (rows - 1) * dy * 1000;
            }
        } else {
            xVals = Array.from({ length: cols }, (_, i) => i);
            yVals = Array.from({ length: rows }, (_, i) => i);
            xTitle = 'X [pixels]';
            yTitle = 'Y [pixels]';
            xMin = 0;
            xMax = cols - 1;
            yMin = 0;
            yMax = rows - 1;
        }

        // Az contour trace
        const traces = [
            {
                z: azFlipped,
                x: xVals,
                y: yVals,
                type: 'contour',
                colorscale: 'Viridis',
                contours: { coloring: 'lines' },
                showscale: false,
                name: 'Az'
            }
        ];

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 35, r: 10, t: 10, b: 35 },
            xaxis: {
                title: xTitle,
                range: [xMin, xMax]
            },
            yaxis: {
                title: yTitle,
                range: [yMin, yMax]
            },
            images: [
                {
                    source: transparentBoundaryUrl,
                    xref: 'x',
                    yref: 'y',
                    x: xMin,
                    y: yMax,
                    sizex: xMax - xMin,
                    sizey: yMax - yMin,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'above'
                }
            ],
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Az+Boundary render error:', error);
        container.innerHTML = `<p style="padding:20px; color:red;">Error: ${error.message}</p>`;
    }
}

async function renderMaterialImage(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Get step input image (from InputImage folder)
        const imgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        // Get or create canvas (keep existing canvas to prevent flickering)
        let canvas = container.querySelector('canvas');
        if (!canvas) {
            container.innerHTML = '<canvas style="width: 100%; height: 100%;"></canvas>';
            canvas = container.querySelector('canvas');
        }
        const ctx = canvas.getContext('2d');

        // Preload image (draw only after fully loaded to prevent flickering)
        const img = new Image();
        img.onload = function() {
            // Set canvas size
            const containerRect = container.getBoundingClientRect();
            const scale = Math.min(containerRect.width / img.width, containerRect.height / img.height);

            canvas.width = img.width * scale;
            canvas.height = img.height * scale;

            // Draw image (image is fully loaded at this point)
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };

        img.onerror = function() {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">Failed to load material image</div>';
        };

        img.src = imgUrl;
    } catch (error) {
        console.error('Material image load error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">Error loading material image</div>';
    }
}

async function renderStepInputImage(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Get step input image
        const imgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        container.innerHTML = '';
        const size = getContainerSize(container);

        // Load image to get dimensions
        const img = new Image();
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = imgUrl;
        });

        const rows = img.height;
        const cols = img.width;

        // Generate physical coordinates if available
        let xTitle, yTitle, xMin, xMax, yMin, yMax;
        if (AppState.analysisConditions) {
            const coordSys = AppState.analysisConditions.coordinate_system || 'cartesian';
            if (coordSys === 'polar') {
                const theta_start = AppState.analysisConditions.theta_start || 0;
                const dr = AppState.analysisConditions.dr || 0.001;
                const dtheta = AppState.analysisConditions.dtheta || 0.001;
                xTitle = 'r - r_start [mm]';
                yTitle = 'θ [rad]';
                xMin = 0;
                xMax = (cols - 1) * dr * 1000;
                yMin = theta_start;
                yMax = theta_start + (rows - 1) * dtheta;
            } else {
                const dx = AppState.analysisConditions.dx || 0.001;
                const dy = AppState.analysisConditions.dy || 0.001;
                xTitle = 'X [mm]';
                yTitle = 'Y [mm]';
                xMin = 0;
                xMax = (cols - 1) * dx * 1000;
                yMin = 0;
                yMax = (rows - 1) * dy * 1000;
            }
        } else {
            xTitle = 'X [pixels]';
            yTitle = 'Y [pixels]';
            xMin = 0;
            xMax = cols - 1;
            yMin = 0;
            yMax = rows - 1;
        }

        // Display image using Plotly
        await Plotly.newPlot(container, [], {
            width: size.width,
            height: size.height,
            margin: { l: 35, r: 10, t: 10, b: 35 },
            xaxis: {
                title: xTitle,
                range: [xMin, xMax],
                showgrid: false
            },
            yaxis: {
                title: yTitle,
                range: [yMin, yMax],
                showgrid: false
            },
            images: [
                {
                    source: imgUrl,
                    xref: 'x',
                    yref: 'y',
                    x: xMin,
                    y: yMax,
                    sizex: xMax - xMin,
                    sizey: yMax - yMin,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'below'
                }
            ],
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Step input image load error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">Error loading image</div>';
    }
}

// Helper: Load force data for a specific step
async function loadForceData(step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) return null;

    try {
        const response = await fetch(`/api/load-csv-raw?result=${encodeURIComponent(resultPath)}&file=Forces/step_${String(step).padStart(4, '0')}.csv`);

        if (!response.ok) {
            console.warn(`Forces data not found for step ${step}`);
            return null;
        }

        const textData = await response.text();

        if (!textData || textData.trim().length === 0) {
            console.warn(`Empty forces data for step ${step}`);
            return null;
        }

        // Parse Forces CSV
        // Format: Material,RGB_R,RGB_G,RGB_B,Force_X[N/m],Force_Y[N/m],Force_Magnitude[N/m],Torque[N],Boundary_Pixels
        const lines = textData.trim().split('\n');

        // Find header line
        let headerIdx = -1;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('Material,')) {
                headerIdx = i;
                break;
            }
        }

        if (headerIdx === -1) {
            console.error(`No header line found in forces file for step ${step}`);
            return null;
        }

        const headers = lines[headerIdx].split(',');

        // Get column indices
        const materialIdx = headers.findIndex(h => h && h.trim() === 'Material');
        const rgbRIdx = headers.findIndex(h => h && h.includes('RGB_R'));
        const rgbGIdx = headers.findIndex(h => h && h.includes('RGB_G'));
        const rgbBIdx = headers.findIndex(h => h && h.includes('RGB_B'));
        const forceXIdx = headers.findIndex(h => h && h.includes('Force_X'));
        const forceYIdx = headers.findIndex(h => h && h.trim().startsWith('Force_Y'));
        const torqueOriginIdx = headers.findIndex(h => h && h.includes('Torque_Origin'));
        const torqueCenterIdx = headers.findIndex(h => h && h.includes('Torque_Center'));
        const energyIdx = headers.findIndex(h => h && h.includes('Magnetic_Energy'));

        // Fallback: old format (Torque only)
        const torqueIdx = torqueOriginIdx !== -1 ? torqueOriginIdx :
                         headers.findIndex(h => h && h.includes('Torque'));

        if (forceXIdx === -1 || forceYIdx === -1 || torqueIdx === -1) {
            console.error(`Missing force columns in step ${step}`);
            return null;
        }

        // Material data and totals
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

                // Get RGB values (for color code creation)
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

        if (dataRowCount === 0) {
            console.log(`No valid data rows found in forces file for step ${step}`);
        }

        return {
            total: {
                force_x: totalForceX,
                force_y: totalForceY,
                torque: totalTorque
            },
            materials: materials
        };
    } catch (error) {
        console.error(`Force data load error for step ${step}:`, error);
        return null;
    }
}

async function renderForceXTime(containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < AppState.totalSteps; i++) {
            const data = await loadForceData(i + 1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No Forces data available</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        // x-axis values: 1..totalSteps array (1-based)
        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        // Get list of material names (from first step)
        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        // Create traces per material
        const traces = [];

        // Calculate marker sizes (always return array)
        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: AppState.totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === AppState.currentStep)) ? highlightSize : baseSize;
            });
        };

        // Trace per material
        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
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

        // Get data range
        const allForces = traces.flatMap(t => t.y);
        const maxForce = Math.max(...allForces.map(Math.abs));
        const yrange = maxForce > 1e-10 ? undefined : [-0.1, 0.1];

        // Legend position: inside plot if traces <= 3, outside otherwise
        const legendConfig = traces.length <= 3
            ? { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
            : { x: 1.02, y: 1, xanchor: 'left' };

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Force X [N/m]', range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Force X time plot error:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`;
        }
    }
}

async function renderForceYTime(containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < AppState.totalSteps; i++) {
            const data = await loadForceData(i + 1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No Forces data available</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        const traces = [];

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: AppState.totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === AppState.currentStep)) ? highlightSize : baseSize;
            });
        };

        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
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

        const allForces = traces.flatMap(t => t.y);
        const maxForce = Math.max(...allForces.map(Math.abs));
        const yrange = maxForce > 1e-10 ? undefined : [-0.1, 0.1];

        // Legend position: inside plot if traces <= 3, outside otherwise
        const legendConfig = traces.length <= 3
            ? { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
            : { x: 1.02, y: 1, xanchor: 'left' };

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Force Y [N/m]', range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Force Y time plot error:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`;
        }
    }
}

async function renderTorqueTime(containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < AppState.totalSteps; i++) {
            const data = await loadForceData(i + 1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No Forces data available</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        const traces = [];

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: AppState.totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === AppState.currentStep)) ? highlightSize : baseSize;
            });
        };

        materialNames.forEach(matName => {
            const torqueData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
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

        const allTorques = traces.flatMap(t => t.y);
        const maxTorque = Math.max(...allTorques.map(Math.abs));
        const yrange = maxTorque > 1e-10 ? undefined : [-0.1, 0.1];

        // Legend position: inside plot if traces <= 3, outside otherwise
        const legendConfig = traces.length <= 3
            ? { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
            : { x: 1.02, y: 1, xanchor: 'left' };

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Torque [Nm/m]', range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Torque time plot error:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`;
        }
    }
}

async function renderEnergyTime(containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < AppState.totalSteps; i++) {
            const data = await loadForceData(i + 1);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No Energy data available</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        const traces = [];

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: AppState.totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === AppState.currentStep)) ? highlightSize : baseSize;
            });
        };

        materialNames.forEach(matName => {
            const energyData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        energyData.push(mat.energy);
                        if (!matColor) matColor = mat.color;
                    } else {
                        energyData.push(0);
                    }
                } else {
                    energyData.push(0);
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

        // Legend position: inside plot if traces <= 3, outside otherwise
        const legendConfig = traces.length <= 3
            ? { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
            : { x: 1.02, y: 1, xanchor: 'left' };

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'Energy [J/m]' },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Energy time plot error:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`;
        }
    }
}

// ===== Plotting Helper Functions =====
function getContainerSize(container) {
    const rect = container.getBoundingClientRect();
    console.log(`Container rect: ${rect.width}x${rect.height}`);

    // Account for padding: 10px * 2 sides = 20px
    // Don't use default if size is too small (before initialization)
    const padding = 20;
    return {
        width: rect.width > 50 ? Math.max(rect.width - padding, 100) : 400,
        height: rect.height > 50 ? Math.max(rect.height - padding, 100) : 400
    };
}

// ===== Plotting Functions =====
function plotContour(elementId, data, title, usePhysicalAxes = false) {
    const container = document.getElementById(elementId);
    if (!container) {
        console.error(`plotContour: Container not found: ${elementId}`);
        return;
    }

    if (!data || data.length === 0) {
        container.innerHTML = '<p>No data available</p>';
        return;
    }

    // Clear container before plotting
    container.innerHTML = '';

    // Get container size
    const size = getContainerSize(container);

    // Colorbar width should be 10% of total width (9:1 ratio)
    const colorbarThickness = Math.floor(size.width * 0.1);

    const trace = {
        z: data,
        type: 'contour',
        colorscale: 'Viridis',
        contours: {
            coloring: 'heatmap'
        },
        colorbar: {
            title: title,
            thickness: colorbarThickness,
            len: 1.0
        }
    };

    // Generate physical axes if requested and conditions are available
    let xaxis, yaxis;
    if (usePhysicalAxes && AppState.analysisConditions) {
        const rows = data.length;
        const cols = data[0]?.length || 0;
        const coordSys = AppState.analysisConditions.coordinate_system || 'cartesian';

        if (coordSys === 'polar') {
            const theta_start = AppState.analysisConditions.theta_start || 0;
            const dr = AppState.analysisConditions.dr || 0.001;
            const dtheta = AppState.analysisConditions.dtheta || 0.001;

            // r: mm (from 0), theta: radians
            const rVals = Array.from({ length: cols }, (_, i) => i * dr * 1000);
            const thetaVals = Array.from({ length: rows }, (_, i) => theta_start + i * dtheta);

            trace.x = rVals;
            trace.y = thetaVals;
            xaxis = { title: 'r - r_start [mm]' };
            yaxis = { title: 'θ [rad]' };
        } else {
            const dx = AppState.analysisConditions.dx || 0.001;
            const dy = AppState.analysisConditions.dy || 0.001;

            // Cartesian: both in mm
            const xVals = Array.from({ length: cols }, (_, i) => i * dx * 1000);
            const yVals = Array.from({ length: rows }, (_, i) => i * dy * 1000);

            trace.x = xVals;
            trace.y = yVals;
            xaxis = { title: 'X [mm]' };
            yaxis = { title: 'Y [mm]' };
        }
    } else {
        xaxis = { title: 'X [pixels]' };
        yaxis = { title: 'Y [pixels]' };
    }

    const layout = {
        width: size.width,
        height: size.height,
        title: title,
        xaxis: xaxis,
        yaxis: yaxis,
        margin: { t: 40, r: colorbarThickness + 15, b: 40, l: 60 },
        dragmode: false
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}

// ===== Plotting Functions =====
function plotHeatmap(elementId, data, title, usePhysicalAxes = false) {
    const container = document.getElementById(elementId);
    if (!container) {
        console.error(`plotHeatmap: Container not found: ${elementId}`);
        return;
    }

    if (!data || data.length === 0) {
        container.innerHTML = '<p>No data available</p>';
        return;
    }

    // Clear container before plotting
    container.innerHTML = '';

    // Get container size
    const size = getContainerSize(container);

    // Colorbar width should be 10% of total width (9:1 ratio)
    const colorbarThickness = Math.floor(size.width * 0.1);

    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {
            title: title,
            thickness: colorbarThickness,
            len: 1.0
        }
    };

    // Generate physical axes if requested and conditions are available
    let xaxis, yaxis;
    if (usePhysicalAxes && AppState.analysisConditions) {
        const rows = data.length;
        const cols = data[0]?.length || 0;
        const coordSys = AppState.analysisConditions.coordinate_system || 'cartesian';

        if (coordSys === 'polar') {
            const theta_start = AppState.analysisConditions.theta_start || 0;
            const dr = AppState.analysisConditions.dr || 0.001;
            const dtheta = AppState.analysisConditions.dtheta || 0.001;

            // r: mm (from 0), theta: radians
            const rVals = Array.from({ length: cols }, (_, i) => i * dr * 1000);
            const thetaVals = Array.from({ length: rows }, (_, i) => theta_start + i * dtheta);

            trace.x = rVals;
            trace.y = thetaVals;
            xaxis = { title: 'r - r_start [mm]' };
            yaxis = { title: 'θ [rad]' };
        } else {
            const dx = AppState.analysisConditions.dx || 0.001;
            const dy = AppState.analysisConditions.dy || 0.001;

            // Cartesian: both in mm
            const xVals = Array.from({ length: cols }, (_, i) => i * dx * 1000);
            const yVals = Array.from({ length: rows }, (_, i) => i * dy * 1000);

            trace.x = xVals;
            trace.y = yVals;
            xaxis = { title: 'X [mm]' };
            yaxis = { title: 'Y [mm]' };
        }
    } else {
        xaxis = { title: 'X [pixels]' };
        yaxis = { title: 'Y [pixels]' };
    }

    const layout = {
        width: size.width,
        height: size.height,
        title: title,
        xaxis: xaxis,
        yaxis: yaxis,
        margin: { t: 40, r: colorbarThickness + 15, b: 40, l: 60 },
        dragmode: false
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
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
