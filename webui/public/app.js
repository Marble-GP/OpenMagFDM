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
    maxCacheEntries: 500,  // Maximum cache entries to prevent memory leak (500 * ~2MB = ~1GB max)
    // Polar coordinate transform options
    isPolarCoordinates: false,  // True if current result uses polar coordinates
    polarCartesianTransform: false,  // Apply cartesian transform (arc/donut view)
    polarFullModel: false,  // Expand to full model
    polarFullModelMultiplier: 1,  // Multiplier for full model (N in 2π/N)
    // Plot zoom state preservation
    plotZoomStates: {},  // { containerId: { xaxis: { range: [min, max] }, yaxis: { range: [min, max] } } }
    // Plotly mode bar visibility
    showPlotlyModeBar: false  // Show/hide Plotly mode bar for all plots
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
    if (tabName === 'files') {
        initializeFileManager();
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
                // Normal parent with defined children (e.g., anderson has children: ["enabled", "depth", "beta"])
                else if (parentInfo && parentInfo.children) {
                    availableKeywords = parentInfo.children;
                }
                // Check if grandparent accepts any child (we're inside a specific material)
                else if (grandparentContext) {
                    const grandparentInfo = keywords[grandparentContext];
                    if (grandparentInfo && grandparentInfo.acceptsAnyChild && grandparentInfo.childrenProperties) {
                        availableKeywords = grandparentInfo.childrenProperties;
                    }
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
        const response = await fetch(`/api/results?userId=${AppState.userId}`);
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

        // Return the results for use in loadResults()
        return result.results;
    } catch (error) {
        console.error('Error loading results list:', error);
        return [];
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

        // Update polar coordinate controls
        updatePolarControls();

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

        // Auto-reload dashboard plots when result selection changes
        await updateAllPlots();
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
        const r_orientation = polar.r_orientation || 'horizontal';

        // Determine nr and ntheta based on r_orientation
        let nr, ntheta;
        if (r_orientation === 'horizontal') {
            // Az[theta_idx][r_idx]: rows = ntheta, cols = nr
            nr = cols;
            ntheta = rows;
        } else {
            // Az[r_idx][theta_idx]: rows = nr, cols = ntheta
            nr = rows;
            ntheta = cols;
        }

        // Calculate dr, dtheta (use from conditions.json if available, otherwise calculate)
        const dr = AppState.analysisConditions.dr || (r_end - r_start) / (nr - 1);
        const dtheta = AppState.analysisConditions.dtheta || polar.theta_range / (ntheta - 1);

        // Determine theta boundary conditions
        const bc = AppState.analysisConditions.boundary_conditions || {};
        const thetaMinBC = bc.theta_min || {};
        const thetaMaxBC = bc.theta_max || {};
        const thetaPeriodic = (thetaMinBC.type === 'periodic' && thetaMaxBC.type === 'periodic');
        const thetaAntiperiodic = thetaPeriodic &&
            ((thetaMinBC.value !== undefined && thetaMinBC.value < 0) ||
             (thetaMaxBC.value !== undefined && thetaMaxBC.value < 0));

        // Generate r coordinate array
        const r_coords = Array(nr).fill(0).map((_, ir) => r_start + ir * dr);

        // Calculate magnetic field in polar coordinates: Br, Bθ
        const Br = Array(rows).fill(0).map(() => Array(cols).fill(0));
        const Btheta = Array(rows).fill(0).map(() => Array(cols).fill(0));

        // Helper function to get Az value at (ir, jt) with r_orientation handling
        const getAz = (ir, jt) => {
            if (r_orientation === 'horizontal') {
                return Az[jt][ir];
            } else {
                return Az[ir][jt];
            }
        };

        // Helper function to set field value at (ir, jt) with r_orientation handling
        const setField = (field, ir, jt, value) => {
            if (r_orientation === 'horizontal') {
                field[jt][ir] = value;
            } else {
                field[ir][jt] = value;
            }
        };

        for (let jt = 0; jt < ntheta; jt++) {
            for (let ir = 0; ir < nr; ir++) {
                const r = r_coords[ir];
                const safe_r = Math.max(r, 1e-15);

                // Br = (1/r) * ∂Az/∂θ
                let jt_next, jt_prev;
                let Az_next, Az_prev;

                if (thetaPeriodic) {
                    // Periodic or anti-periodic boundary
                    jt_next = (jt + 1) % ntheta;
                    jt_prev = (jt - 1 + ntheta) % ntheta;
                    Az_next = getAz(ir, jt_next);
                    Az_prev = getAz(ir, jt_prev);

                    // Apply sign flip for anti-periodic BC
                    if (thetaAntiperiodic) {
                        if (jt === ntheta - 1) Az_next = -Az_next;  // next crosses boundary
                        if (jt === 0) Az_prev = -Az_prev;          // prev crosses boundary
                    }
                } else {
                    // Non-periodic (Dirichlet/Neumann) - use one-sided difference at boundaries
                    if (jt === 0) {
                        jt_next = 1;
                        jt_prev = 0;
                        Az_next = getAz(ir, jt_next);
                        Az_prev = getAz(ir, jt_prev);
                    } else if (jt === ntheta - 1) {
                        jt_next = ntheta - 1;
                        jt_prev = ntheta - 2;
                        Az_next = getAz(ir, jt_next);
                        Az_prev = getAz(ir, jt_prev);
                    } else {
                        jt_next = jt + 1;
                        jt_prev = jt - 1;
                        Az_next = getAz(ir, jt_next);
                        Az_prev = getAz(ir, jt_prev);
                    }
                }

                const denom = (jt === 0 || jt === ntheta - 1) && !thetaPeriodic ? dtheta : (2 * dtheta);
                const dAz_dtheta = (Az_next - Az_prev) / denom;
                setField(Br, ir, jt, dAz_dtheta / safe_r);

                // Bθ = -∂Az/∂r
                let dAz_dr = 0;
                if (ir === 0) {
                    dAz_dr = (getAz(1, jt) - getAz(0, jt)) / dr;
                } else if (ir === nr - 1) {
                    dAz_dr = (getAz(nr-1, jt) - getAz(nr-2, jt)) / dr;
                } else {
                    dAz_dr = (getAz(ir+1, jt) - getAz(ir-1, jt)) / (2 * dr);
                }
                setField(Btheta, ir, jt, -dAz_dr);
            }
        }

        // Polar → Cartesian transformation (for visualization)
        // Physical coordinates: x = r*cos(θ), y = r*sin(θ)
        // Field transformation: Bx = Br*cos(θ) - Bθ*sin(θ), By = Br*sin(θ) + Bθ*cos(θ)
        for (let jt = 0; jt < ntheta; jt++) {
            const theta = jt * dtheta;
            const cos_theta = Math.cos(theta);
            const sin_theta = Math.sin(theta);

            for (let ir = 0; ir < nr; ir++) {
                let Br_val, Btheta_val;
                if (r_orientation === 'horizontal') {
                    Br_val = Br[jt][ir];
                    Btheta_val = Btheta[jt][ir];
                    Bx[jt][ir] = Br_val * cos_theta - Btheta_val * sin_theta;
                    By[jt][ir] = Br_val * sin_theta + Btheta_val * cos_theta;
                } else {
                    Br_val = Br[ir][jt];
                    Btheta_val = Btheta[ir][jt];
                    Bx[ir][jt] = Br_val * cos_theta - Btheta_val * sin_theta;
                    By[ir][jt] = Br_val * sin_theta + Btheta_val * cos_theta;
                }
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
                plotHeatmap('previewPlot2', B, '|B| [T]', true);
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
    const stopBtn = document.getElementById('stopBtn');

    btn.disabled = true;
    btn.textContent = 'Running...';
    stopBtn.style.display = 'inline-block';

    const outputDiv = document.getElementById('solverOutput');
    const progressContainer = document.getElementById('solverProgressContainer');
    const progressBar = document.getElementById('solverProgressBar');
    const progressText = document.getElementById('solverProgressText');
    const progressPercent = document.getElementById('solverProgressPercent');

    outputDiv.textContent = '';
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressBar.style.background = 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)';  // Reset to default color
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
        stopBtn.style.display = 'none';

        // Hide progress bar after 3 seconds if completed successfully
        setTimeout(() => {
            if (progressText.textContent === 'Completed successfully') {
                progressContainer.style.display = 'none';
            }
        }, 3000);
    }
}

async function stopSolver() {
    // Show confirmation dialog
    if (!confirm('解析を停止しますか？\n\nAre you sure you want to stop the calculation?')) {
        return;
    }

    try {
        const response = await fetch('/api/stop-solver', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showStatus('solverStatus', 'Solver stopped by user', 'warning');

            const outputDiv = document.getElementById('solverOutput');
            outputDiv.textContent += '\n\n=== Solver stopped by user ===\n';
            outputDiv.scrollTop = outputDiv.scrollHeight;

            const progressText = document.getElementById('solverProgressText');
            progressText.textContent = 'Stopped by user';

            const progressBar = document.getElementById('solverProgressBar');
            progressBar.style.background = 'linear-gradient(90deg, #ffc107 0%, #ff9800 100%)';
        } else {
            throw new Error(result.error || 'Failed to stop solver');
        }
    } catch (error) {
        console.error('Error stopping solver:', error);
        showStatus('solverStatus', `Error stopping solver: ${error.message}`, 'error');
    } finally {
        // Reset button states
        const btn = document.getElementById('solverBtn');
        const stopBtn = document.getElementById('stopBtn');

        btn.disabled = false;
        btn.textContent = 'Run Solver';
        stopBtn.style.display = 'none';
    }
}

async function loadResults() {
    try {
        // Refresh results list to get the latest results
        const results = await refreshResultsList();

        // Check if we have any results
        if (results.length === 0) {
            showStatus('solverStatus', 'No results found', 'error');
            return;
        }

        // Select the newest result (first in the list)
        const newestResult = results[0];
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
    az_boundary: { name: 'Field Lines (on Material)', render: renderAzBoundary },
    az_edge: { name: 'Field Lines (on Edge)', render: renderAzEdge },
    step_input_image: { name: 'Step Input Image', render: renderStepInputImage },
    coarsening_mask: { name: 'Coarsening Mask', render: renderCoarseningMask },
    line_profile: { name: 'Line Profile', render: renderLineProfile },
    flux_linkage_interactive: { name: 'Flux Linkage', render: renderFluxLinkageInteractive },
    force_x_time: { name: 'Force X-axis', render: renderForceXTime },
    force_y_time: { name: 'Force Y-axis', render: renderForceYTime },
    torque_time: { name: 'Torque', render: renderTorqueTime },
    energy_time: { name: 'Magnetic Energy', render: renderEnergyTime },
    virtual_work: { name: 'Virtual Work (dW/dx)', render: renderVirtualWork }
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
        float: true,
        handle: '.plot-header'  // Only allow dragging from header bar
    }, '#dashboard-grid');

    AppState.gridStack = grid;

    // Setup resize event handler for Plotly plots
    grid.on('resizestop', (_event, element) => {
        // Find the plot container inside the resized widget
        const plotContainer = element.querySelector('.plot-container');
        if (plotContainer && plotContainer._fullLayout) {
            // Get new container size
            const newSize = getContainerSize(plotContainer);

            // Update Plotly layout with new dimensions
            try {
                Plotly.relayout(plotContainer, {
                    width: newSize.width,
                    height: newSize.height
                });
                console.log(`Resized plot ${plotContainer.id} to ${newSize.width}x${newSize.height}`);
            } catch (error) {
                console.error('Error resizing plot:', error);
            }
        }
    });

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
    if (!container) {
        console.warn('Container not found');
        return;
    }

    // Find the actual Plotly container (may be in sub-container for special plots)
    let plotlyContainers = [];
    if (container.data && container.layout) {
        // Direct Plotly plot
        plotlyContainers.push(container);
    } else {
        // Check for special plots with sub-containers
        if (container._lineProfileImageDiv) {
            plotlyContainers.push(container._lineProfileImageDiv);
        }
        if (container._fluxLinkagePlotDiv) {
            plotlyContainers.push(container._fluxLinkagePlotDiv);
        }
        // Also check for sub-divs with Plotly data
        const subDivs = container.querySelectorAll('div[id]');
        subDivs.forEach(div => {
            if (div.data && div.layout && !plotlyContainers.includes(div)) {
                plotlyContainers.push(div);
            }
        });
    }

    if (plotlyContainers.length === 0) {
        console.warn('No Plotly plot found in container');
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

    // Update Plotly dragmode for all found containers
    plotlyContainers.forEach(plotContainer => {
        Plotly.relayout(plotContainer, { dragmode: dragmode }).catch(err => {
            console.error('Failed to update drag mode:', err);
        });
    });

    // Update GridStack tile movability
    const widgetEl = container.closest('.grid-stack-item');
    if (widgetEl && AppState.gridStack) {
        AppState.gridStack.movable(widgetEl, tileMovable);
    }
}

// ===== Reset Interaction Mode =====
function resetInteractionMode(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Find the interaction mode button
    const contentElement = container.closest('.grid-stack-item-content');
    if (!contentElement) return;

    const button = contentElement.querySelector('.interaction-mode-btn');
    if (!button) return;

    // Reset to Move mode (disabled)
    button.dataset.mode = 'disabled';
    button.title = 'Mode: Move';

    // Update button icon
    const img = button.querySelector('img');
    if (img) {
        img.src = '/icon/window.svg';
    }

    // Update Plotly dragmode to false (allows tile movement)
    if (container._fullLayout || container.layout) {
        Plotly.relayout(container, { dragmode: false }).catch(err => {
            console.error('Failed to reset drag mode:', err);
        });
    }

    // Enable GridStack tile movability
    const widgetEl = container.closest('.grid-stack-item');
    if (widgetEl && AppState.gridStack) {
        AppState.gridStack.movable(widgetEl, true);
    }
}

// ===== Reset Plot Zoom =====
function resetPlotZoom(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Clear saved zoom state
    delete AppState.plotZoomStates[containerId];

    // Reset interaction mode to Move
    resetInteractionMode(containerId);

    // Find plot type from parent element and re-render to reset
    const contentElement = container.closest('.grid-stack-item-content');
    if (contentElement) {
        const plotType = contentElement.dataset.plotType;
        const plotDef = plotDefinitions[plotType];

        // Reset special widget states
        if (plotType === 'line_profile' && lineProfileState[containerId]) {
            lineProfileState[containerId].startPoint = null;
            lineProfileState[containerId].endPoint = null;
            lineProfileState[containerId].selectingStart = true;
            lineProfileState[containerId].zoomRange = null;  // Reset zoom
        }
        if (plotType === 'flux_linkage_interactive' && fluxLinkageState[containerId]) {
            fluxLinkageState[containerId].startPoint = null;
            fluxLinkageState[containerId].endPoint = null;
            fluxLinkageState[containerId].selectingStart = true;
            fluxLinkageState[containerId].fluxValue = null;
            fluxLinkageState[containerId].zoomRange = null;  // Reset zoom
        }

        if (plotDef && plotDef.render) {
            // Re-render the plot to reset to initial state
            plotDef.render(containerId, AppState.currentStep).catch(err => {
                console.error('Plot re-render error:', err);
            });
            return;
        }
    }

    // Fallback to relayout method (for plots without render function)
    if (container._fullLayout || container.layout) {
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
        // Clear saved zoom state
        if (container.id) {
            delete AppState.plotZoomStates[container.id];
        }

        // Reset interaction mode to Move
        resetInteractionMode(container.id);

        // Find plot type from parent element and re-render to reset
        const contentElement = container.closest('.grid-stack-item-content');
        if (contentElement) {
            const plotType = contentElement.dataset.plotType;
            const plotDef = plotDefinitions[plotType];

            if (plotDef && plotDef.render) {
                // Re-render the plot to reset to initial state
                plotDef.render(container.id, AppState.currentStep).catch(err => {
                    console.error('Plot re-render error:', err);
                });
                return;
            }
        }

        // Fallback to relayout method (for plots without render function)
        if (container._fullLayout || container.layout) {
            Plotly.relayout(container, {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            }).catch(err => {
                console.error('Plotly reset zoom error:', err);
            });
        }
    });
}

// ===== Toggle Plotly Mode Bar =====
function togglePlotlyModeBar(show) {
    AppState.showPlotlyModeBar = show;

    // Update all existing plots
    const containers = document.querySelectorAll('.plot-container[id^="container-"]');
    containers.forEach(container => {
        // Check if this is a Plotly plot
        if (container._fullLayout) {
            // Update mode bar visibility using Plotly.relayout
            Plotly.relayout(container, {
                'modebar.orientation': 'v'  // Trigger relayout
            }).then(() => {
                // Force mode bar visibility update
                const config = { displayModeBar: show };
                Plotly.react(container, container.data, container.layout, config);
            }).catch(err => {
                console.error('Failed to toggle mode bar:', err);
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
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    plotContour(containerId, flipped, 'Az [Wb/m]', true);
}

async function renderAzHeatmap(containerId, step) {
    const data = await loadCsvData('Az', step);
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    plotHeatmap(containerId, flipped, 'Az [Wb/m]', true);
}

async function renderJzDistribution(containerId, step) {
    const data = await loadCsvData('Jz', step);
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
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

    plotHeatmap(containerId, B, '|B| [T]', true);
}

async function renderHMagnitude(containerId, step) {
    // Check if nonlinear materials are present and enabled
    const hasNonlinear = AppState.analysisConditions?.nonlinear_solver?.has_nonlinear_materials;
    const nlEnabled = AppState.analysisConditions?.nonlinear_solver?.enabled;

    if (hasNonlinear && nlEnabled) {
        // For nonlinear materials: load H directly from solver output (H.csv)
        try {
            const hData = await loadCsvData('H', step);
            // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
            const hFlipped = flipVertical(hData);
            plotHeatmap(containerId, hFlipped, '|H| [A/m] (solver)', true);
            return;
        } catch (error) {
            console.warn('H.csv not found, falling back to calculation from Az and Mu:', error);
        }
    }

    // For linear materials: calculate H from Az and Mu
    const dx = AppState.analysisConditions ? AppState.analysisConditions.dx : 0.001;
    const dy = AppState.analysisConditions ? AppState.analysisConditions.dy : 0.001;

    // Load Az and Mu with caching
    const azData = await loadCsvData('Az', step);
    const muData = await loadCsvData('Mu', step);

    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const azFlipped = flipVertical(azData);
    const muFlipped = flipVertical(muData);

    const { H } = calculateMagneticField(azFlipped, muFlipped, dx, dy);

    plotHeatmap(containerId, H, '|H| [A/m] (calculated)', true);
}

async function renderMuDistribution(containerId, step) {
    const data = await loadCsvData('Mu', step);
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    plotHeatmap(containerId, flipped, 'μ [H/m]', true);
}

async function renderEnergyDensity(containerId, step) {
    const data = await loadCsvData('EnergyDensity', step);
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
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

// Helper: Flip an image URL vertically (for image coordinate to analysis coordinate conversion)
async function flipImageVertical(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            // Flip vertically: translate to bottom, scale y by -1
            ctx.translate(0, img.height);
            ctx.scale(1, -1);
            ctx.drawImage(img, 0, 0);

            // Return as Data URL
            resolve(canvas.toDataURL('image/png'));
        };

        img.onerror = function() {
            reject(new Error('Failed to load image for vertical flip'));
        };

        img.src = url;
    });
}

// Helper: Flip an image URL vertically AND make black pixels transparent
async function flipAndMakeBlackTransparent(url, threshold = 30) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            // Flip vertically: translate to bottom, scale y by -1
            ctx.translate(0, img.height);
            ctx.scale(1, -1);
            ctx.drawImage(img, 0, 0);

            // Reset transform for getImageData
            ctx.setTransform(1, 0, 0, 1, 0, 0);

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
            reject(new Error('Failed to load image for flip and transparency conversion'));
        };

        img.src = url;
    });
}

/**
 * Apply Sobel edge detection to an image URL
 * Returns a canvas with white background and black edges
 * @param {string} url - Image URL
 * @returns {Promise<HTMLCanvasElement>} - Canvas with edge detection result
 */
async function applySobelEdgeDetection(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const width = img.width;
            const height = img.height;

            // Create canvases
            const srcCanvas = document.createElement('canvas');
            srcCanvas.width = width;
            srcCanvas.height = height;
            const srcCtx = srcCanvas.getContext('2d');

            const dstCanvas = document.createElement('canvas');
            dstCanvas.width = width;
            dstCanvas.height = height;
            const dstCtx = dstCanvas.getContext('2d');

            // Draw source image
            srcCtx.drawImage(img, 0, 0);
            const srcData = srcCtx.getImageData(0, 0, width, height);
            const src = srcData.data;

            // Prepare output with white background
            const dstData = dstCtx.createImageData(width, height);
            const dst = dstData.data;

            // Convert to grayscale first (inline)
            const gray = new Float32Array(width * height);
            for (let i = 0; i < width * height; i++) {
                const idx = i * 4;
                gray[i] = 0.299 * src[idx] + 0.587 * src[idx + 1] + 0.114 * src[idx + 2];
            }

            // Sobel kernels
            const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
            const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

            // Apply Sobel filter
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    let gx = 0, gy = 0;

                    for (let ky = -1; ky <= 1; ky++) {
                        for (let kx = -1; kx <= 1; kx++) {
                            const pixel = gray[(y + ky) * width + (x + kx)];
                            gx += pixel * sobelX[ky + 1][kx + 1];
                            gy += pixel * sobelY[ky + 1][kx + 1];
                        }
                    }

                    // Calculate gradient magnitude
                    const magnitude = Math.sqrt(gx * gx + gy * gy);

                    const dstIdx = (y * width + x) * 4;
                    const srcIdx = (y * width + x) * 4;
                    const threshold = 30;

                    if (magnitude > threshold) {
                        // Edge: keep original pixel color
                        dst[dstIdx] = src[srcIdx];          // R
                        dst[dstIdx + 1] = src[srcIdx + 1];  // G
                        dst[dstIdx + 2] = src[srcIdx + 2];  // B
                    } else {
                        // Uniform area: white background
                        dst[dstIdx] = 255;      // R
                        dst[dstIdx + 1] = 255;  // G
                        dst[dstIdx + 2] = 255;  // B
                    }
                    dst[dstIdx + 3] = 255;  // A (opaque)
                }
            }

            // Handle edges (set to white)
            for (let x = 0; x < width; x++) {
                // Top row
                let idx = x * 4;
                dst[idx] = dst[idx + 1] = dst[idx + 2] = 255;
                dst[idx + 3] = 255;
                // Bottom row
                idx = ((height - 1) * width + x) * 4;
                dst[idx] = dst[idx + 1] = dst[idx + 2] = 255;
                dst[idx + 3] = 255;
            }
            for (let y = 0; y < height; y++) {
                // Left column
                let idx = (y * width) * 4;
                dst[idx] = dst[idx + 1] = dst[idx + 2] = 255;
                dst[idx + 3] = 255;
                // Right column
                idx = (y * width + width - 1) * 4;
                dst[idx] = dst[idx + 1] = dst[idx + 2] = 255;
                dst[idx + 3] = 255;
            }

            dstCtx.putImageData(dstData, 0, 0);
            resolve(dstCanvas);
        };

        img.onerror = function() {
            reject(new Error('Failed to load image for edge detection'));
        };

        img.src = url;
    });
}

/**
 * Calculate appropriate number of contour lines based on image size
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {number} - Number of contour lines
 */
function calculateContourLineCount(width, height) {
    // Target: approximately one line per 20-30 pixels (diagonal)
    const diagonal = Math.sqrt(width * width + height * height);
    const lineCount = Math.floor(diagonal / 25);
    // Clamp between 10 and 40 lines
    return Math.max(10, Math.min(40, lineCount));
}

/**
 * Draw contour lines to canvas from 2D data
 * @param {Array<Array<number>>} data - 2D array of values (analysis coordinate: y-up)
 * @param {number} numLines - Number of contour lines (if not provided, auto-calculate)
 * @returns {HTMLCanvasElement} - Canvas with contour lines (image coordinate: y-down, black background, white lines)
 */
function drawContourToCanvas(data, numLines = null) {
    const rows = data.length;
    const cols = data[0].length;

    // Auto-calculate number of contour lines if not specified
    if (numLines === null) {
        numLines = calculateContourLineCount(cols, rows);
    }

    const canvas = document.createElement('canvas');
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d');

    // Fill black background (will be converted to transparent later)
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, cols, rows);

    // Find min/max for contour levels
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const val = data[i][j];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }

    // Draw contour lines with WHITE color (black background will become transparent)
    ctx.fillStyle = 'white';

    console.log(`Drawing ${numLines} contour lines for ${cols}x${rows} image`);

    for (let k = 0; k < numLines; k++) {
        const level = minVal + (maxVal - minVal) * k / (numLines - 1);

        // Simple threshold-based contour (approximate)
        // Note: data is in analysis coordinate (y-up), but canvas is image coordinate (y-down)
        for (let i = 0; i < rows - 1; i++) {
            for (let j = 0; j < cols - 1; j++) {
                const v00 = data[i][j];
                const v10 = data[i][j + 1];
                const v01 = data[i + 1][j];
                const v11 = data[i + 1][j + 1];

                // Check if contour passes through this cell
                const minV = Math.min(v00, v10, v01, v11);
                const maxV = Math.max(v00, v10, v01, v11);

                if (level >= minV && level <= maxV) {
                    // Draw white line segment
                    ctx.fillRect(j, i, 1, 1);
                }
            }
        }
    }

    return canvas;
}

/**
 * Transform polar coordinate image to cartesian
 * @param {HTMLCanvasElement} polarCanvas - Canvas with polar image (image coordinate: y-down)
 * @param {object} conditions - Analysis conditions
 * @param {boolean} fullModel - If true, expand to full model
 * @param {boolean} preserveColors - If true, preserve original colors (for boundary images); if false, convert black to transparent (for contour lines)
 * @returns {HTMLCanvasElement} - Canvas with cartesian image (image coordinate: y-down, black→transparent)
 */
function transformPolarImageToCartesian(polarCanvas, conditions, fullModel = false, preserveColors = false) {
    const r_i = conditions.polar?.r_start || conditions.r_i || 0;
    const r_o = conditions.polar?.r_end || conditions.r_o || 1;
    const thetaRange = conditions.polar?.theta_range || conditions.theta_range || 0;
    const r_orientation = conditions.polar?.r_orientation || 'horizontal';

    const polarWidth = polarCanvas.width;
    const polarHeight = polarCanvas.height;

    // Determine nr and ntheta based on r_orientation
    // r_orientation: 'horizontal' means r is horizontal (cols), theta is vertical (rows)
    // r_orientation: 'vertical' means r is vertical (rows), theta is horizontal (cols)
    let nr, ntheta;
    if (r_orientation === 'horizontal') {
        nr = polarWidth;
        ntheta = polarHeight;
    } else {
        nr = polarHeight;
        ntheta = polarWidth;
    }

    // Determine repetitions for full model
    const N = fullModel ? AppState.polarFullModelMultiplier : 1;

    // Create output canvas
    const resolution = Math.max(polarWidth, polarHeight) * 2;
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = resolution;
    outputCanvas.height = resolution;
    const ctx = outputCanvas.getContext('2d');

    // Fill black background (will be converted to transparent)
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, resolution, resolution);

    // Get polar image data
    const polarCtx = polarCanvas.getContext('2d');
    const polarImageData = polarCtx.getImageData(0, 0, polarWidth, polarHeight);
    const polarPixels = polarImageData.data;

    // Create output image data
    const outputImageData = ctx.createImageData(resolution, resolution);
    const outputPixels = outputImageData.data;

    // Initialize with black (transparent background)
    for (let i = 0; i < outputPixels.length; i += 4) {
        outputPixels[i] = 0;
        outputPixels[i + 1] = 0;
        outputPixels[i + 2] = 0;
        outputPixels[i + 3] = 255;
    }

    // Transform: for each output pixel, find corresponding polar pixel
    const centerX = resolution / 2;
    const centerY = resolution / 2;
    const scale = resolution / (2 * r_o);

    for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
            // Convert image coordinate (y-down) to physical coordinate (y-up)
            // Image: y=0 is top, y=resolution is bottom
            // Physical: py>0 is up, py<0 is down
            const px = (x - centerX) / scale;
            const py = (centerY - y) / scale;  // Y-axis flip: image y-down → physical y-up

            // Convert to polar
            const r = Math.sqrt(px * px + py * py);
            let theta = Math.atan2(py, px);
            if (theta < 0) theta += 2 * Math.PI;

            // Check if within valid range
            if (r < r_i || r > r_o) {
                // Outside domain - keep black (transparent)
                continue;
            }

            // Map theta to sector for full model
            if (fullModel) {
                const sectorAngle = 2 * Math.PI / N;
                theta = theta % sectorAngle;
            } else {
                if (theta > thetaRange) {
                    // Outside sector - keep black (transparent)
                    continue;
                }
            }

            // Calculate fractional indices for r and theta
            const r_frac = (r - r_i) / (r_o - r_i) * (nr - 1);
            const theta_frac = theta / thetaRange * (ntheta - 1);

            // Map to polar image coordinates based on r_orientation
            let polarX, polarY;
            if (r_orientation === 'horizontal') {
                // r is horizontal (x), theta is vertical (y)
                polarX = Math.floor(r_frac);
                polarY = Math.floor(theta_frac);
            } else {
                // r is vertical (y), theta is horizontal (x)
                polarX = Math.floor(theta_frac);
                polarY = Math.floor(r_frac);
            }

            if (polarX >= 0 && polarX < polarWidth && polarY >= 0 && polarY < polarHeight) {
                const polarIdx = (polarY * polarWidth + polarX) * 4;
                const outIdx = (y * resolution + x) * 4;

                outputPixels[outIdx] = polarPixels[polarIdx];
                outputPixels[outIdx + 1] = polarPixels[polarIdx + 1];
                outputPixels[outIdx + 2] = polarPixels[polarIdx + 2];
                outputPixels[outIdx + 3] = polarPixels[polarIdx + 3];
            }
        }
    }

    ctx.putImageData(outputImageData, 0, 0);

    // Convert black pixels to transparent (preserve colors if requested)
    const finalImageData = ctx.getImageData(0, 0, resolution, resolution);
    const finalPixels = finalImageData.data;
    const threshold = 30;  // Black threshold

    if (preserveColors) {
        // For boundary images: preserve original colors, only make black transparent
        for (let i = 0; i < finalPixels.length; i += 4) {
            const r = finalPixels[i];
            const g = finalPixels[i + 1];
            const b = finalPixels[i + 2];

            // If RGB sum is below threshold (black background), make transparent
            if (r + g + b <= threshold * 3) {
                finalPixels[i + 3] = 0;  // Set alpha to transparent
            }
            // Otherwise keep original color and alpha
        }
    } else {
        // For contour lines: convert all black to transparent
        for (let i = 0; i < finalPixels.length; i += 4) {
            const r = finalPixels[i];
            const g = finalPixels[i + 1];
            const b = finalPixels[i + 2];

            // If RGB sum is below threshold, make transparent
            if (r + g + b <= threshold * 3) {
                finalPixels[i + 3] = 0;  // Set alpha to transparent
            }
        }
    }

    ctx.putImageData(finalImageData, 0, 0);
    return outputCanvas;
}

/**
 * Dilate image (8-connectivity morphological dilation)
 * @param {HTMLCanvasElement} canvas - Input canvas (white lines on black background)
 * @param {number} iterations - Number of dilation iterations
 * @returns {HTMLCanvasElement} - Dilated canvas
 */
function dilateImage(canvas, iterations = 1) {
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');

    for (let iter = 0; iter < iterations; iter++) {
        const imageData = ctx.getImageData(0, 0, width, height);
        const pixels = imageData.data;
        const output = new Uint8ClampedArray(pixels);

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;

                // Check if current pixel is WHITE (contour line)
                if (pixels[idx] > 128) {
                    // Dilate white to 8 neighbors
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const nIdx = ((y + dy) * width + (x + dx)) * 4;
                            output[nIdx] = 255;
                            output[nIdx + 1] = 255;
                            output[nIdx + 2] = 255;
                            output[nIdx + 3] = 255;
                        }
                    }
                }
            }
        }

        for (let i = 0; i < pixels.length; i++) {
            pixels[i] = output[i];
        }
        ctx.putImageData(imageData, 0, 0);
    }

    return canvas;
}

/**
 * Merge two images (overlay contour lines on boundary)
 * @param {HTMLCanvasElement} contourCanvas - Contour lines canvas (white lines on black/transparent background)
 * @param {HTMLCanvasElement} boundaryCanvas - Boundary image canvas (after transformation, black→transparent)
 * @returns {HTMLCanvasElement} - Merged canvas (white background, black contour and boundary lines)
 */
function mergeImages(contourCanvas, boundaryCanvas) {
    const width = Math.max(contourCanvas.width, boundaryCanvas.width);
    const height = Math.max(contourCanvas.height, boundaryCanvas.height);

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = width;
    outputCanvas.height = height;
    const ctx = outputCanvas.getContext('2d');

    // Fill white background for final output
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Get boundary image data
    const boundaryCtx = boundaryCanvas.getContext('2d');
    const boundaryData = boundaryCtx.getImageData(0, 0, boundaryCanvas.width, boundaryCanvas.height);
    const boundaryPixels = boundaryData.data;

    // Get contour image data
    const contourCtx = contourCanvas.getContext('2d');
    const contourData = contourCtx.getImageData(0, 0, contourCanvas.width, contourCanvas.height);
    const contourPixels = contourData.data;

    // Create output image with both images merged
    const outputData = ctx.createImageData(width, height);
    const outputPixels = outputData.data;

    // Initialize with white background
    for (let i = 0; i < outputPixels.length; i += 4) {
        outputPixels[i] = 255;
        outputPixels[i + 1] = 255;
        outputPixels[i + 2] = 255;
        outputPixels[i + 3] = 255;
    }

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const outIdx = (y * width + x) * 4;

            // Get corresponding boundary pixel (scale if needed)
            const boundaryX = Math.floor(x * boundaryCanvas.width / width);
            const boundaryY = Math.floor(y * boundaryCanvas.height / height);

            if (boundaryX < boundaryCanvas.width && boundaryY < boundaryCanvas.height) {
                const boundaryIdx = (boundaryY * boundaryCanvas.width + boundaryX) * 4;
                const boundaryAlpha = boundaryPixels[boundaryIdx + 3];

                // If boundary pixel is NOT transparent, draw it with original color
                if (boundaryAlpha > 128) {
                    outputPixels[outIdx] = boundaryPixels[boundaryIdx];
                    outputPixels[outIdx + 1] = boundaryPixels[boundaryIdx + 1];
                    outputPixels[outIdx + 2] = boundaryPixels[boundaryIdx + 2];
                    outputPixels[outIdx + 3] = 255;
                }
            }

            // Get corresponding contour pixel (scale if needed)
            const contourX = Math.floor(x * contourCanvas.width / width);
            const contourY = Math.floor(y * contourCanvas.height / height);

            if (contourX < contourCanvas.width && contourY < contourCanvas.height) {
                const contourIdx = (contourY * contourCanvas.width + contourX) * 4;
                const isWhite = contourPixels[contourIdx] > 128;

                if (isWhite) {
                    // Draw contour line as BLACK
                    outputPixels[outIdx] = 0;
                    outputPixels[outIdx + 1] = 0;
                    outputPixels[outIdx + 2] = 0;
                    outputPixels[outIdx + 3] = 255;
                }
            }
        }
    }

    ctx.putImageData(outputData, 0, 0);
    return outputCanvas;
}

/**
 * Load image to canvas
 * @param {string} url - Image URL
 * @returns {Promise<HTMLCanvasElement>} - Canvas with loaded image
 */
async function loadImageToCanvas(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            resolve(canvas);
        };

        img.onerror = function() {
            reject(new Error(`Failed to load image: ${url}`));
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

        // Get input image URL (material image as background for field lines)
        const inputImgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        container.innerHTML = '';
        const size = getContainerSize(container);

        const coordSys = AppState.analysisConditions?.coordinate_system || 'cartesian';

        // Check if polar coordinate transformation is enabled
        if (coordSys === 'polar' && AppState.polarCartesianTransform) {
            // Image-based approach for polar transformation
            console.log('Using image-based approach for polar coordinate transformation');

            // Step 1: Draw contour lines to canvas (in polar coordinates)
            // numLines is auto-calculated based on image size
            const contourCanvas = drawContourToCanvas(azData);

            // Step 2: Apply dilation to thicken lines (prevent breakage during transformation)
            dilateImage(contourCanvas, 1);

            // Step 3: Load input image (material image) to canvas
            const inputCanvas = await loadImageToCanvas(inputImgUrl);

            // Step 4: Transform both images to cartesian coordinates
            // For contour: preserveColors=false (convert black to transparent)
            const contourCartesian = transformPolarImageToCartesian(
                contourCanvas,
                AppState.analysisConditions,
                AppState.polarFullModel,
                false
            );

            // For input image: preserveColors=true (keep material colors, only black becomes transparent)
            const inputCartesian = transformPolarImageToCartesian(
                inputCanvas,
                AppState.analysisConditions,
                AppState.polarFullModel,
                true
            );

            // Step 5: Merge images (overlay contour lines on material image)
            const mergedCanvas = mergeImages(contourCartesian, inputCartesian);

            // Step 6: Display as image in Plotly
            const mergedImageUrl = mergedCanvas.toDataURL('image/png');

            const r_o = AppState.analysisConditions.polar?.r_end || AppState.analysisConditions.r_o || 1;
            const xMin = -r_o * 1000;
            const xMax = r_o * 1000;
            const yMin = -r_o * 1000;
            const yMax = r_o * 1000;

            let layout = {
                width: size.width,
                height: size.height,
                margin: { l: 35, r: 10, t: 10, b: 35 },
                xaxis: {
                    title: 'X [mm]',
                    range: [xMin, xMax],
                    ...(AppState.polarFullModel && { scaleanchor: 'y', scaleratio: 1 })
                },
                yaxis: {
                    title: 'Y [mm]',
                    range: [yMin, yMax]
                },
                images: [{
                    source: mergedImageUrl,
                    xref: 'x',
                    yref: 'y',
                    x: xMin,
                    y: yMax,
                    sizex: xMax - xMin,
                    sizey: yMax - yMin,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'above'
                }],
                dragmode: false
            };

            // Restore saved zoom state if exists
            layout = restoreZoomState(containerId, layout);

            await Plotly.newPlot(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
            setupZoomTracking(containerId);

        } else {
            // Original Plotly contour approach for non-transformed coordinates
            const azFlipped = flipVertical(azData);
            const transparentInputUrl = await makeBlackTransparent(inputImgUrl);

            const rows = azFlipped.length;
            const cols = azFlipped[0].length;

            let xVals, yVals, zVals, xTitle, yTitle, xMin, xMax, yMin, yMax;

            if (coordSys === 'polar') {
                // Original polar view (r vs theta)
                const theta_start = AppState.analysisConditions.theta_start || 0;
                const dr = AppState.analysisConditions.dr || 0.001;
                const dtheta = AppState.analysisConditions.dtheta || 0.001;
                const r_orientation = AppState.analysisConditions.polar?.r_orientation || 'horizontal';

                // Determine nr and ntheta based on r_orientation
                let nr, ntheta;
                if (r_orientation === 'horizontal') {
                    nr = cols;
                    ntheta = rows;
                } else {
                    nr = rows;
                    ntheta = cols;
                }

                const rVals = Array.from({ length: nr }, (_, i) => i * dr * 1000);
                const thetaVals = Array.from({ length: ntheta }, (_, i) => theta_start + i * dtheta);
                zVals = azFlipped;

                if (r_orientation === 'horizontal') {
                    xVals = rVals;
                    yVals = thetaVals;
                    xTitle = 'r - r_start [mm]';
                    yTitle = 'θ [rad]';
                    xMin = 0;
                    xMax = (nr - 1) * dr * 1000;
                    yMin = theta_start;
                    yMax = theta_start + (ntheta - 1) * dtheta;
                } else {
                    xVals = thetaVals;
                    yVals = rVals;
                    xTitle = 'θ [rad]';
                    yTitle = 'r - r_start [mm]';
                    xMin = theta_start;
                    xMax = theta_start + (ntheta - 1) * dtheta;
                    yMin = 0;
                    yMax = (nr - 1) * dr * 1000;
                }
            } else if (AppState.analysisConditions) {
                // Cartesian coordinates
                const dx = AppState.analysisConditions.dx || 0.001;
                const dy = AppState.analysisConditions.dy || 0.001;
                xVals = Array.from({ length: cols }, (_, i) => i * dx * 1000);
                yVals = Array.from({ length: rows }, (_, i) => i * dy * 1000);
                zVals = azFlipped;
                xTitle = 'X [mm]';
                yTitle = 'Y [mm]';
                xMin = 0;
                xMax = (cols - 1) * dx * 1000;
                yMin = 0;
                yMax = (rows - 1) * dy * 1000;
            } else {
                xVals = Array.from({ length: cols }, (_, i) => i);
                yVals = Array.from({ length: rows }, (_, i) => i);
                zVals = azFlipped;
                xTitle = 'X [pixels]';
                yTitle = 'Y [pixels]';
                xMin = 0;
                xMax = cols - 1;
                yMin = 0;
                yMax = rows - 1;
            }

            const traces = [{
                z: zVals,
                x: xVals,
                y: yVals,
                type: 'contour',
                colorscale: 'Viridis',
                contours: { coloring: 'lines' },
                showscale: false,
                name: 'Az'
            }];

            let layout = {
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
                images: [{
                    source: transparentInputUrl,
                    xref: 'x',
                    yref: 'y',
                    x: xMin,
                    y: yMax,
                    sizex: xMax - xMin,
                    sizey: yMax - yMin,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'below'
                }],
                dragmode: false
            };

            // Restore saved zoom state if exists
            layout = restoreZoomState(containerId, layout);

            await Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
            setupZoomTracking(containerId);
        }
    } catch (error) {
        console.error('Field Lines + Material Image render error:', error);
        container.innerHTML = `<p style="padding:20px; color:red;">Error: ${error.message}</p>`;
    }
}

/**
 * Render field lines (Az contours) overlaid on edge-detected boundary image
 * Uses Sobel edge detection on input image for clearer boundary visualization
 */
async function renderAzEdge(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Load Az data with caching
        const azData = await loadCsvData('Az', step);

        // Get input image URL
        const inputImgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        container.innerHTML = '';
        const size = getContainerSize(container);

        const coordSys = AppState.analysisConditions?.coordinate_system || 'cartesian';

        // Apply Sobel edge detection to input image
        const edgeCanvas = await applySobelEdgeDetection(inputImgUrl);
        const edgeImgUrl = edgeCanvas.toDataURL('image/png');

        // Check if polar coordinate transformation is enabled
        if (coordSys === 'polar' && AppState.polarCartesianTransform) {
            // Image-based approach for polar transformation
            console.log('Using edge detection with polar coordinate transformation');

            // Step 1: Draw contour lines to canvas (in polar coordinates)
            const contourCanvas = drawContourToCanvas(azData);

            // Step 2: Apply dilation to thicken lines
            dilateImage(contourCanvas, 1);

            // Step 3: Transform edge image to cartesian coordinates
            const edgeCartesian = transformPolarImageToCartesian(
                edgeCanvas,
                AppState.analysisConditions,
                AppState.polarFullModel,
                true
            );

            // Step 4: Transform contour to cartesian coordinates
            const contourCartesian = transformPolarImageToCartesian(
                contourCanvas,
                AppState.analysisConditions,
                AppState.polarFullModel,
                false
            );

            // Step 5: Merge images (overlay contour lines on edge image)
            const mergedCanvas = mergeImages(contourCartesian, edgeCartesian);
            const mergedImageUrl = mergedCanvas.toDataURL('image/png');

            const r_o = AppState.analysisConditions.polar?.r_end || AppState.analysisConditions.r_o || 1;
            const xMin = -r_o * 1000;
            const xMax = r_o * 1000;
            const yMin = -r_o * 1000;
            const yMax = r_o * 1000;

            let layout = {
                width: size.width,
                height: size.height,
                margin: { l: 35, r: 10, t: 10, b: 35 },
                xaxis: {
                    title: 'X [mm]',
                    range: [xMin, xMax],
                    ...(AppState.polarFullModel && { scaleanchor: 'y', scaleratio: 1 })
                },
                yaxis: {
                    title: 'Y [mm]',
                    range: [yMin, yMax]
                },
                images: [{
                    source: mergedImageUrl,
                    xref: 'x',
                    yref: 'y',
                    x: xMin,
                    y: yMax,
                    sizex: xMax - xMin,
                    sizey: yMax - yMin,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'above'
                }],
                dragmode: false
            };

            layout = restoreZoomState(containerId, layout);
            await Plotly.newPlot(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
            setupZoomTracking(containerId);

        } else {
            // Original Plotly contour approach with edge-detected background
            const azFlipped = flipVertical(azData);

            const rows = azFlipped.length;
            const cols = azFlipped[0].length;

            let xVals, yVals, zVals, xTitle, yTitle, xMin, xMax, yMin, yMax;

            if (coordSys === 'polar') {
                const theta_start = AppState.analysisConditions.theta_start || 0;
                const dr = AppState.analysisConditions.dr || 0.001;
                const dtheta = AppState.analysisConditions.dtheta || 0.001;
                const r_orientation = AppState.analysisConditions.polar?.r_orientation || 'horizontal';

                let nr, ntheta;
                if (r_orientation === 'horizontal') {
                    nr = cols;
                    ntheta = rows;
                } else {
                    nr = rows;
                    ntheta = cols;
                }

                const rVals = Array.from({ length: nr }, (_, i) => i * dr * 1000);
                const thetaVals = Array.from({ length: ntheta }, (_, i) => theta_start + i * dtheta);
                zVals = azFlipped;

                if (r_orientation === 'horizontal') {
                    xVals = rVals;
                    yVals = thetaVals;
                    xTitle = 'r - r_start [mm]';
                    yTitle = 'θ [rad]';
                    xMin = 0;
                    xMax = (nr - 1) * dr * 1000;
                    yMin = theta_start;
                    yMax = theta_start + (ntheta - 1) * dtheta;
                } else {
                    xVals = thetaVals;
                    yVals = rVals;
                    xTitle = 'θ [rad]';
                    yTitle = 'r - r_start [mm]';
                    xMin = theta_start;
                    xMax = theta_start + (ntheta - 1) * dtheta;
                    yMin = 0;
                    yMax = (nr - 1) * dr * 1000;
                }
            } else if (AppState.analysisConditions) {
                const dx = AppState.analysisConditions.dx || 0.001;
                const dy = AppState.analysisConditions.dy || 0.001;
                xVals = Array.from({ length: cols }, (_, i) => i * dx * 1000);
                yVals = Array.from({ length: rows }, (_, i) => i * dy * 1000);
                zVals = azFlipped;
                xTitle = 'X [mm]';
                yTitle = 'Y [mm]';
                xMin = 0;
                xMax = (cols - 1) * dx * 1000;
                yMin = 0;
                yMax = (rows - 1) * dy * 1000;
            } else {
                xVals = Array.from({ length: cols }, (_, i) => i);
                yVals = Array.from({ length: rows }, (_, i) => i);
                zVals = azFlipped;
                xTitle = 'X [pixels]';
                yTitle = 'Y [pixels]';
                xMin = 0;
                xMax = cols - 1;
                yMin = 0;
                yMax = rows - 1;
            }

            const traces = [{
                z: zVals,
                x: xVals,
                y: yVals,
                type: 'contour',
                colorscale: 'Viridis',
                contours: { coloring: 'lines' },
                showscale: false,
                name: 'Az'
            }];

            let layout = {
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
                images: [{
                    source: edgeImgUrl,
                    xref: 'x',
                    yref: 'y',
                    x: xMin,
                    y: yMax,
                    sizex: xMax - xMin,
                    sizey: yMax - yMin,
                    sizing: 'stretch',
                    opacity: 1.0,
                    layer: 'below'
                }],
                dragmode: false
            };

            layout = restoreZoomState(containerId, layout);
            await Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
            setupZoomTracking(containerId);
        }
    } catch (error) {
        console.error('Field Lines + Edge render error:', error);
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
        let layout = {
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
        };

        // Restore saved zoom state if exists
        layout = restoreZoomState(containerId, layout);

        await Plotly.newPlot(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
        setupZoomTracking(containerId);
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
        let layout = {
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
        };

        // Restore saved zoom state if exists
        layout = restoreZoomState(containerId, layout);

        await Plotly.newPlot(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
        setupZoomTracking(containerId);
    } catch (error) {
        console.error('Step input image load error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">Error loading image</div>';
    }
}

async function renderCoarseningMask(containerId, step) {
    const resultPath = getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Get step-specific coarsening mask (binary: 0=coarsened, 255=active)
        const maskUrl = `/api/get-coarsening-mask?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;
        // Get step-specific input image
        const inputUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        container.innerHTML = '';
        const size = getContainerSize(container);

        // Load both images
        const [maskImg, inputImg] = await Promise.all([
            loadImage(maskUrl).catch(() => null),
            loadImage(inputUrl).catch(() => null)
        ]);

        if (!maskImg) {
            throw new Error('Coarsening mask not available');
        }

        const rows = maskImg.height;
        const cols = maskImg.width;

        // Compute product of input image and mask
        let resultImgUrl;
        if (inputImg && inputImg.width === cols && inputImg.height === rows) {
            // Create canvas for pixel manipulation
            const canvas = document.createElement('canvas');
            canvas.width = cols;
            canvas.height = rows;
            const ctx = canvas.getContext('2d');

            // Draw input image first
            ctx.drawImage(inputImg, 0, 0);
            const inputData = ctx.getImageData(0, 0, cols, rows);

            // Draw mask image to get mask data
            ctx.drawImage(maskImg, 0, 0);
            const maskData = ctx.getImageData(0, 0, cols, rows);

            // Compute product: output = input * (mask / 255)
            // mask is grayscale, so all channels have same value
            const outputData = ctx.createImageData(cols, rows);
            for (let i = 0; i < inputData.data.length; i += 4) {
                const maskValue = maskData.data[i]; // R channel (all channels same for grayscale)
                if (maskValue > 128) {
                    // Active cell: show input image pixel
                    outputData.data[i] = inputData.data[i];       // R
                    outputData.data[i + 1] = inputData.data[i + 1]; // G
                    outputData.data[i + 2] = inputData.data[i + 2]; // B
                    outputData.data[i + 3] = 255;                   // A
                } else {
                    // Coarsened cell: show black
                    outputData.data[i] = 0;
                    outputData.data[i + 1] = 0;
                    outputData.data[i + 2] = 0;
                    outputData.data[i + 3] = 255;
                }
            }
            ctx.putImageData(outputData, 0, 0);
            resultImgUrl = canvas.toDataURL('image/png');
        } else {
            // Fallback: just show mask if input image unavailable or size mismatch
            resultImgUrl = maskUrl;
        }

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
        let layout = {
            width: size.width,
            height: size.height,
            margin: { l: 35, r: 10, t: 25, b: 35 },
            title: {
                text: 'Coarsening Mask (black = coarsened)',
                font: { size: 12 }
            },
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
                    source: resultImgUrl,
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
        };

        // Restore saved zoom state if exists
        layout = restoreZoomState(containerId, layout);

        await Plotly.newPlot(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
        setupZoomTracking(containerId);
    } catch (error) {
        console.error('Coarsening mask load error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">Coarsening mask not available<br><small>(Adaptive mesh may not be enabled for this result)</small></div>';
    }
}

// Helper function to load an image and return a promise
function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load image: ' + url));
        img.src = url;
    });
}

// ===== Interactive Plot Functions =====

// State for interactive line profile
const lineProfileState = {};

async function renderLineProfile(containerId, step) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const resultPath = getCurrentResultPath();
    if (!resultPath) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">No result selected</div>';
        return;
    }

    // Initialize state for this container
    if (!lineProfileState[containerId]) {
        lineProfileState[containerId] = {
            startPoint: null,
            endPoint: null,
            selectingStart: true,
            displayField: 'az',  // 'az', 'mu', 'bn', 'bt', 'hn', 'ht'
            zoomRange: null  // { xRange: [min, max], yRange: [min, max] } - preserved across re-renders
        };
    }

    const state = lineProfileState[containerId];

    try {
        // Load Az and Mu data
        const azData = await loadCsvData('Az', step);
        const muData = await loadCsvData('Mu', step);

        if (!azData || azData.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">No data available</div>';
            return;
        }

        const dx = AppState.analysisConditions?.dx || 0.001;
        const dy = AppState.analysisConditions?.dy || 0.001;

        // Flip data for display (analysis y-up to image y-down)
        const azFlipped = flipVertical(azData);
        const muFlipped = flipVertical(muData);

        // Calculate B and H with components
        const { Bx, By, B, Hx, Hy, H } = calculateMagneticField(azFlipped, muFlipped, dx, dy);

        const rows = azFlipped.length;
        const cols = azFlipped[0].length;

        container.innerHTML = '';
        const size = getContainerSize(container);

        // Create compact control bar (use panel header for mode toggle)
        const controlBar = document.createElement('div');
        controlBar.style.cssText = 'display: flex; align-items: center; gap: 8px; padding: 4px 8px; background: #f8f8f8; border-bottom: 1px solid #eee; font-size: 11px;';
        controlBar.innerHTML = `
            <span style="color: #666;">Display:</span>
            <select id="${containerId}-field-select" style="padding: 2px 4px; font-size: 11px; border: 1px solid #ccc; border-radius: 3px;" onchange="setLineProfileField('${containerId}', this.value)">
                <option value="az" ${state.displayField === 'az' ? 'selected' : ''}>Az [Wb/m]</option>
                <option value="mu" ${state.displayField === 'mu' ? 'selected' : ''}>μ [H/m]</option>
                <option value="bn" ${state.displayField === 'bn' ? 'selected' : ''}>Bn [T]</option>
                <option value="bt" ${state.displayField === 'bt' ? 'selected' : ''}>Bt [T]</option>
                <option value="hn" ${state.displayField === 'hn' ? 'selected' : ''}>Hn [A/m]</option>
                <option value="ht" ${state.displayField === 'ht' ? 'selected' : ''}>Ht [A/m]</option>
            </select>
        `;
        container.appendChild(controlBar);

        // Create main content area
        const contentArea = document.createElement('div');
        contentArea.style.cssText = 'display: flex; height: calc(100% - 28px);';

        // Create two subplots: input image (left) and profile (right)
        const imageDiv = document.createElement('div');
        imageDiv.id = containerId + '-image';
        imageDiv.style.cssText = 'width: 50%; height: 100%;';

        const profileDiv = document.createElement('div');
        profileDiv.id = containerId + '-profile';
        profileDiv.style.cssText = 'width: 50%; height: 100%;';

        contentArea.appendChild(imageDiv);
        contentArea.appendChild(profileDiv);
        container.appendChild(contentArea);

        // Get input image URL
        const imgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        // Create X and Y coordinate arrays
        const xCoords = Array.from({ length: cols }, (_, i) => i * dx * 1000); // mm
        const yCoords = Array.from({ length: rows }, (_, j) => j * dy * 1000); // mm

        // Build traces for input image with line overlay
        const traces = [];

        // Add line trace if both points are selected
        if (state.startPoint && state.endPoint) {
            traces.push({
                x: [state.startPoint.x, state.endPoint.x],
                y: [state.startPoint.y, state.endPoint.y],
                mode: 'lines+markers',
                type: 'scatter',
                line: { color: 'lime', width: 3 },
                marker: { size: 10, color: ['green', 'red'] },
                name: 'Profile Line',
                showlegend: false
            });
        }

        // Add markers for selected points
        if (state.startPoint && !state.endPoint) {
            traces.push({
                x: [state.startPoint.x],
                y: [state.startPoint.y],
                mode: 'markers',
                type: 'scatter',
                marker: { size: 14, color: 'green', symbol: 'circle', line: { color: 'white', width: 2 } },
                showlegend: false
            });
        }

        const xMax = (cols - 1) * dx * 1000;
        const yMax = (rows - 1) * dy * 1000;

        // Status text for point selection (use Move mode in panel header to click)
        const statusText = state.selectingStart ? 'Move mode: click START' : 'Move mode: click END';
        const statusColor = state.selectingStart ? 'green' : 'red';

        // Use saved zoom range if available, otherwise use full range
        const xRangeToUse = state.zoomRange?.xRange || [0, xMax];
        const yRangeToUse = state.zoomRange?.yRange || [0, yMax];

        const imageLayout = {
            width: (size.width / 2) - 5,
            height: size.height - 30,
            margin: { l: 50, r: 10, t: 25, b: 40 },
            title: { text: statusText, font: { size: 10, color: statusColor } },
            xaxis: { title: 'X [mm]', range: xRangeToUse },
            yaxis: { title: 'Y [mm]', range: yRangeToUse },
            images: [{
                source: imgUrl,
                xref: 'x', yref: 'y',
                x: 0, y: yMax,
                sizex: xMax, sizey: yMax,
                sizing: 'stretch',
                opacity: 1.0,
                layer: 'below'
            }],
            dragmode: false  // Default to move/click mode for point selection
        };

        await Plotly.newPlot(imageDiv, traces, imageLayout, { responsive: true, displayModeBar: false });

        // Store reference in main container for panel header toggle
        container._lineProfileImageDiv = imageDiv;

        // Setup click handler using DOM event (works at any zoom level)
        // Remove existing handler if any
        if (imageDiv._clickHandler) {
            imageDiv.removeEventListener('click', imageDiv._clickHandler);
        }
        imageDiv._clickHandler = (evt) => {
            // Check if in move mode (dragmode: false)
            const currentDragmode = imageDiv.layout?.dragmode;
            if (currentDragmode && currentDragmode !== false) {
                return;  // Don't handle clicks in zoom/pan mode
            }

            // Get plot area bounding box
            const plotArea = imageDiv.querySelector('.nsewdrag');
            if (!plotArea) return;

            const rect = plotArea.getBoundingClientRect();
            const mouseX = evt.clientX - rect.left;
            const mouseY = evt.clientY - rect.top;

            // Check if click is within plot area
            if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) {
                return;
            }

            // Convert pixel coordinates to data coordinates using current axis ranges
            const xaxis = imageDiv._fullLayout?.xaxis;
            const yaxis = imageDiv._fullLayout?.yaxis;
            if (!xaxis || !yaxis) return;

            const xRange = xaxis.range;
            const yRange = yaxis.range;

            const clickedX = xRange[0] + (mouseX / rect.width) * (xRange[1] - xRange[0]);
            const clickedY = yRange[1] - (mouseY / rect.height) * (yRange[1] - yRange[0]);  // Y is inverted

            // Save current zoom range before re-rendering
            state.zoomRange = {
                xRange: [...xRange],
                yRange: [...yRange]
            };

            if (state.selectingStart) {
                state.startPoint = { x: clickedX, y: clickedY };
                state.selectingStart = false;
                state.endPoint = null;
            } else {
                state.endPoint = { x: clickedX, y: clickedY };
                state.selectingStart = true;
            }

            renderLineProfile(containerId, AppState.currentStep);
        };
        imageDiv.addEventListener('click', imageDiv._clickHandler);

        // Profile plot
        if (state.startPoint && state.endPoint) {
            // Calculate line direction for normal/tangent decomposition
            const lineVecX = state.endPoint.x - state.startPoint.x;
            const lineVecY = state.endPoint.y - state.startPoint.y;
            const lineLen = Math.sqrt(lineVecX * lineVecX + lineVecY * lineVecY);
            const tangentX = lineVecX / lineLen;  // Tangent unit vector
            const tangentY = lineVecY / lineLen;
            const normalX = -tangentY;  // Normal unit vector (perpendicular)
            const normalY = tangentX;

            // Extract profile data with vector decomposition
            const profileData = extractLineProfileEnhanced(
                state.startPoint, state.endPoint,
                azFlipped, muFlipped, Bx, By, Hx, Hy,
                normalX, normalY, tangentX, tangentY,
                dx, dy
            );

            // Select which data to display based on dropdown
            let yData, yLabel, yColor;
            switch (state.displayField) {
                case 'az':
                    yData = profileData.az;
                    yLabel = 'Az [Wb/m]';
                    yColor = '#1f77b4';
                    break;
                case 'mu':
                    yData = profileData.mu;
                    yLabel = 'μ [H/m]';
                    yColor = '#ff7f0e';
                    break;
                case 'bn':
                    yData = profileData.bn;
                    yLabel = 'Bn (normal) [T]';
                    yColor = '#2ca02c';
                    break;
                case 'bt':
                    yData = profileData.bt;
                    yLabel = 'Bt (tangent) [T]';
                    yColor = '#d62728';
                    break;
                case 'hn':
                    yData = profileData.hn;
                    yLabel = 'Hn (normal) [A/m]';
                    yColor = '#9467bd';
                    break;
                case 'ht':
                    yData = profileData.ht;
                    yLabel = 'Ht (tangent) [A/m]';
                    yColor = '#8c564b';
                    break;
                default:
                    yData = profileData.az;
                    yLabel = 'Az [Wb/m]';
                    yColor = '#1f77b4';
            }

            const profileTraces = [{
                x: profileData.distance,
                y: yData,
                name: yLabel,
                type: 'scatter',
                mode: 'lines',
                line: { color: yColor, width: 2 }
            }];

            const profileLayout = {
                width: (size.width / 2) - 5,
                height: size.height - 40,
                margin: { l: 60, r: 20, t: 30, b: 40 },
                title: { text: 'Line Profile', font: { size: 11 } },
                xaxis: { title: 'Distance [mm]' },
                yaxis: { title: yLabel },
                showlegend: false
            };

            await Plotly.newPlot(profileDiv, profileTraces, profileLayout, { responsive: true, displayModeBar: false });
        } else {
            // Show instructions
            profileDiv.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <p><strong>Line Profile</strong></p>
                    <p style="font-size: 0.85em; margin-top: 10px;">1. Set mode to "Select"</p>
                    <p style="font-size: 0.85em;">2. Click on image to set START point (green)</p>
                    <p style="font-size: 0.85em;">3. Click again to set END point (red)</p>
                    <p style="font-size: 0.85em; margin-top: 15px;">Choose field from dropdown:</p>
                    <p style="font-size: 0.8em; color: #888;">Az, μ, Bn/Bt (B normal/tangent), Hn/Ht (H normal/tangent)</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Line profile error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">Error loading data</div>';
    }
}

// Line profile control functions
function setLineProfileField(containerId, field) {
    if (lineProfileState[containerId]) {
        lineProfileState[containerId].displayField = field;
        renderLineProfile(containerId, AppState.currentStep);
    }
}

function resetLineProfilePoints(containerId) {
    if (lineProfileState[containerId]) {
        lineProfileState[containerId].startPoint = null;
        lineProfileState[containerId].endPoint = null;
        lineProfileState[containerId].selectingStart = true;
        renderLineProfile(containerId, AppState.currentStep);
    }
}

// Helper: Extract field values along a line with vector decomposition
function extractLineProfileEnhanced(start, end, azData, muData, Bx, By, Hx, Hy, normalX, normalY, tangentX, tangentY, dx, dy) {
    const result = { distance: [], az: [], mu: [], bn: [], bt: [], hn: [], ht: [] };

    const rows = azData.length;
    const cols = azData[0].length;

    // Convert mm to pixel indices
    const x0 = Math.round(start.x / (dx * 1000));
    const y0 = Math.round(start.y / (dy * 1000));
    const x1 = Math.round(end.x / (dx * 1000));
    const y1 = Math.round(end.y / (dy * 1000));

    // Bresenham's line algorithm
    const points = [];
    let x = x0, y = y0;
    const dx_line = Math.abs(x1 - x0);
    const dy_line = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx_line - dy_line;

    while (true) {
        if (x >= 0 && x < cols && y >= 0 && y < rows) {
            points.push({ x, y });
        }

        if (x === x1 && y === y1) break;

        const e2 = 2 * err;
        if (e2 > -dy_line) {
            err -= dy_line;
            x += sx;
        }
        if (e2 < dx_line) {
            err += dx_line;
            y += sy;
        }
    }

    // Extract values at each point
    let cumDist = 0;
    for (let i = 0; i < points.length; i++) {
        const pt = points[i];

        if (i > 0) {
            const prev = points[i - 1];
            const ddx = (pt.x - prev.x) * dx * 1000;
            const ddy = (pt.y - prev.y) * dy * 1000;
            cumDist += Math.sqrt(ddx * ddx + ddy * ddy);
        }

        result.distance.push(cumDist);
        result.az.push(azData[pt.y][pt.x]);
        result.mu.push(muData[pt.y][pt.x]);

        // Get B and H components at this point
        const bx = Bx[pt.y] ? Bx[pt.y][pt.x] || 0 : 0;
        const by = By[pt.y] ? By[pt.y][pt.x] || 0 : 0;
        const hx = Hx[pt.y] ? Hx[pt.y][pt.x] || 0 : 0;
        const hy = Hy[pt.y] ? Hy[pt.y][pt.x] || 0 : 0;

        // Decompose into normal and tangent components
        // Normal: projection onto normal vector
        // Tangent: projection onto tangent vector
        result.bn.push(bx * normalX + by * normalY);
        result.bt.push(bx * tangentX + by * tangentY);
        result.hn.push(hx * normalX + hy * normalY);
        result.ht.push(hx * tangentX + hy * tangentY);
    }

    return result;
}

// State for interactive flux linkage
const fluxLinkageState = {};

async function renderFluxLinkageInteractive(containerId, step) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const resultPath = getCurrentResultPath();
    if (!resultPath) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">No result selected</div>';
        return;
    }

    // Initialize state for this container
    if (!fluxLinkageState[containerId]) {
        fluxLinkageState[containerId] = {
            startPoint: null,
            endPoint: null,
            selectingStart: true,
            fluxValue: null,
            zoomRange: null  // { xRange: [min, max], yRange: [min, max] } - preserved across re-renders
        };
    }

    const state = fluxLinkageState[containerId];

    try {
        // Load Az data
        const azData = await loadCsvData('Az', step);

        if (!azData || azData.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">No data available</div>';
            return;
        }

        const dx = AppState.analysisConditions?.dx || 0.001;
        const dy = AppState.analysisConditions?.dy || 0.001;

        // Flip data for display
        const azFlipped = flipVertical(azData);
        const rows = azFlipped.length;
        const cols = azFlipped[0].length;

        container.innerHTML = '';
        const size = getContainerSize(container);

        // Create compact control bar (use panel header for mode toggle)
        const controlBar = document.createElement('div');
        controlBar.style.cssText = 'display: flex; align-items: center; gap: 8px; padding: 4px 8px; background: #f8f8f8; border-bottom: 1px solid #eee; font-size: 11px;';
        controlBar.innerHTML = `
            <span style="color: #666;">Φ = Az(end) - Az(start)</span>
        `;
        container.appendChild(controlBar);

        // Create plot area
        const plotDiv = document.createElement('div');
        plotDiv.id = containerId + '-plot';
        plotDiv.style.cssText = 'width: 100%; height: calc(100% - 28px);';
        container.appendChild(plotDiv);

        // Get input image URL
        const imgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        // Create X and Y coordinate arrays
        const xCoords = Array.from({ length: cols }, (_, i) => i * dx * 1000); // mm
        const yCoords = Array.from({ length: rows }, (_, j) => j * dy * 1000); // mm

        const xMax = (cols - 1) * dx * 1000;
        const yMax = (rows - 1) * dy * 1000;

        // Build traces for input image with point overlay
        const traces = [];

        // Add line between points if both selected
        if (state.startPoint && state.endPoint) {
            traces.push({
                x: [state.startPoint.x, state.endPoint.x],
                y: [state.startPoint.y, state.endPoint.y],
                mode: 'lines+markers',
                type: 'scatter',
                line: { color: 'yellow', width: 3 },
                marker: { size: 12, color: ['green', 'red'] },
                showlegend: false
            });
        } else {
            // Add individual markers
            if (state.startPoint) {
                traces.push({
                    x: [state.startPoint.x],
                    y: [state.startPoint.y],
                    mode: 'markers',
                    type: 'scatter',
                    marker: { size: 14, color: 'green', symbol: 'circle', line: { color: 'white', width: 2 } },
                    showlegend: false
                });
            }
        }

        // Calculate flux linkage if both points are set
        let fluxText = '';
        if (state.startPoint && state.endPoint) {
            // Convert mm to pixel indices
            const i0 = Math.round(state.startPoint.x / (dx * 1000));
            const j0 = Math.round(state.startPoint.y / (dy * 1000));
            const i1 = Math.round(state.endPoint.x / (dx * 1000));
            const j1 = Math.round(state.endPoint.y / (dy * 1000));

            // Clamp to valid range
            const i0c = Math.max(0, Math.min(cols - 1, i0));
            const j0c = Math.max(0, Math.min(rows - 1, j0));
            const i1c = Math.max(0, Math.min(cols - 1, i1));
            const j1c = Math.max(0, Math.min(rows - 1, j1));

            const azStart = azFlipped[j0c][i0c];
            const azEnd = azFlipped[j1c][i1c];
            const fluxLinkage = azEnd - azStart;

            state.fluxValue = fluxLinkage;

            fluxText = `Φ = ${fluxLinkage.toExponential(4)} Wb/m`;
        }

        // Status text (use Move mode in panel header to click)
        let statusText = '';
        let statusColor = '#666';
        if (state.startPoint && state.endPoint) {
            statusText = fluxText;
            statusColor = '#333';
        } else {
            statusText = state.selectingStart ? 'Move mode: click START' : 'Move mode: click END';
            statusColor = state.selectingStart ? 'green' : 'red';
        }

        // Use saved zoom range if available, otherwise use full range
        const xRangeToUse = state.zoomRange?.xRange || [0, xMax];
        const yRangeToUse = state.zoomRange?.yRange || [0, yMax];

        const layout = {
            width: size.width,
            height: size.height - 32,
            margin: { l: 50, r: 20, t: 30, b: 50 },
            title: {
                text: statusText,
                font: { size: 11, color: statusColor }
            },
            xaxis: { title: 'X [mm]', range: xRangeToUse },
            yaxis: { title: 'Y [mm]', range: yRangeToUse },
            showlegend: false,
            images: [{
                source: imgUrl,
                xref: 'x', yref: 'y',
                x: 0, y: yMax,
                sizex: xMax, sizey: yMax,
                sizing: 'stretch',
                opacity: 1.0,
                layer: 'below'
            }],
            dragmode: false,  // Default to move/click mode for point selection
            annotations: state.startPoint && state.endPoint ? [
                {
                    x: (state.startPoint.x + state.endPoint.x) / 2,
                    y: (state.startPoint.y + state.endPoint.y) / 2 + (yMax * 0.03),
                    text: fluxText,
                    showarrow: false,
                    font: { size: 14, color: 'white' },
                    bgcolor: 'rgba(0,0,0,0.7)',
                    borderpad: 4
                }
            ] : []
        };

        await Plotly.newPlot(plotDiv, traces, layout, { responsive: true, displayModeBar: false });

        // Store reference in main container for panel header toggle
        container._fluxLinkagePlotDiv = plotDiv;

        // Setup click handler using DOM event (works at any zoom level)
        // Remove existing handler if any
        if (plotDiv._clickHandler) {
            plotDiv.removeEventListener('click', plotDiv._clickHandler);
        }
        plotDiv._clickHandler = (evt) => {
            // Check if in move mode (dragmode: false)
            const currentDragmode = plotDiv.layout?.dragmode;
            if (currentDragmode && currentDragmode !== false) {
                return;  // Don't handle clicks in zoom/pan mode
            }

            // Get plot area bounding box
            const plotArea = plotDiv.querySelector('.nsewdrag');
            if (!plotArea) return;

            const rect = plotArea.getBoundingClientRect();
            const mouseX = evt.clientX - rect.left;
            const mouseY = evt.clientY - rect.top;

            // Check if click is within plot area
            if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) {
                return;
            }

            // Convert pixel coordinates to data coordinates using current axis ranges
            const xaxis = plotDiv._fullLayout?.xaxis;
            const yaxis = plotDiv._fullLayout?.yaxis;
            if (!xaxis || !yaxis) return;

            const xRange = xaxis.range;
            const yRange = yaxis.range;

            const clickedX = xRange[0] + (mouseX / rect.width) * (xRange[1] - xRange[0]);
            const clickedY = yRange[1] - (mouseY / rect.height) * (yRange[1] - yRange[0]);  // Y is inverted

            // Save current zoom range before re-rendering
            state.zoomRange = {
                xRange: [...xRange],
                yRange: [...yRange]
            };

            if (state.selectingStart) {
                state.startPoint = { x: clickedX, y: clickedY };
                state.selectingStart = false;
                state.endPoint = null;
                state.fluxValue = null;
            } else {
                state.endPoint = { x: clickedX, y: clickedY };
                state.selectingStart = true;
            }

            // Re-render to update
            renderFluxLinkageInteractive(containerId, AppState.currentStep);
        };
        plotDiv.addEventListener('click', plotDiv._clickHandler);

    } catch (error) {
        console.error('Flux linkage interactive error:', error);
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: red;">Error loading data</div>';
    }
}

// Flux linkage control function
function resetFluxLinkagePoints(containerId) {
    if (fluxLinkageState[containerId]) {
        fluxLinkageState[containerId].startPoint = null;
        fluxLinkageState[containerId].endPoint = null;
        fluxLinkageState[containerId].selectingStart = true;
        fluxLinkageState[containerId].fluxValue = null;
        renderFluxLinkageInteractive(containerId, AppState.currentStep);
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
        let systemTotalEnergy = 0;  // System total energy from _SYSTEM_TOTAL row
        let dataRowCount = 0;

        for (let i = headerIdx + 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('#') || line.length === 0) continue;

            const values = line.split(',');
            if (values.length > Math.max(forceXIdx, forceYIdx, torqueIdx)) {
                const materialName = materialIdx !== -1 ? values[materialIdx].trim() : `Material_${dataRowCount}`;

                // Check for special _SYSTEM_TOTAL row
                if (materialName === '_SYSTEM_TOTAL') {
                    systemTotalEnergy = energyIdx !== -1 ? (parseFloat(values[energyIdx]) || 0) : 0;
                    continue;  // Don't add to materials list
                }

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
            materials: materials,
            system_total_energy: systemTotalEnergy
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

        // Full model multiplier for polar coordinates
        const forceMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        // Trace per material
        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        forceData.push(mat.force_x * forceMultiplier);
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

        const yaxisTitle = forceMultiplier > 1
            ? `Force X [N/m] (×${forceMultiplier})`
            : 'Force X [N/m]';

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: yaxisTitle, range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
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

        // Full model multiplier for polar coordinates
        const forceMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        forceData.push(mat.force_y * forceMultiplier);
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

        const yaxisTitle = forceMultiplier > 1
            ? `Force Y [N/m] (×${forceMultiplier})`
            : 'Force Y [N/m]';

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: yaxisTitle, range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
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

        // Full model multiplier for polar coordinates
        const torqueMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        materialNames.forEach(matName => {
            const torqueData = [];
            let matColor = null;

            for (let i = 0; i < AppState.totalSteps; i++) {
                const stepData = allStepsData[i];
                if (stepData && stepData.materials) {
                    const mat = stepData.materials.find(m => m.name === matName);
                    if (mat) {
                        torqueData.push(mat.torque * torqueMultiplier);
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

        const yaxisTitle = torqueMultiplier > 1
            ? `Torque [Nm/m] (×${torqueMultiplier})`
            : 'Torque [Nm/m]';

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: yaxisTitle, range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
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

        // Full model multiplier for polar coordinates (energy is also multiplied for full model)
        const energyMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        // Per-material energy traces
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

        // Add system total energy as black line
        const systemEnergyData = [];
        let hasSystemEnergy = false;
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepData = allStepsData[i];
            if (stepData && stepData.system_total_energy !== undefined) {
                systemEnergyData.push(stepData.system_total_energy * energyMultiplier);
                hasSystemEnergy = true;
            } else {
                systemEnergyData.push(0);
            }
        }

        if (hasSystemEnergy) {
            traces.push({
                x: xSteps,
                y: systemEnergyData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'System Total',
                line: { color: '#000000', width: 2 },
                marker: { color: '#000000', size: getMarkerSizes(6, 14) }
            });
        }

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
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
    } catch (error) {
        console.error('Energy time plot error:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`;
        }
    }
}

async function renderSystemEnergyTime(containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < AppState.totalSteps; i++) {
            const data = await loadForceData(i + 1);
            allStepsData.push(data || null);
            if (data && data.system_total_energy !== undefined) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No System Energy data available</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        // Full model multiplier for polar coordinates (energy is also multiplied for full model)
        const energyMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        // Extract system total energy for each step
        const energyData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepData = allStepsData[i];
            if (stepData && stepData.system_total_energy !== undefined) {
                energyData.push(stepData.system_total_energy * energyMultiplier);
            } else {
                energyData.push(0);
            }
        }

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: AppState.totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === AppState.currentStep)) ? highlightSize : baseSize;
            });
        };

        const traces = [{
            x: xSteps,
            y: energyData,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'System Total',
            line: { color: '#1f77b4', width: 2 },
            marker: { color: '#1f77b4', size: getMarkerSizes(6, 14) }
        }];

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 55, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: 'System Energy [J/m]' },
            showlegend: false,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
    } catch (error) {
        console.error('System energy time plot error:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`;
        }
    }
}

async function renderVirtualWork(containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < AppState.totalSteps; i++) {
            const data = await loadForceData(i + 1);
            allStepsData.push(data || null);
            if (data && data.system_total_energy !== undefined) hasData = true;
        }

        if (!hasData) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No System Energy data available</div>';
            return;
        }

        container.innerHTML = '';
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: AppState.totalSteps }, (_, k) => k + 1);

        // Full model multiplier for polar coordinates
        const energyMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        // Extract system total energy for each step
        const energyData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            const stepData = allStepsData[i];
            if (stepData && stepData.system_total_energy !== undefined) {
                energyData.push(stepData.system_total_energy * energyMultiplier);
            } else {
                energyData.push(0);
            }
        }

        // Determine displacement per step and units based on coordinate system and slide direction
        let displacementPerStep = 1;  // default
        let yAxisTitle = '+dW/dx [N/m]';
        let isAngular = false;

        if (AppState.analysisConditions) {
            const transient = AppState.analysisConditions.transient;
            const slidePixelsPerStep = transient?.slide_pixels_per_step || 1;
            const slideDirection = transient?.slide_direction || 'horizontal';
            const coordSystem = AppState.analysisConditions.coordinate_system || 'cartesian';

            if (coordSystem === 'polar') {
                const polar = AppState.analysisConditions.polar;
                const rOrientation = polar?.r_orientation || 'horizontal';
                const dr = AppState.analysisConditions.dr || 0.001;
                const dtheta = AppState.analysisConditions.dtheta || 0.01;

                // Determine if sliding is in theta direction (angular) or r direction (radial)
                // r_orientation = 'horizontal': r along x-axis, theta along y-axis
                //   slide_direction = 'vertical' → theta direction → torque
                //   slide_direction = 'horizontal' → r direction → force
                // r_orientation = 'vertical': r along y-axis, theta along x-axis
                //   slide_direction = 'vertical' → r direction → force
                //   slide_direction = 'horizontal' → theta direction → torque

                if ((rOrientation === 'horizontal' && slideDirection === 'vertical') ||
                    (rOrientation === 'vertical' && slideDirection === 'horizontal')) {
                    // Theta direction sliding → torque
                    displacementPerStep = slidePixelsPerStep * dtheta;  // [rad]
                    yAxisTitle = '+dW/dθ (Torque) [N·m/m]';
                    isAngular = true;
                } else {
                    // R direction sliding → force
                    displacementPerStep = slidePixelsPerStep * dr;  // [m]
                    yAxisTitle = '+dW/dr [N/m]';
                }
            } else {
                // Cartesian coordinates
                const dx = AppState.analysisConditions.dx || 0.001;
                const dy = AppState.analysisConditions.dy || 0.001;

                if (slideDirection === 'horizontal') {
                    displacementPerStep = slidePixelsPerStep * dx;  // [m]
                    yAxisTitle = '+dW/dx [N/m]';
                } else {
                    displacementPerStep = slidePixelsPerStep * dy;  // [m]
                    yAxisTitle = '+dW/dy [N/m]';
                }
            }
        }

        // Calculate virtual work: F = +dW/dx for constant-current systems (Jz specified)
        // Note: For constant-flux systems, F = -dW/dx. OpenMagFDM uses constant current.
        const virtualWorkData = [];
        for (let i = 0; i < AppState.totalSteps; i++) {
            if (i === 0) {
                // Forward difference for first point
                if (AppState.totalSteps > 1) {
                    const dW = energyData[1] - energyData[0];
                    virtualWorkData.push(dW / displacementPerStep);
                } else {
                    virtualWorkData.push(0);
                }
            } else if (i === AppState.totalSteps - 1) {
                // Backward difference for last point
                const dW = energyData[i] - energyData[i - 1];
                virtualWorkData.push(dW / displacementPerStep);
            } else {
                // Central difference for interior points
                const dW = energyData[i + 1] - energyData[i - 1];
                virtualWorkData.push(dW / (2 * displacementPerStep));
            }
        }

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: AppState.totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === AppState.currentStep)) ? highlightSize : baseSize;
            });
        };

        const traces = [{
            x: xSteps,
            y: virtualWorkData,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Virtual Work',
            line: { color: '#d62728', width: 2 },
            marker: { color: '#d62728', size: getMarkerSizes(6, 14) }
        }];

        await Plotly.newPlot(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 55, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, AppState.totalSteps] },
            yaxis: { title: yAxisTitle },
            showlegend: false,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
    } catch (error) {
        console.error('Virtual work plot error:', error);
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
            const r_orientation = AppState.analysisConditions.polar?.r_orientation || 'horizontal';

            // Determine nr and ntheta based on r_orientation
            let nr, ntheta;
            if (r_orientation === 'horizontal') {
                // data[theta][r]: rows = ntheta, cols = nr
                nr = cols;
                ntheta = rows;
            } else {
                // data[r][theta]: rows = nr, cols = ntheta
                nr = rows;
                ntheta = cols;
            }

            // r: mm (from 0), theta: radians
            const rVals = Array.from({ length: nr }, (_, i) => i * dr * 1000);
            const thetaVals = Array.from({ length: ntheta }, (_, i) => theta_start + i * dtheta);

            if (r_orientation === 'horizontal') {
                trace.x = rVals;
                trace.y = thetaVals;
                xaxis = { title: 'r - r_start [mm]' };
                yaxis = { title: 'θ [rad]' };
            } else {
                trace.x = thetaVals;
                trace.y = rVals;
                xaxis = { title: 'θ [rad]' };
                yaxis = { title: 'r - r_start [mm]' };
            }
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

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar });
}

// ===== Plot Zoom State Management =====
/**
 * Save current zoom state of a plot
 * @param {string} containerId - Plot container ID
 * @param {object} layout - Plotly layout object with xaxis and yaxis
 */
function saveZoomState(containerId, layout) {
    if (layout && layout.xaxis && layout.yaxis) {
        AppState.plotZoomStates[containerId] = {
            xaxis: { range: layout.xaxis.range },
            yaxis: { range: layout.yaxis.range }
        };
    }
}

/**
 * Restore saved zoom state to a plot layout
 * @param {string} containerId - Plot container ID
 * @param {object} layout - Plotly layout object to modify
 * @returns {object} - Modified layout with restored zoom state
 */
function restoreZoomState(containerId, layout) {
    const savedState = AppState.plotZoomStates[containerId];
    if (savedState && savedState.xaxis && savedState.yaxis) {
        if (!layout.xaxis) layout.xaxis = {};
        if (!layout.yaxis) layout.yaxis = {};
        layout.xaxis.range = savedState.xaxis.range;
        layout.yaxis.range = savedState.yaxis.range;
    }
    return layout;
}

/**
 * Setup zoom state tracking for a Plotly container
 * @param {string} containerId - Plot container ID
 */
function setupZoomTracking(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.on('plotly_relayout', (eventData) => {
        // Save zoom state when user zooms/pans
        if (eventData['xaxis.range[0]'] !== undefined || eventData['xaxis.range'] !== undefined) {
            const layout = container.layout;
            saveZoomState(containerId, layout);
        }
    });
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

    // Separate title into graph title and colorbar title (unit)
    // Expected format: "Physical Quantity [Unit]" or just "Title"
    let graphTitle = title;
    let colorbarTitle = '';
    const unitMatch = title.match(/^(.+?)\s+(\[.+?\])(.*)$/);
    if (unitMatch) {
        graphTitle = unitMatch[1] + (unitMatch[3] || ''); // Physical quantity + any suffix
        colorbarTitle = unitMatch[2]; // [Unit]
    }

    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {
            title: colorbarTitle,
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

        if (coordSys === 'polar' && AppState.polarCartesianTransform) {
            // Apply polar to cartesian transformation
            const transformedData = transformPolarToCartesian(
                data,
                AppState.analysisConditions,
                AppState.polarFullModel
            );

            trace.x = transformedData.x;
            trace.y = transformedData.y;
            trace.z = transformedData.z;

            const r_o = AppState.analysisConditions.polar?.r_end || AppState.analysisConditions.r_o || 1;
            xaxis = {
                title: 'X [mm]',
                range: [-r_o * 1000, r_o * 1000],
                ...(AppState.polarFullModel && { scaleanchor: 'y', scaleratio: 1 })
            };
            yaxis = {
                title: 'Y [mm]',
                range: [-r_o * 1000, r_o * 1000]
            };
        } else if (coordSys === 'polar') {
            // Original polar view (r vs theta)
            const theta_start = AppState.analysisConditions.theta_start || 0;
            const dr = AppState.analysisConditions.dr || 0.001;
            const dtheta = AppState.analysisConditions.dtheta || 0.001;
            const r_orientation = AppState.analysisConditions.polar?.r_orientation || 'horizontal';

            // Determine nr and ntheta based on r_orientation
            let nr, ntheta;
            if (r_orientation === 'horizontal') {
                // data[theta][r]: rows = ntheta, cols = nr
                nr = cols;
                ntheta = rows;
            } else {
                // data[r][theta]: rows = nr, cols = ntheta
                nr = rows;
                ntheta = cols;
            }

            // r: mm (from 0), theta: radians
            const rVals = Array.from({ length: nr }, (_, i) => i * dr * 1000);
            const thetaVals = Array.from({ length: ntheta }, (_, i) => theta_start + i * dtheta);

            if (r_orientation === 'horizontal') {
                trace.x = rVals;
                trace.y = thetaVals;
                xaxis = { title: 'r - r_start [mm]' };
                yaxis = { title: 'θ [rad]' };
            } else {
                trace.x = thetaVals;
                trace.y = rVals;
                xaxis = { title: 'θ [rad]' };
                yaxis = { title: 'r - r_start [mm]' };
            }
        } else {
            // Cartesian coordinates
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

    // Update title for full model
    if (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1) {
        graphTitle += ` (Full Model ×${AppState.polarFullModelMultiplier})`;
    }

    let layout = {
        width: size.width,
        height: size.height,
        title: graphTitle,
        xaxis: xaxis,
        yaxis: yaxis,
        margin: { t: 40, r: colorbarThickness + 15, b: 40, l: 60 },
        dragmode: false
    };

    // Restore saved zoom state if exists
    layout = restoreZoomState(elementId, layout);

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }).then(() => {
        setupZoomTracking(elementId);
    });
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

    let layout = {
        title: title,
        xaxis: { title: 'Index' },
        yaxis: { title: 'Force [N/m]' },
        margin: { t: 40, r: 20, b: 40, l: 60 },
        dragmode: false
    };

    // Restore saved zoom state if exists
    layout = restoreZoomState(elementId, layout);

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }).then(() => {
        setupZoomTracking(elementId);
    });
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

// ===== File Manager Functions =====

/**
 * Initialize file manager tab
 */
function initializeFileManager() {
    // Update user ID display
    const userIdDisplay = document.getElementById('fileManagerUserId');
    if (userIdDisplay) {
        userIdDisplay.textContent = AppState.userId;
    }

    // Load outputs list
    refreshOutputsList();
}

/**
 * Refresh the list of output folders
 */
async function refreshOutputsList() {
    const outputsList = document.getElementById('outputsList');
    if (!outputsList) return;

    try {
        outputsList.innerHTML = '<p style="color: #666;">Loading...</p>';

        const response = await fetch(`/api/user-outputs?userId=${AppState.userId}`);

        if (!response.ok) {
            throw new Error('Failed to load outputs');
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Failed to load outputs');
        }

        if (result.outputs.length === 0) {
            outputsList.innerHTML = '<p style="color: #666;">No analysis results found.</p>';
            return;
        }

        // Build output list HTML
        let html = '';
        for (const output of result.outputs) {
            const date = new Date(output.created).toLocaleString('ja-JP');
            // Escape output name for safe HTML embedding
            const escapedName = output.name.replace(/'/g, "\\'");

            html += `
                <div class="output-item" onclick="selectOutput('${escapedName}', event)">
                    <input type="checkbox" class="output-checkbox"
                           data-folder="${escapedName}"
                           onclick="event.stopPropagation(); updateSelectedCount()">
                    <div class="output-info">
                        <div class="output-name">${output.name}</div>
                        <div class="output-details">
                            Created: ${date} | Size: ${output.sizeFormatted} | Steps: ${output.steps}
                        </div>
                    </div>
                    <div class="output-actions">
                        <button class="btn-secondary btn-small" onclick="event.stopPropagation(); renameOutput('${escapedName}')">Rename</button>
                        <button class="btn-secondary btn-small" onclick="event.stopPropagation(); editDescription('${escapedName}')">Memo</button>
                        <button class="btn-delete btn-small" onclick="event.stopPropagation(); deleteOutput('${escapedName}')">Delete</button>
                    </div>
                </div>
            `;
        }

        outputsList.innerHTML = html;

        // Reset selection count
        updateSelectedCount();

    } catch (error) {
        console.error('Error loading outputs:', error);
        outputsList.innerHTML = `<p style="color: #dc3545;">Error: ${error.message}</p>`;
    }
}

/**
 * Delete an output folder
 * @param {string} folderName - Name of the folder to delete
 */
async function deleteOutput(folderName) {
    // Confirmation dialog
    const confirmMsg = `Delete output folder: ${folderName}?\n\nThis action cannot be undone.`;
    if (!confirm(confirmMsg)) {
        return;
    }

    try {
        const response = await fetch(`/api/user-outputs/${folderName}?userId=${AppState.userId}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Failed to delete output folder');
        }

        // Show success message
        alert('Output folder deleted successfully');

        // Refresh the list
        refreshOutputsList();

    } catch (error) {
        console.error('Error deleting output:', error);
        alert(`Error: ${error.message}`);
    }
}

/**
 * Select an output and show preview
 * @param {string} folderName - Folder name
 * @param {Event} event - Click event
 */
async function selectOutput(folderName, event) {
    // Skip if clicking checkbox
    if (event.target.type === 'checkbox') {
        return;
    }

    // Highlight selected item
    document.querySelectorAll('.output-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');

    // Show preview panel
    await showOutputPreview(folderName);
}

/**
 * Show output preview in right panel
 * @param {string} folderName - Folder name
 */
async function showOutputPreview(folderName) {
    const previewPanel = document.getElementById('filePreviewPanel');
    const descDiv = document.getElementById('previewDescription');

    if (!previewPanel || !descDiv) return;

    previewPanel.style.display = 'block';

    try {
        // Fetch description
        const descResponse = await fetch(`/api/user-outputs/${encodeURIComponent(folderName)}/description?userId=${AppState.userId}`);
        const descResult = await descResponse.json();

        if (descResult.success && descResult.description) {
            // Escape HTML and convert newlines to <br>
            const escapedDesc = descResult.description
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\n/g, '<br>');
            descDiv.innerHTML = `<strong>Description:</strong><br>${escapedDesc}`;
        } else {
            descDiv.innerHTML = '<em>No description</em>';
        }

        // Render preview plots (similar to Run & Preview)
        const resultPath = `outputs/${AppState.userId}/${folderName}`;
        await renderFileManagerPreview(resultPath);

    } catch (error) {
        console.error('Error loading preview:', error);
        descDiv.innerHTML = `<em style="color: #dc3545;">Error loading description</em>`;
    }
}

/**
 * Render preview plots for File Manager
 * @param {string} resultPath - Result path
 */
async function renderFileManagerPreview(resultPath) {
    const plot1 = document.getElementById('filePreviewPlot1');
    const plot2 = document.getElementById('filePreviewPlot2');

    if (!plot1 || !plot2) return;

    try {
        // Load step 1 data for preview
        const step = 1;

        // Plot 1: Az heatmap
        const azData = await loadCsvData('Az', step, resultPath);
        const azFlipped = flipVertical(azData);
        plot1.innerHTML = '';
        await plotHeatmapInDiv(plot1, azFlipped, 'Az [Wb/m]', true, false);

        // Plot 2: B magnitude
        const bData = await loadCsvData('B_magnitude', step, resultPath);
        const bFlipped = flipVertical(bData);
        plot2.innerHTML = '';
        await plotHeatmapInDiv(plot2, bFlipped, '|B| [T]', true, false);

    } catch (error) {
        console.error('Error rendering preview:', error);
        plot1.innerHTML = '<div style="padding: 20px; text-align: center;">Preview not available</div>';
        plot2.innerHTML = '';
    }
}

/**
 * Plot heatmap in a specific div
 * @param {HTMLElement} container - Container element
 * @param {Array} data - Data array
 * @param {string} title - Plot title
 * @param {boolean} usePhysicalAxes - Use physical axes
 * @param {boolean} useHarmonicMean - Use harmonic mean for interpolation
 */
async function plotHeatmapInDiv(container, data, title, usePhysicalAxes, useHarmonicMean) {
    // Use a temporary unique ID
    const tempId = 'temp_' + Math.random().toString(36).substr(2, 9);
    container.id = tempId;
    await plotHeatmap(tempId, data, title, usePhysicalAxes, useHarmonicMean);
}

/**
 * Update selected count display
 */
function updateSelectedCount() {
    const checkboxes = document.querySelectorAll('.output-checkbox:checked');
    const count = checkboxes.length;
    const countSpan = document.getElementById('selectedCount');
    const deleteBtn = document.getElementById('bulkDeleteBtn');

    if (countSpan) {
        countSpan.textContent = count;
    }

    if (deleteBtn) {
        deleteBtn.disabled = count === 0;
    }
}

/**
 * Toggle select all checkboxes
 */
function toggleSelectAll() {
    const selectAll = document.getElementById('selectAllOutputs');
    const checkboxes = document.querySelectorAll('.output-checkbox');

    if (!selectAll) return;

    checkboxes.forEach(cb => {
        cb.checked = selectAll.checked;
    });

    updateSelectedCount();
}

/**
 * Delete multiple selected outputs
 */
async function bulkDeleteOutputs() {
    const checkboxes = document.querySelectorAll('.output-checkbox:checked');
    const folderNames = Array.from(checkboxes).map(cb => cb.dataset.folder);

    if (folderNames.length === 0) return;

    const confirmMsg = `Delete ${folderNames.length} folder(s)?\n\nThis action cannot be undone.`;
    if (!confirm(confirmMsg)) {
        return;
    }

    try {
        const response = await fetch('/api/user-outputs/bulk', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, folderNames })
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Bulk delete failed');
        }

        // Show result
        const failed = result.results.filter(r => !r.success).length;
        if (failed > 0) {
            alert(`Deleted ${result.results.length - failed} folders.\n${failed} folders failed.`);
        } else {
            alert(`Successfully deleted ${result.results.length} folders.`);
        }

        // Refresh list
        refreshOutputsList();

        // Hide preview panel
        const previewPanel = document.getElementById('filePreviewPanel');
        if (previewPanel) {
            previewPanel.style.display = 'none';
        }

    } catch (error) {
        console.error('Error in bulk delete:', error);
        alert(`Error: ${error.message}`);
    }
}

/**
 * Rename an output folder
 * @param {string} folderName - Current folder name
 */
async function renameOutput(folderName) {
    const newName = prompt('Enter new folder name:', folderName);
    if (!newName || newName === folderName) return;

    try {
        const response = await fetch(`/api/user-outputs/${encodeURIComponent(folderName)}/rename`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, newName })
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Failed to rename');
        }

        alert('Folder renamed successfully');
        refreshOutputsList();
        refreshResultsList();  // Update Run & Preview tab list

    } catch (error) {
        console.error('Error renaming:', error);
        alert(`Error: ${error.message}`);
    }
}

/**
 * Edit description for an output folder
 * @param {string} folderName - Folder name
 */
async function editDescription(folderName) {
    try {
        // Fetch current description
        const response = await fetch(`/api/user-outputs/${encodeURIComponent(folderName)}/description?userId=${AppState.userId}`);
        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error('Failed to load description');
        }

        const newDesc = prompt('Enter description/memo:', result.description || '');
        if (newDesc === null) return;  // Cancelled

        // Update description
        const updateResponse = await fetch(`/api/user-outputs/${encodeURIComponent(folderName)}/description`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, description: newDesc })
        });

        const updateResult = await updateResponse.json();

        if (!updateResponse.ok || !updateResult.success) {
            throw new Error('Failed to update description');
        }

        // Update preview if currently showing this folder
        const activeItem = document.querySelector('.output-item.active');
        if (activeItem) {
            const checkbox = activeItem.querySelector('.output-checkbox');
            if (checkbox && checkbox.dataset.folder === folderName) {
                await showOutputPreview(folderName);
            }
        }

    } catch (error) {
        console.error('Error editing description:', error);
        alert(`Error: ${error.message}`);
    }
}

// ===== Polar Coordinate Transform Functions =====

/**
 * Update polar controls visibility and info based on loaded conditions
 */
function updatePolarControls() {
    const polarControls = document.getElementById('polarControls');
    const polarInfo = document.getElementById('polarInfo');
    const fullModelMultiplier = document.getElementById('fullModelMultiplier');

    if (!AppState.analysisConditions) {
        polarControls.style.display = 'none';
        return;
    }

    const isPolar = AppState.analysisConditions.coordinate_system === 'polar';
    AppState.isPolarCoordinates = isPolar;

    if (!isPolar) {
        polarControls.style.display = 'none';
        return;
    }

    // Show polar controls
    polarControls.style.display = 'block';

    // Calculate full model multiplier from theta_range
    // Check both locations: top-level theta_range and polar.theta_range
    const thetaRange = AppState.analysisConditions.polar?.theta_range
        || AppState.analysisConditions.theta_range
        || 0;
    const multiplier = calculateFullModelMultiplier(thetaRange);
    AppState.polarFullModelMultiplier = multiplier;

    if (multiplier > 1) {
        fullModelMultiplier.textContent = multiplier;
        polarInfo.textContent = `θ range: ${(thetaRange * 180 / Math.PI).toFixed(1)}° ≈ 2π/${multiplier} → Full model available`;
    } else {
        fullModelMultiplier.textContent = 'N';
        polarInfo.textContent = `θ range: ${(thetaRange * 180 / Math.PI).toFixed(1)}° → Full model not available`;
        // Disable full model checkbox if not available
        document.getElementById('polarFullModel').disabled = (multiplier === 1);
    }
}

/**
 * Calculate full model multiplier N from theta_range
 * Returns N if theta_range ≈ 2π/N, otherwise returns 1
 * @param {number} thetaRange - Theta range in radians
 * @returns {number} - Multiplier N (1 if not applicable)
 */
function calculateFullModelMultiplier(thetaRange) {
    const TWO_PI = 2 * Math.PI;
    const TOLERANCE = 0.02; // 2% tolerance

    // Check for common divisors: 2, 3, 4, 5, 6, 8, 10, 12, 16, 18, 20, 24, 30, 36, 40, 60, 72, 120
    const commonDivisors = [2, 3, 4, 5, 6, 8, 10, 12, 16, 18, 20, 24, 30, 36, 40, 60, 72, 120];

    for (const N of commonDivisors) {
        const expectedAngle = TWO_PI / N;
        const relativeError = Math.abs(thetaRange - expectedAngle) / expectedAngle;

        if (relativeError < TOLERANCE) {
            return N;
        }
    }

    return 1; // Not a clean divisor of 2π
}

/**
 * Toggle cartesian transform for polar coordinates
 */
function toggleCartesianTransform() {
    const checkbox = document.getElementById('polarCartesianTransform');
    AppState.polarCartesianTransform = checkbox.checked;

    // Refresh all plots including dashboard
    refreshAllPlots();
    updateAllPlots();
}

/**
 * Toggle full model expansion for polar coordinates
 */
function toggleFullModel() {
    const checkbox = document.getElementById('polarFullModel');
    AppState.polarFullModel = checkbox.checked;

    // If full model is enabled, cartesian transform should also be enabled
    if (AppState.polarFullModel && !AppState.polarCartesianTransform) {
        document.getElementById('polarCartesianTransform').checked = true;
        AppState.polarCartesianTransform = true;
    }

    // Refresh all plots including dashboard
    refreshAllPlots();
    updateAllPlots();
}

/**
 * Transform polar coordinate data to cartesian (arc or full donut)
 * @param {Array<Array<number>>} polarData - 2D array in polar coordinates
 * @param {object} conditions - Analysis conditions containing r_i, r_o, theta_range, r_orientation, boundary_conditions
 * @param {boolean} fullModel - If true, replicate to full 360 degrees
 * @returns {object} - {x: Array, y: Array, z: Array} for Plotly heatmap
 */
function transformPolarToCartesian(polarData, conditions, fullModel = false) {
    // Extract polar parameters (check both nested and top-level locations)
    const r_i = conditions.polar?.r_start || conditions.r_i || 0;
    const r_o = conditions.polar?.r_end || conditions.r_o || 1;
    const thetaRange = conditions.polar?.theta_range || conditions.theta_range || 0;
    const r_orientation = conditions.polar?.r_orientation || 'horizontal';

    // Determine theta boundary conditions
    const bcTheta = conditions.boundary_conditions || {};
    const thetaMinBC = bcTheta.theta_min || {};
    const thetaMaxBC = bcTheta.theta_max || {};
    const thetaPeriodic = (thetaMinBC.type === 'periodic' && thetaMaxBC.type === 'periodic');
    const thetaAntiperiodic = thetaPeriodic &&
        ((thetaMinBC.value !== undefined && thetaMinBC.value < 0) ||
         (thetaMaxBC.value !== undefined && thetaMaxBC.value < 0));

    // Determine nr and ntheta based on r_orientation
    let ntheta, nr;
    if (r_orientation === 'horizontal') {
        // polarData[theta_idx][r_idx]: rows = ntheta, cols = nr
        ntheta = polarData.length;
        nr = polarData[0].length;
    } else {
        // polarData[r_idx][theta_idx]: rows = nr, cols = ntheta
        nr = polarData.length;
        ntheta = polarData[0].length;
    }

    // Determine number of repetitions
    const N = fullModel ? AppState.polarFullModelMultiplier : 1;

    // Create output grid
    const resolution = Math.max(nr, ntheta) * 2; // Higher resolution for interpolation
    const gridSize = 2 * r_o;
    const dx = gridSize / resolution;
    const dy = gridSize / resolution;

    // Create 1D arrays for x and y coordinates (in mm)
    const x = Array.from({ length: resolution }, (_, j) => (-r_o + j * dx) * 1000);
    const y = Array.from({ length: resolution }, (_, i) => (-r_o + i * dy) * 1000);
    const z = [];

    for (let i = 0; i < resolution; i++) {
        const row_z = [];
        const py = -r_o + i * dy; // in meters

        for (let j = 0; j < resolution; j++) {
            const px = -r_o + j * dx; // in meters

            // Convert to polar
            const r = Math.sqrt(px * px + py * py);
            let theta = Math.atan2(py, px);
            if (theta < 0) theta += 2 * Math.PI;

            // Check if within valid range
            if (r < r_i || r > r_o) {
                row_z.push(null); // Outside domain
                continue;
            }

            // Map theta to sector
            if (fullModel) {
                // Map theta to original sector [0, thetaRange]
                const sectorAngle = 2 * Math.PI / N;
                theta = theta % sectorAngle;
            } else {
                // Single arc
                if (theta > thetaRange) {
                    row_z.push(null);
                    continue;
                }
            }

            // Interpolate from polar data
            const r_idx = (r - r_i) / (r_o - r_i) * (nr - 1);
            const theta_idx = theta / thetaRange * (ntheta - 1);

            // Bilinear interpolation with periodic/anti-periodic BC support
            const value = bilinearInterpolate(polarData, theta_idx, r_idx, thetaPeriodic, thetaAntiperiodic, r_orientation);
            row_z.push(value);
        }

        z.push(row_z);
    }

    return { x, y, z };
}

/**
 * Bilinear interpolation for 2D array with periodic/anti-periodic BC support
 * @param {Array<Array<number>>} data - 2D array [ntheta][nr] or [nr][ntheta] depending on r_orientation
 * @param {number} theta_idx - Theta index (fractional)
 * @param {number} r_idx - R index (fractional)
 * @param {boolean} thetaPeriodic - If true, wrap theta index periodically
 * @param {boolean} thetaAntiperiodic - If true, apply sign flip when crossing theta boundary
 * @param {string} r_orientation - 'horizontal' (data[theta][r]) or 'vertical' (data[r][theta])
 * @returns {number} - Interpolated value
 */
function bilinearInterpolate(data, theta_idx, r_idx, thetaPeriodic = false, thetaAntiperiodic = false, r_orientation = 'horizontal') {
    // Safety checks
    if (!data || data.length === 0 || !data[0] || data[0].length === 0) {
        return 0;
    }

    let ntheta, nr;
    if (r_orientation === 'horizontal') {
        // data[theta_idx][r_idx]: rows = ntheta, cols = nr
        ntheta = data.length;
        nr = data[0].length;
    } else {
        // data[r_idx][theta_idx]: rows = nr, cols = ntheta
        nr = data.length;
        ntheta = data[0].length;
    }

    // Clamp r index to valid range
    r_idx = Math.max(0, Math.min(nr - 1.001, r_idx));

    // Handle theta index based on periodicity
    let crossesBoundary = false;
    let i0, i1, dt;

    if (thetaPeriodic) {
        // Wrap theta index periodically
        theta_idx = ((theta_idx % ntheta) + ntheta) % ntheta;
        i0 = Math.floor(theta_idx);
        i1 = (i0 + 1) % ntheta;
        dt = theta_idx - i0;
        // Check if interpolation crosses the theta boundary
        crossesBoundary = (i1 < i0);
    } else {
        // Clamp theta index
        theta_idx = Math.max(0, Math.min(ntheta - 1.001, theta_idx));
        i0 = Math.floor(theta_idx);
        i1 = Math.min(i0 + 1, ntheta - 1);
        dt = theta_idx - i0;
    }

    const j0 = Math.floor(r_idx);
    const j1 = Math.min(j0 + 1, nr - 1);
    const dr = r_idx - j0;

    // Get corner values with safety checks (handle r_orientation)
    let v00, v01, v10, v11;
    if (r_orientation === 'horizontal') {
        // data[theta_idx][r_idx]
        v00 = (data[i0] && data[i0][j0] !== undefined) ? data[i0][j0] : 0;
        v01 = (data[i0] && data[i0][j1] !== undefined) ? data[i0][j1] : 0;
        v10 = (data[i1] && data[i1][j0] !== undefined) ? data[i1][j0] : 0;
        v11 = (data[i1] && data[i1][j1] !== undefined) ? data[i1][j1] : 0;
    } else {
        // data[r_idx][theta_idx]
        v00 = (data[j0] && data[j0][i0] !== undefined) ? data[j0][i0] : 0;
        v01 = (data[j1] && data[j1][i0] !== undefined) ? data[j1][i0] : 0;
        v10 = (data[j0] && data[j0][i1] !== undefined) ? data[j0][i1] : 0;
        v11 = (data[j1] && data[j1][i1] !== undefined) ? data[j1][i1] : 0;
    }

    // Apply sign flip for anti-periodic BC when crossing boundary
    if (thetaAntiperiodic && crossesBoundary) {
        v10 = -v10;
        v11 = -v11;
    }

    // Bilinear interpolation
    const v0 = v00 * (1 - dr) + v01 * dr;  // at theta=i0
    const v1 = v10 * (1 - dr) + v11 * dr;  // at theta=i1

    return v0 * (1 - dt) + v1 * dt;
}

/**
 * Refresh all plots in the dashboard (for when polar transform settings change)
 */
function refreshAllPlots() {
    // Refresh all gridstack items
    if (AppState.gridStack) {
        const items = AppState.gridStack.getGridItems();
        items.forEach(item => {
            const plotId = item.getAttribute('data-plot-id');
            const plotType = item.getAttribute('data-plot-type');

            if (plotId && plotType) {
                // Find the plot div inside the item
                const plotDiv = item.querySelector('[id^="plot-"]');
                if (plotDiv) {
                    // Re-render the plot
                    renderPlot(plotDiv.id, plotType);
                }
            }
        });
    }
}
