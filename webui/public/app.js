// =====================================================
// OpenMagFDM Integrated Environment - Main Application
// =====================================================

// ===== Global State =====
const AppState = {
    currentTab: 'config',
    configData: null,
    uploadedImage: null,
    currentStep: 1,
    totalSteps: 1,
    resultsData: {},
    gridStack: null,
    aceEditor: null,  // Ace Editor instance
    yamlSchema: null,  // YAML schema for autocomplete
    userId: null  // User identifier (from cookie)
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', async () => {
    initializeUserId();
    initializeTabs();
    await initializeConfigEditor();
    initializeDashboard();
    await refreshConfigList();
    await loadConfig();
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

        const response = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Failed to upload image');

        const result = await response.json();
        showStatus('solverStatus', `Image uploaded: ${result.filename}`, 'success');
    } catch (error) {
        showStatus('solverStatus', `Upload error: ${error.message}`, 'error');
    }
}

async function runSolver() {
    const btn = document.getElementById('solverBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const outputDiv = document.getElementById('solverOutput');
    outputDiv.textContent = 'Starting solver...\n';

    try {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ configFile: 'sample_config.yaml' })
        });

        if (!response.ok) throw new Error('Solver execution failed');

        const result = await response.json();
        outputDiv.textContent += result.output || 'Solver completed successfully\n';

        showStatus('solverStatus', 'Solver completed successfully', 'success');

        // Load quick preview
        await loadQuickPreview();

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
        const response = await fetch('/api/detect-steps');
        if (!response.ok) throw new Error('Failed to detect steps');

        const data = await response.json();
        AppState.totalSteps = data.totalSteps || 1;

        showStatus('solverStatus', `Found ${AppState.totalSteps} steps`, 'info');

        await loadQuickPreview();
    } catch (error) {
        showStatus('solverStatus', `Error loading results: ${error.message}`, 'error');
    }
}

async function loadQuickPreview() {
    const step = 1; // Always show first step in preview

    try {
        // Load Az data
        const azResponse = await fetch(`/api/load-csv?type=Az&step=${step}`);
        if (azResponse.ok) {
            const azData = await azResponse.json();
            plotHeatmap('previewPlot1', azData.data, 'Az (Magnetic Vector Potential)');
        }

        // Load Mu data
        const muResponse = await fetch(`/api/load-csv?type=Mu&step=${step}`);
        if (muResponse.ok) {
            const muData = await muResponse.json();
            plotHeatmap('previewPlot2', muData.data, 'μ (Permeability)');
        }

        // Load energy density
        const energyResponse = await fetch(`/api/load-csv?type=EnergyDensity&step=${step}`);
        if (energyResponse.ok) {
            const energyData = await energyResponse.json();
            plotHeatmap('previewPlot3', energyData.data, 'Energy Density');
        }

    } catch (error) {
        console.error('Preview error:', error);
    }
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

    // Create widget content
    const content = `
        <div class="plot-header">
            <span>${plotDef.name}</span>
            <div class="plot-controls">
                <button onclick="removePlot('${plotId}')">Remove</button>
            </div>
        </div>
        <div class="plot-container" id="${containerId}">
            <div style="text-align: center; padding: 20px; color: #999;">Loading...</div>
        </div>
    `;

    // Add to GridStack
    AppState.gridStack.addWidget({
        x: x,
        y: y,
        w: w,
        h: h,
        content: content,
        id: plotId
    });

    // Render plot
    setTimeout(async () => {
        const step = parseInt(document.getElementById('stepInput').value) || 1;
        try {
            await plotDef.render(containerId, step);
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

function loadStep() {
    const step = parseInt(document.getElementById('stepInput').value) || 1;

    // Reload all plots in dashboard with new step
    const plots = document.querySelectorAll('.grid-stack-item');
    plots.forEach(item => {
        const content = item.querySelector('[id^="plot-"]');
        if (content) {
            // Determine plot type from content and reload
            // This is simplified - you might want to store plot type in data attribute
            loadAndPlot(content.id, 'heatmap', step);
        }
    });
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
// Placeholder implementations - these will call actual data loading and plotting
async function renderAzContour(containerId, step) {
    const response = await fetch(`/api/load-csv?type=Az&step=${step}`);
    if (!response.ok) throw new Error('Failed to load Az data');
    const result = await response.json();
    plotContour(containerId, result.data, 'Az (Magnetic Vector Potential)');
}

async function renderAzHeatmap(containerId, step) {
    const response = await fetch(`/api/load-csv?type=Az&step=${step}`);
    if (!response.ok) throw new Error('Failed to load Az data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Az Heatmap');
}

async function renderJzDistribution(containerId, step) {
    const response = await fetch(`/api/load-csv?type=Jz&step=${step}`);
    if (!response.ok) throw new Error('Failed to load Jz data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Jz (Current Density)');
}

async function renderBMagnitude(containerId, step) {
    const response = await fetch(`/api/load-csv?type=B&step=${step}`);
    if (!response.ok) throw new Error('Failed to load B data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, '|B| Distribution');
}

async function renderHMagnitude(containerId, step) {
    const response = await fetch(`/api/load-csv?type=H&step=${step}`);
    if (!response.ok) throw new Error('Failed to load H data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, '|H| Distribution');
}

async function renderMuDistribution(containerId, step) {
    const response = await fetch(`/api/load-csv?type=Mu&step=${step}`);
    if (!response.ok) throw new Error('Failed to load Mu data');
    const result = await response.json();
    plotHeatmap(containerId, result.data, 'Permeability Distribution');
}

async function renderEnergyDensity(containerId, step) {
    const response = await fetch(`/api/load-csv?type=EnergyDensity&step=${step}`);
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

async function renderForceXTime(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Force X-axis time series coming soon</p>';
}

async function renderForceYTime(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Force Y-axis time series coming soon</p>';
}

async function renderTorqueTime(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Torque time series coming soon</p>';
}

async function renderEnergyTime(containerId, step) {
    document.getElementById(containerId).innerHTML = '<p style="padding:20px;">Energy time series coming soon</p>';
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
