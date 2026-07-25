// =====================================================
// OpenMagFDM Integrated Environment - Main Application
// =====================================================

// ===== Global State =====
// Field data is retained by byte budget rather than entry count.  The values
// are decoded 2D JavaScript arrays, so the estimate below charges each array
// slot as 8 bytes plus a small per-array overhead.  Keep this as a single
// constant so packaged builds can tune the browser memory budget easily.
const FIELD_CACHE_BUDGET_BYTES = 256 * 1024 * 1024;
const FIELD_WORKING_MEMORY_BUDGET_BYTES = 384 * 1024 * 1024;
const FIELD_UNKNOWN_DECODE_RESERVATION_BYTES = 64 * 1024 * 1024;
const FIELD_CACHE_ARRAY_OVERHEAD_BYTES = 64;

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
    animationTimer: null,  // Serialized animation timeout
    animationRunId: 0,     // Invalidates a pending/in-flight animation loop
    isAnimating: false,  // Animation state flag
    analysisConditions: null,  // Analysis conditions from conditions.json
    dataCache: {},       // Decoded field arrays, keyed by resultPath:type:step
    dataMeta: {},        // Per-entry { format, precision }
    dataCacheSizes: {},  // Per-entry estimated retained bytes
    dataCacheLru: new Map(), // Oldest key first; touch by delete+set
    dataCacheBytes: 0,
    dataCacheBudgetBytes: FIELD_CACHE_BUDGET_BYTES,
    fieldLoadGeneration: 0,
    fieldPayloadsInFlight: new Map(), // key -> leased raw decode entry
    fieldPreloadController: null,
    fieldDecodeTail: Promise.resolve(), // serializes memory-heavy TIFF/JSON decoding
    fieldDecodeReservedBytes: 0,
    fieldPayloadStagingBytes: 0,
    fieldViewPins: new Map(), // AbortSignal -> { fields: Map<cacheKey, bytes> }
    fieldViewPinRefs: new WeakMap(), // decoded array -> { bytes, count }
    fieldViewPinnedBytes: 0,
    lastFieldMeta: null,
    resultLoadGeneration: 0,
    resultLoadController: null,
    dashboardRenderGeneration: 0,
    dashboardRenderQueue: Promise.resolve(),
    dashboardRenderController: null,
    containerRenderTokens: new Map(),
    containerRenderControllers: new Map(),
    filePreviewGeneration: 0,
    filePreviewController: null,
    filePreviewResultPath: '',
    bhRenderGeneration: 0,
    bhRenderController: null,
    bhListGeneration: 0,
    bhListController: null,
    // Polar coordinate transform options
    isPolarCoordinates: false,  // True if current result uses polar coordinates
    polarCartesianTransform: false,  // Apply cartesian transform (arc/donut view)
    polarFullModel: false,  // Expand to full model
    polarFullModelMultiplier: 1,  // Multiplier for full model (N in 2π/N)
    // Plot zoom state preservation
    plotZoomStates: {},  // { containerId: { xaxis: { range: [min, max] }, yaxis: { range: [min, max] } } }
    // Plotly mode bar visibility
    showPlotlyModeBar: false,  // Show/hide Plotly mode bar for all plots
    // Plot configuration (ranges, colorscales, etc.)
    plotConfigs: {},  // { plotId: { xRange: 'auto'|[min,max], yRange: 'auto'|[min,max], zRange: 'auto'|[min,max], colorscale: 'Viridis' } }
    // Detect Colors result cache
    lastDetectResult: null,  // Last result from /api/materials/detect
    // Phase D.4: Detect Colors per-chip library / Coil assignments.
    // detectAssign[hex] = { kind: 'none'|'Coil'|<preset name>,
    //                       coilGroup: 'A'..'Z', coilSign: '+'|'-' }
    detectAssign: {},
    // Phase D.4: active library used by the Detect Colors modal.
    // Falls back to AppState.selectedLibrary on every open. The header
    // dropdown overrides it for the duration of the modal.
    detectLibraryName: null,
    detectLibraryPresets: {},    // { presetName: { mu_r, jz, ... }, ... }
    detectLibraryMaterials: {},  // { matName: { rgb, mu_r, jz, ... }, ... }
    detectLibraryRaw: '',        // raw library YAML text (for verbatim splicing)
    // Material Library
    selectedLibrary: null,       // Active library filename (null = none)
    libraryAceEditor: null,      // Ace Editor instance inside Library modal
    currentLibraryFile: null,    // Currently selected filename in Library modal
    currentBHMaterial: null,     // Last rendered BH material {name, props} for axis toggle
    userFiles: [],               // Unified File Manager listing
    userFileSelection: new Set() // category + filename keys selected for bulk actions
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
    initInputImageViewer();
    await initializeConfigEditor();
    initializeDashboard();
    initCustomSelect();
    await refreshConfigList();
    await loadConfig();
    await refreshImageList();
    await refreshResultsList();

    // Restore last active material library from cookie
    const lastLibrary = getCookie('magfdm_last_library');
    if (lastLibrary) {
        try {
            // Verify the library file still exists on server
            const resp = await fetch(
                `/api/material-libraries/${encodeURIComponent(lastLibrary)}?userId=${AppState.userId}`
            );
            if (resp.ok) {
                AppState.selectedLibrary = lastLibrary;
                document.getElementById('activeLibraryName').textContent = lastLibrary;
                document.getElementById('activeLibraryBadge').style.display = 'inline-flex';
                console.log('Restored material library from cookie:', lastLibrary);
            } else {
                // File no longer exists — clear stale cookie
                setCookie('magfdm_last_library', '', -1);
            }
        } catch (e) {
            console.warn('Failed to restore material library:', e);
            setCookie('magfdm_last_library', '', -1);
        }
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

            // Detect whether cursor is in a "value position" (after "key: ") vs a "key position".
            // Value position example: "    type: d"  (lineKey="type", prefix="d")
            // Key position example:   "    ty"       (isValuePosition=false)
            const lineBeforeCursor = session.getLine(pos.row).substring(0, pos.column);
            const valuePositionMatch = lineBeforeCursor.match(/^\s*([\w-]+):\s+\S*$/);
            const isValuePosition = !!valuePositionMatch;
            const lineKey = valuePositionMatch ? valuePositionMatch[1] : null;

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
                // Top level - show only keywords that have no validParents
                // (i.e. genuine root-level keys). Pre-BJ-fix this returned
                // Object.keys(keywords), which surfaced per-material keys
                // like `coarsen`/`coarsen_ratio` at the document root and
                // led users to write `coarsen: true` outside any material
                // block, where the parser silently ignores it.
                availableKeywords = Object.keys(keywords).filter(k => {
                    const info = keywords[k];
                    return !info || !info.validParents || info.validParents.length === 0;
                });
            }

            // Add keyword completions
            for (const keyword of availableKeywords) {
                const info = keywords[keyword];
                if (!info) continue;

                if (!isValuePosition) {
                    // At key position: show "keyword: " completion
                    completions.push({
                        caption: keyword,
                        value: keyword + ': ',
                        meta: info.type || 'keyword',
                        score: 1000,
                        docHTML: `<b>${keyword}</b><br>${info.description}<br><code>${info.example || ''}</code>`
                    });
                }

                // Add value suggestions
                if (info.values) {
                    info.values.forEach(val => {
                        if (isValuePosition && lineKey === keyword) {
                            // In value position for this keyword: suggest only the value (not "keyword: value")
                            completions.push({
                                caption: val,
                                value: val,
                                meta: 'value',
                                score: 900,
                                docHTML: `<b>${keyword}: ${val}</b><br>${info.description || ''}`
                            });
                        } else if (!isValuePosition) {
                            // At key position: suggest full "keyword: value" pair
                            completions.push({
                                caption: `${keyword}: ${val}`,
                                value: `${keyword}: ${val}`,
                                meta: 'value',
                                score: 900
                            });
                        }
                    });
                }
            }

            // Add snippets based on context
            addContextSnippets(completions, parentContext, grandparentContext, isValuePosition, lineKey);

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

    // Phase D.5: inline colour swatches next to material-name lines.
    initMaterialSwatches(editor);
}

// ============================================================
// Phase D.5: inline material-colour swatches in the Ace editor.
// ============================================================
// Strategy: maintain a single position-absolute overlay <div> stacked
// over the editor container. On every YAML change (debounced 200 ms)
// we re-parse the doc, walk doc.materials, and for each entry with an
// rgb: [r,g,b] property we look up the row where its name is defined
// and append a 12 x 12 swatch to the overlay positioned via
// renderer.textToScreenCoordinates(). The overlay is re-positioned
// (cheap path) on every renderer afterRender so the swatches track the
// editor when scrolling / folding / resizing without a YAML re-parse.

function initMaterialSwatches(editor) {
    if (editor._swatchInit) return;
    editor._swatchInit = true;
    editor._swatchEntries = [];   // [{ name, row, color }, ...]
    let parseTimer = null;
    const reparse = () => {
        clearTimeout(parseTimer);
        parseTimer = setTimeout(() => {
            editor._swatchEntries = computeMaterialSwatchEntries(editor);
            positionMaterialSwatches(editor);
        }, 200);
    };
    editor.session.on('change', reparse);
    editor.renderer.on('afterRender', () => positionMaterialSwatches(editor));
    window.addEventListener('resize', () => positionMaterialSwatches(editor));
    editor._swatchEntries = computeMaterialSwatchEntries(editor);
    positionMaterialSwatches(editor);
}

function computeMaterialSwatchEntries(editor) {
    let doc;
    try { doc = jsyaml.load(editor.getValue()) || {}; }
    catch (_) { return []; }
    if (!doc || typeof doc !== 'object') return [];
    const materials = (doc.materials && typeof doc.materials === 'object') ? doc.materials : {};
    const names = Object.keys(materials);
    if (names.length === 0) return [];
    const lines = editor.session.getDocument().getAllLines();
    // The materials: block starts at some row; the keys are nested one
    // indent level under it. We scan for the keys with a regex so the
    // search is robust to varying indentation widths.
    const entries = [];
    for (const name of names) {
        const m = materials[name];
        if (!m || typeof m !== 'object' || !Array.isArray(m.rgb) || m.rgb.length < 3) continue;
        const r = Math.max(0, Math.min(255, Math.round(Number(m.rgb[0]) || 0)));
        const g = Math.max(0, Math.min(255, Math.round(Number(m.rgb[1]) || 0)));
        const b = Math.max(0, Math.min(255, Math.round(Number(m.rgb[2]) || 0)));
        const namePattern = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp(`^\\s+${namePattern}\\s*:\\s*$`);
        const row = lines.findIndex(ln => re.test(ln));
        if (row < 0) continue;
        entries.push({ name, row, color: `rgb(${r}, ${g}, ${b})`, lineLen: lines[row].length });
    }
    return entries;
}

function positionMaterialSwatches(editor) {
    const entries = editor._swatchEntries || [];
    const container = editor.container;
    let overlay = container.querySelector('.mat-swatch-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'mat-swatch-overlay';
        overlay.style.cssText = 'position:absolute; inset:0; pointer-events:none; z-index:5; overflow:hidden;';
        container.appendChild(overlay);
    }
    overlay.innerHTML = '';
    if (entries.length === 0) return;
    const containerRect = container.getBoundingClientRect();
    const visibleTop    = editor.renderer.getFirstVisibleRow();
    const visibleBottom = editor.renderer.getLastVisibleRow();
    for (const e of entries) {
        // Skip rows outside the visible viewport for cheaper layout.
        if (e.row < visibleTop - 1 || e.row > visibleBottom + 1) continue;
        const screen = editor.renderer.textToScreenCoordinates(e.row, e.lineLen);
        const left = screen.pageX - containerRect.left - window.scrollX + 6;
        const top  = screen.pageY - containerRect.top  - window.scrollY + 2;
        const swatch = document.createElement('div');
        swatch.style.cssText =
            'position:absolute; width:12px; height:12px; border:1px solid rgba(0,0,0,0.45); ' +
            'border-radius:2px; box-shadow:0 0 0 1px rgba(255,255,255,0.25);';
        swatch.style.background = e.color;
        swatch.style.left = `${left}px`;
        swatch.style.top  = `${top}px`;
        swatch.title = `${e.name} — ${e.color}`;
        overlay.appendChild(swatch);
    }
}

// Add context-aware snippets
// isValuePosition: true if cursor is after "key: " (in value position)
// lineKey: the key name on the current line when isValuePosition is true
function addContextSnippets(completions, parentContext, grandparentContext, isValuePosition = false, lineKey = null) {
    if (!AppState.yamlSchema || !AppState.yamlSchema.snippets) return;

    const snippets = AppState.yamlSchema.snippets;

    // Add boundary condition snippets (for inner, outer, left, right, top, bottom, theta_min, theta_max)
    const boundaryContexts = ['inner', 'outer', 'left', 'right', 'top', 'bottom', 'theta_min', 'theta_max'];
    if (boundaryContexts.includes(parentContext)) {
        // Each boundary snippet has a "full" form (key position) and a "value-only" form (value position after "type:")
        const boundarySnippets = [
            { key: 'boundary_dirichlet',   caption: '[Snippet] Dirichlet',    valueOnly: 'dirichlet',
              doc: '<b>Snippet: Dirichlet Boundary</b><br>type: dirichlet + value: 0.0' },
            { key: 'boundary_neumann',     caption: '[Snippet] Neumann',      valueOnly: 'neumann',
              doc: '<b>Snippet: Neumann Boundary</b><br>type: neumann + value: 0.0' },
            { key: 'boundary_periodic',    caption: '[Snippet] Periodic',     valueOnly: 'periodic',
              doc: '<b>Snippet: Periodic Boundary</b><br>type: periodic + value: 0.0' },
            { key: 'boundary_antiperiodic',caption: '[Snippet] Anti-Periodic',valueOnly: 'periodic',
              doc: '<b>Snippet: Anti-Periodic Boundary</b><br>type: periodic + value: -1.0' },
            { key: 'boundary_robin',       caption: '[Snippet] Robin',        valueOnly: 'robin',
              doc: '<b>Snippet: Robin Boundary</b><br>type: robin + alpha/beta/gamma' },
        ];
        for (const bs of boundarySnippets) {
            if (!snippets[bs.key]) continue;
            if (isValuePosition && lineKey === 'type') {
                // User is typing the BC type value — suggest just the value word
                completions.push({
                    caption: bs.caption,
                    value: bs.valueOnly,
                    meta: 'value',
                    score: 1100,
                    docHTML: bs.doc
                });
            } else if (!isValuePosition) {
                // User is at a key position — insert the full "type: ...\nvalue: ..." snippet
                completions.push({
                    caption: bs.caption,
                    value: snippets[bs.key].snippet,
                    meta: 'snippet',
                    score: 1100,
                    docHTML: bs.doc
                });
            }
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

    // Add domain decomposition template snippet (when parent is domain_decomposition)
    if (parentContext === 'domain_decomposition') {
        if (snippets.domain_decomposition_template) {
            completions.push({
                caption: '[Snippet] Domain Decomposition Template',
                value: snippets.domain_decomposition_template.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Snippet: Domain Decomposition</b><br>Variable-resolution optimized Schwarz (polar, opt-in accuracy mode): keep the air gap fine, coarsen smooth radial bands'
            });
        }
    }
}

// ---- Library editor completer ----
// Creates a completer scoped to material library YAML files (material_presets structure).
function createLibraryCompleter() {
    return {
        getCompletions: function(editor, session, pos, prefix, callback) {
            if (!AppState.yamlSchema) { callback(null, []); return; }
            const completions = [];
            const keywords = AppState.yamlSchema.keywords;

            // Value-position detection (same logic as main editor)
            const lineBeforeCursor = session.getLine(pos.row).substring(0, pos.column);
            const valuePositionMatch = lineBeforeCursor.match(/^\s*([\w-]+):\s+\S*$/);
            const isValuePosition = !!valuePositionMatch;
            const lineKey = valuePositionMatch ? valuePositionMatch[1] : null;

            const contextPath = getContextPath(editor, session, pos);
            const parentContext  = contextPath[contextPath.length - 1] || null;
            const grandparentContext = contextPath[contextPath.length - 2] || null;

            // Library YAML context rules:
            //   top level                     → suggest material_presets
            //   parent = material_presets      → user names presets freely (no key suggestions)
            //   grandparent = material_presets → inside a preset → suggest preset properties
            //   parent = magnetization         → inside magnetization block → suggest children
            let availableKeywords = [];
            if (!parentContext) {
                availableKeywords = ['material_presets'];
            } else if (grandparentContext === 'material_presets') {
                const presetsInfo = keywords['material_presets'];
                availableKeywords = presetsInfo && presetsInfo.childrenProperties
                    ? presetsInfo.childrenProperties
                    : ['mu_r', 'B-H', 'bh_type'];
            } else if (parentContext === 'magnetization') {
                const magInfo = keywords['magnetization'];
                availableKeywords = magInfo && magInfo.children ? magInfo.children : [];
            }
            // parentContext === 'material_presets': user defines preset names → no key suggestions

            for (const kw of availableKeywords) {
                const info = keywords[kw];
                if (!info) continue;

                if (!isValuePosition) {
                    completions.push({
                        caption: kw,
                        value: kw + ': ',
                        meta: info.type || 'keyword',
                        score: 1000,
                        docHTML: `<b>${kw}</b><br>${info.description || ''}<br><code>${info.example || ''}</code>`
                    });
                }

                if (info.values) {
                    info.values.forEach(val => {
                        if (isValuePosition && lineKey === kw) {
                            completions.push({ caption: val, value: val, meta: 'value', score: 900,
                                docHTML: `<b>${kw}: ${val}</b>` });
                        } else if (!isValuePosition) {
                            completions.push({ caption: `${kw}: ${val}`, value: `${kw}: ${val}`,
                                meta: 'value', score: 900 });
                        }
                    });
                }
            }

            addLibrarySnippets(completions, parentContext, grandparentContext, isValuePosition);
            callback(null, completions);
        }
    };
}

// Snippets for the library editor (material preset templates, magnetization patterns)
function addLibrarySnippets(completions, parentContext, grandparentContext, isValuePosition) {
    if (!AppState.yamlSchema || !AppState.yamlSchema.snippets || isValuePosition) return;
    const snippets = AppState.yamlSchema.snippets;

    if (!parentContext) {
        // Top level: offer a full library file template
        if (snippets.lib_file_template) {
            completions.push({
                caption: '[Template] Material Library',
                value: snippets.lib_file_template.snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: '<b>Material Library Template</b><br>Creates a starter library with soft and magnet presets'
            });
        }
    } else if (grandparentContext === 'material_presets') {
        // Inside a specific preset: show preset-type snippets
        const presetSnippets = [
            { key: 'lib_preset_mur_const',      doc: 'Constant relative permeability' },
            { key: 'lib_preset_mur_formula',     doc: 'Variable μr as a formula of H (tinyexpr)' },
            { key: 'lib_preset_soft_table',      doc: 'Nonlinear B-H curve (measured data table)' },
            { key: 'lib_preset_soft_formula',    doc: 'Nonlinear B-H curve (continuous formula)' },
            { key: 'lib_preset_magnet_parallel', doc: 'Permanent magnet with parallel magnetization + Br' },
            { key: 'lib_preset_magnet_demag',    doc: 'Permanent magnet via B-H demagnetization curve' },
        ];
        for (const ps of presetSnippets) {
            if (!snippets[ps.key]) continue;
            completions.push({
                caption: `[Preset] ${snippets[ps.key].name}`,
                value: snippets[ps.key].snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: `<b>${snippets[ps.key].name}</b><br>${ps.doc}`
            });
        }
    } else if (parentContext === 'magnetization') {
        // Inside magnetization block
        const magSnippets = [
            'magnetization_parallel', 'magnetization_radial', 'magnetization_tangential',
            'magnetization_halbach', 'magnetization_polar_anisotropy', 'magnetization_custom'
        ];
        for (const key of magSnippets) {
            if (!snippets[key]) continue;
            completions.push({
                caption: `[Snippet] ${snippets[key].name}`,
                value: snippets[key].snippet,
                meta: 'snippet',
                score: 1100,
                docHTML: `<b>${snippets[key].name}</b>`
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
        document.getElementById('detectColorsBtn').style.display = 'block';
        document.getElementById('polarizeBtn').style.display = 'block';
        document.getElementById('magPreviewBtn').style.display = 'block';
        document.getElementById('cartesianTemplateBtn').style.display = 'block';
        document.getElementById('imagePropsBtn').style.display = 'block';

        // Refresh image list
        await refreshImageList();
        // Probe for AA-noise so the warning banner + filter button can
        // surface in the Input Image panel without the user having to
        // open Detect Colors or Polar Preprocess first.
        checkInputImageNoise(result.filename);
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
    document.getElementById('detectColorsBtn').style.display = 'block';
    document.getElementById('polarizeBtn').style.display = 'block';
    document.getElementById('cartesianTemplateBtn').style.display = 'block';
    const _mpb = document.getElementById('magPreviewBtn'); if (_mpb) _mpb.style.display = 'block';
    const _ipb = document.getElementById('imagePropsBtn'); if (_ipb) _ipb.style.display = 'block';
    showStatus('solverStatus', `Image loaded: ${filename}`, 'success');
    checkInputImageNoise(filename);
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
            document.getElementById('detectColorsBtn').style.display = 'none';
            document.getElementById('polarizeBtn').style.display = 'none';
            document.getElementById('magPreviewBtn').style.display = 'none';
            document.getElementById('cartesianTemplateBtn').style.display = 'none';
            document.getElementById('imagePropsBtn').style.display = 'none';
            document.getElementById('quantizeFilterBtn').style.display = 'none';
            document.getElementById('inputImageNoiseBanner').style.display = 'none';
        }
    } catch (error) {
        showStatus('solverStatus', `Delete error: ${error.message}`, 'error');
    }
}

// =====================================================
// Detect Colors Feature
// =====================================================

// Internal color detection: POSTs /api/materials/detect with the currently
// selected image. Returns the parsed result or throws. Separated from the
// modal opener so the Polar Preprocess Modal (v1.5) can request the same
// detection in parallel with /api/preprocess-polar/detect without opening
// the Detect Colors modal.
async function detectColorsInternal(opts = {}) {
    if (!AppState.uploadedImageFilename) {
        throw new Error('No image selected');
    }
    const rareThreshold = opts.rareThreshold != null
        ? opts.rareThreshold
        : (parseFloat(document.getElementById('detectRareThreshold').value || '5') / 100);
    const blendTolerance = opts.blendTolerance != null
        ? opts.blendTolerance
        : parseInt(document.getElementById('detectBlendTolerance').value || '8', 10);

    const imgResponse = await fetch(`/uploads/${AppState.userId}/${AppState.uploadedImageFilename}`);
    if (!imgResponse.ok) throw new Error('Failed to fetch image');
    const blob = await imgResponse.blob();

    const formData = new FormData();
    formData.append('image', blob, AppState.uploadedImageFilename);
    formData.append('userId', AppState.userId);

    const params = new URLSearchParams({
        rareThreshold: String(rareThreshold),
        blendTolerance: String(blendTolerance),
    });
    const response = await fetch(`/api/materials/detect?${params}`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({ error: response.statusText }));
        throw new Error(err.error || 'Detection failed');
    }
    return await response.json();
}

async function detectColors() {
    try {
        const result = await detectColorsInternal();
        AppState.lastDetectResult = result;
        // Phase D.4: reset assignments to 'none' for every freshly-
        // detected colour. The user can pick library presets / Coil
        // per chip after the modal opens.
        AppState.detectAssign = {};
        (result.colors || []).forEach(c => {
            const hex = `#${c.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
            AppState.detectAssign[hex] = {
                kind: 'none',
                coilGroup: 'A',
                coilSign: '+',
                // Phase D.7: per-chip magnetization sub-state. Shown only
                // when kind references a magnet preset (isMagnetMaterial).
                // Parameters are kept across pattern switches so the user
                // doesn't lose typed values when toggling.
                magnetization: defaultMagnetizationState(),
            };
        });
        // Phase D.4: default the modal's library picker to the globally
        // active library (if any). The dropdown lets the user override.
        AppState.detectLibraryName = AppState.selectedLibrary || null;
        await detectRefreshLibraryList();
        await detectLoadLibraryByName(AppState.detectLibraryName);
        renderDetectModal(result);
        document.getElementById('detectColorsModal').style.display = 'flex';
    } catch (error) {
        showStatus('solverStatus', `Detect error: ${error.message}`, 'error');
    }
}

// Phase D.4: populate the library picker dropdown from
// /api/material-libraries. Idempotent; preserves the current
// selection across reloads.
async function detectRefreshLibraryList() {
    const sel = document.getElementById('detectActiveLibrary');
    if (!sel) return;
    const previous = sel.value || AppState.detectLibraryName || '';
    try {
        const r = await fetch(`/api/material-libraries?userId=${AppState.userId}`).then(r => r.json());
        const libs = (r && Array.isArray(r.libraries)) ? r.libraries : [];
        sel.innerHTML = '<option value="">(none — preset references will fail)</option>';
        for (const lib of libs) {
            const opt = document.createElement('option');
            opt.value = lib.filename;
            opt.textContent = lib.filename;
            sel.appendChild(opt);
        }
        // Restore prior selection if it still exists in the list.
        const stillThere = Array.from(sel.options).some(o => o.value === previous);
        sel.value = stillThere ? previous : '';
        if (!sel._ppBound) {
            sel.addEventListener('change', async () => {
                AppState.detectLibraryName = sel.value || null;
                await detectLoadLibraryByName(AppState.detectLibraryName);
                renderDetectChips();
                regenerateDetectYamlPreview();
            });
            sel._ppBound = true;
        }
    } catch (e) {
        // Leave the dropdown with the single (none) option so the modal
        // still works without a library.
    }
}

// Phase D.4: fetch + parse the chosen library YAML. Populates
// AppState.detectLibraryPresets / Materials / Raw. Passing null
// clears them so the per-chip dropdowns fall back to (none) / Coil.
async function detectLoadLibraryByName(name) {
    AppState.detectLibraryPresets = {};
    AppState.detectLibraryMaterials = {};
    AppState.detectLibraryRaw = '';
    if (!name) return;
    try {
        const r = await fetch(`/api/material-libraries/${encodeURIComponent(name)}?userId=${AppState.userId}`);
        if (!r.ok) return;
        const text = await r.text();
        AppState.detectLibraryRaw = text;
        const doc = jsyaml.load(text) || {};
        if (doc && typeof doc === 'object') {
            if (doc.material_presets && typeof doc.material_presets === 'object') {
                AppState.detectLibraryPresets = doc.material_presets;
            }
            if (doc.materials && typeof doc.materials === 'object') {
                AppState.detectLibraryMaterials = doc.materials;
            }
        }
    } catch (e) {
        // Silent: the picker is a convenience, not load-critical.
    }
}

function renderDetectModal(result) {
    // Noise / AA-heavy warning banner. Heuristic: a CAD image typically
    // has 5-50 unique RGB values; if we see thousands the image is
    // either JPEG-derived or aggressively anti-aliased and the uniform-
    // colour filter is the recommended cleanup before downstream use.
    const banner = document.getElementById('detectNoisyBanner');
    const detail = document.getElementById('detectNoisyBannerDetail');
    const unique = Number(result.uniqueColors) || 0;
    const aaTotal = Number(result.aaBlendsTotal) || 0;
    if (unique > 1000) {
        detail.textContent =
            `(Number of unique colors: ${unique.toLocaleString()} / AA blends: ${aaTotal.toLocaleString()}).`;
        banner.style.display = 'flex';
    } else {
        banner.style.display = 'none';
    }

    // Sync the picker dropdown to whatever detectColors() set up.
    const libSel = document.getElementById('detectActiveLibrary');
    if (libSel) libSel.value = AppState.detectLibraryName || '';

    // Render dominant color grid (delegated so the chip block can be
    // re-rendered on library change without recomputing the banner).
    renderDetectChips();

    // Render AA blends section
    const aaSection = document.getElementById('detectAASection');
    const aaList = document.getElementById('detectAAList');
    aaList.innerHTML = '';
    const blends = result.aaBlends || [];
    if (blends.length > 0) {
        blends.forEach(b => {
            const hexC = `#${b.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
            const hexA = `#${b.baseA.map(v => v.toString(16).padStart(2, '0')).join('')}`;
            const hexB = `#${b.baseB.map(v => v.toString(16).padStart(2, '0')).join('')}`;
            const row = document.createElement('div');
            row.style.cssText = 'display:flex; align-items:center; gap:8px; font-size:0.8rem; margin-bottom:5px;';
            row.innerHTML = `
                <div style="width:18px; height:18px; background:${hexC}; border:1px solid #ccc; border-radius:2px;"></div>
                <span style="font-family:monospace;">${hexC}</span>
                <span style="color:#6c757d;">(${(b.ratio * 100).toFixed(1)}%)</span>
                <span style="color:#aaa;">≈</span>
                <div style="width:14px; height:14px; background:${hexA}; border:1px solid #ccc; border-radius:2px;"></div>
                <span style="font-family:monospace; color:#6c757d;">${hexA}</span>
                <span style="color:#aaa;">×${(1 - b.t).toFixed(2)} + </span>
                <div style="width:14px; height:14px; background:${hexB}; border:1px solid #ccc; border-radius:2px;"></div>
                <span style="font-family:monospace; color:#6c757d;">${hexB}</span>
                <span style="color:#aaa;">×${b.t.toFixed(2)}</span>
            `;
            aaList.appendChild(row);
        });
        aaSection.style.display = 'block';
    } else {
        aaSection.style.display = 'none';
    }

    // Phase D.4: YAML preview is regenerated from the per-chip
    // assignments rather than echoing the server's stock template.
    regenerateDetectYamlPreview();
}

// Phase D.4: render the dominant colour grid with per-chip kind /
// Coil-group / Coil-sign dropdowns. Splits out of renderDetectModal so
// that picking a different library can re-render only the chips.
function renderDetectChips() {
    const result = AppState.lastDetectResult;
    const grid = document.getElementById('detectColorGrid');
    if (!grid) return;
    grid.innerHTML = '';
    const presetNames = Object.keys(AppState.detectLibraryPresets || {});
    const materialNames = Object.keys(AppState.detectLibraryMaterials || {});
    const libNames = [...presetNames, ...materialNames];
    const presetsAll = AppState.detectLibraryPresets || {};
    const materialsAll = AppState.detectLibraryMaterials || {};
    (result && result.colors || []).forEach(c => {
        const hex = `#${c.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
        const isAA = c.antialias === true;
        const assign = AppState.detectAssign[hex] ||
            (AppState.detectAssign[hex] = {
                kind: 'none', coilGroup: 'A', coilSign: '+',
                magnetization: defaultMagnetizationState(),
            });
        // Older chip records may pre-date Phase D.7 — backfill the
        // magnetization sub-state if it is missing.
        if (!assign.magnetization) assign.magnetization = defaultMagnetizationState();
        // If the prior selection is no longer in the active library,
        // fall back to (none) so the dropdown stays consistent.
        if (assign.kind !== 'none' && assign.kind !== 'Coil' && !libNames.includes(assign.kind)) {
            assign.kind = 'none';
        }
        // Phase D.7: classify whether this kind is a magnet preset to
        // decide whether the magnetization sub-row should be rendered.
        const libProps = presetsAll[assign.kind] || materialsAll[assign.kind] || null;
        const isMagnet = (assign.kind !== 'none' && assign.kind !== 'Coil' && isMagnetMaterial(libProps));
        const item = document.createElement('div');
        item.style.cssText = 'display:flex; flex-direction:column; gap:6px; background:#f8f9fa; border-radius:4px; padding:6px 8px; font-size:0.8rem;';
        const optsKind = ['<option value="none">(none)</option>',
                          '<option value="Coil">Coil</option>']
            .concat(libNames.map(n => `<option value="${n}">${n}</option>`))
            .join('');
        const groupOpts = Array.from({ length: 26 }, (_, i) => {
            const L = String.fromCharCode(65 + i);
            return `<option value="${L}">${L}</option>`;
        }).join('');
        const patternOpts = [
            ['parallel',           'parallel (uniform angle)'],
            ['radial',             'radial (outward/inward from centre)'],
            ['tangential',         'tangential (CCW/CW around centre)'],
            ['halbach_continuous', 'halbach_continuous (p, centre, offset)'],
            ['polar_anisotropy',   'polar_anisotropy (p, Kn=Fn/Rm, centre, offset)'],
            ['radial_array',       'radial_array (NS alternating radial, p poles)'],
            ['parallel_array',     'parallel_array (NS alternating sector-parallel, p poles)'],
            ['custom',             'custom (Mx, My expressions)'],
        ].map(([v, t]) => `<option value="${v}">${t}</option>`).join('');
        item.innerHTML = `
          <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
            <div style="width:20px; height:20px; background:${hex}; border:1px solid #ccc; border-radius:2px; flex-shrink:0;"></div>
            <span style="font-family:monospace;">${hex}</span>
            <span style="color:#6c757d;">${(c.ratio * 100).toFixed(1)}%</span>
            ${isAA ? '<span style="background:#fff3cd; color:#856404; border-radius:10px; padding:1px 6px; font-size:0.75rem;">AA base</span>' : ''}
            <select data-detect-hex="${hex}" data-role="kind" style="padding:2px 4px; font-size:0.78rem;">${optsKind}</select>
            <span data-coil-extras="${hex}" style="display:none; gap:4px; align-items:center;">
                <span style="color:#6c757d;">group</span>
                <select data-detect-hex="${hex}" data-role="coilGroup" style="padding:2px 4px; font-size:0.78rem;">${groupOpts}</select>
                <label style="display:inline-flex; align-items:center; gap:2px;"><input type="radio" name="coilSign-${hex}" data-detect-hex="${hex}" data-role="coilSign" value="+"> +</label>
                <label style="display:inline-flex; align-items:center; gap:2px;"><input type="radio" name="coilSign-${hex}" data-detect-hex="${hex}" data-role="coilSign" value="-"> −</label>
            </span>
          </div>
          <div data-magnet-row="${hex}" style="display:none; padding-left:28px; gap:8px; align-items:center; flex-wrap:wrap; font-size:0.78rem;">
            <svg class="mag-preview" data-detect-hex="${hex}" width="80" height="80"
                 viewBox="0 0 80 80"
                 style="border:1px solid #c8c8c8; border-radius:4px; background:#fff; flex-shrink:0;"></svg>
            <span style="color:#6c757d; font-weight:600;">Magnetization</span>
            <select data-detect-hex="${hex}" data-role="magPattern" style="padding:2px 4px; font-size:0.78rem;">${patternOpts}</select>
            <span data-mag-params="parallel-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>angle <input type="number" step="1" data-detect-hex="${hex}" data-role="magAngle" style="width:64px; padding:1px 3px;"> deg</label>
            </span>
            <span data-mag-params="radial-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>cx <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCx" style="width:80px; padding:1px 3px;"> m</label>
                <label>cy <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCy" style="width:80px; padding:1px 3px;"> m</label>
                <label><input type="radio" name="magDir-${hex}" data-detect-hex="${hex}" data-role="magDir" value="outward"> outward</label>
                <label><input type="radio" name="magDir-${hex}" data-detect-hex="${hex}" data-role="magDir" value="inward"> inward</label>
            </span>
            <span data-mag-params="radialArray-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>p <input type="number" min="1" step="1" data-detect-hex="${hex}" data-role="magPArr" style="width:48px; padding:1px 3px;"></label>
                <label>cx <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCxArr" style="width:80px; padding:1px 3px;"> m</label>
                <label>cy <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCyArr" style="width:80px; padding:1px 3px;"> m</label>
                <label>offset <input type="number" step="1" data-detect-hex="${hex}" data-role="magOffsetArr" style="width:64px; padding:1px 3px;"> deg</label>
                <label><input type="radio" name="magDirArr-${hex}" data-detect-hex="${hex}" data-role="magDirArr" value="outward"> outward</label>
                <label><input type="radio" name="magDirArr-${hex}" data-detect-hex="${hex}" data-role="magDirArr" value="inward"> inward</label>
            </span>
            <span data-mag-params="parallelArray-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>p <input type="number" min="1" step="1" data-detect-hex="${hex}" data-role="magPParr" style="width:48px; padding:1px 3px;"></label>
                <label title="extra rotation off the sector mid-axis">angle <input type="number" step="1" data-detect-hex="${hex}" data-role="magAngleParr" style="width:64px; padding:1px 3px;"> deg</label>
                <label>cx <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCxParr" style="width:80px; padding:1px 3px;"> m</label>
                <label>cy <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCyParr" style="width:80px; padding:1px 3px;"> m</label>
                <label>offset <input type="number" step="1" data-detect-hex="${hex}" data-role="magOffsetParr" style="width:64px; padding:1px 3px;"> deg</label>
                <label><input type="radio" name="magDirParr-${hex}" data-detect-hex="${hex}" data-role="magDirParr" value="outward"> outward</label>
                <label><input type="radio" name="magDirParr-${hex}" data-detect-hex="${hex}" data-role="magDirParr" value="inward"> inward</label>
            </span>
            <span data-mag-params="halbach-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>p <input type="number" min="1" step="1" data-detect-hex="${hex}" data-role="magP" style="width:48px; padding:1px 3px;"></label>
                <label>offset <input type="number" step="1" data-detect-hex="${hex}" data-role="magOffset" style="width:64px; padding:1px 3px;"> deg</label>
                <label>cx <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCx2" style="width:80px; padding:1px 3px;"> m</label>
                <label>cy <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCy2" style="width:80px; padding:1px 3px;"> m</label>
            </span>
            <span data-mag-params="polar-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>p <input type="number" min="1" step="1" data-detect-hex="${hex}" data-role="magP3" style="width:48px; padding:1px 3px;"></label>
                <label title="Kn = Fn/Rm (Kano 2025 §3.2). 1.6 ≈ sinusoidal gap density">Kn <input type="number" min="0.1" step="0.1" data-detect-hex="${hex}" data-role="magKn" style="width:60px; padding:1px 3px;"></label>
                <label>offset <input type="number" step="1" data-detect-hex="${hex}" data-role="magOffset3" style="width:64px; padding:1px 3px;"> deg</label>
                <label>cx <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCx3" style="width:80px; padding:1px 3px;"> m</label>
                <label>cy <input type="number" step="0.001" data-detect-hex="${hex}" data-role="magCy3" style="width:80px; padding:1px 3px;"> m</label>
            </span>
            <span data-mag-params="custom-${hex}" style="display:none; gap:4px; align-items:center;">
                <label>Mx <input type="text" data-detect-hex="${hex}" data-role="magMx" style="width:160px; padding:1px 3px;" placeholder="Hc * cos(2*theta)"></label>
                <label>My <input type="text" data-detect-hex="${hex}" data-role="magMy" style="width:160px; padding:1px 3px;" placeholder="Hc * sin(2*theta)"></label>
            </span>
          </div>
        `;
        grid.appendChild(item);
        // Apply persisted state to the freshly-built controls
        item.querySelector('select[data-role="kind"]').value = assign.kind;
        item.querySelector('select[data-role="coilGroup"]').value = assign.coilGroup || 'A';
        item.querySelectorAll('input[data-role="coilSign"]').forEach(r => {
            r.checked = (r.value === (assign.coilSign || '+'));
        });
        const extras = item.querySelector(`span[data-coil-extras="${hex}"]`);
        if (extras) extras.style.display = (assign.kind === 'Coil') ? 'inline-flex' : 'none';
        // Phase D.7: hydrate the magnetization sub-row visibility and
        // input values from the current state.
        const magRow = item.querySelector(`[data-magnet-row="${hex}"]`);
        if (magRow) magRow.style.display = isMagnet ? 'flex' : 'none';
        applyMagnetizationStateToControls(item, hex, assign.magnetization);
        // Phase H: paint the per-chip vector preview on initial render.
        if (isMagnet) {
            const svgPrev = item.querySelector(`svg.mag-preview[data-detect-hex="${hex}"]`);
            if (svgPrev) renderMagnetizationPreview(svgPrev, assign.magnetization);
        }
    });
    // Wire kind / coil controls (Phase D.4)
    // Wire all per-chip controls to update AppState.detectAssign and
    // regenerate the YAML preview.
    grid.querySelectorAll('select[data-role="kind"]').forEach(el => {
        el.addEventListener('change', () => {
            const hex = el.dataset.detectHex;
            const assign = AppState.detectAssign[hex];
            assign.kind = el.value;
            const extras = grid.querySelector(`span[data-coil-extras="${hex}"]`);
            if (extras) extras.style.display = (el.value === 'Coil') ? 'inline-flex' : 'none';
            // Phase D.7: surface / hide the magnetization sub-row and
            // seed defaults from the library preset on first transition
            // to a magnet kind.
            const libProps = presetsAll[assign.kind] || materialsAll[assign.kind] || null;
            const becomingMagnet = (assign.kind !== 'none' && assign.kind !== 'Coil'
                                    && isMagnetMaterial(libProps));
            const magRow = grid.querySelector(`[data-magnet-row="${hex}"]`);
            if (magRow) magRow.style.display = becomingMagnet ? 'flex' : 'none';
            if (becomingMagnet) {
                seedMagnetizationFromLibrary(assign, libProps);
                // Phase G: also reseed orientation_offset to the
                // cardinal-aligned default if the chip is sitting on a
                // sector pattern and the user hasn't taken control yet.
                reseedOffsetIfAuto(assign);
                applyMagnetizationStateToControls(magRow.parentElement, hex, assign.magnetization);
            }
            regenerateDetectYamlPreview();
        });
    });
    grid.querySelectorAll('select[data-role="coilGroup"]').forEach(el => {
        el.addEventListener('change', () => {
            AppState.detectAssign[el.dataset.detectHex].coilGroup = el.value;
            regenerateDetectYamlPreview();
        });
    });
    grid.querySelectorAll('input[data-role="coilSign"]').forEach(el => {
        el.addEventListener('change', () => {
            if (!el.checked) return;
            AppState.detectAssign[el.dataset.detectHex].coilSign = el.value;
            regenerateDetectYamlPreview();
        });
    });
    // Phase D.7: wire the per-chip magnetization controls.
    bindMagnetizationControls(grid);
}

// Phase D.7: visibility of the per-pattern parameter group within a
// chip's magnetization row. Called on render and whenever the user
// changes the pattern dropdown.
function showMagPatternParams(item, hex, pattern) {
    const map = {
        parallel:           `parallel-${hex}`,
        radial:             `radial-${hex}`,
        tangential:         `radial-${hex}`,   // shares cx/cy + direction with radial
        halbach_continuous: `halbach-${hex}`,
        polar_anisotropy:   `polar-${hex}`,
        radial_array:       `radialArray-${hex}`,
        parallel_array:     `parallelArray-${hex}`,
        custom:             `custom-${hex}`,
    };
    const target = map[pattern];
    item.querySelectorAll('[data-mag-params]').forEach(el => {
        el.style.display = (el.dataset.magParams === target) ? 'inline-flex' : 'none';
    });
}

// Phase D.7: push the current magnetization sub-state into the visible
// inputs. Called after a render so freshly-created controls show the
// persisted values.
function applyMagnetizationStateToControls(item, hex, m) {
    if (!item || !m) return;
    showMagPatternParams(item, hex, m.pattern);
    const setVal = (role, v) => {
        const el = item.querySelector(`[data-role="${role}"][data-detect-hex="${hex}"]`);
        if (el) el.value = (v == null) ? '' : v;
    };
    const setSel = (role, v) => {
        const el = item.querySelector(`[data-role="${role}"][data-detect-hex="${hex}"]`);
        if (el) el.value = v;
    };
    setSel('magPattern', m.pattern);
    setVal('magAngle', m.angle);
    setVal('magCx', m.cx);
    setVal('magCy', m.cy);
    setVal('magP',  m.p);
    setVal('magOffset', m.orientation_offset);
    setVal('magCx2', m.cx);
    setVal('magCy2', m.cy);
    setVal('magP3', m.p);
    setVal('magKn', m.Kn);
    setVal('magOffset3', m.orientation_offset);
    setVal('magCx3', m.cx);
    setVal('magCy3', m.cy);
    setVal('magMx', m.Mx);
    setVal('magMy', m.My);
    // Phase E.4: direction radios (shared radial/tangential) + array
    // patterns.
    item.querySelectorAll(`input[data-role="magDir"][data-detect-hex="${hex}"]`).forEach(r => {
        r.checked = (r.value === (m.direction || 'outward'));
    });
    item.querySelectorAll(`input[data-role="magDirArr"][data-detect-hex="${hex}"]`).forEach(r => {
        r.checked = (r.value === (m.direction || 'outward'));
    });
    item.querySelectorAll(`input[data-role="magDirParr"][data-detect-hex="${hex}"]`).forEach(r => {
        r.checked = (r.value === (m.direction || 'outward'));
    });
    setVal('magPArr', m.p);
    setVal('magCxArr', m.cx);
    setVal('magCyArr', m.cy);
    setVal('magOffsetArr', m.orientation_offset);
    setVal('magPParr', m.p);
    setVal('magAngleParr', m.angle);
    setVal('magCxParr', m.cx);
    setVal('magCyParr', m.cy);
    setVal('magOffsetParr', m.orientation_offset);
}

// Phase D.7: wire all the magnetization controls in the chip grid.
// One listener per data-role; updates the per-chip magnetization
// sub-state and re-renders the YAML preview.
function bindMagnetizationControls(grid) {
    const updateNum = (el, key) => {
        const hex = el.dataset.detectHex;
        const a = AppState.detectAssign[hex];
        if (!a || !a.magnetization) return;
        const v = Number(el.value);
        a.magnetization[key] = Number.isFinite(v) ? v : 0;
        regenerateDetectYamlPreview();
    };
    // Phase G: dedicated p handler. After updating p, reseed the
    // cardinal-aligned offset default so dragging p around keeps the
    // sector boundaries cardinal-aligned until the user takes
    // explicit control of the offset.
    const updateP = (el) => {
        const hex = el.dataset.detectHex;
        const a = AppState.detectAssign[hex];
        if (!a || !a.magnetization) return;
        const v = Number(el.value);
        a.magnetization.p = Number.isFinite(v) ? Math.max(1, Math.round(v)) : 1;
        reseedOffsetIfAuto(a);
        const item = el.closest('[data-magnet-row]') ?
            el.closest('[data-magnet-row]').parentElement : el.parentElement;
        applyMagnetizationStateToControls(item, hex, a.magnetization);
        regenerateDetectYamlPreview();
    };
    // Phase G: dedicated offset handler. The very act of the user
    // typing in any of the offset inputs sets _offset_explicit so
    // subsequent pattern / p edits don't clobber their value.
    const updateOffset = (el) => {
        const hex = el.dataset.detectHex;
        const a = AppState.detectAssign[hex];
        if (!a || !a.magnetization) return;
        const v = Number(el.value);
        a.magnetization.orientation_offset = Number.isFinite(v) ? v : 0;
        a.magnetization._offset_explicit = true;
        regenerateDetectYamlPreview();
    };
    const updateStr = (el, key) => {
        const hex = el.dataset.detectHex;
        const a = AppState.detectAssign[hex];
        if (!a || !a.magnetization) return;
        a.magnetization[key] = el.value;
        regenerateDetectYamlPreview();
    };
    grid.querySelectorAll('select[data-role="magPattern"]').forEach(el => {
        el.addEventListener('change', () => {
            const hex = el.dataset.detectHex;
            const a = AppState.detectAssign[hex];
            if (!a) return;
            a.magnetization.pattern = el.value;
            // Phase G: switching INTO a sector pattern reseeds the
            // orientation_offset to the cardinal default unless the
            // user has explicitly typed a value.
            reseedOffsetIfAuto(a);
            const item = el.closest('[data-magnet-row]') ?
                el.closest('[data-magnet-row]').parentElement : el.parentElement;
            showMagPatternParams(item, hex, el.value);
            applyMagnetizationStateToControls(item, hex, a.magnetization);
            regenerateDetectYamlPreview();
        });
    });
    // angle (parallel)
    grid.querySelectorAll('input[data-role="magAngle"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'angle')));
    // cx, cy (radial / tangential — shared inputs)
    grid.querySelectorAll('input[data-role="magCx"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cx')));
    grid.querySelectorAll('input[data-role="magCy"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cy')));
    // halbach: p / offset / cx / cy
    grid.querySelectorAll('input[data-role="magP"]').forEach(el =>
        el.addEventListener('input', () => updateP(el)));
    grid.querySelectorAll('input[data-role="magOffset"]').forEach(el =>
        el.addEventListener('input', () => updateOffset(el)));
    grid.querySelectorAll('input[data-role="magCx2"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cx')));
    grid.querySelectorAll('input[data-role="magCy2"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cy')));
    // polar_anisotropy: p / Kn / offset / cx / cy
    grid.querySelectorAll('input[data-role="magP3"]').forEach(el =>
        el.addEventListener('input', () => updateP(el)));
    grid.querySelectorAll('input[data-role="magKn"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'Kn')));
    grid.querySelectorAll('input[data-role="magOffset3"]').forEach(el =>
        el.addEventListener('input', () => updateOffset(el)));
    grid.querySelectorAll('input[data-role="magCx3"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cx')));
    grid.querySelectorAll('input[data-role="magCy3"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cy')));
    // custom expressions
    grid.querySelectorAll('input[data-role="magMx"]').forEach(el =>
        el.addEventListener('input', () => updateStr(el, 'Mx')));
    grid.querySelectorAll('input[data-role="magMy"]').forEach(el =>
        el.addEventListener('input', () => updateStr(el, 'My')));
    // Phase E.4: direction radios (radial / tangential share `magDir`,
    // array patterns each have their own role since they're separate
    // visible groups but write to the same state field).
    const wireDirRadios = (role) => {
        grid.querySelectorAll(`input[data-role="${role}"]`).forEach(el => {
            el.addEventListener('change', () => {
                if (!el.checked) return;
                const a = AppState.detectAssign[el.dataset.detectHex];
                if (!a || !a.magnetization) return;
                a.magnetization.direction = el.value;
                regenerateDetectYamlPreview();
            });
        });
    };
    wireDirRadios('magDir');
    wireDirRadios('magDirArr');
    wireDirRadios('magDirParr');
    // Phase E.4: array-pattern-specific param inputs. They share the
    // same magnetization state fields as the polar / halbach controls
    // (p, cx, cy, orientation_offset, angle) so multiple inputs can
    // bind to the same key — that's intentional, the UI just exposes
    // the parameter under whichever pattern is currently visible.
    grid.querySelectorAll('input[data-role="magPArr"]').forEach(el =>
        el.addEventListener('input', () => updateP(el)));
    grid.querySelectorAll('input[data-role="magCxArr"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cx')));
    grid.querySelectorAll('input[data-role="magCyArr"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cy')));
    grid.querySelectorAll('input[data-role="magOffsetArr"]').forEach(el =>
        el.addEventListener('input', () => updateOffset(el)));
    grid.querySelectorAll('input[data-role="magPParr"]').forEach(el =>
        el.addEventListener('input', () => updateP(el)));
    grid.querySelectorAll('input[data-role="magAngleParr"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'angle')));
    grid.querySelectorAll('input[data-role="magCxParr"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cx')));
    grid.querySelectorAll('input[data-role="magCyParr"]').forEach(el =>
        el.addEventListener('input', () => updateNum(el, 'cy')));
    grid.querySelectorAll('input[data-role="magOffsetParr"]').forEach(el =>
        el.addEventListener('input', () => updateOffset(el)));
}

// Phase D.7: initial magnetization sub-state for a chip. Parameters are
// kept so toggling pattern doesn't lose values. Defaults follow
// Kano 2025 §3.2 (Kn=1.6 → near-sinusoidal gap density, THD<1%).
function defaultMagnetizationState() {
    return {
        pattern: 'parallel',
        angle: 0,           // [deg]
        p: 4,               // Phase J: number of poles (4-pole machine default)
        Kn: 1.6,            // = R_pc / Rm
        cx: 0,              // [m]
        cy: 0,              // [m]
        orientation_offset: 0,  // [deg] -- auto-seeded to -180/p when a
                                //         sector pattern is first picked
        // Phase G: false until the user types in the offset input directly.
        // When false, switching pattern / editing p re-applies the
        // cardinal-aligned -180/(2p) default. Once the user has set an
        // explicit offset we stop overwriting it.
        _offset_explicit: false,
        direction: 'outward',
        Mx: '',             // tinyexpr expression
        My: '',
    };
}

// Phase G: patterns whose magnetisation structure is defined in terms
// of pole sectors. For these the natural cardinal-aligned default is
// orientation_offset = -180°/(2p) (sector 0 centred on θ=0).
const SECTOR_PATTERNS = new Set([
    'halbach_continuous', 'polar_anisotropy', 'radial_array', 'parallel_array',
]);

// Phase G: recompute orientation_offset to its cardinal-aligned default
// for the current pattern + p, unless the user has explicitly set the
// offset (_offset_explicit = true). Called from the pattern dropdown
// change handler and from every p input handler.
function reseedOffsetIfAuto(assign) {
    const m = assign && assign.magnetization;
    if (!m || m._offset_explicit) return;
    if (!SECTOR_PATTERNS.has(m.pattern)) return;
    // Phase J: p = poles; sector 0 centred on θ=0 wants -180/p.
    const p = Math.max(1, Math.round(Number(m.p) || 1));
    m.orientation_offset = -180 / p;
}

// Phase D.7: classify whether a library entry (preset or material) is a
// permanent magnet, i.e., whether the per-chip magnetization UI should
// be shown when this entry is the kind selection. Checks:
//   1. an explicit magnetization block with Br or Hc
//   2. a B-H array containing a non-zero B at H=0 (any format)
function isMagnetMaterial(props) {
    if (!props || typeof props !== 'object') return false;
    const m = props.magnetization;
    if (m && typeof m === 'object' && (m.Br != null || m.Hc != null)) return true;
    const bh = props['B-H'];
    if (Array.isArray(bh) && bh.length === 2 && Array.isArray(bh[0]) && Array.isArray(bh[1])) {
        const H = bh[0], B = bh[1];
        for (let i = 0; i < H.length && i < B.length; i++) {
            if (Math.abs(Number(H[i]) || 0) < 1e-6 && Math.abs(Number(B[i]) || 0) > 1e-9) {
                return true;
            }
        }
    }
    return false;
}

// Phase D.7: physical-coordinate context from Polar Preprocess (if any
// detection has been run) used to suggest cx/cy and R_pc defaults.
// Returns { dx_per_px, cx_phys, cy_phys, Rm_phys, polar_origin } or
// nulls when the modal hasn't been run.
function detectMagnetizationContext() {
    const pp = AppState.polarPreprocess;
    if (!pp || !pp.detection) {
        return { dx_per_px: 0, cx_phys: 0, cy_phys: 0, Rm_phys: 0, polar_origin: false };
    }
    const cur = pp.current || {};
    const det = pp.detection || {};
    const r_outer_px = (cur.r_outer_px > 0) ? cur.r_outer_px : (det.r_outer_px || 0);
    const r_outer_m  = (cur.r_outer_physical > 0) ? cur.r_outer_physical : 1.0;
    const dx_per_px = (r_outer_px > 0) ? (r_outer_m / r_outer_px) : 0;
    // For polar save target the warp re-centres on the rotor axis so
    // (cx, cy) = (0, 0) is the physically correct default. For
    // cartesian save the rotor centre stays at its image-px position
    // multiplied by the auto-derived dx.
    const polar_origin = (cur.save_as === 'polar');
    const cx_phys = polar_origin ? 0 : (cur.center_x || det.center_x || 0) * dx_per_px;
    const cy_phys = polar_origin ? 0 : (cur.center_y || det.center_y || 0) * dx_per_px;
    return { dx_per_px, cx_phys, cy_phys, Rm_phys: r_outer_m, polar_origin };
}

// Phase H: evaluate the magnetisation direction at a normalised polar
// position (theta_rad, r_norm) in the magnet's reference frame. The
// preview SVG works on a unit-circle schematic so cx/cy are implicitly
// (0,0) and r_norm is normalised to the magnet outer radius. Returns
// { angle_rad, sign } — `sign` is the alternating-pole sign factor and
// the renderer translates it into either an arrow flip (180°) or a
// colour change so the user can visually trace the N/S layout.
function evalMagnetizationDirection(m, theta_rad, r_norm) {
    if (!m || !m.pattern) return { angle_rad: 0, sign: 1 };
    const orient_rad = (Number(m.orientation_offset) || 0) * Math.PI / 180;
    const dirSign = (m.direction === 'inward') ? -1 : 1;
    if (m.pattern === 'parallel') {
        return { angle_rad: (Number(m.angle) || 0) * Math.PI / 180, sign: 1 };
    }
    if (m.pattern === 'radial') {
        return { angle_rad: theta_rad, sign: dirSign };
    }
    if (m.pattern === 'tangential') {
        return { angle_rad: theta_rad + Math.PI / 2, sign: dirSign };
    }
    if (m.pattern === 'halbach_continuous') {
        // Phase J: p = poles; the rotation rate is p/2.
        const p = Math.max(1, Math.round(Number(m.p) || 1));
        return { angle_rad: (p / 2) * (theta_rad - orient_rad), sign: 1 };
    }
    if (m.pattern === 'radial_array' || m.pattern === 'parallel_array') {
        // Phase J: p sectors of 2π/p each.
        const p = Math.max(1, Math.round(Number(m.p) || 1));
        const twopi = 2 * Math.PI;
        let theta_pos = theta_rad - orient_rad;
        theta_pos = ((theta_pos % twopi) + twopi) % twopi;
        const pole_span = twopi / p;
        let k_pole = Math.floor(theta_pos / pole_span);
        if (k_pole < 0) k_pole = 0;
        if (k_pole >= p) k_pole = p - 1;
        const sign = ((k_pole % 2) === 0) ? dirSign : -dirSign;
        if (m.pattern === 'radial_array') {
            return { angle_rad: theta_rad, sign };
        }
        const angle_mid = (k_pole + 0.5) * pole_span + orient_rad
                        + (Number(m.angle) || 0) * Math.PI / 180;
        return { angle_rad: angle_mid, sign };
    }
    if (m.pattern === 'polar_anisotropy') {
        // Phase J: p OJ centres on the pitch circle.
        const p = Math.max(1, Math.round(Number(m.p) || 1));
        const Kn = (Number(m.Kn) > 0) ? Number(m.Kn) : 1.6;
        const x = r_norm * Math.cos(theta_rad);
        const y = r_norm * Math.sin(theta_rad);
        let Bx = 0, By = 0;
        for (let k = 0; k < p; k++) {
            const theta_k = 2 * Math.PI * k / p + orient_rad;
            const sgn = (k % 2 === 0) ? 1 : -1;
            const dx_w = x - Kn * Math.cos(theta_k);
            const dy_w = y - Kn * Math.sin(theta_k);
            const r2 = dx_w * dx_w + dy_w * dy_w;
            if (r2 < 1e-20) continue;
            Bx += sgn * (-dy_w) / r2;
            By += sgn * dx_w / r2;
        }
        const norm = Math.hypot(Bx, By);
        return {
            angle_rad: norm > 1e-12 ? Math.atan2(By, Bx) : 0,
            sign: 1,
        };
    }
    // custom: cannot eval without a tinyexpr runtime in browser — flat
    // schematic so the preview at least shows something.
    return { angle_rad: 0, sign: 1 };
}

// Phase H: draw a compact schematic of the magnetisation vector field
// into the chip's `<svg class="mag-preview">`. Red arrows mark sign=+1
// (N-out / outward / pole 0 + direction), blue arrows sign=-1
// (S-out / inward / alternating poles). Sector boundary dotted lines
// are drawn for the discrete array patterns so the user can read off
// the pole geometry at a glance.
function renderMagnetizationPreview(svg, m) {
    if (!svg) return;
    const W = 80, H = 80;
    const r_view = 1.18;
    const scale = W / (2 * r_view);
    // SVG y-axis is down; mathematical θ=90° should appear UP. The
    // outer <g> applies scale(1, -1) on y so cos/sin renders match the
    // standard polar convention.
    const parts = [];
    parts.push(`<g transform="translate(${W/2},${H/2}) scale(${scale},${-scale})">`);
    parts.push('<circle cx="0" cy="0" r="1" fill="#fafafa" stroke="#aaa" stroke-width="0.02"/>');
    if (!m || !m.pattern) {
        parts.push('</g>');
        svg.innerHTML = parts.join('');
        return;
    }
    // Sector boundaries for the discrete patterns. Phase J: p sectors.
    if (m.pattern === 'radial_array' || m.pattern === 'parallel_array') {
        const orient_rad = (Number(m.orientation_offset) || 0) * Math.PI / 180;
        const p = Math.max(1, Math.round(Number(m.p) || 1));
        for (let k = 0; k < p; k++) {
            const theta_k = 2 * Math.PI * k / p + orient_rad;
            const x = 1.05 * Math.cos(theta_k);
            const y = 1.05 * Math.sin(theta_k);
            parts.push(`<line x1="0" y1="0" x2="${x.toFixed(3)}" y2="${y.toFixed(3)}" stroke="#c8c8c8" stroke-width="0.015" stroke-dasharray="0.05,0.04"/>`);
        }
    }
    // For polar_anisotropy, show the p OJ centres. Phase J.
    if (m.pattern === 'polar_anisotropy') {
        const orient_rad = (Number(m.orientation_offset) || 0) * Math.PI / 180;
        const p = Math.max(1, Math.round(Number(m.p) || 1));
        const Kn = (Number(m.Kn) > 0) ? Number(m.Kn) : 1.6;
        for (let k = 0; k < p; k++) {
            const theta_k = 2 * Math.PI * k / p + orient_rad;
            const cx = Kn * Math.cos(theta_k);
            const cy = Kn * Math.sin(theta_k);
            if (Math.hypot(cx, cy) < r_view * 1.5) {
                const fill = (k % 2 === 0) ? '#d63333' : '#3366cc';
                parts.push(`<circle cx="${cx.toFixed(3)}" cy="${cy.toFixed(3)}" r="0.05" fill="${fill}"/>`);
            }
        }
    }
    // Sample arrows around the unit circle.
    const r_sample = 0.62;
    const arrowLen = 0.24;
    const useDense = (m.pattern === 'halbach_continuous' || m.pattern === 'polar_anisotropy');
    const nSamples = useDense ? 16 : 12;
    for (let i = 0; i < nSamples; i++) {
        const theta_deg = i * 360 / nSamples;
        const theta_rad = theta_deg * Math.PI / 180;
        const px = r_sample * Math.cos(theta_rad);
        const py = r_sample * Math.sin(theta_rad);
        const res = evalMagnetizationDirection(m, theta_rad, r_sample);
        const angle = res.angle_rad + (res.sign < 0 ? Math.PI : 0);
        const x2 = px + arrowLen * Math.cos(angle);
        const y2 = py + arrowLen * Math.sin(angle);
        const colour = (res.sign >= 0) ? '#d63333' : '#3366cc';
        parts.push(`<line x1="${px.toFixed(3)}" y1="${py.toFixed(3)}" x2="${x2.toFixed(3)}" y2="${y2.toFixed(3)}" stroke="${colour}" stroke-width="0.035"/>`);
        // Arrowhead — small triangle at the tip.
        const ah = 0.08;
        const cosA = Math.cos(angle), sinA = Math.sin(angle);
        const tailX = x2 - ah * cosA;
        const tailY = y2 - ah * sinA;
        const perpX = -sinA * (ah * 0.5);
        const perpY =  cosA * (ah * 0.5);
        parts.push(`<polygon points="${x2.toFixed(3)},${y2.toFixed(3)} ${(tailX + perpX).toFixed(3)},${(tailY + perpY).toFixed(3)} ${(tailX - perpX).toFixed(3)},${(tailY - perpY).toFixed(3)}" fill="${colour}"/>`);
    }
    parts.push('</g>');
    svg.innerHTML = parts.join('');
}

// Phase D.7: emit a magnetization: block at the given indent for the
// per-chip state. Pattern-specific fields are emitted; Kn is converted
// to R_pc using the Polar Preprocess context (Rm). Inline comments
// reference the Kano 2025 §3.2 framing where relevant so users can
// trace the parameter meaning back to the paper.
function appendMagnetizationBlock(lines, indent, m) {
    if (!m || !m.pattern) return;
    lines.push(`${indent}magnetization:`);
    const sub = `${indent}  `;
    lines.push(`${sub}pattern: ${m.pattern}`);
    const dir = (m.direction === 'inward') ? 'inward' : 'outward';
    if (m.pattern === 'parallel') {
        lines.push(`${sub}angle: ${Number(m.angle) || 0}`);
    } else if (m.pattern === 'radial' || m.pattern === 'tangential') {
        lines.push(`${sub}cx: ${Number(m.cx) || 0}`);
        lines.push(`${sub}cy: ${Number(m.cy) || 0}`);
        // Phase E.4: emit direction only when non-default so backward-
        // compatible YAML stays minimal for the common outward case.
        if (dir !== 'outward') lines.push(`${sub}direction: ${dir}`);
    } else if (m.pattern === 'halbach_continuous') {
        lines.push(`${sub}p: ${Math.max(1, Math.round(Number(m.p) || 1))}`);
        lines.push(`${sub}cx: ${Number(m.cx) || 0}`);
        lines.push(`${sub}cy: ${Number(m.cy) || 0}`);
        if (Number(m.orientation_offset) !== 0) {
            lines.push(`${sub}orientation_offset: ${Number(m.orientation_offset)}`);
        }
    } else if (m.pattern === 'polar_anisotropy') {
        const ctx = detectMagnetizationContext();
        const Rm = ctx.Rm_phys || 1.0;
        const Kn = (Number(m.Kn) > 0) ? Number(m.Kn) : 1.6;
        const R_pc = Kn * Rm;
        lines.push(`${sub}p: ${Math.max(1, Math.round(Number(m.p) || 1))}`);
        lines.push(`${sub}# R_pc sets where the polar-anisotropy pattern is anchored; Kn=${Kn} is a starting point.`);
        lines.push(`${sub}R_pc: ${R_pc}`);
        lines.push(`${sub}cx: ${Number(m.cx) || 0}`);
        lines.push(`${sub}cy: ${Number(m.cy) || 0}`);
        if (Number(m.orientation_offset) !== 0) {
            lines.push(`${sub}orientation_offset: ${Number(m.orientation_offset)}`);
        }
    } else if (m.pattern === 'radial_array') {
        lines.push(`${sub}p: ${Math.max(1, Math.round(Number(m.p) || 1))}`);
        lines.push(`${sub}cx: ${Number(m.cx) || 0}`);
        lines.push(`${sub}cy: ${Number(m.cy) || 0}`);
        lines.push(`${sub}direction: ${dir}    # pole 0 is ${dir}-pointing; alternates per sector`);
        if (Number(m.orientation_offset) !== 0) {
            lines.push(`${sub}orientation_offset: ${Number(m.orientation_offset)}`);
        }
    } else if (m.pattern === 'parallel_array') {
        lines.push(`${sub}p: ${Math.max(1, Math.round(Number(m.p) || 1))}`);
        lines.push(`${sub}angle: ${Number(m.angle) || 0}    # extra rotation off sector mid-axis`);
        lines.push(`${sub}cx: ${Number(m.cx) || 0}`);
        lines.push(`${sub}cy: ${Number(m.cy) || 0}`);
        lines.push(`${sub}direction: ${dir}    # pole 0 sign; alternates per sector`);
        if (Number(m.orientation_offset) !== 0) {
            lines.push(`${sub}orientation_offset: ${Number(m.orientation_offset)}`);
        }
    } else if (m.pattern === 'custom') {
        if (m.Mx) lines.push(`${sub}Mx: "${String(m.Mx).replace(/"/g, '\\"')}"`);
        if (m.My) lines.push(`${sub}My: "${String(m.My).replace(/"/g, '\\"')}"`);
    }
}

// Phase D.7: pull library magnetization defaults (if any) into the
// per-chip state when a magnet preset is freshly picked. Non-destructive:
// keeps any field the user already edited.
function seedMagnetizationFromLibrary(assign, libProps) {
    const m = assign.magnetization;
    if (!m) return;
    // Apply Polar Preprocess context defaults first so library values can
    // still override them (the user's library is the authoritative
    // source, the Polar Preprocess is just a sensible fallback for the
    // rotor centre when the library didn't say).
    const ctx = detectMagnetizationContext();
    if (m.cx === 0) m.cx = ctx.cx_phys;
    if (m.cy === 0) m.cy = ctx.cy_phys;
    const lm = libProps && libProps.magnetization;
    if (!lm) return;
    if (lm.pattern && m.pattern === 'parallel' && lm.pattern !== 'parallel') {
        m.pattern = lm.pattern;
    }
    if (lm.angle != null && m.angle === 0) m.angle = Number(lm.angle) || 0;
    if (lm.p != null && m.p === 4) m.p = Math.max(1, Math.round(Number(lm.p) || 4));
    if (lm.cx != null) m.cx = Number(lm.cx) || 0;
    if (lm.cy != null) m.cy = Number(lm.cy) || 0;
    if (lm.orientation_offset != null && m.orientation_offset === 0) {
        m.orientation_offset = Number(lm.orientation_offset) || 0;
    }
    if (lm.R_pc != null && ctx.Rm_phys > 0) {
        m.Kn = Number(lm.R_pc) / ctx.Rm_phys;
    }
}

// Phase D.4: regenerate the YAML preview based on the current per-chip
// assignments + the picked library presets. The result is stashed back
// onto AppState.lastDetectResult.generatedYaml so insertMaterialsSection
// uses it instead of the server's stock template.
function regenerateDetectYamlPreview() {
    const yaml = buildDetectYamlFromAssignments();
    AppState.lastDetectResult.generatedYaml = yaml;
    document.getElementById('detectYamlPreview').textContent = yaml;
    // Phase H: refresh every per-chip magnetisation vector preview.
    document.querySelectorAll('svg.mag-preview[data-detect-hex]').forEach(svg => {
        const hex = svg.dataset.detectHex;
        const a = AppState.detectAssign && AppState.detectAssign[hex];
        if (a && a.magnetization) renderMagnetizationPreview(svg, a.magnetization);
    });
}

// Phase D.4: emit material_presets + materials blocks. Library presets
// referenced by any chip are copied in verbatim; the rest of the
// library's content is left untouched. Coil chips emit
// `jz: ${sign}$J_Coil_${group}` plus a one-line comment so users can
// trace each entry back to its group/sign decision.
function buildDetectYamlFromAssignments() {
    const result = AppState.lastDetectResult;
    if (!result) return '';
    const colors = result.colors || [];
    const assign = AppState.detectAssign || {};
    const presetsAll = AppState.detectLibraryPresets || {};
    const materialsAll = AppState.detectLibraryMaterials || {};

    const usedPresetNames = new Set();
    const lines = [];

    // Preserve the existing header lines from the server's template
    // (coordinate_system + comments) so users still see the framing.
    const stockYaml = result.yamlTemplate || '';
    const headerLines = [];
    for (const raw of stockYaml.split('\n')) {
        if (raw.startsWith('materials:')) break;
        headerLines.push(raw);
    }
    lines.push(...headerLines);

    // Phase F.3: scan the assignments for every Coil group letter used.
    // Each group is paired with a J_Coil_<letter> entry in the
    // variables: block so the $J_Coil_<letter> tokens emitted under
    // materials:.jz actually resolve. insertMaterialsSection merges this
    // into the editor doc's existing variables non-destructively.
    const coilGroupsUsed = new Set();
    for (const c of colors) {
        const hex = `#${c.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
        const a = assign[hex];
        if (a && a.kind === 'Coil') coilGroupsUsed.add(a.coilGroup || 'A');
    }
    if (coilGroupsUsed.size > 0) {
        lines.push('variables:');
        const sorted = Array.from(coilGroupsUsed).sort();
        for (const g of sorted) {
            lines.push(`  J_Coil_${g}: 1.0e6    # Coil-${g} current density [A/m^2] — adjust to match your drive`);
        }
        lines.push('');
    }

    // Phase E.3: material_presets are NOT emitted inline. server.js
    // already merges AppState.selectedLibrary into the config YAML at
    // analysis launch via mergeLibraryIntoConfig (server.js ~L189), so
    // copying the preset definitions here would just produce stale
    // duplicates that drift from the library file the user edits in
    // the Library Manager. We emit a one-line note instead pointing
    // at which presets the materials: block depends on, and the
    // active library that resolves them.
    const referencedPresets = [];
    for (const c of colors) {
        const hex = `#${c.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
        const kind = assign[hex] && assign[hex].kind;
        if (kind && kind !== 'none' && kind !== 'Coil' && presetsAll[kind]) {
            if (!usedPresetNames.has(kind)) {
                usedPresetNames.add(kind);
                referencedPresets.push(kind);
            }
        }
    }
    if (referencedPresets.length > 0) {
        const lib = AppState.detectLibraryName || '(none)';
        lines.push(`# Uses preset(s) ${referencedPresets.join(', ')} from active library "${lib}".`);
        lines.push('# Change the preset or library to change the assigned material properties.');
        lines.push('');
    }

    // Now the materials: block. For each detected colour we either copy
    // the library material verbatim (kind in materialsAll) with rgb
    // overridden to the detected value, reference a preset (kind in
    // presetsAll), emit a Coil expression, or fall back to the default.
    lines.push('materials:');
    for (const c of colors) {
        const [r, g, b] = c.rgb;
        const hex = `#${c.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
        const a = assign[hex] || { kind: 'none' };
        const ratio = (c.ratio * 100).toFixed(1);
        if (a.kind === 'Coil') {
            const grp = a.coilGroup || 'A';
            const sign = (a.coilSign === '-') ? '-' : '';
            const dir = (a.coilSign === '-') ? '-' : '+';
            lines.push(`  coil_${grp}_${sign === '-' ? 'neg' : 'pos'}_${hex.slice(1)}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            lines.push(`    mu_r: 1.0       # coverage: ${ratio}%`);
            lines.push(`    jz: ${sign}$J_Coil_${grp}    # Coil-${grp}, J direction: ${dir}Z`);
            if (c.antialias === true) lines.push(`    anti_aliasing: true`);
        } else if (a.kind && a.kind !== 'none' && presetsAll[a.kind]) {
            lines.push(`  material_${hex.slice(1)}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            lines.push(`    preset: ${a.kind}    # coverage: ${ratio}%`);
            if (c.antialias === true) lines.push(`    anti_aliasing: true`);
            // Phase D.7: append per-chip magnetization block when the
            // selected preset is a magnet. This overrides the preset's
            // own magnetization direction with the user's choice while
            // keeping Br / mu_r etc. inherited from the preset.
            if (isMagnetMaterial(presetsAll[a.kind])) {
                appendMagnetizationBlock(lines, '    ', a.magnetization);
            }
        } else if (a.kind && a.kind !== 'none' && materialsAll[a.kind]) {
            // Library material picked directly (no preset): copy props
            // verbatim but override rgb to the detected colour.
            lines.push(`  material_${hex.slice(1)}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            const props = materialsAll[a.kind] || {};
            const skipMag = isMagnetMaterial(props);
            for (const k of Object.keys(props)) {
                if (k === 'rgb') continue;
                // Phase D.7: skip the library's magnetization block; we
                // emit the user-edited one explicitly below.
                if (skipMag && k === 'magnetization') continue;
                const sub = jsyaml.dump({ [k]: props[k] }, { indent: 2, lineWidth: -1 }).replace(/\n$/, '');
                for (const ln of sub.split('\n')) lines.push('    ' + ln);
            }
            if (c.antialias === true) lines.push(`    anti_aliasing: true`);
            if (skipMag) appendMagnetizationBlock(lines, '    ', a.magnetization);
        } else {
            lines.push(`  material_${hex.slice(1)}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            lines.push(`    mu_r: 1.0       # Set permeability  (coverage: ${ratio}%)`);
            lines.push(`    jz: 0.0`);
            if (c.antialias === true) lines.push(`    anti_aliasing: true`);
        }
    }

    // Auto-emit one flux_linkage entry for every Coil group. A full model
    // normally has both + and - cross-sections; an antiperiodic one-pole
    // model may contain only one. The solver evaluates the signed area mean
    // of whichever side is present.
    const coilByGroup = {};   // { 'A': { pos: '#hex', neg: '#hex' }, ... }
    for (const c of colors) {
        const hex = `#${c.rgb.map(v => v.toString(16).padStart(2, '0')).join('')}`;
        const a = assign[hex];
        if (!a || a.kind !== 'Coil') continue;
        const grp = a.coilGroup || 'A';
        if (!coilByGroup[grp]) coilByGroup[grp] = { pos: null, neg: null };
        if (a.coilSign === '-') coilByGroup[grp].neg = hex;
        else                    coilByGroup[grp].pos = hex;
    }
    const groupsWithCoil = Object.keys(coilByGroup)
        .filter(g => coilByGroup[g].pos || coilByGroup[g].neg)
        .sort();
    if (groupsWithCoil.length > 0) {
        lines.push('');
        lines.push('# Flux linkage: both sides use ⟨Az⟩(+X) - ⟨Az⟩(-X).');
        lines.push('# A single visible side is valid for an antiperiodic half-period model:');
        lines.push('# material_a alone gives +⟨Az⟩; material_b alone gives -⟨Az⟩.');
        lines.push('# Cartesian uses uniform per-cell weight; polar uses the r·dr·dθ');
        lines.push('# Jacobian, so the average is area-weighted in either coord system.');
        lines.push('flux_linkage:');
        for (const g of groupsWithCoil) {
            lines.push(`  - name: Phi_Coil_${g}`);
            if (coilByGroup[g].pos) {
                const posHex = coilByGroup[g].pos.slice(1);
                lines.push(`    material_a: coil_${g}_pos_${posHex}`);
            }
            if (coilByGroup[g].neg) {
                const negHex = coilByGroup[g].neg.slice(1);
                lines.push(`    material_b: coil_${g}_neg_${negHex}`);
            }
        }
    }

    // Preserve the trailing AA-blends comment block from the stock
    // template so the user still has the AA diagnostic info.
    const aaIdx = stockYaml.indexOf('# Anti-aliasing blends detected');
    if (aaIdx >= 0) {
        lines.push('');
        lines.push(stockYaml.slice(aaIdx).replace(/\n$/, ''));
    }
    return lines.join('\n');
}

async function rerunDetect() {
    await detectColors();
}

function closeDetectColorsModal() {
    document.getElementById('detectColorsModal').style.display = 'none';
}

// ============================================================
// Phase I: Magnetization full-image preview modal
// ============================================================
// Opens a side-by-side view of the loaded image with sampled M
// vector arrows overlaid. The arrows come from parsing the current
// editor YAML for materials that carry a magnetization block (preset
// references are resolved through AppState.selectedLibrary the same
// way the WebUI server does at analysis launch, so the preview
// matches what the solver will see).

AppState.magPreview = {
    materials: [],   // resolved magnet entries [{ name, rgb:[r,g,b], magnetization }]
    image: null,     // HTMLImageElement
    pixels: null,    // ImageData
    coordSystem: 'cartesian',
    dx: 1e-3,
    dy: 1e-3,
    rStart: 0,               // polar: annulus inner radius [m]
    rEnd: 0.1,               // polar: annulus outer radius [m]
    thetaRange: 2 * Math.PI, // polar: angular extent [rad]
    rOrientation: 'horizontal',
};

// Evaluate a YAML numeric field that may be a simple expression such as
// "2*pi" (only digits, ., + - * / ( ) and `pi` are allowed).
function evalYamlNumber(v, fallback) {
    if (typeof v === 'number' && isFinite(v)) return v;
    if (typeof v === 'string') {
        const s = v.replace(/\bpi\b/gi, '(' + Math.PI + ')');
        if (/^[0-9eE+\-*/(). ]+$/.test(s)) {
            try {
                const r = Function('"use strict"; return (' + s + ')')();
                if (isFinite(r)) return r;
            } catch (_) { /* fall through */ }
        }
    }
    return fallback;
}

async function openMagnetizationPreviewModal() {
    if (!AppState.uploadedImageFilename) {
        showStatus('solverStatus', 'Select an image first', 'error');
        return;
    }
    if (!AppState.aceEditor) {
        showStatus('solverStatus', 'YAML editor not available', 'error');
        return;
    }
    // Parse YAML
    let doc;
    try {
        doc = jsyaml.load(AppState.aceEditor.getValue()) || {};
    } catch (e) {
        showStatus('solverStatus', `YAML parse error: ${e.message}`, 'error');
        return;
    }
    // Resolve presets via the active library exactly like server.js
    // mergeLibraryIntoConfig does at run time.
    let presets = Object.assign({}, doc.material_presets || {});
    if (AppState.selectedLibrary) {
        try {
            const r = await fetch(`/api/material-libraries/${encodeURIComponent(AppState.selectedLibrary)}?userId=${AppState.userId}`);
            if (r.ok) {
                const lib = jsyaml.load(await r.text()) || {};
                const libPresets = Object.assign({}, lib.material_presets || {});
                for (const [n, p] of Object.entries(lib.materials || {})) {
                    const { rgb: _rgb, ...rest } = (p || {});
                    libPresets[n] = rest;
                }
                presets = Object.assign({}, libPresets, presets);
            }
        } catch (_) { /* best effort */ }
    }
    // Find magnet materials (rgb + magnetization carrying Br/Hc or B-H remanence).
    const magnetMaterials = [];
    const materials = doc.materials || {};
    for (const [name, raw] of Object.entries(materials)) {
        let props = raw || {};
        if (props.preset && presets[props.preset]) {
            // Mirror the solver's mergeMaterialPreset: map-valued keys merge
            // one level deep, so an inline `magnetization: { pattern: ... }`
            // override keeps the preset's Br/Hc instead of dropping them
            // (a shallow merge here made such magnets invisible to the preview).
            const base = presets[props.preset] || {};
            const merged = Object.assign({}, base, props);
            for (const k of Object.keys(props)) {
                const o = props[k], b = base[k];
                if (o && b && typeof o === 'object' && typeof b === 'object' &&
                    !Array.isArray(o) && !Array.isArray(b)) {
                    merged[k] = Object.assign({}, b, o);
                }
            }
            props = merged;
        }
        if (!Array.isArray(props.rgb) || props.rgb.length < 3) continue;
        const m = props.magnetization;
        if (!m || typeof m !== 'object') continue;
        // Preview needs only the DIRECTION field, not the magnitude: accept
        // any magnetization block that carries a pattern even when Br/Hc are
        // unknown (e.g. `preset: NdFeB_N40` whose library isn't loaded in
        // this browser). The unresolved preset is surfaced as a warning so
        // the user knows the RUN still needs the material library selected.
        const hasDirectionInfo = (typeof m.pattern === 'string' && m.pattern !== '') ||
                                 m.Br != null || m.Hc != null;
        if (!hasDirectionInfo && !isMagnetMaterial(props)) continue;
        magnetMaterials.push({
            name,
            rgb: [Number(props.rgb[0]) | 0, Number(props.rgb[1]) | 0, Number(props.rgb[2]) | 0],
            magnetization: m,
            unresolvedPreset: (props.preset && !presets[props.preset]) ? props.preset : null,
        });
    }
    AppState.magPreview.materials = magnetMaterials;
    AppState.magPreview.coordSystem = (doc.coordinate_system === 'polar') ? 'polar' : 'cartesian';
    AppState.magPreview.dx = (doc.mesh && Number(doc.mesh.dx)) || 1e-3;
    AppState.magPreview.dy = (doc.mesh && Number(doc.mesh.dy)) || AppState.magPreview.dx;
    // Polar runs need the physical annulus to evaluate patterns in the true
    // rotor frame (r cosθ, r sinθ) like the solver does. theta_range is often
    // the string "2*pi", so evaluate simple numeric expressions.
    const pd = doc.polar_domain || {};
    AppState.magPreview.rStart = evalYamlNumber(pd.r_start, 0);
    AppState.magPreview.rEnd = evalYamlNumber(pd.r_end, 0.1);
    AppState.magPreview.thetaRange = evalYamlNumber(pd.theta_range, 2 * Math.PI);
    AppState.magPreview.rOrientation = (pd.r_orientation === 'vertical') ? 'vertical' : 'horizontal';
    // Populate sidebar list
    const matList = document.getElementById('magPreviewMaterialList');
    if (magnetMaterials.length === 0) {
        matList.innerHTML = '<div style="color:#856404; background:#fff3cd; padding:6px 8px; border-radius:3px;">No magnet materials found in the current YAML (need rgb + magnetization block).</div>';
    } else {
        matList.innerHTML = magnetMaterials.map(mm => {
            const hex = '#' + mm.rgb.map(v => v.toString(16).padStart(2, '0')).join('');
            const pat = (mm.magnetization && mm.magnetization.pattern) || '?';
            const warn = mm.unresolvedPreset
                ? `<div style="color:#856404; background:#fff3cd; border-radius:3px; padding:2px 6px;
                        font-size:0.75rem; margin:2px 0 4px 20px;">
                        preset "${mm.unresolvedPreset}" not found — direction previewed from the inline
                        pattern; select its material library before running the solver.</div>`
                : '';
            return `<div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                <div style="width:14px; height:14px; background:${hex}; border:1px solid #aaa;"></div>
                <span style="font-family:monospace; font-size:0.78rem;">${hex}</span>
                <span style="color:#495057;">${mm.name}</span>
                <span style="color:#6c757d;">(${pat})</span>
            </div>${warn}`;
        }).join('');
    }
    document.getElementById('magPreviewStatus').textContent =
        `coord_system: ${AppState.magPreview.coordSystem}  ·  ${magnetMaterials.length} magnet material(s)`;
    // Show modal + attach zoom-pan (idempotent)
    document.getElementById('magnetizationPreviewModal').style.display = 'flex';
    attachZoomPan(
        document.getElementById('magPreviewContainer'),
        document.getElementById('magPreviewZoomStage'),
        {
            indicatorEl: document.getElementById('magPreviewZoomIndicator'),
            onScaleChange: () => { /* arrows are in image-coord space; no rescale needed */ },
        }
    );
    // Pixel rulers tracking the zoom-pan transform (idempotent).
    const magRulers = attachPixelRulers({
        topCanvas: document.getElementById('magPreviewRulerTop'),
        leftCanvas: document.getElementById('magPreviewRulerLeft'),
        containerEl: document.getElementById('magPreviewContainer'),
        stageEl: document.getElementById('magPreviewZoomStage'),
        imgEl: document.getElementById('magPreviewImg'),
    });
    // Load image into the preview + into a canvas for pixel sampling
    const imgEl = document.getElementById('magPreviewImg');
    await new Promise((resolve, reject) => {
        imgEl.onload = () => {
            // Mirror onto an off-screen canvas so we can read pixels.
            const canvas = document.createElement('canvas');
            canvas.width = imgEl.naturalWidth;
            canvas.height = imgEl.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(imgEl, 0, 0);
            AppState.magPreview.image = imgEl;
            try {
                AppState.magPreview.pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
            } catch (e) {
                AppState.magPreview.pixels = null;
            }
            const svg = document.getElementById('magPreviewOverlay');
            svg.setAttribute('viewBox', `0 0 ${imgEl.naturalWidth} ${imgEl.naturalHeight}`);
            resolve();
        };
        imgEl.onerror = () => reject(new Error('image load failed'));
        imgEl.src = `/uploads/${AppState.userId}/${AppState.uploadedImageFilename}?t=${Date.now()}`;
    });
    renderMagnetizationFieldOverlay();
    // Rulers need the final layout (image decoded + modal visible).
    if (magRulers) requestAnimationFrame(() => magRulers.redraw());
}

function closeMagnetizationPreviewModal() {
    document.getElementById('magnetizationPreviewModal').style.display = 'none';
    AppState.magPreview.image = null;
    AppState.magPreview.pixels = null;
}

// Phase I: paint sampled M-vector arrows onto the modal's SVG overlay
// based on AppState.magPreview state. Cheap to re-run, so the user
// can tweak grid stride / arrow scale without reloading the image.
function renderMagnetizationFieldOverlay() {
    const pix = AppState.magPreview.pixels;
    const mats = AppState.magPreview.materials || [];
    const svg = document.getElementById('magPreviewOverlay');
    if (!svg) return;
    if (!pix || mats.length === 0) {
        svg.innerHTML = '';
        return;
    }
    const W = pix.width, H = pix.height;
    const data = pix.data;
    const stride = Math.max(2, parseInt(document.getElementById('magPreviewStride').value, 10) || 12);
    const arrowScale = Math.max(0.1, parseFloat(document.getElementById('magPreviewArrowScale').value) || 1.0);
    const arrowLen = stride * 0.6 * arrowScale;
    const isPolar = (AppState.magPreview.coordSystem === 'polar');
    const dx = AppState.magPreview.dx || 1e-3;
    const dy = AppState.magPreview.dy || dx;
    // Index the magnet materials by packed RGB for O(1) per-pixel lookup.
    const matByRgb = new Map();
    for (const mm of mats) {
        const key = (mm.rgb[0] << 16) | (mm.rgb[1] << 8) | mm.rgb[2];
        matByRgb.set(key, mm);
    }
    const parts = [];
    let arrowCount = 0;
    for (let j = Math.floor(stride / 2); j < H; j += stride) {
        for (let i = Math.floor(stride / 2); i < W; i += stride) {
            const p = (j * W + i) * 4;
            const key = (data[p] << 16) | (data[p + 1] << 8) | data[p + 2];
            const mm = matByRgb.get(key);
            if (!mm) continue;
            // Convert image pixel (i, j) to the physical (x, y) in metres
            // that the solver evaluates patterns at.
            // Cartesian: image y is down, analysis y is up (the solver does
            // cv::flip), so flip j here too.
            // Polar: the warped strip's axes are (r, θ); the solver maps
            // image-up to +θ (horizontal r) / +r (vertical r) and evaluates
            // at the true rotor-frame point (r cosθ, r sinθ). Mirror that,
            // and remember θ so the resulting lab-frame vector can be
            // decomposed into strip-local (r̂, θ̂) components for display.
            let x_phys, y_phys, thetaAtPixel = 0;
            if (isPolar) {
                const mp = AppState.magPreview;
                let r_idx, th_idx, nR, nTh;
                if (mp.rOrientation === 'vertical') {
                    r_idx = H - 1 - j; th_idx = i; nR = H; nTh = W;
                } else {
                    r_idx = i; th_idx = H - 1 - j; nR = W; nTh = H;
                }
                const r_phys = mp.rStart + (mp.rEnd - mp.rStart) * (r_idx / Math.max(1, nR - 1));
                thetaAtPixel = mp.thetaRange * (th_idx / Math.max(1, nTh));
                x_phys = r_phys * Math.cos(thetaAtPixel);
                y_phys = r_phys * Math.sin(thetaAtPixel);
            } else {
                x_phys = i * dx;
                y_phys = (H - 1 - j) * dy;
            }
            const m = mm.magnetization;
            const cx = Number(m.cx) || 0;
            const cy = Number(m.cy) || 0;
            const r_local = Math.hypot(x_phys - cx, y_phys - cy);
            const theta_local = Math.atan2(y_phys - cy, x_phys - cx);
            // Use unit-circle-normalised r for polar_anisotropy so its
            // Kn-based Biot-Savart computation works regardless of
            // physical scale.
            const Rm_estimate = Math.max(r_local, 1e-9);
            const res = evalMagnetizationDirection(m, theta_local,
                r_local / (Number(m.R_pc) || Rm_estimate));
            if (!res) continue;
            let angle = res.angle_rad + (res.sign < 0 ? Math.PI : 0);
            if (isPolar) {
                // Lab-frame angle φ → strip-local: components (M_r, M_θ) =
                // (cos(φ−θ), sin(φ−θ)). Horizontal r: x_img = r̂, image-up = θ̂
                // → display angle = φ−θ. Vertical r: x_img = θ̂, image-up = r̂
                // → display angle = π/2 − (φ−θ).
                const rel = angle - thetaAtPixel;
                angle = (AppState.magPreview.rOrientation === 'vertical')
                    ? (Math.PI / 2 - rel) : rel;
            }
            // Render in IMAGE coords (y down) so the arrow visually
            // matches the image. Physical +y is image -y.
            const cos_a = Math.cos(angle);
            const sin_a = Math.sin(angle);
            const x_tip = i + arrowLen * cos_a;
            const y_tip = j - arrowLen * sin_a;   // flip y for SVG
            const colour = (res.sign >= 0) ? '#d63333' : '#3366cc';
            parts.push(`<line x1="${i}" y1="${j}" x2="${x_tip.toFixed(2)}" y2="${y_tip.toFixed(2)}" stroke="${colour}" stroke-width="${(stride * 0.08).toFixed(2)}" stroke-linecap="round" opacity="0.9"/>`);
            // Arrowhead triangle
            const ah = arrowLen * 0.33;
            const tail_x = x_tip - ah * cos_a;
            const tail_y = y_tip + ah * sin_a;   // image-coord flip
            const perp_x =  sin_a * ah * 0.5;
            const perp_y =  cos_a * ah * 0.5;
            parts.push(`<polygon points="${x_tip.toFixed(2)},${y_tip.toFixed(2)} ${(tail_x + perp_x).toFixed(2)},${(tail_y + perp_y).toFixed(2)} ${(tail_x - perp_x).toFixed(2)},${(tail_y - perp_y).toFixed(2)}" fill="${colour}" opacity="0.9"/>`);
            arrowCount++;
        }
    }
    svg.innerHTML = parts.join('');
    document.getElementById('magPreviewStatus').textContent =
        `coord_system: ${AppState.magPreview.coordSystem}  ·  ${mats.length} material(s)  ·  ${arrowCount} arrow(s) painted (stride=${stride}px${isPolar ? ', polar' : ''})`;
}

// ============================================================
// Uniform-colour quantization filter modal (Phase 5c.2)
// ============================================================
// Triggered from the warning banner on Detect Colors modal (or any other
// caller that knows there's a loaded image). Live preview is generated
// server-side via /api/preprocess-filter/quantize?preview=true with a
// 350 ms debounce on slider changes; Apply persists a non-preview output
// and swaps it in as the loaded image.
AppState.quantizeFilter = {
    sourceFilename: null,
    previewFilename: null,
    busy: false,
    debounceTimer: null,
    lastResult: null,
    closing: false,
};

async function openQuantizeFilterModal() {
    const qf = AppState.quantizeFilter;
    if (!AppState.uploadedImageFilename) {
        showStatus('solverStatus', 'No image selected', 'error');
        return;
    }
    qf.sourceFilename = AppState.uploadedImageFilename;
    qf.previewFilename = null;
    qf.lastResult = null;
    qf.closing = false;

    const modal = document.getElementById('quantizeFilterModal');
    modal.style.display = 'flex';

    // Attach zoom/pan once and reset to 100 %.
    const zp = attachZoomPan(
        document.getElementById('qfilterPreviewContainer'),
        document.getElementById('qfilterZoomStage'),
        { indicatorEl: document.getElementById('qfilterZoomIndicator') }
    );
    if (zp) zp.reset();

    // Initialise preview to source so the user sees the un-filtered image
    // before the first server round-trip lands.
    document.getElementById('qfilterPreviewImg').src =
        `/uploads/${AppState.userId}/${qf.sourceFilename}?t=${Date.now()}`;
    document.getElementById('qfilterStatus').textContent = '';
    document.getElementById('qfilterFooterNote').textContent =
        `source: ${qf.sourceFilename}`;
    bindQuantizeFilterControls();

    // Fire an initial preview render with the default slider values.
    scheduleQuantizePreview(50);
}

function closeQuantizeFilterModal() {
    const qf = AppState.quantizeFilter;
    qf.closing = true;
    if (qf.debounceTimer) { clearTimeout(qf.debounceTimer); qf.debounceTimer = null; }
    document.getElementById('quantizeFilterModal').style.display = 'none';
    document.getElementById('qfilterPreviewImg').src = '';
}

// Wire range<->number pairs and slider->preview debounce. Idempotent.
function bindQuantizeFilterControls() {
    const pairs = [
        ['qfilterThresholdRange', 'qfilterThreshold'],
        ['qfilterNRange',         'qfilterN'],
        ['qfilterMinDistRange',   'qfilterMinDist'],
        ['qfilterDespeckleRange', 'qfilterDespeckle'],
        ['qfilterMinIslandRange', 'qfilterMinIsland'],
        ['qfilterBilSpaceRange',  'qfilterBilSpace'],
        ['qfilterBilColorRange',  'qfilterBilColor'],
    ];
    for (const [rangeId, numId] of pairs) {
        const r = document.getElementById(rangeId);
        const n = document.getElementById(numId);
        if (r._qfBound) continue;
        r.addEventListener('input', () => {
            n.value = r.value;
            scheduleQuantizePreview();
        });
        n.addEventListener('input', () => {
            const v = Number(n.value);
            if (Number.isFinite(v)) r.value = v;
            scheduleQuantizePreview();
        });
        r._qfBound = true;
    }
}

function scheduleQuantizePreview(delayMs = 350) {
    const qf = AppState.quantizeFilter;
    if (qf.debounceTimer) clearTimeout(qf.debounceTimer);
    qf.debounceTimer = setTimeout(() => runQuantizePreview(), delayMs);
}

async function runQuantizePreview() {
    const qf = AppState.quantizeFilter;
    if (qf.busy || qf.closing || !qf.sourceFilename) return;
    qf.busy = true;
    const loading = document.getElementById('qfilterLoading');
    const status = document.getElementById('qfilterStatus');
    loading.classList.add('active');
    status.textContent = 'Generating preview…';

    const params = currentQuantizeParams();
    try {
        const res = await fetch('/api/preprocess-filter/quantize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId,
                filename: qf.sourceFilename,
                rareThreshold: params.rareThreshold,
                nTargets: params.nTargets,
                minTargetDist: params.minTargetDist,
                despeckleRadius: params.despeckleRadius,
                minIslandSize:   params.minIslandSize,
                bilateralSigmaSpatial: params.bilateralSigmaSpatial,
                bilateralSigmaColor:   params.bilateralSigmaColor,
                preview: true,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'preview failed');
        qf.previewFilename = res.filename;
        qf.lastResult = res;
        const img = document.getElementById('qfilterPreviewImg');
        // Bust the cache: the preview file is overwritten on each request
        // so we have to force the browser to re-fetch the same URL.
        img.src = `${res.path}?t=${Date.now()}`;
        renderQuantizeTargetChips(res.targets || []);
        document.getElementById('qfilterTargetCount').textContent =
            `(N=${res.n_targets_used}, source uniqueColors=${res.unique_colors_in.toLocaleString()})`;
        status.textContent = '';
    } catch (err) {
        status.textContent = `Error: ${err.message}`;
    } finally {
        qf.busy = false;
        loading.classList.remove('active');
    }
}

function currentQuantizeParams() {
    return {
        rareThreshold:         Math.max(0, Number(document.getElementById('qfilterThreshold').value || 0)) / 100,
        nTargets:              Math.max(1, Number(document.getElementById('qfilterN').value || 8)),
        minTargetDist:         Math.max(0, Number(document.getElementById('qfilterMinDist').value || 30)),
        despeckleRadius:       Math.max(0, Math.min(10, Math.floor(Number(document.getElementById('qfilterDespeckle').value || 1)))),
        minIslandSize:         Math.max(0, Math.min(200, Math.floor(Number(document.getElementById('qfilterMinIsland').value || 0)))),
        bilateralSigmaSpatial: Math.max(0, Number(document.getElementById('qfilterBilSpace').value || 0)),
        bilateralSigmaColor:   Math.max(1, Number(document.getElementById('qfilterBilColor').value || 20)),
    };
}

function renderQuantizeTargetChips(targets) {
    const host = document.getElementById('qfilterTargetChips');
    host.innerHTML = '';
    targets.forEach(t => {
        const [r, g, b] = t.rgb;
        const hex = '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
        const chip = document.createElement('span');
        chip.className = 'qfilter-chip';
        chip.innerHTML = `
            <span class="qfilter-chip-swatch" style="background:${hex}"></span>
            <span>${hex}</span>
            <span style="color:#6c757d;">${(t.ratio * 100).toFixed(2)}%</span>`;
        host.appendChild(chip);
    });
}

async function applyQuantizeFilter() {
    const qf = AppState.quantizeFilter;
    if (qf.busy || !qf.sourceFilename) return;
    qf.busy = true;
    const status = document.getElementById('qfilterStatus');
    const applyBtn = document.getElementById('qfilterApplyBtn');
    applyBtn.disabled = true;
    status.textContent = 'Generating and saving image…';
    const params = currentQuantizeParams();
    try {
        const res = await fetch('/api/preprocess-filter/quantize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId,
                filename: qf.sourceFilename,
                rareThreshold: params.rareThreshold,
                nTargets: params.nTargets,
                minTargetDist: params.minTargetDist,
                despeckleRadius: params.despeckleRadius,
                minIslandSize:   params.minIslandSize,
                bilateralSigmaSpatial: params.bilateralSigmaSpatial,
                bilateralSigmaColor:   params.bilateralSigmaColor,
                preview: false,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'apply failed');

        // Swap the new filtered image in as the loaded image and re-render
        // the YAML/material flows that depend on it.
        AppState.uploadedImageFilename = res.filename;
        await refreshImageList();
        const sel = document.getElementById('imageSelect');
        if (sel) sel.value = res.filename;
        await loadSelectedImage();

        showStatus('solverStatus',
            `Color uniformization filter applied: ${res.filename} (${res.n_targets_used} colors)`,
            'success');
        closeQuantizeFilterModal();
        // If the Detect Colors modal is still open, rerun it so the user
        // sees the new "noise" count drop to ~the materials count.
        const dm = document.getElementById('detectColorsModal');
        if (dm && dm.style.display === 'flex') {
            await detectColors();
        }
    } catch (err) {
        status.textContent = `Error: ${err.message}`;
    } finally {
        qf.busy = false;
        applyBtn.disabled = false;
    }
}

// Helper for runAutoTune(): set both the range and the number input of
// a slider pair to the same value (without triggering the input event
// 5 times in a row, which would re-fire the preview debounce).
function setQuantizeSlider(numId, rangeId, value) {
    const num = document.getElementById(numId);
    const rng = document.getElementById(rangeId);
    if (num) num.value = value;
    if (rng) rng.value = value;
}

async function runAutoTune() {
    const qf = AppState.quantizeFilter;
    if (qf.busy || !qf.sourceFilename) return;
    qf.busy = true;
    const status   = document.getElementById('qfilterStatus');
    const tuneBtn  = document.getElementById('qfilterAutoTuneBtn');
    const applyBtn = document.getElementById('qfilterApplyBtn');
    tuneBtn.disabled = true;
    applyBtn.disabled = true;
    status.textContent = 'Optimising (this may take a few seconds)…';

    // Start the search from the user's current slider values so they can
    // pre-seed the optimiser. N and rareThreshold are held fixed by the
    // backend (they define "what counts as a material" -- a user
    // decision -- so optimising them tends to drive the output toward
    // the dominant background colour).
    const cur = currentQuantizeParams();
    const seed = [
        cur.minTargetDist,
        cur.despeckleRadius,
        cur.bilateralSigmaSpatial,
        cur.bilateralSigmaColor,
        cur.minIslandSize,
    ];

    try {
        const res = await fetch('/api/preprocess-filter/auto-tune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId,
                filename: qf.sourceFilename,
                nTargets:      cur.nTargets,
                rareThreshold: cur.rareThreshold,
                maxEvals:      60,
                subsampleSize: 256,
                initial:       seed,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'auto-tune failed');

        const p = res.params;
        setQuantizeSlider('qfilterMinDist',   'qfilterMinDistRange',   Math.round(p.minTargetDist));
        setQuantizeSlider('qfilterDespeckle', 'qfilterDespeckleRange', p.despeckleRadius);
        setQuantizeSlider('qfilterMinIsland', 'qfilterMinIslandRange', p.minIslandSize);
        setQuantizeSlider('qfilterBilSpace',  'qfilterBilSpaceRange',  p.bilateralSigmaSpatial.toFixed(1));
        setQuantizeSlider('qfilterBilColor',  'qfilterBilColorRange',  Math.round(p.bilateralSigmaColor));

        const before = res.score_initial;
        const after  = res.score_final;
        const pct = before > 0 ? Math.round(100 * (1 - after / before)) : 0;
        status.textContent =
            `Auto-tune: scattered noise ${(before * 100).toFixed(2)}% → ${(after * 100).toFixed(2)}% ` +
            `(${pct}% reduction, ${res.evals} evals, ${res.elapsed_ms} ms, subsample ${res.subsample_w}×${res.subsample_h}).`;
        // Re-render the preview at full resolution with the new params.
        scheduleQuantizePreview(50);
    } catch (err) {
        status.textContent = `Auto-tune error: ${err.message}`;
    } finally {
        qf.busy = false;
        tuneBtn.disabled = false;
        applyBtn.disabled = false;
    }
}

async function copyDetectedYaml() {
    if (!AppState.lastDetectResult) return;
    // Phase D.4: copy the regenerated YAML (assignments + presets) when
    // available so the clipboard matches the preview.
    const text = AppState.lastDetectResult.generatedYaml ||
                 AppState.lastDetectResult.yamlTemplate || '';
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        showStatus('solverStatus', 'YAML template copied to clipboard', 'success');
    } catch (e) {
        showStatus('solverStatus', 'Clipboard write failed', 'error');
    }
}

function insertMaterialsSection() {
    if (!AppState.lastDetectResult) return;
    // Phase D.4: prefer the regenerated YAML (carries the per-chip
    // assignments + library presets) over the server's stock template.
    const sourceYaml = AppState.lastDetectResult.generatedYaml ||
                       AppState.lastDetectResult.yamlTemplate || '';
    if (!sourceYaml) return;
    if (!AppState.aceEditor) {
        showStatus('solverStatus', 'Config editor not initialized', 'error');
        return;
    }

    try {
        // Parse detected materials
        const detectedDoc = jsyaml.load(sourceYaml);
        if (!detectedDoc || !detectedDoc.materials) {
            showStatus('solverStatus', 'No materials found in detected template', 'error');
            return;
        }

        // Parse current editor content
        const currentYaml = AppState.aceEditor.getValue();
        let currentDoc = {};
        try {
            currentDoc = jsyaml.load(currentYaml) || {};
        } catch (e) {
            currentDoc = {};
        }

        // Replace only the materials section
        currentDoc.materials = detectedDoc.materials;
        // Phase E.3: material_presets are intentionally NOT merged here
        // — the server merges AppState.selectedLibrary at run time, so
        // inlining the presets would create a stale copy of the library
        // contents inside every config file. The user keeps the library
        // YAML authoritative.
        // Phase F.3: merge detected variables (J_Coil_<group>) into the
        // editor's existing variables non-destructively. The user's
        // previously-set values win; only missing keys are added so
        // re-running Detect Colors doesn't clobber drive-tuned current
        // densities the user already set.
        if (detectedDoc.variables && typeof detectedDoc.variables === 'object') {
            if (!currentDoc.variables || typeof currentDoc.variables !== 'object') {
                currentDoc.variables = {};
            }
            for (const k of Object.keys(detectedDoc.variables)) {
                if (currentDoc.variables[k] == null) {
                    currentDoc.variables[k] = detectedDoc.variables[k];
                }
            }
        }
        // Phase L: merge auto-generated Phi_Coil_X flux_linkage entries
        // into the editor's existing array by name. Re-running Detect
        // Colors after the user tuned an entry won't reset that entry,
        // and the entries from new Coil chips get appended.
        if (Array.isArray(detectedDoc.flux_linkage) && detectedDoc.flux_linkage.length > 0) {
            if (!Array.isArray(currentDoc.flux_linkage)) currentDoc.flux_linkage = [];
            const existing = new Set(
                currentDoc.flux_linkage.map(e => (e && e.name) || '').filter(Boolean));
            for (const entry of detectedDoc.flux_linkage) {
                if (entry && entry.name && !existing.has(entry.name)) {
                    currentDoc.flux_linkage.push(entry);
                    existing.add(entry.name);
                }
            }
        }

        // Phase BA: jsyaml.dump strips every comment from the document,
        // including the polar / cartesian Insert YAML's hint block. Re-
        // append it so the user keeps seeing the optional-controls
        // section after every Detect Colors round-trip. ensureSolverHintBlock
        // is idempotent so running Detect Colors twice in a row doesn't
        // duplicate the block.
        const merged = ensureSolverHintBlock(
            jsyaml.dump(currentDoc, { indent: 2, lineWidth: -1 }));
        AppState.aceEditor.setValue(merged, -1);

        closeDetectColorsModal();
        switchTab('config');
        showStatus('configStatus', 'materials: section updated from detected colors', 'success');
    } catch (e) {
        showStatus('solverStatus', `Insert failed: ${e.message}`, 'error');
    }
}

// =====================================================
// Polar Preprocess Modal (v1.5) — Phase 5c (skeleton)
// =====================================================
// Phase 5c implements: state holder, open/close, read-only overlay drawn
// from the auto-detect response, numeric input one-way sync (state -> input,
// no edit handling yet), color chip list with non-functional dropdowns.
// Phase 5d adds mouse/wheel/key editing + backend color_groups; Phase 5e
// adds preview + save & insert. Keep this section self-contained.

AppState.polarPreprocess = {
    sourceFilename: null,
    imageNaturalWidth: 0,
    imageNaturalHeight: 0,
    detection: null,
    current: {
        center_x: 0, center_y: 0,
        // r_inner_px is the *warp* inner radius. Defaults to 0 so the
        // polar warp covers the entire machine; the detected air gap is
        // tracked separately in air_gap_px and shown only as a yellow
        // dashed visual marker (the user can copy it to r_inner via the
        // dropdown button if they want to cut the inside off).
        r_inner_px: 0, r_outer_px: 0,
        air_gap_px: 0,
        // Phase D.3: user-adjustable offset on top of the dropdown-selected
        // dip radius. The yellow dashed marker is rendered at
        // (air_gap_px + air_gap_offset_px). Reset to 0 on dropdown change
        // or re-detect. Can be negative.
        air_gap_offset_px: 0,
        // Phase D.3: when true and save_as === 'polar', Insert YAML emits
        // a transient.slides: skeleton anchored at the effective air gap.
        // Default ON (user request): the emission is still gated on an
        // air-gap marker actually existing, so it is a no-op without one.
        air_gap_as_slide: true,
        air_gap_slide_side: 'inside',      // 'inside' | 'outside'
        air_gap_slide_pixels_per_step: 1,
        theta_start: 0, theta_end: 2 * Math.PI,
        is_sector: false,
        nr: 0, ntheta: 0,
        snap_ntheta: true,
        r_orientation: 'horizontal',
        r_outer_physical: 1.0,
        save_as: 'polar',
    },
    lastPreview: { filename: null, path: null, polar_domain: null, width: 0, height: 0 },
    lastSaved:   null,  // Set by Save image; { filename, path, polar_domain, width, height }
    isDirty: false,
    isWarping: false,
    isDetecting: false,
    _debounceTimer: null,
    _activeDrag: null,
    _rafScheduled: false,
};

// ============================================================
// Reusable zoom + pan helper (used by the Polar Preprocess preview and the
// uniform-colour Filter preview). Wheel zooms around the cursor, middle
// mouse drag pans, double-click resets. SVG overlays still get correct
// image-pixel coords through getScreenCTM().inverse() because CSS
// transforms are reflected in the SVG CTM, so handle hit-tests don't need
// any adjustment when zoomed.
function attachZoomPan(containerEl, stageEl, opts = {}) {
    if (!containerEl || !stageEl) return null;
    if (containerEl._zoomPan) return containerEl._zoomPan;
    const indicatorEl = opts.indicatorEl || null;
    const minScale = opts.minScale ?? 0.25;
    const maxScale = opts.maxScale ?? 16;
    const zoomFactor = opts.zoomFactor ?? 1.15;
    // Phase D.1: optional callback fired after every transform update so
    // consumers (e.g. the Polar Preprocess overlay) can re-render their
    // SVG content at sizes inversely proportional to the zoom level.
    const onScaleChange = typeof opts.onScaleChange === 'function' ? opts.onScaleChange : null;
    const state = { scale: 1, tx: 0, ty: 0, panning: false, panStart: null };
    let indicatorTimer = null;
    // Full-transform listeners (scale AND pan) — used by the pixel rulers.
    // Registered via api.addTransformListener so late consumers can hook a
    // container whose zoom-pan was attached earlier (attach is idempotent).
    const transformListeners = [];

    function apply() {
        stageEl.style.transform =
            `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`;
        if (indicatorEl) {
            indicatorEl.textContent = Math.round(state.scale * 100) + '%';
            indicatorEl.classList.add('visible');
            clearTimeout(indicatorTimer);
            indicatorTimer = setTimeout(
                () => indicatorEl.classList.remove('visible'), 1200);
        }
        if (onScaleChange) onScaleChange(state.scale);
        for (const fn of transformListeners) { try { fn(state); } catch (_) {} }
    }
    function reset() {
        state.scale = 1; state.tx = 0; state.ty = 0;
        apply();
    }
    function onWheel(e) {
        e.preventDefault();
        const rect = containerEl.getBoundingClientRect();
        const dx = e.clientX - rect.left;
        const dy = e.clientY - rect.top;
        // Cursor position in stage-local coords (pre-transform).
        const wx = (dx - state.tx) / state.scale;
        const wy = (dy - state.ty) / state.scale;
        const f = e.deltaY < 0 ? zoomFactor : 1 / zoomFactor;
        const next = Math.max(minScale, Math.min(maxScale, state.scale * f));
        if (next === state.scale) return;
        state.scale = next;
        // Keep the cursor over the same stage point.
        state.tx = dx - wx * state.scale;
        state.ty = dy - wy * state.scale;
        apply();
    }
    function onMouseDown(e) {
        if (e.button !== 1) return;  // middle button only
        e.preventDefault();
        state.panning = true;
        state.panStart = { x: e.clientX, y: e.clientY, tx: state.tx, ty: state.ty };
        containerEl.classList.add('panning');
    }
    function onMouseMove(e) {
        if (!state.panning) return;
        state.tx = state.panStart.tx + (e.clientX - state.panStart.x);
        state.ty = state.panStart.ty + (e.clientY - state.panStart.y);
        apply();
    }
    function onMouseUp(e) {
        if (!state.panning) return;
        if (e.button === 1 || e.button === undefined) {
            state.panning = false;
            containerEl.classList.remove('panning');
        }
    }
    function onAuxClick(e) {
        // Prevent the browser's middle-click "auto-scroll" handler kicking in.
        if (e.button === 1) e.preventDefault();
    }
    function onDblClick(e) {
        // Only reset on background dbl-click; let handles handle their own.
        if (e.target.closest('.handle')) return;
        reset();
    }
    containerEl.addEventListener('wheel', onWheel, { passive: false });
    containerEl.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    containerEl.addEventListener('auxclick', onAuxClick);
    containerEl.addEventListener('dblclick', onDblClick);
    apply();
    const api = {
        reset,
        get scale() { return state.scale; },
        get tx() { return state.tx; },
        get ty() { return state.ty; },
        addTransformListener(fn) { if (typeof fn === 'function') transformListeners.push(fn); },
    };
    containerEl._zoomPan = api;
    return api;
}

// ===== Pixel rulers =====
// Canvas rulers (top = x, left = y) in IMAGE PIXEL coordinates that track the
// container's zoom-pan transform. Works for any zoom-stage viewer: reads the
// transform from containerEl._zoomPan and maps image px -> screen px through
// the img's position/CSS-scale inside the stage. Call .redraw() after image
// or layout changes; transform changes re-draw automatically.
function attachPixelRulers({ topCanvas, leftCanvas, containerEl, stageEl, imgEl }) {
    if (!topCanvas || !leftCanvas || !containerEl || !imgEl) return null;
    if (containerEl._pixelRulers) return containerEl._pixelRulers;

    function drawAxis(canvas, horizontal) {
        const cw = canvas.clientWidth, ch = canvas.clientHeight;
        const ctx = canvas.getContext('2d');
        if (!cw || !ch) return;
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
            canvas.width = Math.round(cw * dpr);
            canvas.height = Math.round(ch * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, ch);
        const naturalLen = horizontal ? imgEl.naturalWidth : imgEl.naturalHeight;
        if (!naturalLen) return;
        // Map image px -> ruler px via bounding rects: rects already include
        // the zoom-pan transform AND any wrapper elements between the img and
        // the stage, so this stays correct regardless of DOM structure.
        const contRect = containerEl.getBoundingClientRect();
        const imgRect = imgEl.getBoundingClientRect();
        const rectLen = horizontal ? imgRect.width : imgRect.height;
        if (!rectLen) return;
        const off = horizontal ? (imgRect.left - contRect.left) : (imgRect.top - contRect.top);
        const pxPer = rectLen / naturalLen;               // screen px per image px
        const toScreen = p => off + p * pxPer;
        const fromScreen = x => (x - off) / pxPer;
        const axisLen = horizontal ? cw : ch;
        // Tick step: smallest 1/2/5×10^k giving >= 55 screen px between labels.
        let step = 1;
        while (step * pxPer < 55) {
            const mant = step.toPrecision(1)[0];
            step = (mant === '1') ? step * 2 : (mant === '2') ? step * 2.5 : step * 2;
        }
        const minor = step / 5;
        const pStart = Math.max(0, Math.floor(fromScreen(0) / minor) * minor);
        const pEnd = Math.min(naturalLen, fromScreen(axisLen));
        ctx.strokeStyle = '#adb5bd';
        ctx.fillStyle = '#495057';
        ctx.font = '9px monospace';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let p = pStart; p <= pEnd + 1e-9; p += minor) {
            const x = toScreen(p);
            if (x < -1 || x > axisLen + 1) continue;
            const isMajor = Math.abs(p / step - Math.round(p / step)) < 1e-6;
            const tick = isMajor ? 7 : (minor * pxPer >= 4 ? 4 : 0);
            if (!tick) continue;
            const xr = Math.round(x) + 0.5;
            if (horizontal) { ctx.moveTo(xr, ch); ctx.lineTo(xr, ch - tick); }
            else            { ctx.moveTo(cw, xr); ctx.lineTo(cw - tick, xr); }
            if (isMajor) {
                const label = String(Math.round(p));
                if (horizontal) ctx.fillText(label, xr + 2, ch - 9);
                else            ctx.fillText(label, 1, xr - 2);
            }
        }
        ctx.stroke();
    }

    const api = {
        redraw() { drawAxis(topCanvas, true); drawAxis(leftCanvas, false); },
    };
    if (containerEl._zoomPan) containerEl._zoomPan.addTransformListener(() => api.redraw());
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => api.redraw()).observe(containerEl);
    }
    imgEl.addEventListener('load', () => api.redraw());
    containerEl._pixelRulers = api;
    return api;
}

// Wire the Run & Preview tab's Input Image into the zoom-pan viewer with
// pixel rulers. The wrapper's visibility mirrors the img's 'hidden' class
// (kept in sync via MutationObserver so the existing show/hide sites work).
function initInputImageViewer() {
    const wrap = document.getElementById('inputImageRulerWrap');
    const container = document.getElementById('inputImageZoomContainer');
    const stage = document.getElementById('inputImageZoomStage');
    const img = document.getElementById('uploadedImage');
    if (!wrap || !container || !stage || !img) return;
    attachZoomPan(container, stage, {
        indicatorEl: document.getElementById('inputImageZoomIndicator'),
    });
    const rulers = attachPixelRulers({
        topCanvas: document.getElementById('inputImageRulerTop'),
        leftCanvas: document.getElementById('inputImageRulerLeft'),
        containerEl: container, stageEl: stage, imgEl: img,
    });
    const sync = () => {
        const hidden = img.classList.contains('hidden');
        wrap.style.display = hidden ? 'none' : 'grid';
        if (!hidden && rulers) requestAnimationFrame(() => rulers.redraw());
    };
    new MutationObserver(sync).observe(img, { attributes: true, attributeFilter: ['class'] });
    img.addEventListener('load', () => {
        // A newly loaded image gets a fresh viewport.
        if (container._zoomPan) container._zoomPan.reset();
    });
    sync();
}

function resetPolarPreprocessState() {
    const pp = AppState.polarPreprocess;
    pp.sourceFilename = null;
    pp.imageNaturalWidth = 0;
    pp.imageNaturalHeight = 0;
    pp.detection = null;
    Object.assign(pp.current, {
        center_x: 0, center_y: 0,
        r_inner_px: 0, r_outer_px: 0,
        air_gap_px: 0,
        air_gap_offset_px: 0,
        air_gap_as_slide: true,   // default ON (no-op until a gap marker exists)
        air_gap_slide_side: 'inside',
        air_gap_slide_pixels_per_step: 1,
        theta_start: 0, theta_end: 2 * Math.PI,
        is_sector: false,
        nr: 0, ntheta: 0,
        snap_ntheta: true,
        r_orientation: 'horizontal',
        r_outer_physical: 1.0,
        save_as: 'polar',
    });
    pp.lastPreview = { filename: null, path: null, polar_domain: null, width: 0, height: 0 };
    pp.lastSaved = null;
    pp.isDirty = false;
    pp.isWarping = false;
    pp.isDetecting = false;
    if (pp._debounceTimer) { clearTimeout(pp._debounceTimer); pp._debounceTimer = null; }
    pp._activeDrag = null;
    pp._rafScheduled = false;
}

async function openPolarPreprocessModal() {
    const modal = document.getElementById('polarPreprocessModal');
    if (modal.style.display === 'flex') return;
    if (!AppState.uploadedImageFilename) {
        showStatus('solverStatus', 'Please upload or select an image first', 'error');
        return;
    }
    resetPolarPreprocessState();
    const pp = AppState.polarPreprocess;
    pp.sourceFilename = AppState.uploadedImageFilename;

    // Show modal first so user sees instant feedback
    modal.style.display = 'flex';
    setPolarLoading(true, 'Loading image…');

    // Wire up wheel-zoom / middle-drag-pan / dbl-click-reset on the preview
    // (idempotent: attachZoomPan returns the existing handle if already set).
    // Phase D.1: re-render the SVG overlay whenever the zoom changes so
    // handle dots / dashed circles stay roughly constant-on-screen instead
    // of inflating into pixel blobs at 5-10x zoom.
    const zp = attachZoomPan(
        document.getElementById('polarPreviewContainer'),
        document.getElementById('polarZoomStage'),
        {
            indicatorEl: document.getElementById('polarZoomIndicator'),
            onScaleChange: () => renderPolarOverlay(),
        }
    );
    if (zp) zp.reset();
    // Reset the view to the Source tab on every open, since the warp
    // output is regenerated for the freshly-loaded image.
    switchPolarView('source');
    // Interactivity wiring (Phase 5d.1 / 5f). All idempotent.
    bindPolarInputs();
    bindPolarAirGapTuneInputs();
    attachPolarSvgDrag();
    attachPolarKeyboard();
    attachPolarEscKey();

    try {
        await loadPolarSourceImage();
        const detectRes = await fetch('/api/preprocess-polar/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId,
                filename: pp.sourceFilename,
            }),
        }).then(r => r.json());
        if (!detectRes.success) {
            throw new Error(detectRes.error || 'detect failed');
        }
        pp.detection = detectRes;

        applyDetectionToCurrent(pp.detection);
        renderPolarStatusSummary();
        renderPolarOverlay();
        syncPolarInputsFromState();
        renderPolarDipDropdown();
        // Phase 5f.2: fire an initial warp so the "Warp output" tab has
        // something to show without the user having to wait for the 1.5 s
        // dirty debounce. Best-effort -- failures fall through silently.
        triggerPolarPreviewWarp();
    } catch (err) {
        showStatus('solverStatus', `Polar detect failed: ${err.message}`, 'error');
        renderPolarStatusSummary(err.message);
    } finally {
        setPolarLoading(false);
    }
}

// AA-noise probe + warning + filter-button toggle. Runs on every image
// upload / load via /api/preprocess-filter/quick-stats (lightweight: just
// counts distinct RGB values, no AA blend classification). If the count
// exceeds the noisy threshold an inline banner appears in the Input
// Image panel, independent of any other modal so the user sees it the
// moment they pick the image. The "Apply Color Uniformization Filter"
// button is exposed regardless so the user can also pre-emptively run
// the filter on borderline images.
async function checkInputImageNoise(filename) {
    const banner = document.getElementById('inputImageNoiseBanner');
    const detail = document.getElementById('inputImageNoiseBannerDetail');
    const filterBtn = document.getElementById('quantizeFilterBtn');
    if (!banner || !filterBtn) return;
    banner.style.display = 'none';
    filterBtn.style.display = 'none';
    if (!filename) return;
    try {
        const res = await fetch('/api/preprocess-filter/quick-stats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, filename }),
        }).then(r => r.json());
        if (!res.success) return;
        filterBtn.style.display = 'block';
        if (res.looks_noisy) {
            if (detail) {
                detail.textContent =
                    ` (Unique color count: ${Number(res.unique_colors).toLocaleString()}).`;
            }
            banner.style.display = 'flex';
        }
    } catch (err) {
        // Best-effort: stay silent on failure (the rest of the app still
        // works without the noise banner).
        console.warn('quick-stats failed:', err);
    }
}

function closePolarPreprocessModal(saved = false) {
    const pp = AppState.polarPreprocess;
    // The "dirty" guard now means "user has been editing geometry but
    // hasn't saved a permanent warp file" -- the preview file is
    // disposable. We keep the confirmation only when there's actually
    // unsaved geometry work (isDirty AND we never produced a saved
    // image). After Save image the user can close freely.
    if (!saved && pp.isDirty && !(pp.lastSaved && pp.lastSaved.filename)) {
        if (!confirm('There are unsaved changes. Close anyway?')) return;
    }
    if (pp._debounceTimer) { clearTimeout(pp._debounceTimer); pp._debounceTimer = null; }
    document.getElementById('polarPreprocessModal').style.display = 'none';
    // Release image src so we don't keep the bitmap in memory
    const img = document.getElementById('polarSourceImg');
    if (img) img.src = '';
    const warpImg = document.getElementById('polarWarpImg');
    if (warpImg) warpImg.src = '';
    const svg = document.getElementById('polarOverlay');
    if (svg) svg.innerHTML = '';
    // Best-effort cleanup of the deterministic warp preview file (named
    // from the source filename, e.g. motor.__warp_preview__.png). Fire-
    // and-forget; failure (permissions, race with another session) is
    // non-fatal because the file is overwritten on every future open
    // and `enforceImageLimit` will eventually evict it anyway.
    if (pp.sourceFilename) {
        fetch('/api/preprocess-polar/cleanup-preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, filename: pp.sourceFilename }),
        }).catch(() => {});
    }
    // Reset the lastSaved / lastPreview tracking so the next open starts
    // clean. (resetPolarPreprocessState() will also do this when the
    // modal reopens.)
    pp.lastPreview = { filename: null, path: null, polar_domain: null, width: 0, height: 0 };
    pp.lastSaved   = null;
}

function setPolarLoading(on, msg) {
    const overlay = document.getElementById('polarPreviewLoading');
    if (!overlay) return;
    overlay.textContent = msg || 'Loading…';
    overlay.classList.toggle('active', !!on);
}

async function loadPolarSourceImage() {
    const pp = AppState.polarPreprocess;
    const img = document.getElementById('polarSourceImg');
    return new Promise((resolve, reject) => {
        img.onload = () => {
            pp.imageNaturalWidth = img.naturalWidth;
            pp.imageNaturalHeight = img.naturalHeight;
            const svg = document.getElementById('polarOverlay');
            svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
            resolve();
        };
        img.onerror = () => reject(new Error('image load failed'));
        img.src = `/uploads/${AppState.userId}/${pp.sourceFilename}`;
    });
}

// Pull center / radii / theta / nr / ntheta initial values from the detect
// response into the editable state object.
function applyDetectionToCurrent(detection) {
    const cur = AppState.polarPreprocess.current;
    cur.center_x = detection.center_x;
    cur.center_y = detection.center_y;
    // Phase 5f.2: the backend's r_inner_px is the *air-gap dip* and that's
    // a visual marker, not the start of the polar warp. Warping the whole
    // machine means r_inner = 0; the rotor (or inner stator) is then
    // preserved in the warp output. The user can still copy the air-gap
    // marker into r_inner explicitly via the dip dropdown's "use as
    // r_inner" button.
    cur.r_inner_px = 0;
    cur.air_gap_px = (detection.r_inner_px && detection.r_inner_px > 0)
        ? detection.r_inner_px : 0;
    // Phase D.3: a re-detect invalidates any prior offset / side choice.
    // The slide checkbox itself returns to its default (ON).
    cur.air_gap_offset_px = 0;
    cur.air_gap_as_slide = true;
    cur.air_gap_slide_side = 'inside';
    cur.r_outer_px = detection.r_outer_px;
    cur.theta_start = detection.theta_start;
    cur.theta_end = detection.theta_end;
    cur.is_sector = !detection.is_full_circle;
    const per = detection.periodicity || {};
    if (per.recommended_ntheta && per.recommended_ntheta > 1) {
        cur.ntheta = per.recommended_ntheta;
    } else {
        cur.ntheta = Math.max(2, Math.round(2 * Math.PI * detection.r_outer_px));
    }
    // nr covers the entire radial extent (r=0 to r_outer) since we no
    // longer cut at the air gap.
    cur.nr = Math.max(2, detection.r_outer_px);
}

// Phase D.3: yellow-dashed marker radius is the detected dip plus the
// user's offset; clamped at >= 0. Reused by the overlay, the slide-region
// emission in buildPolarYamlBlock, and the airgap-handle drag math.
function airGapEffectiveR() {
    const cur = AppState.polarPreprocess.current;
    return Math.max(0, (cur.air_gap_px || 0) + (cur.air_gap_offset_px || 0));
}

function renderPolarStatusSummary(errMsg) {
    const el = document.getElementById('polarStatusSummary');
    if (errMsg) {
        el.textContent = `Detection error: ${errMsg}`;
        return;
    }
    const det = AppState.polarPreprocess.detection;
    if (!det) { el.textContent = '—'; return; }
    const per = det.periodicity || {};
    const gray = per.grayscale || {};
    const rgb = per.rgb || {};
    const lines = [];
    lines.push(`shape: ${det.shape}  center: (${det.center_x}, ${det.center_y})  r=[${det.r_inner_px}, ${det.r_outer_px}]`);
    if (det.hough) lines.push(`Hough gain: ${det.hough.gain > 99 ? '>99' : det.hough.gain.toFixed(2)}x`);
    const gN = gray.n_fold != null ? `N≈${gray.n_fold} (int ${gray.n_integer})` : '—';
    const rN = rgb.n_fold  != null ? `N≈${rgb.n_fold} (int ${rgb.n_integer})`   : '—';
    lines.push(`periodicity grayscale: ${gN}`);
    lines.push(`periodicity rgb     : ${rN}`);
    const grp = per.grouped;
    if (grp && grp.n_fold != null) {
        lines.push(`periodicity grouped : N≈${grp.n_fold} (int ${grp.n_integer})`);
    }
    if (per.recommended_ntheta) lines.push(`recommended ntheta: ${per.recommended_ntheta}`);
    // Phase 5f edge-case hints. These surface once per detection in the
    // auto-detect status block so the user has the relevant warning
    // right where they read the rest of the detection summary.
    const pp = AppState.polarPreprocess;
    const W = pp.imageNaturalWidth || det.image_width || 0;
    const H = pp.imageNaturalHeight || det.image_height || 0;
    if (W * H > 4_000_000) {
        lines.push(`<span style="color:#8a6d3b">⚠ large image (${W}×${H}) — preview warp may take 1–2 s per change.</span>`);
    }
    if (det.shape === 'rectangular') {
        lines.push(
            `<span style="color:#8a6d3b">ℹ rectangular stator — r_inner is seeded from the Hough rotor lock-on; ` +
            `verify it traces the actual rotor surface and adjust manually if needed.</span>`);
    }
    el.innerHTML = lines.map(s => `<div>${s}</div>`).join('');
    // Enable sector snap button if N is detected
    const snapBtn = document.getElementById('polarSnapSectorBtn');
    if (snapBtn) snapBtn.disabled = !(gray.n_fold || rgb.n_fold);
}

// Render SVG overlay from the current state. Phase 5c is read-only: this is
// just visualisation, no event handlers attached. Phase 5d will add the
// drag handlers and call renderPolarOverlay() again whenever current changes.
function renderPolarOverlay() {
    const svg = document.getElementById('polarOverlay');
    if (!svg) return;
    const cur = AppState.polarPreprocess.current;
    const det = AppState.polarPreprocess.detection;
    const W = AppState.polarPreprocess.imageNaturalWidth || 1;
    const H = AppState.polarPreprocess.imageNaturalHeight || 1;
    const cx = cur.center_x, cy = cur.center_y;
    const rIn = cur.r_inner_px, rOut = cur.r_outer_px;
    const crossArm = Math.max(8, Math.min(W, H) * 0.02);

    // Phase D.1: at zoom = s, an SVG element with image-coord dimension d
    // appears on screen as d * s pixels. Dividing every literal size by s
    // keeps handles / dashed strokes a constant on-screen size while the
    // background image itself zooms. Floor at 0.25 so very-zoomed-out
    // overlays don't render at zero stroke width.
    const zp = document.getElementById('polarPreviewContainer') &&
               document.getElementById('polarPreviewContainer')._zoomPan;
    const scale = (zp && zp.scale) ? zp.scale : 1;
    const sigma = (base) => base / Math.max(0.25, scale);
    const HANDLE_R       = sigma(6);
    const HANDLE_RXL     = sigma(7);
    const STROKE_GUIDE   = sigma(1.8);
    const STROKE_OUTER   = sigma(2);
    const STROKE_AIRGAP  = sigma(2.2);
    const STROKE_THETA   = sigma(1.8);
    const STROKE_PERIOD  = sigma(1);
    const STROKE_CROSS   = sigma(1.5);
    const STROKE_HANDLE  = sigma(1.5);
    const DASH_GUIDE_IN  = `${sigma(6)},${sigma(4)}`;
    const DASH_AIRGAP    = `${sigma(9)},${sigma(5)}`;
    const DASH_PERIOD    = `${sigma(2)},${sigma(3)}`;

    let html = '';

    // Period guides (drawn first so they sit behind the rings)
    const periodGray = det && det.periodicity && det.periodicity.grayscale;
    const N = periodGray && periodGray.n_integer;
    if (N && N >= 2 && N <= 64 && rOut > 0) {
        for (let k = 0; k < N; k++) {
            const ang = 2 * Math.PI * k / N + (cur.theta_start || 0);
            const x2 = cx + (rOut + 10) * Math.cos(ang);
            const y2 = cy + (rOut + 10) * Math.sin(ang);
            html += `<line class="guide-period" x1="${cx}" y1="${cy}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke-width="${STROKE_PERIOD}" stroke-dasharray="${DASH_PERIOD}"/>`;
        }
    }

    // Inner / outer radius circles (or arcs for sector mode)
    if (rIn > 0) {
        html += `<circle class="guide-inner" cx="${cx}" cy="${cy}" r="${rIn}" stroke-width="${STROKE_GUIDE}" stroke-dasharray="${DASH_GUIDE_IN}"/>`;
    }
    if (rOut > 0) {
        html += `<circle class="guide-outer" cx="${cx}" cy="${cy}" r="${rOut}" stroke-width="${STROKE_OUTER}"/>`;
    }
    // Air-gap visual marker (Phase 5f.2). Yellow dashed circle so the
    // detected dip stays obvious without dictating the warp range. Phase
    // D.3 extends this to render at the *effective* radius (detected dip
    // + user offset) and to expose a drag handle on the right side so the
    // marker can be nudged interactively.
    const rAG = airGapEffectiveR();
    if (cur.air_gap_px > 0 && rAG > 0 && rAG < rOut) {
        html += `<circle class="guide-airgap" cx="${cx}" cy="${cy}" r="${rAG}" stroke-width="${STROKE_AIRGAP}" stroke-dasharray="${DASH_AIRGAP}"/>`;
    }

    // Sector boundary lines
    if (cur.is_sector) {
        const r = Math.max(rOut, rIn || 0) + 4;
        const xs = cx + r * Math.cos(cur.theta_start);
        const ys = cy + r * Math.sin(cur.theta_start);
        const xe = cx + r * Math.cos(cur.theta_end);
        const ye = cy + r * Math.sin(cur.theta_end);
        html += `<line class="guide-theta" x1="${cx}" y1="${cy}" x2="${xs.toFixed(2)}" y2="${ys.toFixed(2)}" stroke-width="${STROKE_THETA}"/>`;
        html += `<line class="guide-theta" x1="${cx}" y1="${cy}" x2="${xe.toFixed(2)}" y2="${ye.toFixed(2)}" stroke-width="${STROKE_THETA}"/>`;
    }

    // Center cross + handle (drawn last so it sits on top)
    html += `<line class="center-cross" x1="${cx - crossArm}" y1="${cy}" x2="${cx + crossArm}" y2="${cy}" stroke-width="${STROKE_CROSS}"/>`;
    html += `<line class="center-cross" x1="${cx}" y1="${cy - crossArm}" x2="${cx}" y2="${cy + crossArm}" stroke-width="${STROKE_CROSS}"/>`;
    html += `<circle class="handle handle-center" data-handle="center" cx="${cx}" cy="${cy}" r="${HANDLE_R}" stroke-width="${STROKE_HANDLE}"/>`;

    // Radius handles at 0 rad (right side)
    if (rIn > 0) {
        html += `<circle class="handle handle-inner" data-handle="inner" cx="${cx + rIn}" cy="${cy}" r="${HANDLE_RXL}" stroke-width="${STROKE_HANDLE}"/>`;
    }
    if (rOut > 0) {
        html += `<circle class="handle handle-outer" data-handle="outer" cx="${cx + rOut}" cy="${cy}" r="${HANDLE_RXL}" stroke-width="${STROKE_HANDLE}"/>`;
    }
    // Phase D.3: airgap drag handle on the right side of the yellow
    // dashed circle. Drag updates air_gap_offset_px so the visual marker
    // moves but the detected dip radius itself stays intact.
    if (cur.air_gap_px > 0 && rAG > 0 && rAG < rOut) {
        html += `<circle class="handle handle-airgap" data-handle="airgap" cx="${cx + rAG}" cy="${cy}" r="${HANDLE_R}" stroke-width="${STROKE_HANDLE}"/>`;
    }
    // Sector theta handles at the ring outer edge
    if (cur.is_sector && rOut > 0) {
        const xs = cx + rOut * Math.cos(cur.theta_start);
        const ys = cy + rOut * Math.sin(cur.theta_start);
        const xe = cx + rOut * Math.cos(cur.theta_end);
        const ye = cy + rOut * Math.sin(cur.theta_end);
        html += `<circle class="handle handle-theta" data-handle="theta_start" cx="${xs.toFixed(2)}" cy="${ys.toFixed(2)}" r="${HANDLE_RXL}" stroke-width="${STROKE_HANDLE}"/>`;
        html += `<circle class="handle handle-theta" data-handle="theta_end"   cx="${xe.toFixed(2)}" cy="${ye.toFixed(2)}" r="${HANDLE_RXL}" stroke-width="${STROKE_HANDLE}"/>`;
    }

    svg.innerHTML = html;
}

// State -> input one-way sync (Phase 5c is read-only). Phase 5d will add
// the reverse direction.
function syncPolarInputsFromState() {
    const cur = AppState.polarPreprocess.current;
    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    setVal('polarCenterX', cur.center_x);
    setVal('polarCenterY', cur.center_y);
    setVal('polarRInner', cur.r_inner_px);
    setVal('polarROuter', cur.r_outer_px);
    setVal('polarROuterPhys', cur.r_outer_physical);
    setVal('polarNr', cur.nr);
    setVal('polarNtheta', cur.ntheta);
    const snap = document.getElementById('polarSnapNtheta');
    if (snap) snap.checked = cur.snap_ntheta;
    // theta inputs (deg, CCW math convention -- negated from the
    // internal image-coord state).
    setVal('polarThetaStart', -radToDegRounded(cur.theta_start));
    setVal('polarThetaEnd',   -radToDegRounded(cur.theta_end));
    // theta mode radio
    document.querySelectorAll('input[name="polarThetaMode"]').forEach(r => {
        r.checked = (r.value === (cur.is_sector ? 'sector' : 'full'));
    });
    document.getElementById('polarSectorInputs').style.display = cur.is_sector ? '' : 'none';
    // orientation
    document.querySelectorAll('input[name="polarROrient"]').forEach(r => {
        r.checked = (r.value === cur.r_orientation);
    });
    // save target
    document.querySelectorAll('input[name="polarSaveAs"]').forEach(r => {
        r.checked = (r.value === cur.save_as);
    });
    // Phase D.3: air-gap tune controls
    setVal('polarAirGapOffset', cur.air_gap_offset_px || 0);
    const slideEl = document.getElementById('polarAirGapAddSlide');
    if (slideEl) slideEl.checked = !!cur.air_gap_as_slide;
    document.querySelectorAll('input[name="polarAirGapSide"]').forEach(r => {
        r.checked = (r.value === (cur.air_gap_slide_side || 'inside'));
    });
}

function radToDegRounded(rad) { return Math.round(rad * 180 / Math.PI * 100) / 100; }

// ============================================================
// Polar Preprocess interactivity (Phase 5d.1)
// ============================================================
// Three independent wirings, all idempotent so the open-modal flow can
// call them every time without piling up duplicate listeners:
//   - bindPolarInputs(): every number / radio / checkbox edits the
//     state and triggers a re-render. Wheel on number inputs steps
//     +/- (Shift = x10).
//   - attachPolarSvgDrag(): left-click on any data-handle element on
//     the SVG overlay starts a drag that updates the corresponding
//     centre / r_inner / r_outer / theta in the state and re-renders.
//   - attachPolarKeyboard(): arrow keys on the focused preview
//     container nudge the centre by +/-1 px (Shift = +/-10).

// markPolarDirty: every state edit feeds through here. The yellow
// "unupdated" badge surfaces immediately so the user knows the on-
// screen overlay no longer matches the last warped preview; a 1.5 s
// debounce then auto-fires a fresh warp. The user can also press
// "Apply Transform" to skip the wait.
function markPolarDirty() {
    const pp = AppState.polarPreprocess;
    pp.isDirty = true;
    const badge = document.getElementById('polarDirtyBadge');
    if (badge) badge.classList.add('active');
    if (pp._debounceTimer) clearTimeout(pp._debounceTimer);
    pp._debounceTimer = setTimeout(triggerPolarPreviewWarp, 1500);
}

async function triggerPolarPreviewWarp() {
    const pp = AppState.polarPreprocess;
    if (pp.isWarping || !pp.sourceFilename) return;
    const cur = pp.current;
    if (!(cur.r_outer_px > cur.r_inner_px)) {
        // Geometry is invalid; leave the badge on and skip the warp.
        return;
    }
    pp.isWarping = true;
    setPolarWarpLoading(true);
    try {
        const res = await fetch('/api/preprocess-polar/warp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId:           AppState.userId,
                filename:         pp.sourceFilename,
                center_x:         cur.center_x,
                center_y:         cur.center_y,
                r_start_px:       cur.r_inner_px,
                r_end_px:         cur.r_outer_px,
                theta_start:      cur.theta_start,
                theta_end:        cur.theta_end,
                nr:               cur.nr,
                ntheta:           cur.ntheta,
                r_orientation:    cur.r_orientation,
                r_outer_physical: cur.r_outer_physical,
                // Always overwrite the deterministic preview file -- the
                // user gets a permanent file only when they press
                // "Save image".
                preview: true,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'warp failed');
        pp.lastPreview = {
            filename:     res.filename,
            path:         res.path,
            polar_domain: res.polar_domain,
            width:        res.output_width,
            height:       res.output_height,
        };
        pp.isDirty = false;
        const badge = document.getElementById('polarDirtyBadge');
        if (badge) badge.classList.remove('active');
        renderPolarPreviewThumb(res.path);
    } catch (err) {
        showStatus('solverStatus', `Polar warp failed: ${err.message}`, 'error');
    } finally {
        pp.isWarping = false;
        setPolarWarpLoading(false);
    }
}

function renderPolarPreviewThumb(path) {
    const canvas = document.getElementById('polarPreviewThumb');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            const w = canvas.width;
            const h = canvas.height;
            ctx.fillStyle = '#222';
            ctx.fillRect(0, 0, w, h);
            const ratio = Math.min(w / img.naturalWidth, h / img.naturalHeight);
            const dw = img.naturalWidth * ratio;
            const dh = img.naturalHeight * ratio;
            ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
        };
        img.onerror = () => {
            ctx.fillStyle = '#222'; ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#888'; ctx.font = '12px sans-serif';
            ctx.fillText('preview unavailable', 8, 80);
        };
        img.src = `${path}?t=${Date.now()}`;
    }
    // Phase 5f.2: the big <img> in the "Warp output" tab gets the same
    // URL with a cache-busting query so the user sees the result of every
    // warp at full size + zoom/pan.
    const warpImg = document.getElementById('polarWarpImg');
    if (warpImg) {
        warpImg.src = `${path}?t=${Date.now()}`;
    }
    // Width x Height badge next to the tab.
    const badge = document.getElementById('polarWarpDimsBadge');
    const pp = AppState.polarPreprocess;
    if (badge && pp.lastPreview && pp.lastPreview.width) {
        badge.textContent = `${pp.lastPreview.width} × ${pp.lastPreview.height} px`;
    }
}

// Tab switch in the preview pane. Each view has its own zoom/pan
// instance so the source overlay stays at 100 % when the user is
// inspecting the warp at 5x, and vice versa.
function switchPolarView(mode) {
    const srcWrap = document.getElementById('polarPreviewContainer');
    const wrpWrap = document.getElementById('polarWarpContainer');
    const srcTab  = document.getElementById('polarViewTabSource');
    const wrpTab  = document.getElementById('polarViewTabWarp');
    if (!srcWrap || !wrpWrap) return;
    if (mode === 'warp') {
        srcWrap.style.display = 'none';
        wrpWrap.style.display = 'flex';
        srcTab.classList.remove('polar-view-tab-active');
        wrpTab.classList.add('polar-view-tab-active');
        attachZoomPan(
            wrpWrap,
            document.getElementById('polarWarpZoomStage'),
            { indicatorEl: document.getElementById('polarWarpZoomIndicator') }
        );
    } else {
        wrpWrap.style.display = 'none';
        srcWrap.style.display = 'flex';
        srcTab.classList.add('polar-view-tab-active');
        wrpTab.classList.remove('polar-view-tab-active');
    }
}

function setPolarWarpLoading(on) {
    const el = document.getElementById('polarWarpLoading');
    if (el) el.classList.toggle('active', !!on);
}

// Dip-candidate dropdown (Phase 5d.2). Backend returns
// detection.dip_candidates = [{r, ratio, inner_band, outer_band, score}]
// sorted best-first; the dropdown lets the user pick one as r_inner when
// the highest-scored guess isn't the physical air gap (e.g. an aux
// mid-yoke gap outscoring the rotor/stator gap on a multi-gap design).
function renderPolarDipDropdown() {
    const row = document.getElementById('polarDipRow');
    const sel = document.getElementById('polarDipSelect');
    const tuneRow = document.getElementById('polarAirGapTuneRow');
    if (!row || !sel) return;
    const det = AppState.polarPreprocess.detection;
    const list = (det && Array.isArray(det.dip_candidates)) ? det.dip_candidates : [];
    // Always show the row when there is at least one candidate so the
    // marker can be turned off too.
    if (list.length === 0) {
        row.style.display = 'none';
        sel.innerHTML = '';
        if (tuneRow) tuneRow.style.display = 'none';
        return;
    }
    row.style.display = 'flex';
    const opts = ['<option value="0">— none (marker off) —</option>']
        .concat(list.map((c, i) => {
            const marker = (i === 0) ? '★' : ' ';
            return `<option value="${c.r}">${marker} r=${c.r}  ratio=${c.ratio.toFixed(2)}  score=${c.score}</option>`;
        }));
    sel.innerHTML = opts.join('');
    const cur = AppState.polarPreprocess.current;
    const match = list.findIndex(c => c.r === cur.air_gap_px);
    sel.selectedIndex = (match >= 0) ? match + 1 : (cur.air_gap_px > 0 ? 1 : 0);
    if (!sel._ppBound) {
        sel.addEventListener('change', () => {
            const v = Number(sel.value);
            AppState.polarPreprocess.current.air_gap_px = Number.isFinite(v) && v > 0 ? v : 0;
            // Phase D.3: a different dip resets the offset (offset is
            // relative to the selected dip, not absolute).
            AppState.polarPreprocess.current.air_gap_offset_px = 0;
            updatePolarAirGapTuneRow();
            renderPolarOverlay();
            syncPolarInputsFromState();
            // No markPolarDirty -- air-gap marker is purely informational
            // and doesn't affect the warp output.
        });
        sel._ppBound = true;
    }
    updatePolarAirGapTuneRow();
}

// Phase D.3: tune-row visibility is gated on (a) at least one dip
// candidate being available and (b) save target being 'polar' (slide
// regions belong to the polar warp pipeline). Called from
// renderPolarDipDropdown and from the save_as radio handler.
function updatePolarAirGapTuneRow() {
    const tuneRow = document.getElementById('polarAirGapTuneRow');
    const sideRow = document.getElementById('polarAirGapSideRow');
    if (!tuneRow) return;
    const det = AppState.polarPreprocess.detection;
    const list = (det && Array.isArray(det.dip_candidates)) ? det.dip_candidates : [];
    const cur = AppState.polarPreprocess.current;
    const visible = list.length > 0 && cur.save_as === 'polar';
    tuneRow.style.display = visible ? 'flex' : 'none';
    if (sideRow) {
        sideRow.style.display = (visible && cur.air_gap_as_slide) ? 'inline-flex' : 'none';
    }
}

// Phase D.3: tune-row listeners are bound once per modal lifetime.
// Called from openPolarPreprocessModal alongside the other bindings.
function bindPolarAirGapTuneInputs() {
    const offsetEl = document.getElementById('polarAirGapOffset');
    const slideEl  = document.getElementById('polarAirGapAddSlide');
    const sideEls  = document.querySelectorAll('input[name="polarAirGapSide"]');
    if (offsetEl && !offsetEl._ppBound) {
        offsetEl.addEventListener('input', () => {
            const v = Number(offsetEl.value);
            AppState.polarPreprocess.current.air_gap_offset_px =
                Number.isFinite(v) ? Math.round(v) : 0;
            renderPolarOverlay();
        });
        offsetEl.addEventListener('wheel', e => {
            e.preventDefault();
            const step = e.shiftKey ? 10 : 1;
            const dir = e.deltaY < 0 ? 1 : -1;
            offsetEl.value = Number(offsetEl.value || 0) + step * dir;
            offsetEl.dispatchEvent(new Event('input', { bubbles: true }));
        }, { passive: false });
        offsetEl._ppBound = true;
    }
    if (slideEl && !slideEl._ppBound) {
        slideEl.addEventListener('change', () => {
            AppState.polarPreprocess.current.air_gap_as_slide = slideEl.checked;
            updatePolarAirGapTuneRow();
        });
        slideEl._ppBound = true;
    }
    sideEls.forEach(r => {
        if (r._ppBound) return;
        r.addEventListener('change', () => {
            if (r.checked) {
                AppState.polarPreprocess.current.air_gap_slide_side = r.value;
            }
        });
        r._ppBound = true;
    });
}

// Copies the currently-selected air-gap marker into r_inner so the warp
// excludes the inside region. Reverse direction from the new default
// (full machine in the warp). The user can always reset r_inner to 0
// by typing it in or by clicking the dip dropdown's "none" entry first.
function useAirGapAsInner() {
    const cur = AppState.polarPreprocess.current;
    if (!(cur.air_gap_px > 0)) return;
    // Phase D.3: copy the *effective* radius (dip + offset) so a manually
    // nudged marker still snaps to r_inner. Then zero the offset so the
    // dropdown choice stays the visible anchor.
    cur.r_inner_px = airGapEffectiveR();
    cur.air_gap_offset_px = 0;
    cur.nr = Math.max(2, cur.r_outer_px - cur.r_inner_px);
    renderPolarOverlay();
    syncPolarInputsFromState();
    markPolarDirty();
}

function bindPolarInputs() {
    const root = document.getElementById('polarPreprocessModal');
    if (!root || root._ppInputsBound) return;
    const cur = () => AppState.polarPreprocess.current;

    // numericMap: data-pp key -> [state field, parser]
    const numericMap = {
        center_x:           ['center_x',           v => Math.round(Number(v))],
        center_y:           ['center_y',           v => Math.round(Number(v))],
        r_inner_px:         ['r_inner_px',         v => Math.max(0, Math.round(Number(v)))],
        r_outer_px:         ['r_outer_px',         v => Math.max(1, Math.round(Number(v)))],
        r_outer_physical:   ['r_outer_physical',   v => Math.max(0.001, Number(v))],
        nr:                 ['nr',                 v => Math.max(2, Math.round(Number(v)))],
        ntheta:             ['ntheta',             v => Math.max(2, Math.round(Number(v)))],
        // Theta inputs are shown in CCW (math) convention. The internal
        // state and the SVG / warp pipelines use the image-coord (Y-down,
        // CW visually) convention, so we negate at the UI boundary.
        theta_start_deg:    ['theta_start',        v => -Number(v) * Math.PI / 180],
        theta_end_deg:      ['theta_end',          v => -Number(v) * Math.PI / 180],
    };
    root.querySelectorAll('input[type=number]').forEach(input => {
        const key = input.dataset.pp;
        const spec = numericMap[key];
        if (!spec) return;
        const [stateField, parse] = spec;
        input.addEventListener('input', () => {
            cur()[stateField] = parse(input.value);
            renderPolarOverlay();
            markPolarDirty();
        });
        input.addEventListener('wheel', e => {
            e.preventDefault();
            const step = Number(input.step) || 1;
            const factor = e.shiftKey ? 10 : 1;
            const dir = e.deltaY < 0 ? 1 : -1;
            const v = Number(input.value) + step * factor * dir;
            input.value = v;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }, { passive: false });
    });

    const snap = document.getElementById('polarSnapNtheta');
    if (snap) snap.addEventListener('change', () => {
        cur().snap_ntheta = snap.checked;
        markPolarDirty();
    });

    root.querySelectorAll('input[name=polarThetaMode]').forEach(r => {
        r.addEventListener('change', () => {
            if (!r.checked) return;
            cur().is_sector = (r.value === 'sector');
            document.getElementById('polarSectorInputs').style.display =
                cur().is_sector ? '' : 'none';
            // Initialise sector range if just switched on with full-circle
            // theta. State stays in image-coord (Y-down) convention but
            // the chosen pair displays as a clean increasing CCW range
            // (-45°, +45° in the user-facing math convention).
            if (cur().is_sector && Math.abs(cur().theta_end - cur().theta_start - 2 * Math.PI) < 1e-6) {
                cur().theta_start =  Math.PI / 4;   // displays as -45°
                cur().theta_end   = -Math.PI / 4;   // displays as +45°
            }
            renderPolarOverlay();
            syncPolarInputsFromState();
            markPolarDirty();
        });
    });
    root.querySelectorAll('input[name=polarROrient]').forEach(r => {
        r.addEventListener('change', () => {
            if (r.checked) { cur().r_orientation = r.value; markPolarDirty(); }
        });
    });
    root.querySelectorAll('input[name=polarSaveAs]').forEach(r => {
        r.addEventListener('change', () => {
            if (r.checked) {
                cur().save_as = r.value;
                // Phase D.3: slide-region tuning UI is polar-only.
                updatePolarAirGapTuneRow();
            }
        });
    });
    root._ppInputsBound = true;
}

function attachPolarSvgDrag() {
    const svg = document.getElementById('polarOverlay');
    if (!svg || svg._ppDragBound) return;
    function svgPoint(e) {
        const pt = svg.createSVGPoint();
        pt.x = e.clientX; pt.y = e.clientY;
        const ctm = svg.getScreenCTM();
        return ctm ? pt.matrixTransform(ctm.inverse()) : { x: 0, y: 0 };
    }
    svg.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        const handle = e.target.closest('[data-handle]');
        if (!handle) return;
        e.preventDefault();
        e.stopPropagation();
        const kind = handle.dataset.handle;
        const p0 = svgPoint(e);
        const cur = AppState.polarPreprocess.current;
        AppState.polarPreprocess._activeDrag = {
            kind,
            p0,
            start: {
                cx: cur.center_x, cy: cur.center_y,
                rIn: cur.r_inner_px, rOut: cur.r_outer_px,
                ts: cur.theta_start, te: cur.theta_end,
            },
        };
        document.body.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', e => {
        const drag = AppState.polarPreprocess._activeDrag;
        if (!drag) return;
        const p = svgPoint(e);
        const cur = AppState.polarPreprocess.current;
        const W = AppState.polarPreprocess.imageNaturalWidth;
        const H = AppState.polarPreprocess.imageNaturalHeight;
        switch (drag.kind) {
            case 'center': {
                cur.center_x = Math.max(0, Math.min(W, Math.round(drag.start.cx + (p.x - drag.p0.x))));
                cur.center_y = Math.max(0, Math.min(H, Math.round(drag.start.cy + (p.y - drag.p0.y))));
                break;
            }
            case 'inner': {
                const dx = p.x - drag.start.cx, dy = p.y - drag.start.cy;
                cur.r_inner_px = Math.max(0, Math.round(Math.hypot(dx, dy)));
                break;
            }
            case 'outer': {
                const dx = p.x - drag.start.cx, dy = p.y - drag.start.cy;
                cur.r_outer_px = Math.max(1, Math.round(Math.hypot(dx, dy)));
                break;
            }
            case 'theta_start':
                cur.theta_start = Math.atan2(p.y - drag.start.cy, p.x - drag.start.cx);
                break;
            case 'theta_end':
                cur.theta_end   = Math.atan2(p.y - drag.start.cy, p.x - drag.start.cx);
                break;
            case 'airgap': {
                // Phase D.3: adjust the offset relative to the detected
                // dip so the dropdown choice stays the anchor and re-
                // selecting a different dip drops the offset cleanly.
                const dx = p.x - drag.start.cx, dy = p.y - drag.start.cy;
                const rTotal = Math.max(0, Math.round(Math.hypot(dx, dy)));
                cur.air_gap_offset_px = rTotal - (cur.air_gap_px || 0);
                break;
            }
        }
        renderPolarOverlay();
        syncPolarInputsFromState();
        markPolarDirty();
    });
    window.addEventListener('mouseup', () => {
        if (AppState.polarPreprocess._activeDrag) {
            AppState.polarPreprocess._activeDrag = null;
            document.body.style.cursor = '';
        }
    });
    svg._ppDragBound = true;
}

function attachPolarKeyboard() {
    const container = document.getElementById('polarPreviewContainer');
    if (!container || container._ppKeysBound) return;
    container.addEventListener('keydown', e => {
        if (document.activeElement && /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
        e.preventDefault();
        const cur = AppState.polarPreprocess.current;
        const step = e.shiftKey ? 10 : 1;
        if (e.key === 'ArrowLeft')  cur.center_x -= step;
        if (e.key === 'ArrowRight') cur.center_x += step;
        if (e.key === 'ArrowUp')    cur.center_y -= step;
        if (e.key === 'ArrowDown')  cur.center_y += step;
        renderPolarOverlay();
        syncPolarInputsFromState();
        markPolarDirty();
    });
    container._ppKeysBound = true;
}

// Esc anywhere in the document closes the polar modal if it is open
// (after the existing dirty-confirmation in closePolarPreprocessModal).
// One global listener -- idempotent flag on document.body so reloads
// don't pile up duplicates.
function attachPolarEscKey() {
    if (document.body._ppEscBound) return;
    document.addEventListener('keydown', e => {
        if (e.key !== 'Escape') return;
        const modal = document.getElementById('polarPreprocessModal');
        if (modal && modal.style.display === 'flex') {
            e.preventDefault();
            closePolarPreprocessModal(false);
        }
    });
    document.body._ppEscBound = true;
}

// Re-runs /api/preprocess-polar/detect with the current image and shape
// hint. Used by the "Re-detect" button in the Auto-detect section.
async function rerunPolarDetect() {
    const pp = AppState.polarPreprocess;
    if (pp.isDetecting || !pp.sourceFilename) return;
    pp.isDetecting = true;
    setPolarLoading(true, 'Re-detecting…');
    try {
        const res = await fetch('/api/preprocess-polar/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId:   AppState.userId,
                filename: pp.sourceFilename,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'detect failed');
        pp.detection = res;
        applyDetectionToCurrent(res);
        renderPolarStatusSummary();
        renderPolarOverlay();
        syncPolarInputsFromState();
        renderPolarDipDropdown();
    } catch (err) {
        showStatus('solverStatus', `Re-detect failed: ${err.message}`, 'error');
    } finally {
        pp.isDetecting = false;
        setPolarLoading(false);
    }
}

function snapToDetectedPeriod() {
    const pp = AppState.polarPreprocess;
    const per = pp.detection && pp.detection.periodicity;
    const N = per && per.grayscale && per.grayscale.n_integer;
    if (!N || N < 2) {
        showStatus('solverStatus', 'No detected period to snap to', 'error');
        return;
    }
    const cur = pp.current;
    cur.is_sector = true;
    // Snap one sector with start=0 and width 2pi/N going CCW in the
    // user-facing math convention. The internal state is in image-coord
    // (Y-down) so the end angle is negated.
    cur.theta_start = 0;
    cur.theta_end   = -2 * Math.PI / N;
    document.getElementById('polarSectorInputs').style.display = '';
    renderPolarOverlay();
    syncPolarInputsFromState();
    markPolarDirty();
}

function applyPolarTransform() {
    const pp = AppState.polarPreprocess;
    if (pp._debounceTimer) { clearTimeout(pp._debounceTimer); pp._debounceTimer = null; }
    triggerPolarPreviewWarp();
}

// "Save image" -- polar mode only. Triggers a fresh warp with preview=false
// so the backend writes a permanent unique filename (e.g. motor_polar.png,
// motor_polar_1.png, ...). The cartesian case has nothing to save here --
// the source file already exists in /uploads -- so we show a hint instead.
async function savePolarImage() {
    const pp = AppState.polarPreprocess;
    const cur = pp.current;
    if (!pp.sourceFilename) {
        showStatus('solverStatus', 'No source image loaded', 'error');
        return;
    }
    if (cur.save_as === 'cartesian') {
        showStatus('solverStatus',
            'Cartesian save target: nothing to save here (the source image already exists in /uploads).',
            'info');
        return;
    }
    if (!(cur.r_outer_px > cur.r_inner_px)) {
        showStatus('solverStatus', 'r_outer must be greater than r_inner', 'error');
        return;
    }
    const btn = document.getElementById('polarSaveImageBtn');
    if (btn) btn.disabled = true;
    setPolarWarpLoading(true);
    try {
        const res = await fetch('/api/preprocess-polar/warp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId:           AppState.userId,
                filename:         pp.sourceFilename,
                center_x:         cur.center_x,
                center_y:         cur.center_y,
                r_start_px:       cur.r_inner_px,
                r_end_px:         cur.r_outer_px,
                theta_start:      cur.theta_start,
                theta_end:        cur.theta_end,
                nr:               cur.nr,
                ntheta:           cur.ntheta,
                r_orientation:    cur.r_orientation,
                r_outer_physical: cur.r_outer_physical,
                preview: false,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'save failed');
        // Track the saved image separately from the live preview. Insert
        // YAML will reference this one if present.
        pp.lastSaved = {
            filename:     res.filename,
            path:         res.path,
            polar_domain: res.polar_domain,
            width:        res.output_width,
            height:       res.output_height,
        };
        await refreshImageList();
        // Surface the new file in the image dropdown selection.
        const sel = document.getElementById('imageSelect');
        if (sel) sel.value = res.filename;
        AppState.uploadedImageFilename = res.filename;
        try { loadSelectedImage(); } catch { /* best effort */ }
        showStatus('solverStatus',
            `Saved warped image: ${res.filename} (${res.output_width} × ${res.output_height} px). ` +
            `Press "Insert YAML" to add the polar_domain block to the editor.`,
            'success');
    } catch (err) {
        showStatus('solverStatus', `Save image failed: ${err.message}`, 'error');
    } finally {
        if (btn) btn.disabled = false;
        setPolarWarpLoading(false);
    }
}

// "Insert YAML" -- writes coordinate_system + polar_domain + image_path
// (polar mode) or coordinate_system + image_path + mesh (cartesian) into
// the YAML editor. Independent from "Save image"; the polar branch
// prefers the most recently saved image, falling back to the preview
// filename and warning the user that they're referring to a transient
// preview file.
// Pick the θ boundary condition from Δθ relative to the detected polar
// pitch Θp = 2π/N. Even-integer multiples → periodic, odd-integer →
// anti-periodic (sign-flipped periodic), otherwise → Dirichlet.
// Returns { kind, label, bcFlow } where bcFlow is the inline yaml fragment
// "type: ..., value: ..." that goes inside `{ ... }`.
function pickThetaBoundary(theta_range, N) {
    if (!(N >= 2)) {
        return {
            kind: 'dirichlet',
            label: 'no rotational period detected — defaulting to Az = 0 on θ boundaries',
            bcFlow: 'type: dirichlet, value: 0.0',
        };
    }
    const Theta_p = 2 * Math.PI / N;
    const ratio = Math.abs(theta_range) / Theta_p;
    const rounded = Math.round(ratio);
    const tol = 0.05;
    if (Math.abs(ratio - rounded) > tol || rounded < 1) {
        return {
            kind: 'dirichlet',
            label: `Δθ / Θp = ${ratio.toFixed(3)} (Θp = 2π/${N}), not an integer multiple → Az = 0 on θ boundaries`,
            bcFlow: 'type: dirichlet, value: 0.0',
        };
    }
    if (rounded % 2 === 0) {
        return {
            kind: 'periodic',
            label: `Δθ = ${rounded}·Θp (even multiple of pole pitch 2π/${N}) → periodic θ boundary`,
            bcFlow: 'type: periodic,  value: 1.0',
        };
    }
    return {
        kind: 'antiperiodic',
        label: `Δθ = ${rounded}·Θp (odd multiple of pole pitch 2π/${N}) → anti-periodic θ boundary`,
        bcFlow: 'type: periodic,  value: -1.0',
    };
}

// Phase K: split the theta-axis pixel count T into integer (N_step,
// N_slide) such that N_step*N_slide simulates exactly one full rotor
// revolution. Constraints, in order of priority:
//   1. Both are integers (loop over divisors).
//   2. N_step * N_slide = T (exact factorisation if possible).
//   3. 32 ≤ N_step ≤ 256 (so the time resolution sits in a sensible
//      band: not coarse, not absurdly fine).
//   4. Prefer the SMALLEST N_slide so individual steps are smooth.
//      With N_step = T fitting in range, N_slide = 1 (the natural
//      "one pixel per step" cadence) is chosen.
// Fallback when T has no suitable divisor: N_step = 100,
// N_slide = round(T/100). N_step*N_slide then approximates T but
// won't land exactly -- the user gets a reasonable schedule and can
// retune in the editor.
function computePolarSlideSchedule(theta_size) {
    const T = Math.max(1, Math.round(theta_size));
    const k_max = Math.max(1, Math.floor(T / 32));
    for (let k = 1; k <= k_max; k++) {
        if (T % k !== 0) continue;
        const N_step = T / k;
        if (N_step >= 32 && N_step <= 256) {
            return { N_step, N_slide: k, exact: true };
        }
    }
    return { N_step: 100, N_slide: Math.max(1, Math.round(T / 100)), exact: false };
}

// Phase K: return the slide schedule (N_step, N_slide) for the current
// polar warp, based on the current ntheta pixel count.
function getPolarSlideSchedule() {
    const cur = AppState.polarPreprocess.current;
    const T = Math.max(1, Math.round(Number(cur.ntheta) || 0));
    return computePolarSlideSchedule(T);
}

// Phase F.4 / N: render a numeric theta_range value as a tinyexpr
// expression using bare `pi` (tinyexpr built-in) when it matches a
// simple rational multiple of π. `pi` works directly inside any field
// that the solver routes through tinyexpr (Phase N moved transient
// fields onto that path); $pi also works via the global substitution
// pass, but `pi` is shorter and renders cleaner in the editor.
function thetaToTinyExpr(theta_range) {
    if (!isFinite(theta_range) || theta_range <= 0) return String(theta_range);
    const r = theta_range / Math.PI;
    const tol = 1e-6;
    const intMul = Math.round(r);
    if (Math.abs(r - intMul) < tol && intMul >= 1 && intMul <= 12) {
        return (intMul === 1) ? 'pi' : `${intMul}*pi`;
    }
    for (let den = 2; den <= 16; den++) {
        for (let num = 1; num < den * 4; num++) {
            if (Math.abs(r - num / den) < tol) {
                if (num === 1) return `pi/${den}`;
                return `${num}*pi/${den}`;
            }
        }
    }
    return String(theta_range);
}

// =====================================================================
// Phase BA: optional-controls hint block
// =====================================================================
//
// A canonical block of commented YAML that surfaces the existence of
// nonlinear-solver + coarsening knobs the solver supports but that
// don't need to be on in every config. We append it to every YAML
// emitted by the polar / cartesian "Insert YAML" path AND re-append
// it after Detect Colors's jsyaml.dump round-trip (which strips
// comments). The marker on the first line lets ensureSolverHintBlock
// detect prior insertion so we never append twice.
const SOLVER_HINT_MARKER = '# --- Optional controls (uncomment + edit as needed) ---';
const SOLVER_HINT_BLOCK = `
${SOLVER_HINT_MARKER}
# These controls matter only when a material has a B-H curve or field-dependent
# permeability. Uncomment the block when you want to trade accuracy, robustness,
# and run time explicitly; for linear materials, leaving it commented is best.
#
# nonlinear_solver:
#   enabled: true                   # enables the nonlinear material response
#   solver_type: newton-krylov      # picard is slower but can be more forgiving
#   max_iterations: 100             # reaching this limit completes with a nonlinear convergence warning (exit code 2)
#   tolerance: 1.0e-3               # accept results only when the reported residual is below this
#   verbose: false                  # true prints iteration diagnostics for troubleshooting
#   anderson:
#     enabled: false                # safest NK baseline; safeguarded AA remains experimental
#     depth: 5
#     beta: 0.3                     # NK default when omitted; smaller is more conservative
#   # Eisenstat-Walker adapts the inner linear-solver tolerance: usually faster,
#   # while preserving the requested outer tolerance. Disable only for diagnostics.
#   eisenstat_walker:
#     enabled: true
#     gamma: 0.9
#     alpha: 2.0
#     eta_min: 1.0e-6
#     eta_max: 0.1
#
# If a step says NOT CONVERGED, its fields are diagnostic only. Increase the
# iteration limit, justify a looser tolerance, or disable EW for comparison.
# Never compare field maps from runs unless both satisfy their tolerance.
#
# Do not add the old coarsen/coarsening keys: they are ignored in v1.6.1.
`;

// Append the SOLVER_HINT_BLOCK to the YAML string iff the marker isn't
// already present. Used by every code path that hands the user a YAML
// document so the hint survives Detect Colors' jsyaml.dump round-trip
// (which would otherwise drop every comment in the file).
function ensureSolverHintBlock(yamlString) {
    if (typeof yamlString !== 'string') return yamlString;
    if (yamlString.indexOf(SOLVER_HINT_MARKER) !== -1) return yamlString;
    const sep = yamlString.endsWith('\n') ? '' : '\n';
    return yamlString + sep + SOLVER_HINT_BLOCK;
}

// v1.6 domain decomposition: optional, COMMENTED-OUT block appended to the
// auto-generated POLAR config (polar only -- DD is a polar feature). The user
// uncomments + tunes the bands to enable a variable-resolution Schwarz solve.
const DD_HINT_MARKER = '# --- Optional: domain decomposition (polar, variable-resolution accuracy mode) ---';
const DD_HINT_BLOCK = `
${DD_HINT_MARKER}
# Opt-in accuracy mode for polar models. It keeps the air gap and coils fine
# while reducing resolution in smooth iron. This can lower memory, but is not
# guaranteed to be faster. Band limits are RADIAL PIXEL INDICES (0..nr).
# Keep every material interface inside a cf=1 band; cf>1 means fewer cells.
# domain_decomposition:
#   enabled: true
#   bands:
#     - [0, 52, 4, 4]      # example only: replace limits for your image
#     - [52, 330, 1, 1]    # example active band: preserve full resolution
#     - [330, 450, 4, 4]   # example smooth yoke
#   robin_p: 12.0          # larger couples bands more strongly, but costs work
#   overlap: 4             # larger is more stable, but uses more memory
#   max_outer: 8           # more sweeps improve agreement between bands
#   tol: 1.0e-3            # smaller gives a stricter interface match
`;
function ensureDDHintBlock(yamlString) {
    if (typeof yamlString !== 'string') return yamlString;
    if (yamlString.indexOf(DD_HINT_MARKER) !== -1) return yamlString;
    const sep = yamlString.endsWith('\n') ? '' : '\n';
    return yamlString + sep + DD_HINT_BLOCK;
}

// Build the polar coordinate_system / polar_domain / boundary block as a
// hand-rolled YAML string. `jsyaml.dump` strips comments, so the block
// has to be assembled as text instead of through the dumper.
function buildPolarYamlBlock(filename, polarDomain) {
    const pp = AppState.polarPreprocess;
    const det = pp.detection || {};
    const per = det.periodicity || {};
    const N = (per.grouped && per.grouped.n_integer)
           || (per.rgb && per.rgb.n_integer)
           || (per.grayscale && per.grayscale.n_integer)
           || null;
    const theta_range = (polarDomain && polarDomain.theta_range) || 2 * Math.PI;
    const bc = pickThetaBoundary(theta_range, N);
    const rs = (polarDomain && polarDomain.r_start  != null) ? polarDomain.r_start  : 0;
    const re = (polarDomain && polarDomain.r_end    != null) ? polarDomain.r_end    : 1;
    const ro = (polarDomain && polarDomain.r_orientation)    ? polarDomain.r_orientation : 'horizontal';
    const lines = [];
    lines.push('# Auto-generated starting point. Review scale, materials, and boundaries before solving.');
    if (N) {
        const Theta_p_deg = (360 / N).toFixed(3);
        const dtheta_deg = (theta_range * 180 / Math.PI).toFixed(3);
        lines.push(`# Detected periodicity N = ${N} → polar pitch Θp = 360°/${N} ≈ ${Theta_p_deg}°.`);
        lines.push(`# Sector Δθ = ${dtheta_deg}°.`);
        lines.push(`# ${bc.label}`);
    } else {
        lines.push('# No rotational periodicity detected; θ boundaries set to Dirichlet (Az = 0).');
    }
    lines.push('coordinate_system: polar');
    lines.push('polar_domain:');
    lines.push('  # Physical radii in metres: changing them changes the image scale and field gradients.');
    lines.push('  # r_start: 0 models the full disc; use a positive value for an annular cutout.');
    lines.push(`  r_start: ${rs}`);
    lines.push(`  r_end: ${re}`);
    lines.push(`  r_orientation: ${ro}`);
    lines.push(`  theta_range: ${thetaToTinyExpr(theta_range)}`);
    lines.push('polar_boundary_conditions:');
    lines.push('  inner:     { type: dirichlet, value: 0.0 }   # rotor axis / r_inner — Az = 0');
    lines.push('  outer:     { type: dirichlet, value: 0.0 }   # stator OD / r_outer — Az = 0');
    lines.push(`  theta_min: { ${bc.bcFlow} }`);
    lines.push(`  theta_max: { ${bc.bcFlow} }`);
    // Phase E.2: image_path here is documentation only — the solver
    // takes the image as argv[2] from the CLI (which the WebUI auto-
    // sets to AppState.uploadedImageFilename = the saved warp output).
    // We still write the filename so the YAML self-describes which
    // image the polar_domain was authored for, and so re-importing the
    // YAML elsewhere preserves that link.
    lines.push('# The CLI image argument is authoritative; this filename records the image used to size the mesh.');
    lines.push(`image_path: ${filename}`);

    // Phase D.3 / Phase F.2: optional transient slide skeleton from the
    // air-gap marker. Polar save target only; the cartesian path emits
    // nothing here (the slide region wouldn't map cleanly without a
    // per-row angular axis). Phase F.2 switched to the legacy single-
    // slide format so the inserted transient block uses
    // `slide_direction / slide_region_* / slide_pixels_per_step` with
    // `total_steps: $N_step` -- the N_step variable is auto-added to
    // the editor's variables block by insertPolarYaml.
    const cur = AppState.polarPreprocess.current;
    const rAG = airGapEffectiveR();
    if (cur.air_gap_as_slide && rAG > 0) {
        const inside = (cur.air_gap_slide_side === 'inside');
        const rOuter = Math.max(1, cur.r_outer_px || 0);
        const region_start = inside ? 0   : rAG;
        const region_end   = inside ? rAG : Math.max(rAG + 1, rOuter);
        // Emit the slide region as PHYSICAL radii in metres (consistent with
        // polar_domain). Decimal literals are the solver's metre marker;
        // integer literals keep the legacy pixel meaning.
        // The air-gap marker is measured in SOURCE image pixels, while nr is
        // the user-selected OUTPUT resolution.  Do not divide by nr: that
        // makes changing nr silently move the physical slide band.  Map the
        // source radius through the actual warp interval instead.
        const warpStartPx = Math.max(0, Math.round(Number(cur.r_inner_px) || 0));
        const warpEndPx = Math.max(warpStartPx + 1, Math.round(Number(cur.r_outer_px) || 0));
        const pxToR = (px) => {
            const t = Math.max(0, Math.min(1, (Number(px) - warpStartPx) / (warpEndPx - warpStartPx)));
            return (rs + (re - rs) * t).toPrecision(8);
        };
        // Phase K: derive N_step / N_slide from ntheta so the slide
        // simulates one full rotation in N_step steps of N_slide pixels
        // each (subject to 32 ≤ N_step ≤ 256). The values themselves
        // live in the variables: block; this YAML only references the
        // $N_step / $N_slide tokens so the user can retune by editing
        // a single variable.
        // Slide direction = the warp axis perpendicular to r.
        const slideDir = (ro === 'horizontal') ? 'vertical' : 'horizontal';
        lines.push('');
        lines.push(`# Sliding ${inside ? 'the inner' : 'the outer'} air-gap band changes the field at each transient step.`);
        lines.push(`# Side: ${inside ? 'inside the air gap' : 'outside the air gap'}`);
        lines.push('# Decimal slide bounds are metres; integer bounds are legacy pixel indices.');
        lines.push('#   decimal literal (0.05)  = PHYSICAL metres — the solver converts to pixels');
        lines.push('#                             via the mesh (polar: radius; cartesian: x/y)');
        lines.push('#   integer literal (212)   = pixel index (legacy)');
        lines.push('# Increase total_steps for smoother motion (longer runtime); increase slide_pixels_per_step for larger jumps.');
        lines.push('transient:');
        lines.push('  enabled: true');
        lines.push('  enable_sliding: true');
        lines.push('  total_steps: $N_step');
        lines.push(`  slide_direction: ${slideDir}`);
        lines.push(`  slide_region_start: ${pxToR(region_start)}`);
        lines.push(`  slide_region_end: ${pxToR(region_end)}`);
        lines.push('  slide_pixels_per_step: $N_slide');
        lines.push('  # parallel_chunks: 3   # optional: shorter wall time, but roughly 3x memory');
    }
    // Phase BA: surface the nonlinear_solver / coarsening knobs even when
    // they aren't active in this template, so a user reading the inserted
    // YAML in Ace sees them as a discoverable optional section.
    // v1.6: also append the optional (commented) domain_decomposition block --
    // polar only, so the DD accuracy mode is discoverable from the generated YAML.
    return ensureDDHintBlock(ensureSolverHintBlock(lines.join('\n') + '\n'));
}

// Build the cartesian coordinate_system / mesh block. mesh.dx and .dy are
// auto-sized from the detected r_outer (in pixels) and the user-supplied
// r_outer_physical (in metres) so the user gets a meaningful default.
// Build a cartesian coordinate_system / mesh block. Used by two callers:
//   1) the Polar Preprocess modal's "save as cartesian" target (opts omitted;
//      the mesh is sized from the detected outer radius), and
//   2) the standalone "Cartesian / Linear template" button (opts carries the
//      image dimensions + physical scale so a linear-motor-style config —
//      periodic travel axis, linear slide skeleton — is emitted directly,
//      without going through polar detection).
// v1.6.1: enriched to emit boundary_conditions, a commented linear-slide
// transient skeleton, and a materials hint so the generated cartesian config
// is a usable starting point rather than a bare mesh.
function buildCartesianYamlBlock(filename, opts) {
    opts = opts || {};
    const lines = [];
    let dxdy;
    if (opts.dxdy != null && opts.dxdy > 0) {
        // Standalone path: mesh sized from a user-supplied physical width.
        dxdy = opts.dxdy;
        lines.push('# Auto-generated starting point for a Cartesian/linear model. Review scale and materials before solving.');
        if (opts.widthPx > 0 && opts.widthM > 0) {
            lines.push('# Smaller dx/dy resolves finer features but increases memory and runtime:');
            lines.push(`#   dx = dy = width_m / width_px = ${opts.widthM} / ${opts.widthPx} ≈ ${dxdy.toExponential(4)} m`);
        } else {
            lines.push(`# Mesh: dx = dy = ${dxdy.toExponential(4)} m (edit to match your image scale).`);
        }
    } else {
        // Polar-modal "save as cartesian" path: size from the detected radius.
        const pp = AppState.polarPreprocess;
        const cur = (pp && pp.current) || {};
        const det = (pp && pp.detection) || {};
        const r_outer_px = (cur.r_outer_px > 0) ? cur.r_outer_px : (det.r_outer_px || 0);
        const r_outer_m  = (cur.r_outer_physical > 0) ? cur.r_outer_physical : 1.0;
        lines.push('# Auto-generated by Polar Preprocess (cartesian save target).');
        if (r_outer_px > 0) {
            dxdy = r_outer_m / r_outer_px;
            lines.push('# Initial cell size from the detected physical radius; smaller cells increase resolution and cost:');
            lines.push(`#   dx = dy = r_outer_physical / r_outer_px = ${r_outer_m} / ${r_outer_px} ≈ ${dxdy.toExponential(4)} m`);
        } else {
            dxdy = 0.2e-3;
            lines.push('# Outer radius detection unavailable; falling back to dx = dy = 0.2 mm.');
            lines.push('# Adjust mesh.dx / mesh.dy to match your image scale.');
        }
    }
    lines.push('coordinate_system: cartesian');
    lines.push('mesh:');
    lines.push(`  dx: ${dxdy}`);
    lines.push(`  dy: ${dxdy}`);
    // Boundary conditions. For a linear machine (or an unwrapped rotary strip)
    // the travel axis is periodic (one pole pitch => anti-periodic via value: -1)
    // and the cross-stack axis is Dirichlet (Az = 0 on the outer yoke surfaces).
    lines.push('boundary_conditions:');
    lines.push('  # Travel axis (direction of motion): periodic for an infinite linear array.');
    lines.push('  # Use value: -1.0 for ANTI-periodic when the image is a single pole pitch.');
    lines.push('  left:   { type: periodic,  value: 1.0 }');
    lines.push('  right:  { type: periodic,  value: 1.0 }');
    lines.push('  # Cross-stack axis (across the air gap / yokes): Az = 0 on the outer surfaces.');
    lines.push('  top:    { type: dirichlet, value: 0.0 }');
    lines.push('  bottom: { type: dirichlet, value: 0.0 }');
    // Phase E.2: image_path is documentation only (solver reads argv[2]).
    lines.push('# The CLI image argument is authoritative; this filename records the image used to choose the mesh scale.');
    lines.push(`image_path: ${filename}`);
    // Materials: when the caller supplied a colour-detection result, embed a
    // materials: entry per detected colour (defaults: mu_r 1.0 / jz 0) so the
    // template is runnable immediately; the Detect Colors modal (opened right
    // after) refines these into presets / coils / magnets and its insert
    // REPLACES this block. Fall back to the old hint comment when detection
    // was unavailable.
    const detCols = opts.detectedColors;
    if (Array.isArray(detCols) && detCols.length > 0) {
        lines.push('# Detected regions start as air-like (mu_r 1, jz 0). Set steel, coil, and magnet properties before solving.');
        lines.push('materials:');
        for (const c of detCols) {
            const [r, g, b] = c.rgb;
            const hex = c.rgb.map(v => v.toString(16).padStart(2, '0')).join('');
            const ratio = (typeof c.ratio === 'number') ? ` (coverage: ${(c.ratio * 100).toFixed(1)}%)` : '';
            lines.push(`  material_${hex}:`);
            lines.push(`    rgb: [${r}, ${g}, ${b}]`);
            lines.push(`    mu_r: 1.0       # Set permeability${ratio}`);
            lines.push('    jz: 0.0');
            if (c.antialias === true) lines.push('    anti_aliasing: true');
        }
    } else {
        lines.push('# materials: run "Detect Colors & Generate YAML Template" on this image to');
        lines.push('#            populate rgb -> {mu_r | B-H | magnetization | jz} for each region.');
    }
    // Commented LINEAR-slide transient skeleton (the linear-motor analogue of the
    // rotary sweep). Left commented because the slide region is image-specific.
    lines.push('# --- Optional: linear motion (mover translation along the travel axis) ---');
    lines.push('# transient:');
    lines.push('#   enabled: true');
    lines.push('#   enable_sliding: true');
    lines.push('#   total_steps: $N_step');
    lines.push('#   slide_direction: horizontal   # shift columns along the travel axis');
    lines.push('#   # UNITS: decimal literal (0.05) = PHYSICAL metres (auto-converted to pixels');
    lines.push('#   #        via mesh dx/dy); integer literal (110) = pixel index (legacy).');
    lines.push('#   slide_region_start: 0.0       # start of the moving band [m]');
    lines.push(`#   slide_region_end: ${(opts.heightPx > 0 && opts.dxdy > 0 ? (opts.heightPx * opts.dxdy).toPrecision(6) : '0.1')}    # end of the moving band [m] (= full image height)`);
    lines.push('#   slide_pixels_per_step: $N_slide');
    lines.push('#   # parallel_chunks: 3          # shorter wall time, but roughly 3x memory');
    lines.push('# variables:');
    lines.push('#   N_step: 100');
    lines.push('#   N_slide: 2');
    // Optional-controls hint block (nonlinear_solver defaults, EW recommended).
    return ensureSolverHintBlock(lines.join('\n') + '\n');
}

// Standalone "Cartesian / Linear template" entry point. Generates a usable
// cartesian config from the currently uploaded image WITHOUT the polar
// detection flow (the user's linear-motor / unwrapped-strip use case). Mesh
// scale comes from a physical-width prompt; the rest is a linear-machine
// skeleton (periodic travel axis, Dirichlet cross-stack, commented linear
// slide). Saved as a new config and loaded, mirroring insertPolarYaml.
async function insertCartesianTemplate() {
    const filename = AppState.uploadedImageFilename;
    if (!filename) {
        showStatus('solverStatus', 'Upload an image first', 'error');
        return;
    }
    // Image pixel dimensions from the already-loaded preview element.
    const imgEl = document.getElementById('uploadedImage');
    const widthPx  = (imgEl && imgEl.naturalWidth)  || 0;
    const heightPx = (imgEl && imgEl.naturalHeight) || 0;
    // Physical width sets the mesh scale (the single most important number for
    // a cartesian run). Prompt with a sensible default; blank -> 0.2 mm,
    // Cancel -> abort without inserting anything.
    let dxdy = 0.2e-3, widthM = 0;
    const ans = window.prompt(
        'Physical width of the image in millimetres?\n' +
        '(sets the mesh: dx = dy = width / pixels. Leave blank for a 0.2 mm default.)',
        '100');
    if (ans === null) return;   // user pressed Cancel
    if (ans.trim() !== '' && widthPx > 0) {
        const mm = Number(ans);
        if (isFinite(mm) && mm > 0) { widthM = mm / 1000; dxdy = widthM / widthPx; }
    }
    // Colour detection (same engine as the Detect Colors modal) so the
    // generated template ships with a materials: entry per region — parity
    // with the polar flow's detection-driven material setup. Best-effort:
    // on failure the template falls back to the hint comment.
    showStatus('solverStatus', 'Detecting material colors…', 'info');
    let detectedColors = null;
    try {
        const det = await detectColorsInternal({ rareThreshold: 0.05, blendTolerance: 8 });
        if (det && Array.isArray(det.colors) && det.colors.length > 0) {
            detectedColors = det.colors;
        }
    } catch (_) { /* fall back to hint comment */ }

    const block = buildCartesianYamlBlock(filename, { dxdy, widthPx, heightPx, widthM, detectedColors });

    const baseName = String(filename).replace(/\.[^./\\]+$/, '');
    const newConfigName = `${baseName}_cartesian.yaml`;
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, file: newConfigName, content: block }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'save failed');
    } catch (err) {
        showStatus('solverStatus', `Failed to save "${newConfigName}": ${err.message}`, 'error');
        return;
    }
    try { await refreshConfigList(); } catch (_) { /* best effort */ }
    const select = document.getElementById('configFileSelect');
    if (select) { select.value = newConfigName; try { await loadConfig(); } catch (_) {} }
    else if (AppState.aceEditor) { AppState.aceEditor.setValue(block, -1); }
    if (typeof switchTab === 'function') switchTab('config');
    if (detectedColors) {
        showStatus('solverStatus',
            `New config "${newConfigName}" created with ${detectedColors.length} detected material(s). ` +
            `Assign presets / coils / magnetization in the Detect Colors dialog.`,
            'success');
        // Open the full assignment UI (library presets, coil groups,
        // magnetization editors). Its "Insert" REPLACES the placeholder
        // materials block we just wrote, so there is no duplication.
        try { await detectColors(); } catch (_) { /* modal is optional */ }
    } else {
        showStatus('solverStatus',
            `New config "${newConfigName}" created and loaded. ` +
            `Run Detect Colors to fill in materials, then adjust boundary_conditions / mesh.`,
            'success');
    }
}

// ===== Image Properties modal =====
// Inspect the loaded image: pixel + physical dimensions (pulled from the
// current YAML; sensible defaults when fields are missing), exact-RGB colour
// statistics matched against the YAML's materials, and a nearest-neighbour
// polar <-> cartesian transform preview (client-side canvas, no server call).

function imagePropsFmtM(v) {
    // metres -> readable mm / µm string
    if (!isFinite(v)) return '?';
    const mm = v * 1000;
    if (Math.abs(mm) >= 0.1 || mm === 0) return `${mm.toPrecision(4)} mm`;
    return `${(mm * 1000).toPrecision(4)} µm`;
}

function imagePropsEscape(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function closeImagePropertiesModal() {
    document.getElementById('imagePropsModal').style.display = 'none';
}

async function openImagePropertiesModal() {
    if (!AppState.uploadedImageFilename) {
        showStatus('solverStatus', 'Select an image first', 'error');
        return;
    }
    const modal = document.getElementById('imagePropsModal');
    modal.style.display = 'flex';
    document.getElementById('imagePropsDims').innerHTML = 'Loading…';
    document.getElementById('imagePropsColorTable').innerHTML = '';
    document.getElementById('imagePropsColorSummary').textContent = '';
    document.getElementById('imagePropsXformCaption').textContent = '';

    // Current YAML (best effort — missing/broken YAML falls back to defaults).
    let doc = {};
    try {
        doc = jsyaml.load(AppState.aceEditor ? AppState.aceEditor.getValue() : '') || {};
    } catch (_) { doc = {}; }

    // Load the image and mirror it onto the source canvas for pixel access.
    const img = new Image();
    try {
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = () => reject(new Error('image load failed'));
            img.src = `/uploads/${AppState.userId}/${AppState.uploadedImageFilename}?t=${Date.now()}`;
        });
    } catch (err) {
        document.getElementById('imagePropsDims').innerHTML =
            `<span style="color:#c00;">Failed to load image: ${imagePropsEscape(err.message)}</span>`;
        return;
    }
    const W = img.naturalWidth, H = img.naturalHeight;
    const srcCanvas = document.getElementById('imagePropsSrcCanvas');
    srcCanvas.width = W; srcCanvas.height = H;
    const sctx = srcCanvas.getContext('2d');
    sctx.drawImage(img, 0, 0);
    let pix = null;
    try { pix = sctx.getImageData(0, 0, W, H); } catch (_) { pix = null; }

    // ---- Dimensions & representative physical sizes ----
    const isPolar = (doc.coordinate_system === 'polar');
    const lines = [];
    lines.push(`<b>${imagePropsEscape(AppState.uploadedImageFilename)}</b> — ${W} × ${H} px`);
    const pd = doc.polar_domain || {};
    const rOrient = (pd.r_orientation === 'vertical') ? 'vertical' : 'horizontal';
    const rs = evalYamlNumber(pd.r_start, 0);
    const re_ = evalYamlNumber(pd.r_end, 0.1);
    const thr = evalYamlNumber(pd.theta_range, 2 * Math.PI);
    if (isPolar) {
        const nR = (rOrient === 'vertical') ? H : W;
        const nTh = (rOrient === 'vertical') ? W : H;
        const dr = (re_ - rs) / Math.max(1, nR - 1);
        const dth = thr / Math.max(1, nTh);
        const defaulted = (pd.r_start == null || pd.r_end == null)
            ? ' <span style="color:#b58900;">(polar_domain incomplete — defaults substituted)</span>' : '';
        lines.push(`coordinate_system: <b>polar</b>, r_orientation: ${rOrient}${defaulted}`);
        lines.push(`annulus: r = ${imagePropsFmtM(rs)} … ${imagePropsFmtM(re_)}` +
                   ` (radial depth ${imagePropsFmtM(re_ - rs)}), θ range = ${(thr * 180 / Math.PI).toFixed(1)}°`);
        lines.push(`resolution: Δr ≈ ${imagePropsFmtM(dr)}/px, Δθ ≈ ${(dth * 180 / Math.PI).toFixed(4)}°/px,` +
                   ` arc length @ mean radius ≈ ${imagePropsFmtM(dth * (rs + re_) / 2)}/px`);
    } else {
        const mesh = doc.mesh || {};
        const dx = evalYamlNumber(mesh.dx, 1e-3);
        const dy = evalYamlNumber(mesh.dy, dx);
        const defaulted = (mesh.dx == null)
            ? ' <span style="color:#b58900;">(mesh missing — default dx = dy = 1 mm)</span>' : '';
        lines.push(`coordinate_system: <b>cartesian</b>${defaulted}`);
        lines.push(`mesh: dx = ${imagePropsFmtM(dx)}, dy = ${imagePropsFmtM(dy)}` +
                   ` → physical ${imagePropsFmtM(W * dx)} × ${imagePropsFmtM(H * dy)}`);
    }
    document.getElementById('imagePropsDims').innerHTML = lines.join('<br>');

    // ---- Colour statistics (exact RGB, matched against YAML materials) ----
    if (pix) {
        const counts = new Map();
        const stride = (W * H > 6e6) ? 2 : 1;   // stay responsive on huge images
        let sampled = 0;
        for (let j = 0; j < H; j += stride) {
            for (let i = 0; i < W; i += stride) {
                const p = (j * W + i) * 4;
                const key = (pix.data[p] << 16) | (pix.data[p + 1] << 8) | pix.data[p + 2];
                counts.set(key, (counts.get(key) || 0) + 1);
                sampled++;
            }
        }
        // Materials in the YAML, by packed RGB, for name badges.
        const matByKey = new Map();
        for (const [name, props] of Object.entries(doc.materials || {})) {
            const rgb = props && props.rgb;
            if (Array.isArray(rgb) && rgb.length >= 3) {
                matByKey.set(((rgb[0] & 255) << 16) | ((rgb[1] & 255) << 8) | (rgb[2] & 255), name);
            }
        }
        const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
        const TOP = 30;
        const rows = [];
        rows.push('<table style="width:100%; border-collapse:collapse; font-size:0.82rem;">' +
            '<tr style="background:#f1f3f5;"><th style="padding:4px 8px; text-align:left;">Colour</th>' +
            '<th style="padding:4px 8px; text-align:left;">RGB</th>' +
            '<th style="padding:4px 8px; text-align:right;">Pixels</th>' +
            '<th style="padding:4px 8px; text-align:right;">Share</th>' +
            '<th style="padding:4px 8px; text-align:left;">Material in YAML</th></tr>');
        for (const [key, n] of sorted.slice(0, TOP)) {
            const r = (key >> 16) & 255, g = (key >> 8) & 255, b = key & 255;
            const hex = '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
            const share = (100 * n / sampled).toFixed(2);
            const mat = matByKey.get(key);
            rows.push(`<tr style="border-top:1px solid #eee;">` +
                `<td style="padding:3px 8px;"><span style="display:inline-block; width:14px; height:14px;` +
                ` background:${hex}; border:1px solid #aaa; vertical-align:middle;"></span>` +
                ` <span style="font-family:monospace;">${hex}</span></td>` +
                `<td style="padding:3px 8px; font-family:monospace;">${r}, ${g}, ${b}</td>` +
                `<td style="padding:3px 8px; text-align:right;">${(n * stride * stride).toLocaleString()}</td>` +
                `<td style="padding:3px 8px; text-align:right;">${share}%</td>` +
                `<td style="padding:3px 8px;">${mat ? `<span style="background:#e3f2fd; border-radius:8px;` +
                ` padding:1px 8px;">${imagePropsEscape(mat)}</span>` : '<span style="color:#adb5bd;">—</span>'}</td></tr>`);
        }
        rows.push('</table>');
        document.getElementById('imagePropsColorTable').innerHTML = rows.join('');
        document.getElementById('imagePropsColorSummary').textContent =
            ` — ${counts.size.toLocaleString()} unique colour(s)` +
            (counts.size > TOP ? `, showing top ${TOP}` : '') +
            (stride > 1 ? ` (sampled every ${stride}px)` : '');
    } else {
        document.getElementById('imagePropsColorTable').innerHTML =
            '<div style="padding:10px; color:#888;">Pixel data unavailable (canvas access blocked).</div>';
    }

    // ---- Coordinate-transform preview ----
    if (pix) {
        renderImagePropsTransform(pix, { isPolar, rs, re_, thr, rOrient });
    }
}

// Nearest-neighbour coordinate-transform preview.
// polar strip -> wrapped annulus (what the machine section looks like), or
// cartesian section -> unwrapped strip (what the polar solver would see).
// Conventions mirror the solver: image-up is +θ (horizontal r) / +r (vertical).
function renderImagePropsTransform(pix, geo) {
    const W = pix.width, H = pix.height;
    const dst = document.getElementById('imagePropsDstCanvas');
    const label = document.getElementById('imagePropsDstLabel');
    const caption = document.getElementById('imagePropsXformCaption');
    const dctx = dst.getContext('2d');

    const sample = (x, y) => {
        const i = Math.round(x), j = Math.round(y);
        if (i < 0 || i >= W || j < 0 || j >= H) return null;
        const p = (j * W + i) * 4;
        return [pix.data[p], pix.data[p + 1], pix.data[p + 2], pix.data[p + 3]];
    };

    if (geo.isPolar) {
        // Wrap the (r, θ) strip back into an annulus.
        label.textContent = 'Wrapped to cartesian (annulus)';
        const S = 560;
        dst.width = S; dst.height = S;
        const out = dctx.createImageData(S, S);
        const rEndPx = S / 2 - 2;
        const nR = (geo.rOrient === 'vertical') ? H : W;
        const nTh = (geo.rOrient === 'vertical') ? W : H;
        for (let y = 0; y < S; y++) {
            for (let x = 0; x < S; x++) {
                const xc = x - S / 2 + 0.5;
                const yc = (S / 2 - y) - 0.5;   // canvas y down -> physical y up
                const rr = Math.hypot(xc, yc) / rEndPx * geo.re_;
                if (rr < geo.rs || rr > geo.re_) continue;
                let th = Math.atan2(yc, xc);
                if (th < 0) th += 2 * Math.PI;
                if (th > geo.thr) continue;     // sector models: leave the rest empty
                const rIdx = (rr - geo.rs) / Math.max(1e-30, geo.re_ - geo.rs) * (nR - 1);
                const thIdx = th / geo.thr * nTh;
                let sx, sy;
                if (geo.rOrient === 'vertical') { sx = thIdx; sy = H - 1 - rIdx; }
                else { sx = rIdx; sy = H - 1 - thIdx; }
                const c = sample(sx, sy);
                if (!c) continue;
                const q = (y * S + x) * 4;
                out.data[q] = c[0]; out.data[q + 1] = c[1]; out.data[q + 2] = c[2]; out.data[q + 3] = 255;
            }
        }
        dctx.putImageData(out, 0, 0);
        caption.textContent = ` — annulus reconstructed from polar_domain (r ${imagePropsFmtM(geo.rs)} … ${imagePropsFmtM(geo.re_)})`;
    } else {
        // Unwrap the section into a (r, θ) strip. Geometry defaults: centre =
        // image centre, r_out = min(W, H)/2 (no YAML fields describe this for
        // a cartesian run, so this is a visual aid, not a measurement).
        label.textContent = 'Unwrapped to polar (r × θ strip)';
        const cx = W / 2, cy = H / 2;
        const rOutPx = Math.min(W, H) / 2;
        const rInPx = 0;
        const Wo = 280, Ho = 720;
        dst.width = Wo; dst.height = Ho;
        const out = dctx.createImageData(Wo, Ho);
        for (let j = 0; j < Ho; j++) {
            const th = 2 * Math.PI * (Ho - 1 - j) / Ho;   // image-up = +θ
            const cth = Math.cos(th), sth = Math.sin(th);
            for (let i = 0; i < Wo; i++) {
                const rPx = rInPx + (rOutPx - rInPx) * i / (Wo - 1);
                const sx = cx + rPx * cth;
                const sy = cy - rPx * sth;   // canvas y down
                const c = sample(sx, sy);
                if (!c) continue;
                const q = (j * Wo + i) * 4;
                out.data[q] = c[0]; out.data[q + 1] = c[1]; out.data[q + 2] = c[2]; out.data[q + 3] = 255;
            }
        }
        dctx.putImageData(out, 0, 0);
        caption.textContent = ' — defaults: centre = image centre, r_out = min(W,H)/2 (visual aid; run Polarize for calibrated geometry)';
    }
}

// ===== User content backup: export / import =====
// Your configs, uploaded images, and material libraries live server-side keyed
// by a browser cookie, and are pruned after long inactivity. Export downloads
// them all as one JSON file; Import restores them into the current browser's
// session (useful after clearing the cache or moving to a new device).
// ---- Export modal: pick individual files (configs / images / libraries /
// results) to include in a .zip backup, with per-file sizes, a running total,
// and incremental rendering so a large results tree stays responsive. ----
const BACKUP_CAT_LABEL = {
    configs: 'Configs', images: 'Uploaded images',
    libraries: 'Material libraries', results: 'Analysis results',
};
const BACKUP_RENDER_BATCH = 150;

function fmtBytes(n) {
    if (n < 1024) return n + ' B';
    if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
    return (n / 1073741824).toFixed(2) + ' GB';
}

async function openBackupModal() {
    if (!AppState.userId) { fileManagerStatus('No user session yet', 'error'); return; }
    const modal = document.getElementById('backupModal');
    const list = document.getElementById('backupModalList');
    list.innerHTML = '<div style="color:#888; padding:12px;">Loading…</div>';
    modal.style.display = 'flex';
    let manifest;
    try {
        manifest = await fetch(`/api/backup-manifest?userId=${encodeURIComponent(AppState.userId)}`).then(r => r.json());
        if (!manifest.success) throw new Error(manifest.error || 'failed');
    } catch (err) {
        list.innerHTML = `<div style="color:#c00; padding:12px;">Failed to list your content: ${err.message}</div>`;
        return;
    }
    // Build the flat model: ordered rows (category headers + items), a size map,
    // per-category entry lists, and a selection Set (default = all but results).
    const rows = [], sizeByEntry = new Map(), catItems = {};
    const selected = new Set();
    let totalSize = 0, totalCount = 0;
    for (const cat of Object.keys(BACKUP_CAT_LABEL)) {
        const files = (manifest.categories && manifest.categories[cat]) || [];
        catItems[cat] = [];
        if (!files.length) continue;
        rows.push({ type: 'header', cat });
        for (const f of files) {
            const entry = `${cat}/${f.path}`;
            rows.push({ type: 'item', cat, entry, size: f.size });
            sizeByEntry.set(entry, f.size);
            catItems[cat].push(entry);
            totalSize += f.size; totalCount++;
            if (cat !== 'results') selected.add(entry);   // results default OFF (large)
        }
    }
    AppState.backup = { rows, sizeByEntry, catItems, selected, totalSize, totalCount, rendered: 0 };
    list.innerHTML = '';
    list.onscroll = () => {
        if (list.scrollTop + list.clientHeight >= list.scrollHeight - 120) backupRenderMore();
    };
    if (totalCount === 0) {
        list.innerHTML = '<div style="color:#888; padding:12px;">No saved content yet.</div>';
    } else {
        backupRenderMore();
    }
    backupUpdateSummary();
}

function closeBackupModal() { document.getElementById('backupModal').style.display = 'none'; }

function backupRenderMore() {
    const b = AppState.backup; if (!b) return;
    const list = document.getElementById('backupModalList');
    const end = Math.min(b.rendered + BACKUP_RENDER_BATCH, b.rows.length);
    const frag = document.createDocumentFragment();
    for (let i = b.rendered; i < end; i++) {
        const row = b.rows[i];
        if (row.type === 'header') {
            const items = b.catItems[row.cat];
            const catSize = items.reduce((s, e) => s + (b.sizeByEntry.get(e) || 0), 0);
            const h = document.createElement('div');
            h.style.cssText = 'margin:8px 0 4px; padding-top:6px; border-top:1px solid #eee; font-weight:600; font-size:0.85rem; display:flex; align-items:center; gap:6px;';
            h.innerHTML =
                `<input type="checkbox" data-cat-header="${row.cat}" onchange="backupToggleCategory('${row.cat}', this.checked)"> ` +
                `${BACKUP_CAT_LABEL[row.cat]} <span style="color:#888; font-weight:400;">(${items.length} file(s), ${fmtBytes(catSize)})</span>`;
            frag.appendChild(h);
        } else {
            const label = document.createElement('label');
            label.style.cssText = 'display:flex; align-items:center; gap:8px; font-size:0.82rem; padding:1px 0 1px 18px;';
            const shown = row.entry.slice(row.cat.length + 1);
            label.innerHTML =
                `<input type="checkbox" data-entry="${row.entry.replace(/"/g, '&quot;')}" ${b.selected.has(row.entry) ? 'checked' : ''} onchange="backupToggleItem(this)"> ` +
                `<span style="flex:1; word-break:break-all;">${shown}</span>` +
                `<span style="color:#888; white-space:nowrap;">${fmtBytes(row.size)}</span>`;
            frag.appendChild(label);
        }
    }
    list.appendChild(frag);
    b.rendered = end;
    backupSyncHeaderChecks();
}

function backupToggleItem(cb) {
    const b = AppState.backup; if (!b) return;
    const entry = cb.getAttribute('data-entry').replace(/&quot;/g, '"');
    if (cb.checked) b.selected.add(entry); else b.selected.delete(entry);
    backupSyncHeaderChecks();
    backupUpdateSummary();
}

function backupToggleCategory(cat, on) {
    const b = AppState.backup; if (!b) return;
    for (const e of b.catItems[cat]) { if (on) b.selected.add(e); else b.selected.delete(e); }
    // Update any rendered item checkboxes for this category.
    document.querySelectorAll(`#backupModalList input[data-entry]`).forEach(cb => {
        const entry = cb.getAttribute('data-entry').replace(/&quot;/g, '"');
        if (entry.startsWith(cat + '/')) cb.checked = on;
    });
    backupSyncHeaderChecks();
    backupUpdateSummary();
}

function backupSelectAll(on) {
    const b = AppState.backup; if (!b) return;
    b.selected.clear();
    if (on) for (const row of b.rows) if (row.type === 'item') b.selected.add(row.entry);
    document.querySelectorAll('#backupModalList input[data-entry]').forEach(cb => { cb.checked = on; });
    backupSyncHeaderChecks();
    backupUpdateSummary();
}

function backupSelectAllExceptResults() {
    const b = AppState.backup; if (!b) return;
    b.selected.clear();
    for (const row of b.rows) if (row.type === 'item' && row.cat !== 'results') b.selected.add(row.entry);
    document.querySelectorAll('#backupModalList input[data-entry]').forEach(cb => {
        const entry = cb.getAttribute('data-entry').replace(/&quot;/g, '"');
        cb.checked = !entry.startsWith('results/');
    });
    backupSyncHeaderChecks();
    backupUpdateSummary();
}

// Reflect per-category selection state on the (rendered) header checkboxes:
// checked = all selected, unchecked = none, indeterminate = partial.
function backupSyncHeaderChecks() {
    const b = AppState.backup; if (!b) return;
    document.querySelectorAll('#backupModalList input[data-cat-header]').forEach(h => {
        const cat = h.getAttribute('data-cat-header');
        const items = b.catItems[cat] || [];
        const sel = items.filter(e => b.selected.has(e)).length;
        h.checked = sel === items.length && items.length > 0;
        h.indeterminate = sel > 0 && sel < items.length;
    });
}

function backupUpdateSummary() {
    const b = AppState.backup; if (!b) return;
    let selSize = 0;
    for (const e of b.selected) selSize += (b.sizeByEntry.get(e) || 0);
    document.getElementById('backupModalSummary').textContent =
        `Selected ${b.selected.size} / ${b.totalCount} file(s), ${fmtBytes(selSize)} of ${fmtBytes(b.totalSize)}`;
    const btn = document.getElementById('backupDownloadBtn');
    if (btn) btn.disabled = b.selected.size === 0;
}

async function downloadBackup() {
    const b = AppState.backup; if (!b) return;
    const entries = [...b.selected];
    if (!entries.length) { fileManagerStatus('Select at least one item', 'error'); return; }
    const btn = document.getElementById('backupDownloadBtn');
    const old = btn.textContent; btn.disabled = true; btn.textContent = 'Preparing…';
    try {
        const resp = await fetch('/api/export', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, entries }),
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `omfdm-backup-${new Date().toISOString().slice(0, 10)}.zip`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        closeBackupModal();
        fileManagerStatus(`Backup downloaded (${entries.length} file(s), ${fmtBytes(blob.size)}).`, 'success');
    } catch (err) {
        fileManagerStatus(`Export failed: ${err.message}`, 'error');
    } finally {
        btn.disabled = false; btn.textContent = old;
    }
}

async function importUserData(input) {
    const file = input && input.files && input.files[0];
    input.value = '';   // reset so re-selecting the same file fires change again
    if (!file) return;
    if (!AppState.userId) { fileManagerStatus('No user session yet', 'error'); return; }
    try {
        const fd = new FormData();
        fd.append('backup', file, file.name);
        fd.append('userId', AppState.userId);
        const res = await fetch('/api/import', { method: 'POST', body: fd }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'import failed');
        const w = res.written || {};
        const nSkip = (res.skipped || []).filter(Boolean).length;
        // Refresh the lists so the restored content shows up immediately.
        try { await refreshConfigList(); } catch (_) {}
        try { await refreshImageList(); } catch (_) {}
        try { await refreshUserFiles(); } catch (_) {}
        fileManagerStatus(
            `Imported ${w.configs || 0} config(s), ${w.images || 0} image(s), ` +
            `${w.libraries || 0} library file(s), ${w.results || 0} result file(s)` +
            (nSkip ? ` (${nSkip} skipped)` : '') +
            `. Material libraries appear in the Library Manager.`,
            'success');
    } catch (err) {
        fileManagerStatus(`Import failed: ${err.message}`, 'error');
    }
}

async function insertPolarYaml() {
    const pp = AppState.polarPreprocess;
    const cur = pp.current;
    if (!pp.sourceFilename) {
        showStatus('solverStatus', 'No source image loaded', 'error');
        return;
    }
    if (!AppState.aceEditor) {
        showStatus('solverStatus', 'YAML editor not available', 'error');
        return;
    }
    let targetFilename;
    let block;
    if (cur.save_as === 'polar') {
        let polarDomain;
        // Phase E.2: ensure the polar warp lives on disk before
        // referencing it from image_path.
        if (!(pp.lastSaved && pp.lastSaved.filename) &&
            pp.lastPreview && pp.lastPreview.filename) {
            try {
                await savePolarImage();
            } catch (err) {
                showStatus('solverStatus',
                    `Auto-save before insert failed: ${err.message}. ` +
                    `Press "Save image" then try Insert YAML again.`,
                    'error');
                return;
            }
        }
        if (pp.lastSaved && pp.lastSaved.filename) {
            targetFilename = pp.lastSaved.filename;
            polarDomain    = pp.lastSaved.polar_domain;
        } else {
            showStatus('solverStatus',
                'No polar warp available. Press "Apply Transform" then "Save image" first.',
                'error');
            return;
        }
        block = buildPolarYamlBlock(targetFilename, polarDomain);
    } else {
        block = buildCartesianYamlBlock(pp.sourceFilename);
        targetFilename = pp.sourceFilename;
    }

    // Phase S: build a complete, self-contained YAML from scratch and
    // persist it as a new config file rather than merging into the
    // editor's existing content. Rationale: when the user warps an
    // image and runs Detect Colors against the warped image, any
    // materials / variables / transient blocks left over from the
    // pre-warp config are stale (colours and pixel indices changed).
    // Starting from a fresh YAML file makes the workflow
    //   1) Polar Preprocess → 2) Detect Colors → 3) Run
    // produce a clean, image-specific config every time. The new file
    // is named after the warped image so the user can tell which YAML
    // matches which image at a glance.
    let yamlText = block;
    if (cur.save_as === 'polar' && cur.air_gap_as_slide && airGapEffectiveR() > 0) {
        // Phase F.2 / K: emit the variables: block alongside the
        // transient: section so $N_step / $N_slide resolve. The values
        // come from the polar warp's ntheta schedule.
        const sched = getPolarSlideSchedule();
        yamlText += `\nvariables:\n`;
        yamlText += `  N_step: ${sched.N_step}\n`;
        yamlText += `  N_slide: ${sched.N_slide}\n`;
    }

    // Compose the new config filename from the image base name.
    const baseName = String(targetFilename).replace(/\.[^./\\]+$/, '');
    const newConfigName = `${baseName}.yaml`;

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId,
                file: newConfigName,
                content: yamlText,
            }),
        }).then(r => r.json());
        if (!res.success) throw new Error(res.error || 'save failed');
    } catch (err) {
        showStatus('solverStatus',
            `Failed to save new config "${newConfigName}": ${err.message}`,
            'error');
        return;
    }

    // Refresh the config dropdown, switch to the new file, load it.
    try { await refreshConfigList(); } catch (_) { /* best effort */ }
    const select = document.getElementById('configFileSelect');
    if (select) {
        select.value = newConfigName;
        try { await loadConfig(); } catch (_) { /* best effort */ }
    } else if (AppState.aceEditor) {
        AppState.aceEditor.setValue(yamlText, -1);
    }
    if (typeof switchTab === 'function') switchTab('config');
    showStatus('solverStatus',
        `New config "${newConfigName}" created and loaded. ` +
        `Run Detect Colors on the warped image to fill in materials.`,
        'success');
}

// Legacy entry point preserved so anything still calling
// savePolarAndInsert() does the closest thing: save image (if polar)
// then insert YAML. The footer no longer wires this.
async function savePolarAndInsert() {
    const pp = AppState.polarPreprocess;
    if (pp.current.save_as === 'polar') await savePolarImage();
    await insertPolarYaml();
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
        const customOptions = document.getElementById('resultSelectOptions');
        const currentValue = select.value;

        // Clear both native and custom selects
        select.innerHTML = '<option value="">Select result...</option>';
        customOptions.innerHTML = '';

        // Add default option to custom select
        const defaultOption = document.createElement('div');
        defaultOption.className = 'custom-select-option';
        defaultOption.setAttribute('data-value', '');
        defaultOption.innerHTML = '<div class="custom-select-option-main">Select result...</div>';
        customOptions.appendChild(defaultOption);

        // Fetch descriptions for all results in parallel
        const resultsWithDescriptions = await Promise.all(
            result.results.map(async (res) => {
                try {
                    const folderName = res.path.split('/').pop();
                    const descResponse = await fetch(`/api/user-outputs/${encodeURIComponent(folderName)}/description?userId=${AppState.userId}`);
                    const descData = await descResponse.json();
                    return {
                        ...res,
                        description: descData.description || ''
                    };
                } catch (error) {
                    console.error(`Failed to fetch description for ${res.name}:`, error);
                    return {
                        ...res,
                        description: ''
                    };
                }
            })
        );

        resultsWithDescriptions.forEach(res => {
            // Add to native select (for compatibility)
            const option = document.createElement('option');
            option.value = res.path;
            option.textContent = `${res.name} (${res.steps} steps)`;
            select.appendChild(option);

            // Add to custom select with styled description
            const customOption = document.createElement('div');
            customOption.className = 'custom-select-option';
            customOption.setAttribute('data-value', res.path);

            const mainText = document.createElement('div');
            mainText.className = 'custom-select-option-main';
            mainText.textContent = `${res.name} (${res.steps} steps)`;

            customOption.appendChild(mainText);

            // Add description if available
            if (res.description && res.description.trim()) {
                const descText = document.createElement('div');
                descText.className = 'custom-select-option-description';

                // Truncate description if too long (max 60 characters)
                let description = res.description.trim();
                if (description.length > 60) {
                    description = description.substring(0, 57) + '...';
                }
                descText.textContent = description;

                // Add full description as title (shows on hover)
                customOption.title = res.description;

                customOption.appendChild(descText);
            }

            customOptions.appendChild(customOption);
        });

        // Restore selection if it still exists
        if (result.results.map(r => r.path).includes(currentValue)) {
            select.value = currentValue;
            updateCustomSelectDisplay(currentValue);
        }

        // Return the results for use in loadResults()
        return result.results;
    } catch (error) {
        console.error('Error loading results list:', error);
        return [];
    }
}

// Update custom select display to match selected value
function updateCustomSelectDisplay(value) {
    const display = document.getElementById('resultSelectDisplay');
    const options = document.querySelectorAll('#resultSelectOptions .custom-select-option');

    // Remove 'selected' class from all options
    options.forEach(opt => opt.classList.remove('selected'));

    if (!value) {
        display.textContent = 'Select result...';
        return;
    }

    // Find and mark the selected option
    const selectedOption = Array.from(options).find(opt => opt.getAttribute('data-value') === value);
    if (selectedOption) {
        selectedOption.classList.add('selected');
        // Get the main text (without description)
        const mainText = selectedOption.querySelector('.custom-select-option-main');
        display.textContent = mainText ? mainText.textContent : 'Select result...';
    }
}

// Initialize custom select event handlers
function initCustomSelect() {
    const container = document.getElementById('resultSelectContainer');
    const trigger = document.getElementById('resultSelectTrigger');
    const options = document.getElementById('resultSelectOptions');
    const nativeSelect = document.getElementById('resultSelect');

    // Toggle dropdown on trigger click
    trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        container.classList.toggle('open');
    });

    // Handle option selection
    options.addEventListener('click', (e) => {
        const option = e.target.closest('.custom-select-option');
        if (option) {
            const value = option.getAttribute('data-value');

            // Update native select
            nativeSelect.value = value;

            // Update custom select display
            updateCustomSelectDisplay(value);

            // Close dropdown
            container.classList.remove('open');

            // Trigger change event on native select
            nativeSelect.dispatchEvent(new Event('change'));
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!container.contains(e.target)) {
            container.classList.remove('open');
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && container.classList.contains('open')) {
            container.classList.remove('open');
        }
    });
}

function createAbortError(message) {
    const error = new Error(message);
    error.name = 'AbortError';
    return error;
}

function isResultLoadCurrent(context) {
    if (!context || context.controller.signal.aborted) return false;
    return AppState.resultLoadGeneration === context.generation &&
           AppState.resultLoadController === context.controller;
}

function assertResultLoadCurrent(context) {
    if (!isResultLoadCurrent(context)) {
        throw createAbortError('Result load superseded');
    }
}

async function loadSelectedResult() {
    const select = document.getElementById('resultSelect');
    const resultPath = select.value;

    if (!resultPath) {
        showStatus('solverStatus', 'Please select a result to load', 'error');
        return;
    }

    if (AppState.resultLoadController) {
        AppState.resultLoadController.abort();
    }
    const controller = new AbortController();
    const loadContext = {
        generation: ++AppState.resultLoadGeneration,
        controller,
        resultPath,
    };
    AppState.resultLoadController = controller;

    const previousResult = AppState.resultsData.currentResult;
    if (previousResult !== resultPath) {
        // Stop scheduling old dashboard work immediately. Field caches remain
        // usable until the new result has passed its validation phase.
        pauseAnimation();
    }
    invalidateDashboardRenders(`result load requested: ${resultPath}`);

    try {
        const defaultConditions = { coordinate_system: 'cartesian', dx: 0.001, dy: 0.001 };
        const stepsPromise = fetch(
            `/api/detect-steps?result=${encodeURIComponent(resultPath)}`,
            { signal: controller.signal }
        ).then(async response => {
            if (!response.ok) throw new Error('Failed to detect steps');
            return await response.json();
        });
        const conditionsPromise = fetch(
            `/api/load-conditions?result=${encodeURIComponent(resultPath)}`,
            { signal: controller.signal }
        ).then(async response => {
            if (!response.ok) {
                console.warn('conditions.json not found, assuming default (cartesian)');
                return defaultConditions;
            }
            return await response.json();
        }).catch(error => {
            if (error && error.name === 'AbortError') throw error;
            console.warn('Failed to load conditions.json:', error);
            return defaultConditions;
        });

        const [stepsData, analysisConditions] = await Promise.all([stepsPromise, conditionsPromise]);
        assertResultLoadCurrent(loadContext);

        // Commit result-dependent state atomically only after both requests
        // have completed and this selection is still the newest request.
        if (previousResult !== resultPath) {
            invalidateFieldLoads(`result switch: ${previousResult || '(none)'} -> ${resultPath}`);
            clearFieldDataCache('result switch');
            clearCoarseningMaskCache();
        }
        AppState.totalSteps = stepsData.steps || 1;
        AppState.currentStep = 1;
        AppState.resultsData.currentResult = resultPath;
        AppState.analysisConditions = analysisConditions;
        loadContext.analysisConditions = analysisConditions;
        console.log('Analysis conditions loaded:', AppState.analysisConditions);

        updateExportFormatBadge();

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

        await finalizeCommittedResult(loadContext);
    } catch (error) {
        if (error && error.name === 'AbortError') return;
        if (isResultLoadCurrent(loadContext)) {
            const committedResult = AppState.resultsData.currentResult;
            controller.abort();
            if (committedResult) {
                await restoreCommittedResultAfterValidationFailure(
                    loadContext,
                    committedResult,
                    error
                );
            } else {
                showStatus('solverStatus', `Error loading result: ${error.message}`, 'error');
            }
        }
    } finally {
        if (AppState.resultLoadController === controller) {
            controller.abort();
            AppState.resultLoadController = null;
        }
    }
}

async function finalizeCommittedResult(loadContext) {
    assertResultLoadCurrent(loadContext);
    await Promise.all([
        loadQuickPreviewFromResult(loadContext.resultPath, loadContext),
        loadResultLog(loadContext.resultPath, loadContext),
    ]);
    assertResultLoadCurrent(loadContext);

    // Auto-reload dashboard plots only after preview and log finalization for
    // this committed result have survived all generation checks.
    await updateAllPlots();
    assertResultLoadCurrent(loadContext);
}

async function restoreCommittedResultAfterValidationFailure(failedContext, committedResult, validationError) {
    const select = document.getElementById('resultSelect');
    if (select) select.value = committedResult;
    updateCustomSelectDisplay(committedResult);

    const controller = new AbortController();
    const recoveryContext = {
        generation: ++AppState.resultLoadGeneration,
        controller,
        resultPath: committedResult,
        analysisConditions: AppState.analysisConditions,
    };
    AppState.resultLoadController = controller;
    invalidateDashboardRenders(`restoring committed result: ${committedResult}`);

    const failureMessage = `Error loading result ${failedContext.resultPath}: ${validationError.message}`;
    try {
        // A failed superseding validation may have interrupted preview, log,
        // or dashboard finalization for the last committed result. Re-run all
        // three under a fresh result generation instead of leaving a partial UI.
        await finalizeCommittedResult(recoveryContext);
        showStatus(
            'solverStatus',
            `${failureMessage}. Restored current result: ${committedResult}`,
            'error'
        );
    } catch (error) {
        if (!error || error.name !== 'AbortError') {
            console.error('Failed to restore committed result UI:', error);
            if (isResultLoadCurrent(recoveryContext)) {
                showStatus(
                    'solverStatus',
                    `${failureMessage}. Failed to restore ${committedResult}: ${error.message}`,
                    'error'
                );
            }
        }
    } finally {
        if (AppState.resultLoadController === controller) {
            controller.abort();
            AppState.resultLoadController = null;
        }
    }
}

async function loadResultLog(resultPath, loadContext = null) {
    try {
        const response = await fetch(
            `/api/get-log?result=${encodeURIComponent(resultPath)}`,
            loadContext ? { signal: loadContext.controller.signal } : undefined
        );
        if (response.ok) {
            const logContent = await response.text();
            if (loadContext) assertResultLoadCurrent(loadContext);
            const logOutput = document.getElementById('logOutput');
            const logPanel = document.getElementById('logPanel');

            if (logOutput) logOutput.textContent = logContent;
            if (logPanel) logPanel.style.display = 'block';
        } else {
            // Log file not found, hide panel
            if (loadContext) assertResultLoadCurrent(loadContext);
            const logPanel = document.getElementById('logPanel');
            if (logPanel) logPanel.style.display = 'none';
        }
    } catch (error) {
        if ((error && error.name === 'AbortError') ||
            (loadContext && !isResultLoadCurrent(loadContext))) return;
        console.error('Error loading log:', error);
        const logPanel = document.getElementById('logPanel');
        if (logPanel) logPanel.style.display = 'none';
    }
}

// Helper: Harmonic mean interpolation of μ at inactive cells
// Two-pass approach:
//   Pass 1: 1D harmonic interpolation along x for inactive cells on active rows
//   Pass 2: Column-wise Hermite (C^1) interpolation in y for inactive rows
// Returns a new 2D array with interpolated values (does not modify input)
function interpolateMuHarmonic(Mu, activeMask) {
    const rows = Mu.length;
    const cols = Mu[0].length;
    const result = Mu.map(row => [...row]);

    // Precompute which rows have interior active cells
    const activeRowFlag = Array(rows).fill(false);
    activeRowFlag[0] = true;
    activeRowFlag[rows - 1] = true;
    for (let j = 1; j < rows - 1; j++) {
        for (let i = 1; i < cols - 1; i++) {
            if (activeMask[j][i]) { activeRowFlag[j] = true; break; }
        }
    }

    // Pass 1: 1D harmonic interpolation along x for inactive cells on active rows
    for (let j = 0; j < rows; j++) {
        if (!activeRowFlag[j]) continue;
        for (let i = 0; i < cols; i++) {
            if (activeMask[j][i]) continue;
            let il = -1, ir = -1;
            for (let ii = i - 1; ii >= 0; ii--) {
                if (activeMask[j][ii]) { il = ii; break; }
            }
            for (let ii = i + 1; ii < cols; ii++) {
                if (activeMask[j][ii]) { ir = ii; break; }
            }
            if (il >= 0 && ir >= 0 && Mu[j][il] > 0 && Mu[j][ir] > 0) {
                const t = (i - il) / (ir - il);
                result[j][i] = 1.0 / ((1 - t) / Mu[j][il] + t / Mu[j][ir]);
            } else if (il >= 0) {
                result[j][i] = Mu[j][il];
            } else if (ir >= 0) {
                result[j][i] = Mu[j][ir];
            }
        }
    }

    // Collect active row list
    const activeRowList = [];
    for (let j = 0; j < rows; j++) {
        if (activeRowFlag[j]) activeRowList.push(j);
    }

    // Pass 2: Column-wise Hermite (C^1) interpolation in y for inactive rows
    // Uses result[] values from Pass 1 (active rows now fully populated)
    if (activeRowList.length >= 2) {
        const nAR = activeRowList.length;
        for (let i = 0; i < cols; i++) {
            // Compute dμ/dy tangents at active rows (Catmull-Rom)
            const dMu_dy = new Float64Array(nAR);
            for (let k = 0; k < nAR; k++) {
                const jc = activeRowList[k];
                const jp = k > 0 ? activeRowList[k - 1] : -1;
                const jn = k < nAR - 1 ? activeRowList[k + 1] : -1;
                if (jp >= 0 && jn >= 0) {
                    dMu_dy[k] = (result[jn][i] - result[jp][i]) / (jn - jp);
                } else if (jn >= 0) {
                    dMu_dy[k] = (result[jn][i] - result[jc][i]) / (jn - jc);
                } else if (jp >= 0) {
                    dMu_dy[k] = (result[jc][i] - result[jp][i]) / (jc - jp);
                }
            }

            // Hermite fill between consecutive active rows
            for (let k = 0; k < nAR - 1; k++) {
                const j1 = activeRowList[k];
                const j2 = activeRowList[k + 1];
                if (j2 - j1 <= 1) continue;
                const span = j2 - j1;
                const mu0 = result[j1][i], mu1 = result[j2][i];
                const m0 = dMu_dy[k] * span, m1 = dMu_dy[k + 1] * span;
                for (let j = j1 + 1; j < j2; j++) {
                    const t = (j - j1) / span;
                    const t2 = t * t, t3 = t2 * t;
                    const H00 = 2 * t3 - 3 * t2 + 1;
                    const H10 = t3 - 2 * t2 + t;
                    const H01 = -2 * t3 + 3 * t2;
                    const H11 = t3 - t2;
                    result[j][i] = H00 * mu0 + H10 * m0 + H01 * mu1 + H11 * m1;
                }
            }
        }
    }

    return result;
}

// Helper: Calculate magnetic fields from Az and Mu (supports both polar and Cartesian coordinates)
// activeMask: optional 2D boolean array (same size as Az) indicating active cells for coarsening-aware differentiation
function calculateMagneticField(
    Az,
    Mu,
    dx = 0.001,
    dy = 0.001,
    activeMask = null,
    analysisConditions = AppState.analysisConditions
) {
    const rows = Az.length;
    const cols = Az[0].length;

    const Bx = Array(rows).fill(0).map(() => Array(cols).fill(0));
    const By = Array(rows).fill(0).map(() => Array(cols).fill(0));

    // Determine coordinate system (if analysisConditions is loaded)
    const coordSystem = analysisConditions ? analysisConditions.coordinate_system : 'cartesian';

    if (coordSystem === 'polar') {
        // Polar coordinate magnetic field calculation
        const polar = analysisConditions.polar;
        // r_start=0 is valid for a full-disc polar model; use null checks,
        // not truthiness, or every such run is reported as missing conditions.
        if (!polar || polar.r_start == null || polar.r_end == null || polar.theta_range == null) {
            console.error('Polar coordinate parameters missing in analysisConditions:', analysisConditions);
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
        const dr = analysisConditions.dr || (r_end - r_start) / (nr - 1);
        const dtheta = analysisConditions.dtheta || polar.theta_range / (ntheta - 1);

        // Determine theta boundary conditions
        const bc = analysisConditions.boundary_conditions || {};
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

        // Helper: check if cell (ir, jt) is active (true when no coarsening mask)
        const getActivePolar = (ir, jt) => {
            if (!activeMask) return true;
            return r_orientation === 'horizontal' ? activeMask[jt][ir] : activeMask[ir][jt];
        };

        if (activeMask) {
            // Full-grid step=1 stencil at active cells.
            // C++ exports interpolated Az at ALL cells (active + inactive via interpolateToFullGrid),
            // so immediate ±1 neighbors are always valid. This matches the non-coarsened solver
            // behavior exactly at material boundaries, eliminating surface concentration artifacts.
            // Inactive cells remain 0; _fillInactiveScalar() fills them after Bx/By conversion.
            for (let jt = 0; jt < ntheta; jt++) {
                for (let ir = 0; ir < nr; ir++) {
                    if (!getActivePolar(ir, jt)) continue;

                    const r = r_coords[ir];
                    const safe_r = Math.max(r, 1e-15);

                    // Br = (1/r) * ∂Az/∂θ — full-grid step=1 stencil.
                    // C++ exports interpolated Az at ALL cells (active + inactive),
                    // so use immediate ±1 neighbors to match non-coarsened behavior exactly.
                    let jt_prev = jt - 1, jt_next = jt + 1;
                    if (jt_prev < 0) jt_prev = thetaPeriodic ? ntheta - 1 : -1;
                    if (jt_next >= ntheta) jt_next = thetaPeriodic ? 0 : -1;

                    let Br_val = 0;
                    if (jt_prev >= 0 && jt_next >= 0) {
                        let Az_next = getAz(ir, jt_next), Az_prev = getAz(ir, jt_prev);
                        if (thetaAntiperiodic) {
                            if (jt_next < jt) Az_next = -Az_next;
                            if (jt_prev > jt) Az_prev = -Az_prev;
                        }
                        Br_val = (Az_next - Az_prev) / (2 * dtheta) / safe_r;
                    } else if (jt_next >= 0) {
                        Br_val = (getAz(ir, jt_next) - getAz(ir, jt)) / dtheta / safe_r;
                    } else if (jt_prev >= 0) {
                        Br_val = (getAz(ir, jt) - getAz(ir, jt_prev)) / dtheta / safe_r;
                    }
                    setField(Br, ir, jt, Br_val);

                    // Bθ = -∂Az/∂r — full-grid step=1 stencil (same rationale as Br).
                    const ir_prev = ir > 0 ? ir - 1 : -1;
                    const ir_next = ir < nr - 1 ? ir + 1 : -1;

                    let Btheta_val = 0;
                    if (ir_prev >= 0 && ir_next >= 0) {
                        Btheta_val = -(getAz(ir_next, jt) - getAz(ir_prev, jt)) / (2 * dr);
                    } else if (ir_next >= 0) {
                        Btheta_val = -(getAz(ir_next, jt) - getAz(ir, jt)) / dr;
                    } else if (ir_prev >= 0) {
                        Btheta_val = -(getAz(ir, jt) - getAz(ir_prev, jt)) / dr;
                    }
                    setField(Btheta, ir, jt, Btheta_val);
                }
            }
        } else {
            // Uniform grid (no coarsening)
            for (let jt = 0; jt < ntheta; jt++) {
                for (let ir = 0; ir < nr; ir++) {
                    const r = r_coords[ir];
                    const safe_r = Math.max(r, 1e-15);

                    // Br = (1/r) * ∂Az/∂θ
                    let jt_next, jt_prev;
                    let Az_next, Az_prev;

                    if (thetaPeriodic) {
                        jt_next = (jt + 1) % ntheta;
                        jt_prev = (jt - 1 + ntheta) % ntheta;
                        Az_next = getAz(ir, jt_next);
                        Az_prev = getAz(ir, jt_prev);
                        if (thetaAntiperiodic) {
                            if (jt === ntheta - 1) Az_next = -Az_next;
                            if (jt === 0) Az_prev = -Az_prev;
                        }
                    } else {
                        if (jt === 0) {
                            jt_next = 1; jt_prev = 0;
                        } else if (jt === ntheta - 1) {
                            jt_next = ntheta - 1; jt_prev = ntheta - 2;
                        } else {
                            jt_next = jt + 1; jt_prev = jt - 1;
                        }
                        Az_next = getAz(ir, jt_next);
                        Az_prev = getAz(ir, jt_prev);
                    }

                    const denom = (jt === 0 || jt === ntheta - 1) && !thetaPeriodic ? dtheta : (2 * dtheta);
                    setField(Br, ir, jt, (Az_next - Az_prev) / denom / safe_r);

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
    } else if (activeMask) {
        // Cartesian with coarsening: full-grid step=1 stencil.
        // C++ exports interpolated Az at ALL cells (active + inactive via interpolateToFullGrid),
        // so use immediate ±1 neighbors to match non-coarsened behavior at material boundaries.
        const bc = analysisConditions ? analysisConditions.boundary_conditions : null;
        const x_periodic = bc && bc.left && bc.right &&
                          bc.left.type === 'periodic' && bc.right.type === 'periodic';
        const y_periodic = bc && bc.bottom && bc.top &&
                          bc.bottom.type === 'periodic' && bc.top.type === 'periodic';

        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                if (!activeMask[j][i]) continue;

                // Bx = ∂Az/∂y — full-grid step=1 stencil
                let j_prev = j - 1, j_next = j + 1;
                if (j_prev < 0) j_prev = y_periodic ? rows - 1 : -1;
                if (j_next >= rows) j_next = y_periodic ? 0 : -1;

                if (j_prev >= 0 && j_next >= 0) {
                    Bx[j][i] = (Az[j_next][i] - Az[j_prev][i]) / (2 * dy);
                } else if (j_next >= 0) {
                    Bx[j][i] = (Az[j_next][i] - Az[j][i]) / dy;
                } else if (j_prev >= 0) {
                    Bx[j][i] = (Az[j][i] - Az[j_prev][i]) / dy;
                }

                // By = -∂Az/∂x — full-grid step=1 stencil
                let i_prev = i - 1, i_next = i + 1;
                if (i_prev < 0) i_prev = x_periodic ? cols - 1 : -1;
                if (i_next >= cols) i_next = x_periodic ? 0 : -1;

                if (i_prev >= 0 && i_next >= 0) {
                    By[j][i] = -(Az[j][i_next] - Az[j][i_prev]) / (2 * dx);
                } else if (i_next >= 0) {
                    By[j][i] = -(Az[j][i_next] - Az[j][i]) / dx;
                } else if (i_prev >= 0) {
                    By[j][i] = -(Az[j][i] - Az[j][i_prev]) / dx;
                }
            }
        }

        // Inactive cells: Bx/By left as 0 (initialized value).
        // |B|/|H| magnitudes are computed at active cells, then _fillInactiveScalar()
        // interpolates the scalar values directly — avoids vector-component artifacts.
    } else {
        // Cartesian coordinate magnetic field calculation (standard uniform grid)
        const bc = analysisConditions ? analysisConditions.boundary_conditions : null;
        const x_periodic = bc && bc.left && bc.right &&
                          bc.left.type === 'periodic' && bc.right.type === 'periodic';
        const y_periodic = bc && bc.bottom && bc.top &&
                          bc.bottom.type === 'periodic' && bc.top.type === 'periodic';

        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                // Bx = ∂Az/∂y
                if (j === 0) {
                    if (y_periodic) {
                        Bx[j][i] = (Az[1][i] - Az[rows-1][i]) / (2 * dy);
                    } else {
                        Bx[j][i] = (Az[1][i] - Az[0][i]) / dy;
                    }
                } else if (j === rows - 1) {
                    if (y_periodic) {
                        Bx[j][i] = (Az[0][i] - Az[rows-2][i]) / (2 * dy);
                    } else {
                        Bx[j][i] = (Az[rows-1][i] - Az[rows-2][i]) / dy;
                    }
                } else {
                    Bx[j][i] = (Az[j+1][i] - Az[j-1][i]) / (2 * dy);
                }

                // By = -∂Az/∂x
                if (i === 0) {
                    if (x_periodic) {
                        By[j][i] = -(Az[j][1] - Az[j][cols-1]) / (2 * dx);
                    } else {
                        By[j][i] = -(Az[j][1] - Az[j][0]) / dx;
                    }
                } else if (i === cols - 1) {
                    if (x_periodic) {
                        By[j][i] = -(Az[j][0] - Az[j][cols-2]) / (2 * dx);
                    } else {
                        By[j][i] = -(Az[j][cols-1] - Az[j][cols-2]) / dx;
                    }
                } else {
                    By[j][i] = -(Az[j][i+1] - Az[j][i-1]) / (2 * dx);
                }
            }
        }
    }

    // H = B / μ (use harmonic-mean-interpolated μ for coarsened grids)
    const MuFinal = activeMask ? interpolateMuHarmonic(Mu, activeMask) : Mu;
    const Hx = Bx.map((row, j) => row.map((val, i) => val / MuFinal[j][i]));
    const Hy = By.map((row, j) => row.map((val, i) => val / MuFinal[j][i]));

    // Fill inactive cells for all field arrays using Gauss-Seidel diffusion.
    // Components (Bx, By, Hx, Hy) are filled for Line Profile vector decomposition.
    // Magnitudes (B, H) are filled separately from scalar values to avoid
    // vector-component artifacts (interpolating |B| ≠ |interpolated B|).
    if (activeMask) {
        _fillInactiveScalar(Bx, activeMask);
        _fillInactiveScalar(By, activeMask);
        _fillInactiveScalar(Hx, activeMask);
        _fillInactiveScalar(Hy, activeMask);
    }

    const B = Bx.map((row, j) => row.map((val, i) => Math.sqrt(val**2 + By[j][i]**2)));
    const H = Hx.map((row, j) => row.map((val, i) => Math.sqrt(val**2 + Hy[j][i]**2)));

    return { Bx, By, B, Hx, Hy, H };
}

// Interpolate inactive cells using iterative Gauss-Seidel diffusion.
// Same algorithm as Plotly's interp2d (connectgaps) but without Plotly overhead.
// Phase 1: Flood-fill nulls from active cell boundaries (neighbor averaging).
// Phase 2: Relaxation sweeps to smooth (active cells pinned).
// Operates on scalar values (|B|, |H|, Az) to avoid vector-component artifacts.
function _fillInactiveScalar(data, mask) {
    const rows = data.length;
    const cols = data[0].length;

    // Mark inactive cells as null (distinguishes "unfilled" from "value = 0")
    for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
            if (!mask[j][i]) data[j][i] = null;
        }
    }

    // Phase 1: Iterative flood-fill from active cells outward
    // Each iteration fills cells adjacent to already-filled cells.
    // For coarsening ratio R, needs ~R iterations to reach all cells.
    for (let iter = 0; iter < 100; iter++) {
        let filled = 0;
        for (let j = 0; j < rows; j++) {
            for (let i = 0; i < cols; i++) {
                if (data[j][i] !== null) continue;
                let sum = 0, n = 0;
                if (j > 0 && data[j - 1][i] !== null) { sum += data[j - 1][i]; n++; }
                if (j < rows - 1 && data[j + 1][i] !== null) { sum += data[j + 1][i]; n++; }
                if (i > 0 && data[j][i - 1] !== null) { sum += data[j][i - 1]; n++; }
                if (i < cols - 1 && data[j][i + 1] !== null) { sum += data[j][i + 1]; n++; }
                if (n > 0) { data[j][i] = sum / n; filled++; }
            }
        }
        if (filled === 0) break;
    }

    // Phase 2: Gauss-Seidel relaxation to smooth (active cells pinned)
    // Converges quickly for small gaps between active cells.
    for (let iter = 0; iter < 10; iter++) {
        for (let j = 1; j < rows - 1; j++) {
            for (let i = 1; i < cols - 1; i++) {
                if (mask[j][i]) continue;
                data[j][i] = (data[j - 1][i] + data[j + 1][i] +
                              data[j][i - 1] + data[j][i + 1]) * 0.25;
            }
        }
    }
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

async function loadQuickPreviewFromResult(resultPath, loadContext = null) {
    const step = 1; // Always show first step in preview
    const previewB = document.getElementById('previewPlot2');
    const previewH = document.getElementById('previewPlot3');
    const bContext = loadContext
        ? createResultPreviewRenderContext(loadContext, previewB)
        : null;
    const hContext = loadContext
        ? createResultPreviewRenderContext(loadContext, previewH)
        : null;

    try {
        if (loadContext) assertResultLoadCurrent(loadContext);
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

        // Load Az and Mu data via the shared loader (handles TIFF/CSV
        // and populates the cache for the rest of the session).
        let azFlat = null, muFlat = null;
        try {
            [azFlat, muFlat] = await Promise.all([
                loadFieldData('Az', 1, resultPath, loadContext?.controller.signal || null),
                loadFieldData('Mu', 1, resultPath, loadContext?.controller.signal || null),
            ]);
        } catch (e) {
            if ((e && e.name === 'AbortError') || (loadContext && !isResultLoadCurrent(loadContext))) throw e;
            console.warn('Quick preview load failed:', e.message);
        }

        if (azFlat && muFlat) {
            // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
            const azFlipped = flipVertical(azFlat);
            const muFlipped = flipVertical(muFlat);

            console.log('Az data dimensions:', azFlipped.length, 'x', azFlipped[0]?.length);
            console.log('Mu data dimensions:', muFlipped.length, 'x', muFlipped[0]?.length);
            console.log('Grid spacing: dx =', dx, ', dy =', dy);

            // Load coarsening mask for coarsening-aware B/H computation
            const maskResult = await getCoarseningMaskArray(
                resultPath,
                1,
                loadContext?.controller.signal || null
            ).catch(error => {
                if ((error && error.name === 'AbortError') ||
                    error?.code === 'FIELD_MEMORY_BUDGET') throw error;
                return null;
            });
            if (loadContext) assertResultLoadCurrent(loadContext);
            const activeMask = maskResult ? maskResult.mask : null;

            const { Bx, By, B, Hx, Hy, H } = calculateMagneticField(
                azFlipped,
                muFlipped,
                dx,
                dy,
                activeMask,
                loadContext?.analysisConditions || AppState.analysisConditions
            );
            console.log('B field calculated, Bx dimensions:', Bx.length, 'x', Bx[0]?.length);
            console.log('B magnitude dimensions:', B.length, 'x', B[0]?.length);
            console.log('H magnitude dimensions:', H.length, 'x', H[0]?.length);
            console.log('B magnitude sample values:', B[0]?.slice(0, 3));

            // Plot |B| and |H|
            await Promise.all([
                plotHeatmap(previewB, B, '|B| [T]', true, false, bContext),
                plotHeatmap(previewH, H, '|H| [A/m]', true, false, hContext),
            ]);
        } else {
            if (loadContext) assertResultLoadCurrent(loadContext);
            showPlotMessage(
                previewB,
                '<p style="text-align:center; padding:20px;">Az/Mu data not available</p>',
                bContext
            );
            showPlotMessage(
                previewH,
                '<p style="text-align:center; padding:20px;">Az/Mu data not available</p>',
                hContext
            );
        }

    } catch (error) {
        if ((error && error.name === 'AbortError') || (loadContext && !isResultLoadCurrent(loadContext))) return;
        console.error('Preview error:', error);
        showPlotMessage(
            previewB,
            '<p style="text-align:center; padding:20px;">Error loading preview</p>',
            bContext
        );
        showPlotMessage(
            previewH,
            '<p style="text-align:center; padding:20px;">Error loading preview</p>',
            hContext
        );
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
                userId: AppState.userId,
                materialLibraryFile: AppState.selectedLibrary || null
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

                        if (data.exitCode === 0 && data.success) {
                            progressBar.style.width = '100%';
                            progressPercent.textContent = '100%';
                            progressText.textContent = 'Completed successfully';
                            showStatus('solverStatus', 'Solver completed successfully', 'success');
                            // Load results and update dashboard
                            await loadResults();
                        } else if (data.exitCode === 2 && data.completed) {
                            progressBar.style.width = '100%';
                            progressPercent.textContent = '100%';
                            progressBar.style.background = 'linear-gradient(90deg, #ffc107 0%, #ff9800 100%)';
                            progressText.textContent = 'Completed with nonlinear convergence warnings';
                            // Exit code 2 still has intentional diagnostic
                            // output. Make it available exactly like a
                            // successful run, then leave the warning visible.
                            await loadResults();
                            showStatus(
                                'solverStatus',
                                'Solver completed with a nonlinear convergence warning. Diagnostic results were loaded; do not treat them as converged.',
                                'warning'
                            );
                        } else if (data.exitCode === null || data.stopped) {
                            progressBar.style.background = 'linear-gradient(90deg, #ffc107 0%, #ff9800 100%)';
                            progressText.textContent = 'Stopped by user';
                            showStatus('solverStatus', 'Solver stopped by user', 'warning');
                        } else {
                            throw new Error(data.message || 'Solver failed');
                        }
                    } else if (data.type === 'error') {
                        // Stderr lines (including nonlinear WARNINGs) are log
                        // entries. A startup/spawn failure carries an explicit
                        // `error` field because no final `done` event is
                        // guaranteed in that case.
                        if (data.success === false && data.error) {
                            throw new Error(data.error);
                        }
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

        // Hide only an unambiguously successful run. Warning-complete runs
        // keep the amber progress state visible for review.
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
    virtual_work: { name: 'Virtual Work (dW/dx)', render: renderVirtualWork },
    flux_linkage_time: { name: 'Flux Linkage Timeline', render: renderFluxLinkageTime },
    back_emf_time: { name: 'Back-EMF Timeline', render: renderBackEMFTime }
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

    // Set data attributes on the grid-stack-item element for Plot Configure
    if (widget) {
        widget.setAttribute('data-plot-id', plotId);
        widget.setAttribute('data-plot-type', plotType);
    }

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
            showPlotMessage(container, '<p style="color:red; padding:20px;">Please select a result first</p>');
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
                const retryContainer = document.getElementById(containerId);
                if (!retryContainer) return;
                const step = AppState.currentStep;
                const renderContext = createDashboardRenderContext(containerId, getCurrentResultPath(), step);
                try {
                    console.log(`addPlotWidget: Rendering ${plotType} in ${containerId} for step ${step}`);
                    await plotDef.render(containerId, step, renderContext);
                    assertRenderContextCurrent(renderContext, retryContainer);
                    console.log(`addPlotWidget: Successfully rendered ${plotType}`);
                } catch (error) {
                    if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, retryContainer)) {
                        discardStaleRender(retryContainer, renderContext);
                        return;
                    }
                    console.error(`Error rendering ${plotType}:`, error);
                    console.error('Error stack:', error.stack);
                    showPlotMessage(retryContainer, `<p style="color:red; padding:20px;">Error: ${error.message}</p>`, renderContext);
                }
            }, 200);
            return;
        }

        const step = AppState.currentStep;
        const renderContext = createDashboardRenderContext(containerId, getCurrentResultPath(), step);
        try {
            console.log(`addPlotWidget: Rendering ${plotType} in ${containerId} for step ${step}`);
            await plotDef.render(containerId, step, renderContext);
            assertRenderContextCurrent(renderContext, container);
            console.log(`addPlotWidget: Successfully rendered ${plotType}`);
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
                discardStaleRender(container, renderContext);
                return;
            }
            console.error(`Error rendering ${plotType}:`, error);
            console.error('Error stack:', error.stack);
            showPlotMessage(container, `<p style="color:red; padding:20px;">Error: ${error.message}</p>`, renderContext);
        }
    }, 100);
}

function removeZoomTracking(plotDiv) {
    if (!plotDiv || !plotDiv._zoomTrackingHandler) return;
    if (typeof plotDiv.removeListener === 'function') {
        plotDiv.removeListener('plotly_relayout', plotDiv._zoomTrackingHandler);
    }
    delete plotDiv._zoomTrackingHandler;
}

function purgePlotlyTree(root) {
    if (!root || typeof Plotly === 'undefined' || typeof Plotly.purge !== 'function') return;

    const graphDivs = new Set();
    if (root._fullLayout || root.data || root.classList?.contains('js-plotly-plot')) {
        graphDivs.add(root);
    }
    root.querySelectorAll?.('.js-plotly-plot').forEach(div => graphDivs.add(div));
    if (root._lineProfileImageDiv) graphDivs.add(root._lineProfileImageDiv);
    if (root._fluxLinkagePlotDiv) graphDivs.add(root._fluxLinkagePlotDiv);

    for (const graphDiv of graphDivs) {
        try {
            removeZoomTracking(graphDiv);
            if (graphDiv._clickHandler) {
                graphDiv.removeEventListener('click', graphDiv._clickHandler);
                delete graphDiv._clickHandler;
            }
            Plotly.purge(graphDiv);
        } catch (error) {
            console.warn('Failed to purge Plotly graph:', error);
        }
    }

    delete root._lineProfileImageDiv;
    delete root._fluxLinkagePlotDiv;
}

function preparePlotlyContainer(container, renderContext = null) {
    if (!container) return;
    assertRenderContextCurrent(renderContext, container);
    purgePlotlyTree(container);
    container.innerHTML = '';
    // Claim the cleared container for the new render immediately. Otherwise a
    // superseded Plotly.newPlot() may resolve later, still see its old token,
    // and purge a newer no-data/error message from the shared container.
    if (renderContext) {
        container._plotlyRenderToken = renderContext.token;
    } else {
        delete container._plotlyRenderToken;
    }
}

function showPlotMessage(container, html, renderContext = null) {
    if (!container) return;
    assertRenderContextCurrent(renderContext, container);
    preparePlotlyContainer(container, renderContext);
    container.innerHTML = html;
}

function invalidateDashboardRenders(reason = '') {
    AppState.dashboardRenderGeneration++;
    if (AppState.dashboardRenderController) {
        AppState.dashboardRenderController.abort();
        AppState.dashboardRenderController = null;
    }
    for (const control of AppState.containerRenderControllers.values()) {
        control.detachParentAbort?.();
        control.controller.abort();
    }
    AppState.containerRenderControllers.clear();
    if (reason) console.log(`Invalidated dashboard renders (${reason})`);
}

function createDashboardRenderContext(
    containerId,
    resultPath,
    step,
    generation = AppState.dashboardRenderGeneration,
    requestController = null
) {
    const container = document.getElementById(containerId);
    const previousControl = AppState.containerRenderControllers.get(containerId);
    if (previousControl) {
        previousControl.detachParentAbort?.();
        previousControl.controller.abort();
    }

    const controller = new AbortController();
    let detachParentAbort = null;
    if (requestController) {
        if (requestController.signal.aborted) {
            controller.abort();
        } else {
            const abortFromParent = () => controller.abort();
            requestController.signal.addEventListener('abort', abortFromParent, { once: true });
            detachParentAbort = () => requestController.signal.removeEventListener('abort', abortFromParent);
        }
    }
    AppState.containerRenderControllers.set(containerId, { controller, detachParentAbort });

    const token = {};
    const context = {
        scope: 'dashboard',
        generation,
        resultPath,
        step,
        containerId,
        container,
        token,
        controller,
        requestController,
        analysisConditions: AppState.analysisConditions,
        polarView: snapshotPolarView(AppState.analysisConditions),
    };
    AppState.containerRenderTokens.set(containerId, token);
    if (container) container._activeRenderToken = token;
    return context;
}

function createFilePreviewRenderContext(previewContext, container) {
    const token = {};
    const context = {
        scope: 'file-preview',
        generation: previewContext.generation,
        controller: previewContext.controller,
        resultPath: previewContext.resultPath,
        analysisConditions: previewContext.analysisConditions,
        polarView: snapshotPolarView(previewContext.analysisConditions),
        containerId: container.id,
        container,
        token,
    };
    container._activeRenderToken = token;
    return context;
}

function createBHRenderContext(container) {
    if (AppState.bhRenderController) AppState.bhRenderController.abort();
    const controller = new AbortController();
    const token = {};
    const generation = ++AppState.bhRenderGeneration;
    if (!container.id) container.id = `libBHPlot-${generation}`;
    const context = {
        scope: 'bh-library',
        generation,
        controller,
        containerId: container.id,
        container,
        token,
    };
    AppState.bhRenderController = controller;
    container._activeRenderToken = token;
    return context;
}

function createResultPreviewRenderContext(loadContext, container) {
    const token = {};
    const context = {
        scope: 'result-load',
        resultLoadContext: loadContext,
        resultPath: loadContext.resultPath,
        analysisConditions: loadContext.analysisConditions || AppState.analysisConditions,
        polarView: snapshotPolarView(loadContext.analysisConditions || AppState.analysisConditions),
        containerId: container.id,
        container,
        token,
    };
    container._activeRenderToken = token;
    return context;
}

function snapshotPolarView(analysisConditions) {
    const isPolar = analysisConditions?.coordinate_system === 'polar';
    const thetaRange = analysisConditions?.polar?.theta_range
        || analysisConditions?.theta_range
        || 0;
    return Object.freeze({
        isPolar,
        cartesianTransform: isPolar && AppState.polarCartesianTransform,
        fullModel: isPolar && AppState.polarFullModel,
        fullModelMultiplier: isPolar ? calculateFullModelMultiplier(thetaRange) : 1,
    });
}

function ensureDashboardRenderContext(containerId, step, renderContext = null) {
    if (renderContext) return renderContext;
    return createDashboardRenderContext(
        containerId,
        getCurrentResultPath(),
        step,
        AppState.dashboardRenderGeneration
    );
}

function isRenderContextCurrent(renderContext, container = null) {
    if (!renderContext) return true;
    const expectedContainer = renderContext.container;
    const target = container || expectedContainer;
    if (!target || target !== expectedContainer || !target.isConnected ||
        document.getElementById(renderContext.containerId) !== target ||
        target._activeRenderToken !== renderContext.token) {
        return false;
    }

    if (renderContext.scope === 'result-load') {
        return isResultLoadCurrent(renderContext.resultLoadContext) &&
               getCurrentResultPath() === renderContext.resultPath;
    }

    if (renderContext.scope === 'file-preview') {
        return !renderContext.controller.signal.aborted &&
               AppState.filePreviewGeneration === renderContext.generation &&
               AppState.filePreviewController === renderContext.controller &&
               AppState.filePreviewResultPath === renderContext.resultPath;
    }

    if (renderContext.scope === 'bh-library') {
        return !renderContext.controller.signal.aborted &&
               AppState.bhRenderGeneration === renderContext.generation &&
               AppState.bhRenderController === renderContext.controller;
    }

    return renderContext.scope === 'dashboard' &&
           !renderContext.controller.signal.aborted &&
           AppState.dashboardRenderGeneration === renderContext.generation &&
           AppState.containerRenderTokens.get(renderContext.containerId) === renderContext.token &&
           getCurrentResultPath() === renderContext.resultPath &&
           AppState.currentStep === renderContext.step;
}

function assertRenderContextCurrent(renderContext, container = null) {
    if (!isRenderContextCurrent(renderContext, container)) {
        throw createAbortError('Plot render superseded');
    }
}

function purgeOwnedStaleRender(graphDiv, renderContext) {
    if (!graphDiv || !renderContext) return;
    if (graphDiv._plotlyRenderToken === renderContext.token ||
        graphDiv._activeRenderToken === renderContext.token) {
        purgePlotlyTree(graphDiv);
    }
}

async function newPlotForRender(graphDiv, traces, layout, config, renderContext = null) {
    assertRenderContextCurrent(renderContext);
    if (renderContext) graphDiv._plotlyRenderToken = renderContext.token;
    let plot;
    try {
        plot = await Plotly.newPlot(graphDiv, traces, layout, config);
    } catch (error) {
        // Widget cleanup can run while Plotly is still constructing a graph.
        // If that detached graph later rejects, purge it regardless of token
        // ownership because no newer connected render can own this node.
        if (!graphDiv.isConnected) purgePlotlyTree(graphDiv);
        throw error;
    }
    if (!graphDiv.isConnected) {
        // cleanupPlotWidgetElement may already have removed both ownership
        // tokens before Plotly.newPlot resolves. A detached graph is always
        // disposable, so do not gate this purge on the old render token.
        purgePlotlyTree(graphDiv);
        throw createAbortError('Plot container detached during Plotly commit');
    }
    if (!isRenderContextCurrent(renderContext)) {
        purgeOwnedStaleRender(graphDiv, renderContext);
        throw createAbortError('Plot render superseded after Plotly commit');
    }
    return plot;
}

function discardStaleRender(container, renderContext) {
    if (!container || !renderContext) return;
    if (container._activeRenderToken === renderContext.token) {
        preparePlotlyContainer(container);
    }
}

function cleanupPlotWidgetElement(widgetEl) {
    if (!widgetEl) return;
    const content = widgetEl.querySelector('.grid-stack-item-content');
    const containerId = content?.dataset.containerId;
    const plotId = widgetEl.dataset.plotId || widgetEl.gridstackNode?.id ||
                   (containerId ? containerId.replace(/^container-/, '') : null);
    const container = containerId ? document.getElementById(containerId) : null;

    if (containerId) {
        AppState.containerRenderTokens.delete(containerId);
        const control = AppState.containerRenderControllers.get(containerId);
        if (control) {
            control.detachParentAbort?.();
            control.controller.abort();
            AppState.containerRenderControllers.delete(containerId);
        }
    }
    purgePlotlyTree(container || widgetEl);
    if (container) {
        delete container._activeRenderToken;
        delete container._plotlyRenderToken;
    }
    if (containerId) {
        delete AppState.plotZoomStates[containerId];
        delete lineProfileState[containerId];
        delete fluxLinkageState[containerId];
    }
    if (plotId) delete AppState.plotConfigs[plotId];
}

function removePlot(plotId) {
    const items = AppState.gridStack.engine.nodes;
    const item = items.find(n => n.id === plotId);
    if (item && item.el) {
        cleanupPlotWidgetElement(item.el);
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
    // Re-render through the same serialized generation/controller pipeline as
    // step and result changes. Direct relayout().then(react) can otherwise
    // recreate Plotly resources after a widget has already been removed.
    refreshAllPlots();
}

// ===== Update All Plots =====
async function runDashboardRenderRequest(request) {
    if (request.controller.signal.aborted ||
        request.generation !== AppState.dashboardRenderGeneration ||
        request.resultPath !== getCurrentResultPath() ||
        request.step !== AppState.currentStep) {
        return;
    }

    const contentElements = Array.from(
        document.querySelectorAll('.grid-stack-item-content[data-plot-type]')
    );
    console.log(`updateAllPlots: Found ${contentElements.length} plots, currentStep=${request.step}`);

    for (const contentElement of contentElements) {
        if (request.controller.signal.aborted ||
            request.generation !== AppState.dashboardRenderGeneration ||
            request.resultPath !== getCurrentResultPath() ||
            request.step !== AppState.currentStep) break;

        const plotType = contentElement.dataset.plotType;
        const containerId = contentElement.dataset.containerId;
        const container = document.getElementById(containerId);
        const plotDefinition = plotDefinitions[plotType];

        if (!plotDefinition) {
            console.warn(`updateAllPlots: Skipping invalid plot type: ${plotType}`);
            continue;
        }
        if (!container || !container.isConnected ||
            contentElement.dataset.containerId !== container.id) {
            console.warn(`updateAllPlots: Container not available: ${containerId}`);
            continue;
        }

        const renderContext = createDashboardRenderContext(
            containerId,
            request.resultPath,
            request.step,
            request.generation,
            request.controller
        );
        try {
            console.log(`updateAllPlots: Rendering ${plotType} in ${containerId} for step ${request.step}`);
            await plotDefinition.render(containerId, request.step, renderContext);
            if (!isRenderContextCurrent(renderContext, container)) {
                discardStaleRender(container, renderContext);
                if (request.controller.signal.aborted ||
                    request.generation !== AppState.dashboardRenderGeneration ||
                    request.resultPath !== getCurrentResultPath() ||
                    request.step !== AppState.currentStep) break;
                // A removed/re-rendered container is local staleness; it must
                // not prevent the remaining widgets in this request rendering.
                continue;
            }
            console.log(`updateAllPlots: Successfully rendered ${plotType}`);
        } catch (error) {
            if ((error && error.name === 'AbortError') ||
                !isRenderContextCurrent(renderContext, container)) {
                discardStaleRender(container, renderContext);
                if (request.controller.signal.aborted ||
                    request.generation !== AppState.dashboardRenderGeneration ||
                    request.resultPath !== getCurrentResultPath() ||
                    request.step !== AppState.currentStep) break;
                continue;
            }
            console.error(`Error updating ${plotType}:`, error);
            console.error('Error stack:', error.stack);
            showPlotMessage(
                container,
                `<p style="color:red; padding:20px;">Error: ${error.message}</p>`,
                renderContext
            );
        }
    }
}

function updateAllPlots() {
    const resultPath = getCurrentResultPath();
    if (!resultPath) {
        console.log('updateAllPlots: No result selected');
        return Promise.resolve();
    }

    if (AppState.dashboardRenderController) AppState.dashboardRenderController.abort();
    for (const control of AppState.containerRenderControllers.values()) {
        control.detachParentAbort?.();
        control.controller.abort();
    }
    AppState.containerRenderControllers.clear();

    const controller = new AbortController();
    AppState.dashboardRenderController = controller;
    const request = {
        generation: ++AppState.dashboardRenderGeneration,
        resultPath,
        step: AppState.currentStep,
        controller,
    };
    const queued = AppState.dashboardRenderQueue
        .catch(error => console.error('Previous dashboard render failed:', error))
        .then(() => runDashboardRenderRequest(request))
        .finally(() => {
            if (AppState.dashboardRenderController === controller) {
                AppState.dashboardRenderController = null;
            }
        });
    AppState.dashboardRenderQueue = queued;
    return queued;
}

// ===== Animation Controls =====
function playAnimation() {
    if (AppState.isAnimating || AppState.totalSteps <= 1) return;

    AppState.isAnimating = true;
    const runId = ++AppState.animationRunId;
    document.getElementById('playBtn').style.display = 'none';
    document.getElementById('pauseBtn').style.display = 'inline-block';

    const scheduleNext = (delay) => {
        AppState.animationTimer = setTimeout(async () => {
            if (!AppState.isAnimating || AppState.animationRunId !== runId) return;
            const startedAt = performance.now();

            AppState.currentStep++;
            if (AppState.currentStep > AppState.totalSteps) {
                AppState.currentStep = 1; // Loop
            }

            document.getElementById('stepSlider').value = AppState.currentStep;
            document.getElementById('currentStep').textContent = AppState.currentStep;

            try {
                // Schedule the next frame only after this render completes, so
                // full-grid calculations and Plotly updates never overlap.
                await updateAllPlots();
            } catch (error) {
                if (!error || error.name !== 'AbortError') {
                    console.error('Animation render failed:', error);
                }
                pauseAnimation();
            } finally {
                if (!AppState.isAnimating || AppState.animationRunId !== runId) return;
                const speed = Math.max(16, parseInt(document.getElementById('animSpeed').value, 10) || 1000);
                const elapsed = performance.now() - startedAt;
                scheduleNext(Math.max(0, speed - elapsed));
            }
        }, delay);
    };

    const initialSpeed = Math.max(16, parseInt(document.getElementById('animSpeed').value, 10) || 1000);
    scheduleNext(initialSpeed);
}

function pauseAnimation() {
    AppState.animationRunId++;
    if (AppState.animationTimer) {
        clearTimeout(AppState.animationTimer);
        AppState.animationTimer = null;
    }

    AppState.isAnimating = false;
    document.getElementById('playBtn').style.display = 'inline-block';
    document.getElementById('pauseBtn').style.display = 'none';
}

function estimateFieldDataBytes(value) {
    const seen = new Set();
    const stack = [value];
    let bytes = 0;

    while (stack.length > 0) {
        const current = stack.pop();
        if (current == null || seen.has(current)) continue;

        if (ArrayBuffer.isView(current)) {
            seen.add(current);
            bytes += FIELD_CACHE_ARRAY_OVERHEAD_BYTES + current.byteLength;
            continue;
        }
        if (current instanceof ArrayBuffer) {
            seen.add(current);
            bytes += FIELD_CACHE_ARRAY_OVERHEAD_BYTES + current.byteLength;
            continue;
        }
        if (!Array.isArray(current)) continue;

        seen.add(current);
        bytes += FIELD_CACHE_ARRAY_OVERHEAD_BYTES + current.length * 8;
        for (const item of current) {
            if (Array.isArray(item) || ArrayBuffer.isView(item) || item instanceof ArrayBuffer) {
                stack.push(item);
            }
        }
    }

    return Math.max(bytes, FIELD_CACHE_ARRAY_OVERHEAD_BYTES);
}

function formatFieldCacheBytes(bytes) {
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${bytes} B`;
}

function retainPinnedFieldData(data, bytes) {
    let ref = AppState.fieldViewPinRefs.get(data);
    if (!ref) {
        ref = { bytes, count: 0 };
        AppState.fieldViewPinRefs.set(data, ref);
        AppState.fieldViewPinnedBytes += bytes;
    }
    ref.count++;
}

function releasePinnedFieldData(data) {
    const ref = AppState.fieldViewPinRefs.get(data);
    if (!ref) return;
    ref.count = Math.max(0, ref.count - 1);
    if (ref.count !== 0) return;
    AppState.fieldViewPinnedBytes = Math.max(
        0,
        AppState.fieldViewPinnedBytes - ref.bytes
    );
    AppState.fieldViewPinRefs.delete(data);
}

function pinFieldDataForSignal(cacheKey, data, signal) {
    if (!signal || signal.aborted || data == null) return;
    let pin = AppState.fieldViewPins.get(signal);
    if (!pin) {
        pin = { fields: new Map() };
        AppState.fieldViewPins.set(signal, pin);
        signal.addEventListener('abort', () => {
            const current = AppState.fieldViewPins.get(signal);
            if (current !== pin) return;
            for (const field of pin.fields.values()) {
                releasePinnedFieldData(field.data);
            }
            pin.fields.clear();
            AppState.fieldViewPins.delete(signal);
        }, { once: true });
    }

    const existing = pin.fields.get(cacheKey);
    if (existing?.data === data) return;
    if (existing) {
        releasePinnedFieldData(existing.data);
    }
    const bytes = estimateFieldDataBytes(data);
    pin.fields.set(cacheKey, { data, bytes });
    retainPinnedFieldData(data, bytes);
}

function evictFieldCacheEntry(cacheKey) {
    if (!Object.prototype.hasOwnProperty.call(AppState.dataCache, cacheKey)) return;
    AppState.dataCacheBytes = Math.max(
        0,
        AppState.dataCacheBytes - (AppState.dataCacheSizes[cacheKey] || 0)
    );
    delete AppState.dataCache[cacheKey];
    delete AppState.dataMeta[cacheKey];
    delete AppState.dataCacheSizes[cacheKey];
    AppState.dataCacheLru.delete(cacheKey);
}

function getCachedFieldData(cacheKey) {
    if (!Object.prototype.hasOwnProperty.call(AppState.dataCache, cacheKey)) return undefined;
    const data = AppState.dataCache[cacheKey];
    AppState.dataCacheLru.delete(cacheKey);
    AppState.dataCacheLru.set(cacheKey, true);
    return data;
}

function cacheFieldData(cacheKey, data, meta) {
    const sizeBytes = estimateFieldDataBytes(data);
    const budgetBytes = AppState.dataCacheBudgetBytes;

    if (Object.prototype.hasOwnProperty.call(AppState.dataCache, cacheKey)) {
        evictFieldCacheEntry(cacheKey);
    }
    if (sizeBytes > budgetBytes) {
        console.warn(
            `Field ${cacheKey} is ${formatFieldCacheBytes(sizeBytes)}, larger than the ` +
            `${formatFieldCacheBytes(budgetBytes)} cache budget; returning it without caching`
        );
        return false;
    }

    while (AppState.dataCacheBytes + sizeBytes > budgetBytes && AppState.dataCacheLru.size > 0) {
        const oldestKey = AppState.dataCacheLru.keys().next().value;
        evictFieldCacheEntry(oldestKey);
    }

    AppState.dataCache[cacheKey] = data;
    AppState.dataMeta[cacheKey] = meta;
    AppState.dataCacheSizes[cacheKey] = sizeBytes;
    AppState.dataCacheLru.set(cacheKey, true);
    AppState.dataCacheBytes += sizeBytes;
    return true;
}

function cacheFieldPairAtomically(entries) {
    const pending = entries.filter(entry =>
        !Object.prototype.hasOwnProperty.call(AppState.dataCache, entry.cacheKey)
    ).map(entry => ({ ...entry, sizeBytes: estimateFieldDataBytes(entry.data) }));
    const pendingBytes = pending.reduce((sum, entry) => sum + entry.sizeBytes, 0);

    // Preload never evicts existing entries speculatively. Both decoded fields
    // are committed in one synchronous section only when the pair fits; a
    // failed/oversized pair therefore needs no rollback and cannot delete a
    // pre-existing Az or Mu entry.
    if (pending.some(entry => entry.sizeBytes > AppState.dataCacheBudgetBytes) ||
        AppState.dataCacheBytes + pendingBytes > AppState.dataCacheBudgetBytes) {
        return false;
    }

    for (const entry of pending) {
        AppState.dataCache[entry.cacheKey] = entry.data;
        AppState.dataMeta[entry.cacheKey] = entry.meta;
        AppState.dataCacheSizes[entry.cacheKey] = entry.sizeBytes;
        AppState.dataCacheLru.set(entry.cacheKey, true);
        AppState.dataCacheBytes += entry.sizeBytes;
    }
    for (const entry of entries) {
        AppState.dataCacheLru.delete(entry.cacheKey);
        AppState.dataCacheLru.set(entry.cacheKey, true);
    }
    AppState.lastFieldMeta = entries[entries.length - 1]?.meta || AppState.lastFieldMeta;
    updateExportFormatBadge();
    return entries.every(entry =>
        Object.prototype.hasOwnProperty.call(AppState.dataCache, entry.cacheKey)
    );
}

function clearFieldDataCache(reason = '') {
    const previousBytes = AppState.dataCacheBytes;
    AppState.dataCache = {};
    AppState.dataMeta = {};
    AppState.dataCacheSizes = {};
    AppState.dataCacheLru.clear();
    AppState.dataCacheBytes = 0;
    AppState.lastFieldMeta = null;
    if (previousBytes > 0) {
        console.log(
            `Cleared ${formatFieldCacheBytes(previousBytes)} of field cache` +
            (reason ? ` (${reason})` : '')
        );
    }
}

function invalidateFieldLoads(reason = '') {
    AppState.fieldLoadGeneration++;
    // Result changes invalidate dashboard/preload consumers, not the shared
    // physical payload itself. A File Manager preview may legitimately be
    // reading a different result at the same time. Shared requests are
    // cancelled by releaseSharedFieldPayload() only after their last consumer
    // has left.
    AppState.fieldPreloadController?.abort();
    AppState.fieldPreloadController = null;
    if (reason) console.log(`Invalidated field loads (${reason})`);
}

async function preloadAllSteps() {
    const currentResult = AppState.resultsData.currentResult;
    if (!currentResult || AppState.totalSteps <= 0) {
        alert('Please select analysis results first');
        return;
    }
    const preloadGeneration = AppState.fieldLoadGeneration;
    const preloadResultLoadGeneration = AppState.resultLoadGeneration;
    AppState.fieldPreloadController?.abort();
    const preloadController = new AbortController();
    AppState.fieldPreloadController = preloadController;

    const btn = document.getElementById('preloadBtn');
    if (!btn) return;

    btn.disabled = true;
    const originalText = btn.textContent;

    try {
        console.log(
            `Starting preload of ${AppState.totalSteps} steps with ` +
            `${formatFieldCacheBytes(AppState.dataCacheBudgetBytes)} cache budget`
        );
        let previousStepBytes = 0;
        let loadedSteps = 0;
        let stoppedForBudget = false;
        let stoppedForError = false;

        for (let step = 1; step <= AppState.totalSteps; step++) {
            if (preloadGeneration !== AppState.fieldLoadGeneration ||
                preloadResultLoadGeneration !== AppState.resultLoadGeneration ||
                preloadController.signal.aborted ||
                getCurrentResultPath() !== currentResult) {
                const error = new Error('Preload invalidated');
                error.name = 'AbortError';
                throw error;
            }
            btn.textContent = `Loading ${step}/${AppState.totalSteps}`;
            const azKey = `${currentResult}:Az:${step}`;
            const muKey = `${currentResult}:Mu:${step}`;
            const pairAlreadyCached =
                Object.prototype.hasOwnProperty.call(AppState.dataCache, azKey) &&
                Object.prototype.hasOwnProperty.call(AppState.dataCache, muKey);

            // Field dimensions are stable within a result. Once one step has
            // been measured, stop before the next step would force LRU
            // eviction; decoding every step only to evict earlier ones wastes
            // both I/O and peak memory.
            if (!pairAlreadyCached && previousStepBytes > 0 &&
                AppState.dataCacheBytes + previousStepBytes > AppState.dataCacheBudgetBytes) {
                stoppedForBudget = true;
                break;
            }

            // Decode missing halves sequentially to limit simultaneous TIFF
            // decode pressure, then commit Az+Mu as one cache transaction.
            let azLease = null;
            let muLease = null;
            let cachedAzPinController = null;
            try {
                const azWasCached = Object.prototype.hasOwnProperty.call(AppState.dataCache, azKey);
                if (!azWasCached) {
                    azLease = await loadFieldDataForPreload(
                        'Az',
                        step,
                        currentResult,
                        preloadController.signal
                    );
                }
                const azEntry = azWasCached
                    ? { data: AppState.dataCache[azKey], meta: AppState.dataMeta[azKey] }
                    : azLease.payload;
                const azPendingBytes = azWasCached ? 0 : estimateFieldDataBytes(azEntry.data);
                const muWasCached = Object.prototype.hasOwnProperty.call(AppState.dataCache, muKey);
                if (azWasCached && !muWasCached) {
                    // The next decode may evict Az from the LRU, but this
                    // step-local reference remains live until the pair commit.
                    // Charge it explicitly so eviction cannot hide it from the
                    // working-memory decision.
                    cachedAzPinController = new AbortController();
                    pinFieldDataForSignal(
                        azKey,
                        azEntry.data,
                        cachedAzPinController.signal
                    );
                }
                // Az and Mu share one grid shape, so Az's measured decoded size
                // is a conservative predictor for a missing Mu field. Reserve
                // one additional Mu-sized block for its decode workspace. This
                // prevents preload from retaining Az and only then discovering
                // that Mu pushes the browser far beyond the cache budget.
                const projectedMuBytes = muWasCached
                    ? 0
                    : (AppState.dataCacheSizes[azKey] || azPendingBytes);
                const projectedMuPeakBytes = muWasCached
                    ? 0
                    : Math.max(
                        FIELD_UNKNOWN_DECODE_RESERVATION_BYTES,
                        projectedMuBytes * 3
                    );
                const pairWouldExceedCache =
                    AppState.dataCacheBytes +
                    azPendingBytes +
                    projectedMuBytes >
                    AppState.dataCacheBudgetBytes;
                const decodeWouldExceedWorkingBudget =
                    AppState.dataCacheBytes +
                    AppState.fieldPayloadStagingBytes +
                    AppState.fieldViewPinnedBytes +
                    projectedMuPeakBytes >
                    FIELD_WORKING_MEMORY_BUDGET_BYTES;
                if (pairWouldExceedCache || decodeWouldExceedWorkingBudget) {
                    stoppedForBudget = true;
                    break;
                }
                if (!muWasCached) {
                    muLease = await loadFieldDataForPreload(
                        'Mu',
                        step,
                        currentResult,
                        preloadController.signal
                    );
                }
                const muEntry = muWasCached
                    ? { data: AppState.dataCache[muKey], meta: AppState.dataMeta[muKey] }
                    : muLease.payload;

                const committed = cacheFieldPairAtomically([
                    { cacheKey: azKey, ...azEntry },
                    { cacheKey: muKey, ...muEntry },
                ]);
                if (!committed) {
                    stoppedForBudget = true;
                    break;
                }
                azLease?.markCached();
                muLease?.markCached();
                previousStepBytes = (AppState.dataCacheSizes[azKey] || 0) +
                                    (AppState.dataCacheSizes[muKey] || 0);
                loadedSteps++;
            } catch (e) {
                if (e && e.name === 'AbortError') throw e;
                if (e && e.code === 'FIELD_MEMORY_BUDGET') {
                    console.warn(`Preload stopped at step ${step}: ${e.message}`);
                    stoppedForBudget = true;
                    break;
                }
                console.warn(`Preload failed at step ${step}:`, e.message);
                stoppedForError = true;
                break;
            } finally {
                // Keep decoded payloads leased through the pair commit so a
                // normal widget requesting the same field reuses them instead
                // of starting a duplicate fetch/decode in the staging window.
                muLease?.release();
                azLease?.release();
                cachedAzPinController?.abort();
            }

            // Small delay to prevent overwhelming the server
            if (step < AppState.totalSteps) {
                await awaitWithAbortSignal(
                    new Promise(resolve => setTimeout(resolve, 10)),
                    preloadController.signal,
                    'Preload invalidated'
                );
            }
        }

        const cacheEntries = AppState.dataCacheLru.size;
        console.log(
            `Preload complete: ${loadedSteps}/${AppState.totalSteps} steps, ` +
            `${cacheEntries} fields / ${formatFieldCacheBytes(AppState.dataCacheBytes)}`
        );
        btn.textContent = stoppedForBudget
            ? `Cached ${loadedSteps}/${AppState.totalSteps} (memory limit)`
            : stoppedForError
                ? `Stopped ${loadedSteps}/${AppState.totalSteps}`
                : 'Preloaded ✓';
        setTimeout(() => {
            btn.textContent = originalText;
        }, (stoppedForBudget || stoppedForError) ? 3000 : 2000);

    } catch (error) {
        if (error && error.name === 'AbortError') {
            console.log('Preload cancelled because the field-data context changed');
        } else {
            console.error('Preload error:', error);
            alert(`Preload failed: ${error.message}`);
        }
        btn.textContent = originalText;
    } finally {
        if (AppState.fieldPreloadController === preloadController) {
            AppState.fieldPreloadController = null;
        }
        btn.disabled = false;
    }
}

function setStep(step) {
    pauseAnimation();
    AppState.currentStep = step;
    const slider = document.getElementById('stepSlider');
    if (slider) slider.value = step;
    document.getElementById('currentStep').textContent = step;
    return updateAllPlots();
}

function clearDashboard() {
    if (AppState.gridStack) {
        pauseAnimation();
        invalidateDashboardRenders('dashboard cleared');
        document.querySelectorAll('#dashboard-grid .grid-stack-item')
            .forEach(cleanupPlotWidgetElement);
        AppState.containerRenderTokens.clear();
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

// Update the small diagnostic badge under the result selector with the
// export format used by the current run. Reads conditions.json's export
// block (written by the solver via exportConditionsJSON) for the "intended"
// format, then refines with dataMeta (what was actually served by the
// server) once any field has been fetched.
function updateExportFormatBadge() {
    const el = document.getElementById('resultMetaBadge');
    if (!el) return;
    const cond = AppState.analysisConditions || {};
    const exp = cond.export || {};
    const fmt = exp.format || 'unknown';
    const prec = exp.precision || 'double';
    const asyncFlag = exp.async === true ? ' / async' : '';
    let line = `Source intent: ${fmt} (${prec})${asyncFlag}`;
    // If we've already loaded any field, append what the server actually
    // returned (useful when format=both — server picks tiff first).
    const last = AppState.lastFieldMeta;
    if (last) {
        if (last && last.format) {
            line += ` — served: ${last.format} (${last.precision || '?'})`;
        }
    }
    el.textContent = line;
}

// Decode a TIFF ArrayBuffer (single-channel float32/float64, NaN-aware) into
// a 2D JS array. NaN bit patterns become null so the existing
// _fillInactiveScalar() Gauss-Seidel filler treats them as gaps. Y-axis is
// reversed to match the legacy CSV path (image coords).
function reserveFieldDecodeWorkspace(bytes, label = 'field decode') {
    const requested = Math.max(
        FIELD_UNKNOWN_DECODE_RESERVATION_BYTES,
        Math.ceil(Number(bytes) || 0)
    );
    AppState.fieldDecodeReservedBytes = Math.max(
        AppState.fieldDecodeReservedBytes,
        requested
    );

    // Keep one bounded decode workspace in addition to retained/staged field
    // arrays. Cache entries are safe to discard because renderers already hold
    // their own references while drawing.
    while (
        AppState.dataCacheLru.size > 0 &&
        AppState.dataCacheBytes +
            AppState.fieldPayloadStagingBytes +
            AppState.fieldViewPinnedBytes +
            AppState.fieldDecodeReservedBytes >
            FIELD_WORKING_MEMORY_BUDGET_BYTES
    ) {
        const oldestKey = AppState.dataCacheLru.keys().next().value;
        evictFieldCacheEntry(oldestKey);
    }

    const estimatedWorkingBytes =
        AppState.dataCacheBytes +
        AppState.fieldPayloadStagingBytes +
        AppState.fieldViewPinnedBytes +
        AppState.fieldDecodeReservedBytes;
    if (estimatedWorkingBytes > FIELD_WORKING_MEMORY_BUDGET_BYTES) {
        const error = new Error(
            `${label} was stopped before allocating about ` +
            `${formatFieldCacheBytes(requested)} because active views and ` +
            `field data would exceed the ` +
            `${formatFieldCacheBytes(FIELD_WORKING_MEMORY_BUDGET_BYTES)} ` +
            'browser working-memory budget'
        );
        error.name = 'FieldMemoryBudgetError';
        error.code = 'FIELD_MEMORY_BUDGET';
        throw error;
    }
}

async function runSerializedFieldDecode(task, signal) {
    const previous = AppState.fieldDecodeTail;
    const turn = previous.catch(() => {}).then(async () => {
        if (signal?.aborted) throw createAbortError('Field decode cancelled');
        AppState.fieldDecodeReservedBytes = 0;
        reserveFieldDecodeWorkspace(
            FIELD_UNKNOWN_DECODE_RESERVATION_BYTES,
            'Field decode'
        );
        try {
            return await task();
        } finally {
            AppState.fieldDecodeReservedBytes = 0;
        }
    });
    // Keep the queue usable after either a decode failure or cancellation.
    AppState.fieldDecodeTail = turn.then(() => undefined, () => undefined);
    return await awaitWithAbortSignal(turn, signal, 'Field decode cancelled');
}

async function decodeTiffArrayBuffer(arrayBuffer, signal = null) {
    if (typeof GeoTIFF === 'undefined') {
        throw new Error('GeoTIFF library not loaded (expected at /lib/geotiff.js)');
    }
    if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
    const image = await tiff.getImage();
    if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
    const width = image.getWidth();
    const height = image.getHeight();
    const bps = image.getBitsPerSample();
    const bytesPerSample = bps === 64 ? 8 : 4;
    // Account for the source ArrayBuffer, GeoTIFF/raster workspace and the
    // final JS row arrays. The multiplier deliberately leaves headroom for
    // decoder temporaries that are not directly observable from JavaScript.
    reserveFieldDecodeWorkspace(
        arrayBuffer.byteLength * 2 +
            width * height * (bytesPerSample + 24) +
            height * FIELD_CACHE_ARRAY_OVERHEAD_BYTES,
        'TIFF field decode'
    );
    const rasters = await image.readRasters();
    if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
    const raster = rasters[0];
    const precision = bps === 64 ? 'double' : 'float';

    const data = new Array(height);
    for (let j = 0; j < height; j++) {
        if (j > 0 && j % 64 === 0) {
            // Yield to the browser so a superseding view can deliver its abort
            // event instead of waiting for the full row conversion.
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
        const row = new Array(width);
        for (let i = 0; i < width; i++) {
            const v = raster[j * width + i];
            row[i] = Number.isNaN(v) ? null : v;
        }
        data[j] = row;
    }
    data.reverse();
    return { data, precision };
}

async function fetchDecodedFieldPayload(
    dataType,
    step,
    resultPath,
    signal,
    onDecoded = null
) {
    return await runSerializedFieldDecode(async () => {
        const file = `${dataType}/${formatStepFilename(step)}`;
        const response = await fetch(
            `/api/load-field?result=${encodeURIComponent(resultPath)}&file=${file}`,
            { signal }
        );
        if (!response.ok) {
            throw new Error(`Failed to load ${dataType} data (HTTP ${response.status})`);
        }

        const contentLength = Math.max(
            0,
            Number(response.headers.get('Content-Length')) || 0
        );
        const contentType = (response.headers.get('Content-Type') || '').toLowerCase();
        if (contentType.startsWith('image/tiff')) {
            reserveFieldDecodeWorkspace(
                Math.max(
                    FIELD_UNKNOWN_DECODE_RESERVATION_BYTES,
                    contentLength * 8
                ),
                'TIFF response'
            );
            const arrayBuffer = await response.arrayBuffer();
            if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
            const decoded = await decodeTiffArrayBuffer(arrayBuffer, signal);
            if (signal?.aborted) throw createAbortError('TIFF decode cancelled');
            const payload = {
                data: decoded.data,
                meta: { format: 'tiff', precision: decoded.precision },
            };
            onDecoded?.(payload);
            return payload;
        }

        reserveFieldDecodeWorkspace(
            Math.max(
                FIELD_UNKNOWN_DECODE_RESERVATION_BYTES,
                contentLength * 6
            ),
            'JSON field response'
        );
        const result = await response.json();
        if (signal?.aborted) throw createAbortError('JSON field decode cancelled');
        if (!result.success) {
            throw new Error(`Failed to parse ${dataType} data: ${result.error || 'unknown error'}`);
        }
        reserveFieldDecodeWorkspace(
            contentLength * 2 + estimateFieldDataBytes(result.data) * 2,
            'JSON field decode'
        );
        const payload = {
            data: result.data,
            meta: {
                format: result.format || 'csv',
                precision: result.precision || 'double',
            },
        };
        onDecoded?.(payload);
        return payload;
    }, signal);
}

function awaitWithAbortSignal(promise, signal, message = 'Request superseded') {
    if (!signal) return promise;
    if (signal.aborted) return Promise.reject(createAbortError(message));
    return new Promise((resolve, reject) => {
        const onAbort = () => reject(createAbortError(message));
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
            value => {
                signal.removeEventListener('abort', onAbort);
                resolve(value);
            },
            error => {
                signal.removeEventListener('abort', onAbort);
                reject(error);
            }
        );
    });
}

function getSharedFieldPayloadEntry(dataType, step, resultPath) {
    const cacheKey = `${resultPath}:${dataType}:${step}`;
    const existing = AppState.fieldPayloadsInFlight.get(cacheKey);
    if (existing) return existing;

    const controller = new AbortController();
    const entry = {
        cacheKey,
        controller,
        consumers: 0,
        settled: false,
        stagingBytes: 0,
        promise: null,
    };
    entry.promise = (async () => {
        try {
            const payload = await fetchDecodedFieldPayload(
                dataType,
                step,
                resultPath,
                controller.signal,
                decodedPayload => {
                    if (controller.signal.aborted || entry.consumers === 0) return;
                    entry.stagingBytes = estimateFieldDataBytes(decodedPayload.data);
                    AppState.fieldPayloadStagingBytes += entry.stagingBytes;
                }
            );
            if (controller.signal.aborted) {
                throw createAbortError('Field payload invalidated');
            }
            return payload;
        } finally {
            entry.settled = true;
        }
    })();
    AppState.fieldPayloadsInFlight.set(cacheKey, entry);
    // A rejection may happen after every caller has cancelled its await.
    // Attach a handler here so abort-driven disposal never produces an
    // unhandled rejection.
    entry.promise.catch(() => {});
    return entry;
}

function releaseSharedFieldPayload(entry) {
    if (!entry) return;
    entry.consumers = Math.max(0, entry.consumers - 1);
    if (entry.consumers !== 0) return;

    if (AppState.fieldPayloadsInFlight.get(entry.cacheKey) === entry) {
        AppState.fieldPayloadsInFlight.delete(entry.cacheKey);
    }
    if (entry.stagingBytes > 0) {
        AppState.fieldPayloadStagingBytes = Math.max(
            0,
            AppState.fieldPayloadStagingBytes - entry.stagingBytes
        );
        entry.stagingBytes = 0;
    }
    // If the final consumer leaves before fetch/decode settles, no useful
    // owner remains. Abort the physical request as well as the caller await.
    if (!entry.settled && !entry.controller.signal.aborted) {
        entry.controller.abort();
    }
}

function markSharedFieldPayloadCached(entry) {
    if (!entry) return;
    // Keep staging bytes charged until the final lease is released. Another
    // consumer (for example File Preview) may still hold the same array after
    // the cache copy is evicted. Temporary double-accounting is intentional
    // and safer than hiding that live reference from the working-set budget.
    entry.cached = true;
}

async function acquireSharedFieldPayload(
    dataType,
    step,
    resultPath,
    callerSignal = null
) {
    const entry = getSharedFieldPayloadEntry(dataType, step, resultPath);
    entry.consumers++;
    try {
        const payload = await awaitWithAbortSignal(
            entry.promise,
            callerSignal,
            'Field payload consumer superseded'
        );
        let released = false;
        return {
            payload,
            markCached() {
                markSharedFieldPayloadCached(entry);
            },
            release() {
                if (released) return;
                released = true;
                releaseSharedFieldPayload(entry);
            },
        };
    } catch (error) {
        releaseSharedFieldPayload(entry);
        throw error;
    }
}

async function loadFieldDataForPreload(dataType, step, resultPath, callerSignal = null) {
    const generation = AppState.fieldLoadGeneration;
    const lease = await acquireSharedFieldPayload(
        dataType,
        step,
        resultPath,
        callerSignal
    );
    try {
        if (generation !== AppState.fieldLoadGeneration || getCurrentResultPath() !== resultPath) {
            throw createAbortError('Preload field decode invalidated');
        }
        return lease;
    } catch (error) {
        lease.release();
        throw error;
    }
}

// Helper function to load field data (CSV or TIFF) with caching.
// Server returns either application/json (CSV path, decoded server-side) or
// image/tiff (raw TIFF stream, decoded here via GeoTIFF). The cache key does
// not include format so a format=both run hits cache regardless of which
// branch served it last time.
async function loadFieldData(dataType, step, providedResultPath = null, callerSignal = null) {
    const resultPath = providedResultPath || getCurrentResultPath();
    if (!resultPath) throw new Error('No result selected');
    if (callerSignal?.aborted) throw createAbortError('Field consumer superseded');
    const selectedResultAtStart = getCurrentResultPath();

    const cacheKey = `${resultPath}:${dataType}:${step}`;
    const cached = getCachedFieldData(cacheKey);
    if (cached !== undefined) {
        if (callerSignal?.aborted) throw createAbortError('Field consumer superseded');
        pinFieldDataForSignal(cacheKey, cached, callerSignal);
        console.log(`Cache hit: ${cacheKey}`);
        return cached;
    }

    const generation = AppState.fieldLoadGeneration;
    // Every actual caller owns a lease. Once all caller signals are aborted,
    // releaseSharedFieldPayload() aborts the physical fetch/decode instead of
    // allowing an invisible cache-populating request to run to completion.
    const lease = await acquireSharedFieldPayload(
        dataType,
        step,
        resultPath,
        callerSignal
    );
    try {
        const { data, meta } = lease.payload;
        const selectedResultChanged = getCurrentResultPath() !== selectedResultAtStart;
        const implicitResultIsStale = !providedResultPath && getCurrentResultPath() !== resultPath;
        if (callerSignal?.aborted ||
            generation !== AppState.fieldLoadGeneration ||
            selectedResultChanged ||
            implicitResultIsStale) {
            throw createAbortError('Field load invalidated');
        }

        const alreadyCached = getCachedFieldData(cacheKey);
        if (alreadyCached !== undefined) {
            pinFieldDataForSignal(cacheKey, alreadyCached, callerSignal);
            return alreadyCached;
        }

        AppState.lastFieldMeta = meta;
        if (cacheFieldData(cacheKey, data, meta)) {
            lease.markCached();
        }
        pinFieldDataForSignal(cacheKey, data, callerSignal);
        updateExportFormatBadge();
        return data;
    } finally {
        lease.release();
    }
}

// Placeholder implementations - these will call actual data loading and plotting
async function renderAzContour(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const data = await loadFieldData(
        'Az', step, renderContext.resultPath, renderContext.controller.signal
    );
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    await plotContour(containerId, flipped, 'Az [Wb/m]', true, renderContext);
}

async function renderAzHeatmap(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const data = await loadFieldData(
        'Az', step, renderContext.resultPath, renderContext.controller.signal
    );
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    // C++ solver already outputs fully-interpolated Az grid (bilinear at inactive cells).
    // No JS re-interpolation needed — Gauss-Seidel would create visible dots at active cells.
    await plotHeatmap(containerId, flipped, 'Az [Wb/m]', true, false, renderContext);
}

async function renderJzDistribution(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const data = await loadFieldData(
        'Jz', step, renderContext.resultPath, renderContext.controller.signal
    );
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    await plotHeatmap(containerId, flipped, 'Jz [A/m²]', true, false, renderContext);
}

async function renderBMagnitude(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const analysisConditions = AppState.analysisConditions;
    // Use grid spacing from analysis conditions
    const dx = analysisConditions ? analysisConditions.dx : 0.001;
    const dy = analysisConditions ? analysisConditions.dy : 0.001;

    // Load Az and Mu with caching
    const azData = await loadFieldData(
        'Az', step, renderContext.resultPath, renderContext.controller.signal
    );
    const muData = await loadFieldData(
        'Mu', step, renderContext.resultPath, renderContext.controller.signal
    );

    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const azFlipped = flipVertical(azData);
    const muFlipped = flipVertical(muData);

    // Load coarsening mask for coarsening-aware B computation (image coords, matches flipped Az)
    const resultPath = renderContext.resultPath;
    const maskResult = await getCoarseningMaskArray(
        resultPath,
        step,
        renderContext.controller.signal
    ).catch(error => {
        if ((error && error.name === 'AbortError') ||
            error?.code === 'FIELD_MEMORY_BUDGET') throw error;
        return null;
    });
    const activeMask = maskResult ? maskResult.mask : null;

    const { B } = calculateMagneticField(
        azFlipped, muFlipped, dx, dy, activeMask, renderContext.analysisConditions
    );

    await plotHeatmap(containerId, B, '|B| [T]', true, false, renderContext);
}

async function renderHMagnitude(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const analysisConditions = AppState.analysisConditions;
    // Check if nonlinear materials are present and enabled
    const hasNonlinear = analysisConditions?.nonlinear_solver?.has_nonlinear_materials;
    const nlEnabled = analysisConditions?.nonlinear_solver?.enabled;

    if (hasNonlinear && nlEnabled) {
        // For nonlinear materials: load H directly from solver output (H.csv)
        try {
            const hData = await loadFieldData(
                'H', step, renderContext.resultPath, renderContext.controller.signal
            );
            // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
            const hFlipped = flipVertical(hData);
            await plotHeatmap(containerId, hFlipped, '|H| [A/m] (solver)', true, false, renderContext);
            return;
        } catch (error) {
            if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext)) throw error;
            console.warn('H.csv not found, falling back to calculation from Az and Mu:', error);
        }
    }

    // For linear materials: calculate H from Az and Mu
    const dx = analysisConditions ? analysisConditions.dx : 0.001;
    const dy = analysisConditions ? analysisConditions.dy : 0.001;

    // Load Az and Mu with caching
    const azData = await loadFieldData(
        'Az', step, renderContext.resultPath, renderContext.controller.signal
    );
    const muData = await loadFieldData(
        'Mu', step, renderContext.resultPath, renderContext.controller.signal
    );

    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const azFlipped = flipVertical(azData);
    const muFlipped = flipVertical(muData);

    // Load coarsening mask for coarsening-aware H computation
    const resultPath = renderContext.resultPath;
    const maskResult = await getCoarseningMaskArray(
        resultPath,
        step,
        renderContext.controller.signal
    ).catch(error => {
        if ((error && error.name === 'AbortError') ||
            error?.code === 'FIELD_MEMORY_BUDGET') throw error;
        return null;
    });
    const activeMask = maskResult ? maskResult.mask : null;

    const { H } = calculateMagneticField(
        azFlipped, muFlipped, dx, dy, activeMask, renderContext.analysisConditions
    );

    await plotHeatmap(containerId, H, '|H| [A/m] (calculated)', true, false, renderContext);
}

async function renderMuDistribution(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const data = await loadFieldData(
        'Mu', step, renderContext.resultPath, renderContext.controller.signal
    );
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    let flipped = flipVertical(data);

    // C++ solver outputs fully-interpolated Mu grid — no frontend interpolation needed

    await plotHeatmap(containerId, flipped, 'μ [H/m]', true, true, renderContext);
}

async function renderEnergyDensity(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const data = await loadFieldData(
        'EnergyDensity', step, renderContext.resultPath, renderContext.controller.signal
    );
    // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
    const flipped = flipVertical(data);
    await plotHeatmap(containerId, flipped, 'Energy [J/m³]', true, false, renderContext);
}

// Helper: Convert black pixels in image to transparent
async function makeBlackTransparent(url, threshold = 30, signal = null) {
    const sourceUrl = signal ? await loadImageDataUrl(url, signal) : url;
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

        img.src = sourceUrl;
    });
}

// Helper: Flip an image URL vertically (for image coordinate to analysis coordinate conversion)
async function flipImageVertical(url, signal = null) {
    const sourceUrl = signal ? await loadImageDataUrl(url, signal) : url;
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

        img.src = sourceUrl;
    });
}

// Helper: Flip an image URL vertically AND make black pixels transparent
async function flipAndMakeBlackTransparent(url, threshold = 30, signal = null) {
    const sourceUrl = signal ? await loadImageDataUrl(url, signal) : url;
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

        img.src = sourceUrl;
    });
}

/**
 * Apply Sobel edge detection to an image URL
 * Returns a canvas with white background and black edges
 * @param {string} url - Image URL
 * @returns {Promise<HTMLCanvasElement>} - Canvas with edge detection result
 */
async function applySobelEdgeDetection(url, signal = null) {
    const sourceUrl = signal ? await loadImageDataUrl(url, signal) : url;
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

        img.src = sourceUrl;
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
async function loadImageToCanvas(url, signal = null) {
    const sourceUrl = signal ? await loadImageDataUrl(url, signal) : url;
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

        img.src = sourceUrl;
    });
}

async function renderAzBoundary(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Load Az data with caching
        const azData = await loadFieldData(
            'Az', step, resultPath, renderContext.controller.signal
        );

        // Get input image URL (material image as background for field lines)
        const inputImgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        preparePlotlyContainer(container, renderContext);
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
            const inputCanvas = await loadImageToCanvas(inputImgUrl, renderContext.controller.signal);

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

            await newPlotForRender(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
            setupZoomTracking(containerId);

        } else {
            // Original Plotly contour approach for non-transformed coordinates
            const azFlipped = flipVertical(azData);
            const transparentInputUrl = await makeBlackTransparent(inputImgUrl, 30, renderContext.controller.signal);

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

            await newPlotForRender(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
            setupZoomTracking(containerId);
        }
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Field Lines + Material Image render error:', error);
        showPlotMessage(container, `<p style="padding:20px; color:red;">Error: ${error.message}</p>`, renderContext);
    }
}

/**
 * Render field lines (Az contours) overlaid on edge-detected boundary image
 * Uses Sobel edge detection on input image for clearer boundary visualization
 */
async function renderAzEdge(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Load Az data with caching
        const azData = await loadFieldData(
            'Az', step, resultPath, renderContext.controller.signal
        );

        // Get input image URL
        const inputImgUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        const coordSys = AppState.analysisConditions?.coordinate_system || 'cartesian';

        // Apply Sobel edge detection to input image
        const edgeCanvas = await applySobelEdgeDetection(inputImgUrl, renderContext.controller.signal);
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
            await newPlotForRender(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
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
            await newPlotForRender(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
            setupZoomTracking(containerId);
        }
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Field Lines + Edge render error:', error);
        showPlotMessage(container, `<p style="padding:20px; color:red;">Error: ${error.message}</p>`, renderContext);
    }
}

async function renderMaterialImage(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const analysisConditions = AppState.analysisConditions;
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Get step input image (from InputImage folder)
        const imgUrl = await loadImageDataUrl(
            `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`,
            renderContext.controller.signal
        );

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        // Load image to get dimensions
        const img = await loadImage(imgUrl, renderContext.controller.signal);

        const rows = img.height;
        const cols = img.width;

        // Generate physical coordinates if available
        let xTitle, yTitle, xMin, xMax, yMin, yMax;
        if (analysisConditions) {
            const coordSys = analysisConditions.coordinate_system || 'cartesian';
            if (coordSys === 'polar') {
                const theta_start = analysisConditions.theta_start || 0;
                const dr = analysisConditions.dr || 0.001;
                const dtheta = analysisConditions.dtheta || 0.001;
                xTitle = 'r - r_start [mm]';
                yTitle = 'θ [rad]';
                xMin = 0;
                xMax = (cols - 1) * dr * 1000;
                yMin = theta_start;
                yMax = theta_start + (rows - 1) * dtheta;
            } else {
                const dx = analysisConditions.dx || 0.001;
                const dy = analysisConditions.dy || 0.001;
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

        await newPlotForRender(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
        setupZoomTracking(containerId);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Material image load error:', error);
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">Error loading material image</div>', renderContext);
    }
}

async function renderStepInputImage(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const analysisConditions = AppState.analysisConditions;
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Get step input image
        const imgUrl = await loadImageDataUrl(
            `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`,
            renderContext.controller.signal
        );

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        // Load image to get dimensions
        const img = await loadImage(imgUrl, renderContext.controller.signal);

        const rows = img.height;
        const cols = img.width;

        // Generate physical coordinates if available
        let xTitle, yTitle, xMin, xMax, yMin, yMax;
        if (analysisConditions) {
            const coordSys = analysisConditions.coordinate_system || 'cartesian';
            if (coordSys === 'polar') {
                const theta_start = analysisConditions.theta_start || 0;
                const dr = analysisConditions.dr || 0.001;
                const dtheta = analysisConditions.dtheta || 0.001;
                xTitle = 'r - r_start [mm]';
                yTitle = 'θ [rad]';
                xMin = 0;
                xMax = (cols - 1) * dr * 1000;
                yMin = theta_start;
                yMax = theta_start + (rows - 1) * dtheta;
            } else {
                const dx = analysisConditions.dx || 0.001;
                const dy = analysisConditions.dy || 0.001;
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

        await newPlotForRender(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
        setupZoomTracking(containerId);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Step input image load error:', error);
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">Error loading image</div>', renderContext);
    }
}

// Shared utility: fetch a coarsening mask PNG and retain compact Uint8Array
// rows in analysis coordinates plus one compact image-coordinate overlay mask.
// Cached per resultPath:step — same mask is reused across Az/B/H renders in a single step
const COARSENING_MASK_CACHE_LIMIT = 4;
const COARSENING_MASK_CACHE_BUDGET_BYTES = 32 * 1024 * 1024;
const COARSENING_MASK_WORKING_BUDGET_BYTES = 128 * 1024 * 1024;
const _maskCache = new Map();
let _maskCacheBytes = 0;

function estimateCoarseningMaskBytes(result) {
    if (!result) return FIELD_CACHE_ARRAY_OVERHEAD_BYTES;
    let bytes = FIELD_CACHE_ARRAY_OVERHEAD_BYTES;
    if (result.imageMask instanceof Uint8Array) {
        bytes += FIELD_CACHE_ARRAY_OVERHEAD_BYTES + result.imageMask.byteLength;
    }
    if (Array.isArray(result.mask)) {
        bytes += FIELD_CACHE_ARRAY_OVERHEAD_BYTES +
            result.mask.length * FIELD_CACHE_ARRAY_OVERHEAD_BYTES;
        for (const row of result.mask) {
            bytes += row instanceof Uint8Array
                ? row.byteLength
                : FIELD_CACHE_ARRAY_OVERHEAD_BYTES;
        }
    }
    return bytes;
}

function removeCoarseningMaskEntry(cacheKey, entry) {
    if (_maskCache.get(cacheKey) !== entry) return false;
    _maskCache.delete(cacheKey);
    _maskCacheBytes = Math.max(0, _maskCacheBytes - (entry.sizeBytes || 0));
    entry.sizeBytes = 0;
    return true;
}

function touchCoarseningMaskEntry(cacheKey, entry) {
    if (_maskCache.get(cacheKey) !== entry) return;
    _maskCache.delete(cacheKey);
    _maskCache.set(cacheKey, entry);
}

function trimCoarseningMaskCache() {
    let settledCount = Array.from(_maskCache.values())
        .filter(entry => entry.settled).length;
    if (settledCount <= COARSENING_MASK_CACHE_LIMIT &&
        _maskCacheBytes <= COARSENING_MASK_CACHE_BUDGET_BYTES) return;

    for (const [cacheKey, entry] of _maskCache) {
        if (!entry.settled || entry.consumers > 0) continue;
        removeCoarseningMaskEntry(cacheKey, entry);
        settledCount--;
        if (settledCount <= COARSENING_MASK_CACHE_LIMIT &&
            _maskCacheBytes <= COARSENING_MASK_CACHE_BUDGET_BYTES) break;
    }
}

function clearCoarseningMaskCache() {
    for (const [cacheKey, entry] of _maskCache) {
        entry.retain = false;
        if (entry.settled || entry.consumers === 0) {
            removeCoarseningMaskEntry(cacheKey, entry);
            if (!entry.settled) entry.controller.abort();
        }
    }
}

async function getCoarseningMaskArray(resultPath, step, signal = null) {
    if (signal?.aborted) throw createAbortError('Coarsening mask load aborted');
    const cacheKey = `${resultPath}:${step}`;
    let entry = _maskCache.get(cacheKey);

    // Return cached result (including null = no coarsening).
    if (entry?.settled) {
        touchCoarseningMaskEntry(cacheKey, entry);
        return entry.result;
    }

    if (!entry) {
        const controller = new AbortController();
        entry = {
            cacheKey,
            controller,
            consumers: 0,
            settled: false,
            retain: true,
            result: null,
            sizeBytes: 0,
            promise: null,
        };
        entry.promise = (async () => {
            try {
                const result = await _fetchCoarseningMask(
                    resultPath,
                    step,
                    controller.signal
                );
                entry.result = result;
                entry.settled = true;
                entry.promise = null;
                entry.sizeBytes = estimateCoarseningMaskBytes(result);
                _maskCacheBytes += entry.sizeBytes;
                touchCoarseningMaskEntry(cacheKey, entry);
                trimCoarseningMaskCache();
                return result;
            } catch (error) {
                removeCoarseningMaskEntry(cacheKey, entry);
                throw error;
            }
        })();
        _maskCache.set(cacheKey, entry);
        entry.promise.catch(() => {});
    } else {
        // A new caller arriving after a cache clear makes this active entry
        // useful again without restarting the same physical image request.
        entry.retain = true;
    }

    entry.consumers++;
    const requestPromise = entry.promise;
    try {
        return await awaitWithAbortSignal(
            requestPromise,
            signal,
            'Coarsening mask consumer superseded'
        );
    } finally {
        entry.consumers = Math.max(0, entry.consumers - 1);
        if (entry.consumers === 0 && !entry.settled) {
            removeCoarseningMaskEntry(cacheKey, entry);
            entry.controller.abort();
        } else if (entry.consumers === 0 && entry.settled && !entry.retain) {
            removeCoarseningMaskEntry(cacheKey, entry);
        } else if (entry.consumers === 0 && entry.settled) {
            trimCoarseningMaskCache();
        }
    }
}

async function _fetchCoarseningMask(resultPath, step, signal = null) {
    const maskUrl = `/api/get-coarsening-mask?result=${encodeURIComponent(resultPath)}&step=${step}`;
    const img = await loadImage(maskUrl, signal);
    if (signal?.aborted) throw createAbortError('Coarsening mask load aborted');

    const rows = img.height;
    const cols = img.width;
    const cellCount = rows * cols;
    // Peak includes the canvas RGBA backing store, ImageData, compact
    // image-coordinate mask and compact analysis-coordinate mask.
    const estimatedWorkingBytes = cellCount * 14 +
        rows * FIELD_CACHE_ARRAY_OVERHEAD_BYTES;
    if (estimatedWorkingBytes > COARSENING_MASK_WORKING_BUDGET_BYTES) {
        const error = new Error(
            `Coarsening mask ${cols}x${rows} requires about ` +
            `${formatFieldCacheBytes(estimatedWorkingBytes)}, exceeding the ` +
            `${formatFieldCacheBytes(COARSENING_MASK_WORKING_BUDGET_BYTES)} ` +
            'mask working-memory budget'
        );
        error.name = 'FieldMemoryBudgetError';
        error.code = 'FIELD_MEMORY_BUDGET';
        throw error;
    }

    const canvas = document.createElement('canvas');
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const pixelData = ctx.getImageData(0, 0, cols, rows);
    const imageMask = new Uint8Array(cellCount);
    const mask = Array.from({ length: rows }, () => new Uint8Array(cols));
    let activeCount = 0;

    for (let j = 0; j < rows; j++) {
        if (j > 0 && j % 64 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        if (signal?.aborted) throw createAbortError('Coarsening mask load aborted');
        for (let i = 0; i < cols; i++) {
            // Active=255, inactive=max 127 (255/skip). Threshold at 200 to avoid
            // grayscale PNG color-space conversion artifacts near 128.
            const cellIndex = j * cols + i;
            const isActive = pixelData.data[cellIndex * 4] > 200;
            imageMask[cellIndex] = isActive ? 255 : 0;
            // Store analysis coordinates directly, avoiding a second reverse
            // pass and the high overhead of JavaScript boolean arrays.
            mask[rows - 1 - j][i] = isActive ? 1 : 0;
            if (isActive) activeCount++;
        }
    }

    // Release the large RGBA canvas backing store before retaining the compact
    // masks in the LRU.
    canvas.width = 0;
    canvas.height = 0;
    img.removeAttribute?.('src');

    // All cells active means no coarsening
    if (activeCount === cellCount) return null;

    console.log(`Coarsening mask: ${activeCount}/${cellCount} active cells`);
    return { mask, imageMask, width: cols, height: rows };
}

async function renderCoarseningMask(containerId, step, renderContext = null) {
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const analysisConditions = AppState.analysisConditions;
    if (!resultPath) throw new Error('No result selected');

    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        // Load coarsening mask via shared utility
        const maskResult = await getCoarseningMaskArray(
            resultPath,
            step,
            renderContext.controller.signal
        ).catch(error => {
            if ((error && error.name === 'AbortError') ||
                error?.code === 'FIELD_MEMORY_BUDGET') throw error;
            return null;
        });
        if (!maskResult) throw new Error('Coarsening mask not available');

        const rows = maskResult.height;
        const cols = maskResult.width;

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        // Load input image for overlay
        const inputUrl = `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;
        const inputImg = await loadImage(inputUrl, renderContext.controller.signal).catch(error => {
            if (error && error.name === 'AbortError') throw error;
            return null;
        });

        // Compute product of input image and mask
        let resultImgUrl;
        if (inputImg && inputImg.width === cols && inputImg.height === rows) {
            const canvas = document.createElement('canvas');
            canvas.width = cols;
            canvas.height = rows;
            const ctx = canvas.getContext('2d');

            ctx.drawImage(inputImg, 0, 0);
            const inputData = ctx.getImageData(0, 0, cols, rows);

            const outputData = ctx.createImageData(cols, rows);
            const maskPixels = maskResult.imageMask; // image coords (not flipped)
            for (let pi = 0; pi < inputData.data.length; pi += 4) {
                if (maskPixels[pi / 4] > 200) {
                    outputData.data[pi] = inputData.data[pi];         // R
                    outputData.data[pi + 1] = inputData.data[pi + 1]; // G
                    outputData.data[pi + 2] = inputData.data[pi + 2]; // B
                    outputData.data[pi + 3] = 255;                     // A
                } else {
                    outputData.data[pi] = 0;
                    outputData.data[pi + 1] = 0;
                    outputData.data[pi + 2] = 0;
                    outputData.data[pi + 3] = 255;
                }
            }
            ctx.putImageData(outputData, 0, 0);
            resultImgUrl = canvas.toDataURL('image/png');
        } else {
            const maskUrl = `/api/get-coarsening-mask?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`;
            resultImgUrl = await loadImageDataUrl(maskUrl, renderContext.controller.signal);
        }

        // Generate physical coordinates if available
        let xTitle, yTitle, xMin, xMax, yMin, yMax;
        if (analysisConditions) {
            const coordSys = analysisConditions.coordinate_system || 'cartesian';
            if (coordSys === 'polar') {
                const theta_start = analysisConditions.theta_start || 0;
                const dr = analysisConditions.dr || 0.001;
                const dtheta = analysisConditions.dtheta || 0.001;
                xTitle = 'r - r_start [mm]';
                yTitle = 'θ [rad]';
                xMin = 0;
                xMax = (cols - 1) * dr * 1000;
                yMin = theta_start;
                yMax = theta_start + (rows - 1) * dtheta;
            } else {
                const dx = analysisConditions.dx || 0.001;
                const dy = analysisConditions.dy || 0.001;
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

        await newPlotForRender(container, [], layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
        setupZoomTracking(containerId);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Coarsening mask load error:', error);
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #666;">Coarsening mask not available<br><small>(Adaptive mesh may not be enabled for this result)</small></div>', renderContext);
    }
}

// Helper function to load an image and return a promise
async function loadImage(url, signal = null) {
    if (signal?.aborted) throw createAbortError('Image load aborted');
    const response = await fetch(url, signal ? { signal } : undefined);
    if (!response.ok) throw new Error(`Failed to load image (HTTP ${response.status}): ${url}`);
    const blob = await response.blob();
    if (signal?.aborted) throw createAbortError('Image load aborted');

    const objectUrl = URL.createObjectURL(blob);
    try {
        return await new Promise((resolve, reject) => {
            const img = new Image();
            let settled = false;
            const cleanup = () => {
                img.onload = null;
                img.onerror = null;
                signal?.removeEventListener('abort', onAbort);
            };
            const onAbort = () => {
                if (settled) return;
                settled = true;
                img.src = '';
                cleanup();
                reject(createAbortError('Image decode aborted'));
            };
            img.onload = () => {
                if (settled) return;
                settled = true;
                cleanup();
                resolve(img);
            };
            img.onerror = () => {
                if (settled) return;
                settled = true;
                cleanup();
                reject(new Error('Failed to decode image: ' + url));
            };
            signal?.addEventListener('abort', onAbort, { once: true });
            img.src = objectUrl;
        });
    } finally {
        URL.revokeObjectURL(objectUrl);
    }
}

async function loadImageDataUrl(url, signal = null) {
    if (signal?.aborted) throw createAbortError('Image load aborted');
    const response = await fetch(url, signal ? { signal } : undefined);
    if (!response.ok) throw new Error(`Failed to load image (HTTP ${response.status}): ${url}`);
    const blob = await response.blob();
    if (signal?.aborted) throw createAbortError('Image load aborted');
    return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        const onAbort = () => {
            reader.abort();
            reject(createAbortError('Image conversion aborted'));
        };
        reader.onload = () => {
            signal?.removeEventListener('abort', onAbort);
            resolve(reader.result);
        };
        reader.onerror = () => {
            signal?.removeEventListener('abort', onAbort);
            reject(reader.error || new Error('Failed to convert image'));
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        reader.readAsDataURL(blob);
    });
}

// ===== Interactive Plot Functions =====

// State for interactive line profile
const lineProfileState = {};

async function renderLineProfile(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);

    const resultPath = renderContext.resultPath;
    if (!resultPath) {
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">No result selected</div>', renderContext);
        return;
    }
    const analysisConditions = AppState.analysisConditions;

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
        const azData = await loadFieldData(
            'Az', step, resultPath, renderContext.controller.signal
        );
        const muData = await loadFieldData(
            'Mu', step, resultPath, renderContext.controller.signal
        );

        if (!azData || azData.length === 0) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">No data available</div>', renderContext);
            return;
        }

        const dx = analysisConditions?.dx || 0.001;
        const dy = analysisConditions?.dy || 0.001;

        // Flip data for display (analysis y-up to image y-down)
        const azFlipped = flipVertical(azData);
        const muFlipped = flipVertical(muData);

        // Load coarsening mask for coarsening-aware B/H computation
        const maskResult = await getCoarseningMaskArray(
            resultPath,
            step,
            renderContext.controller.signal
        ).catch(error => {
            if ((error && error.name === 'AbortError') ||
                error?.code === 'FIELD_MEMORY_BUDGET') throw error;
            return null;
        });
        const activeMask = maskResult ? maskResult.mask : null;

        const { Bx, By, B, Hx, Hy, H } = calculateMagneticField(
            azFlipped, muFlipped, dx, dy, activeMask, renderContext.analysisConditions
        );

        const rows = azFlipped.length;
        const cols = azFlipped[0].length;

        preparePlotlyContainer(container, renderContext);
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
        const imgUrl = await loadImageDataUrl(
            `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`,
            renderContext.controller.signal
        );

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

        await newPlotForRender(imageDiv, traces, imageLayout, { responsive: true, displayModeBar: false }, renderContext);

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

            void renderLineProfile(containerId, AppState.currentStep).catch(error => {
                if (!error || error.name !== 'AbortError') console.error('Line profile re-render error:', error);
            });
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

            await newPlotForRender(profileDiv, profileTraces, profileLayout, { responsive: true, displayModeBar: false }, renderContext);
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
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Line profile error:', error);
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">Error loading data</div>', renderContext);
    }
}

// Line profile control functions
function setLineProfileField(containerId, field) {
    if (lineProfileState[containerId]) {
        lineProfileState[containerId].displayField = field;
        void renderLineProfile(containerId, AppState.currentStep).catch(error => {
            if (!error || error.name !== 'AbortError') console.error('Line profile field update error:', error);
        });
    }
}

function resetLineProfilePoints(containerId) {
    if (lineProfileState[containerId]) {
        lineProfileState[containerId].startPoint = null;
        lineProfileState[containerId].endPoint = null;
        lineProfileState[containerId].selectingStart = true;
        void renderLineProfile(containerId, AppState.currentStep).catch(error => {
            if (!error || error.name !== 'AbortError') console.error('Line profile reset error:', error);
        });
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

async function renderFluxLinkageInteractive(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);

    const resultPath = renderContext.resultPath;
    if (!resultPath) {
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">No result selected</div>', renderContext);
        return;
    }
    const analysisConditions = AppState.analysisConditions;

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
        const azData = await loadFieldData(
            'Az', step, resultPath, renderContext.controller.signal
        );

        if (!azData || azData.length === 0) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">No data available</div>', renderContext);
            return;
        }

        const dx = analysisConditions?.dx || 0.001;
        const dy = analysisConditions?.dy || 0.001;

        // Flip data for display
        const azFlipped = flipVertical(azData);
        const rows = azFlipped.length;
        const cols = azFlipped[0].length;

        preparePlotlyContainer(container, renderContext);
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
        const imgUrl = await loadImageDataUrl(
            `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`,
            renderContext.controller.signal
        );

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

        await newPlotForRender(plotDiv, traces, layout, { responsive: true, displayModeBar: false }, renderContext);

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
            void renderFluxLinkageInteractive(containerId, AppState.currentStep).catch(error => {
                if (!error || error.name !== 'AbortError') console.error('Flux linkage re-render error:', error);
            });
        };
        plotDiv.addEventListener('click', plotDiv._clickHandler);

    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Flux linkage interactive error:', error);
        showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: red;">Error loading data</div>', renderContext);
    }
}

// Flux linkage control function
function resetFluxLinkagePoints(containerId) {
    if (fluxLinkageState[containerId]) {
        fluxLinkageState[containerId].startPoint = null;
        fluxLinkageState[containerId].endPoint = null;
        fluxLinkageState[containerId].selectingStart = true;
        fluxLinkageState[containerId].fluxValue = null;
        void renderFluxLinkageInteractive(containerId, AppState.currentStep).catch(error => {
            if (!error || error.name !== 'AbortError') console.error('Flux linkage reset error:', error);
        });
    }
}

// Helper: Load force data for a specific step
async function loadForceData(step, providedResultPath = null, signal = null) {
    const resultPath = providedResultPath || getCurrentResultPath();
    if (!resultPath) return null;

    try {
        const response = await fetch(
            `/api/load-csv-raw?result=${encodeURIComponent(resultPath)}&file=Forces/step_${String(step).padStart(4, '0')}.csv`,
            signal ? { signal } : undefined
        );

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
        if (error && error.name === 'AbortError') throw error;
        console.error(`Force data load error for step ${step}:`, error);
        return null;
    }
}

async function renderForceXTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const totalSteps = AppState.totalSteps;
    try {
        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i + 1, resultPath, renderContext.controller.signal);
            assertRenderContextCurrent(renderContext, container);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">No Forces data available</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        // x-axis values: 1..totalSteps array (1-based)
        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

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
            return Array.from({ length: totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === renderContext.step)) ? highlightSize : baseSize;
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

            for (let i = 0; i < totalSteps; i++) {
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

        // Get plot configuration if available
        const plotId = containerId.replace('container-', '');
        const plotConfig = AppState.plotConfigs[plotId] || {};

        console.log(`renderForceXTime: containerId=${containerId}, plotId=${plotId}, config=`, plotConfig);

        const layout = {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, totalSteps] },
            yaxis: { title: yaxisTitle, range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        };

        // Apply plot configuration ranges if specified
        if (plotConfig.xRange && plotConfig.xRange !== 'auto') {
            layout.xaxis.range = plotConfig.xRange;
        }
        if (plotConfig.yRange && plotConfig.yRange !== 'auto') {
            layout.yaxis.range = plotConfig.yRange;
        }

        await newPlotForRender(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Force X time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

async function renderForceYTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const totalSteps = AppState.totalSteps;
    try {
        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i + 1, resultPath, renderContext.controller.signal);
            assertRenderContextCurrent(renderContext, container);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">No Forces data available</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        const traces = [];

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === renderContext.step)) ? highlightSize : baseSize;
            });
        };

        // Full model multiplier for polar coordinates
        const forceMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        materialNames.forEach(matName => {
            const forceData = [];
            let matColor = null;

            for (let i = 0; i < totalSteps; i++) {
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

        // Get plot configuration if available
        const plotId = containerId.replace('container-', '');
        const plotConfig = AppState.plotConfigs[plotId] || {};

        const layout = {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, totalSteps] },
            yaxis: { title: yaxisTitle, range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        };

        // Apply plot configuration ranges if specified
        if (plotConfig.xRange && plotConfig.xRange !== 'auto') {
            layout.xaxis.range = plotConfig.xRange;
        }
        if (plotConfig.yRange && plotConfig.yRange !== 'auto') {
            layout.yaxis.range = plotConfig.yRange;
        }

        await newPlotForRender(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Force Y time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

async function renderTorqueTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const totalSteps = AppState.totalSteps;
    try {
        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i + 1, resultPath, renderContext.controller.signal);
            assertRenderContextCurrent(renderContext, container);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">No Forces data available</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        const traces = [];

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === renderContext.step)) ? highlightSize : baseSize;
            });
        };

        // Full model multiplier for polar coordinates
        const torqueMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        materialNames.forEach(matName => {
            const torqueData = [];
            let matColor = null;

            for (let i = 0; i < totalSteps; i++) {
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

        // Get plot configuration if available
        const plotId = containerId.replace('container-', '');
        const plotConfig = AppState.plotConfigs[plotId] || {};

        const layout = {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, totalSteps] },
            yaxis: { title: yaxisTitle, range: yrange },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        };

        // Apply plot configuration ranges if specified
        if (plotConfig.xRange && plotConfig.xRange !== 'auto') {
            layout.xaxis.range = plotConfig.xRange;
        }
        if (plotConfig.yRange && plotConfig.yRange !== 'auto') {
            layout.yaxis.range = plotConfig.yRange;
        }

        await newPlotForRender(container, traces, layout, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Torque time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

// =====================================================================
// Phase Z: Flux Linkage / Back-EMF timeline (CSV-driven)
// =====================================================================
//
// Source: <resultPath>/FluxLinkage/flux_linkage.csv written by the solver's
// exportFluxLinkageCSV() (MagneticFieldAnalyzer.cpp:2347+). Header row is
//   step,<Phi_name_1>,<Phi_name_2>,...
// Each subsequent row is a step index and one Φ value per defined path.
//
// The solver rewrites this CSV after every completed step. Never retain it
// across dashboard reloads: a result path can keep growing during analysis.
// Two palette items consume this:
//   - flux_linkage_time  → Φ(step)
//   - back_emf_time      → -dΦ/dstep (back-EMF convention, proportional to
//                          the per-phase induced voltage when the rotor
//                          advances uniformly per step)
//
// Plotly's built-in legend click hides/shows traces, so the user can
// inspect one phase at a time without us writing a custom legend.
async function loadFluxLinkageData(providedResultPath = null, signal = null) {
    const resultPath = providedResultPath || getCurrentResultPath();
    if (!resultPath) return null;

    try {
        const response = await fetch(
            `/api/load-csv-raw?result=${encodeURIComponent(resultPath)}`
            + `&file=FluxLinkage/flux_linkage.csv`,
            signal ? { cache: 'no-store', signal } : { cache: 'no-store' });
        if (!response.ok) return null;
        const text = await response.text();
        if (!text || !text.trim()) return null;

        const lines = text.trim().split(/\r?\n/);
        if (lines.length < 2) return null;
        const headers = lines[0].split(',').map(s => s.trim());
        if (headers[0].toLowerCase() !== 'step' || headers.length < 2) return null;
        const phiNames = headers.slice(1);
        const steps = [];
        const phiSeries = {};
        phiNames.forEach(n => { phiSeries[n] = []; });
        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split(',');
            if (parts.length < 1) continue;
            const step = Number(parts[0]);
            if (!Number.isFinite(step)) continue;
            steps.push(step);
            for (let k = 0; k < phiNames.length; k++) {
                const v = Number(parts[k + 1]);
                phiSeries[phiNames[k]].push(Number.isFinite(v) ? v : null);
            }
        }
        if (steps.length === 0) return null;

        return { headers, phiNames, steps, phiSeries };
    } catch (e) {
        if (e && e.name === 'AbortError') throw e;
        console.error('loadFluxLinkageData failed:', e);
        return null;
    }
}

// Map "Phi_Coil_A" / "Phi_A" / "PhiA" → red, B → green, C → blue.
// IPMSM / electrical-engineering convention. Falls back to Plotly auto
// for anything that doesn't pattern-match.
function fluxPhaseColor(name) {
    const u = name.toUpperCase();
    if (/(^|[_-])A$|COIL[_-]?A|PHI[_-]?A|_PHASEA/.test(u)) return '#d62728';
    if (/(^|[_-])B$|COIL[_-]?B|PHI[_-]?B|_PHASEB/.test(u)) return '#2ca02c';
    if (/(^|[_-])C$|COIL[_-]?C|PHI[_-]?C|_PHASEC/.test(u)) return '#1f77b4';
    return null;
}

async function renderFluxLinkageTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    try {
        const data = await loadFluxLinkageData(renderContext.resultPath, renderContext.controller.signal);
        assertRenderContextCurrent(renderContext, container);
        if (!data || data.steps.length === 0) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">'
                + 'No flux_linkage.csv in this result.<br>'
                + 'Define <code>flux_linkage:</code> in the YAML and rerun the transient analysis.'
                + '</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        // CSV step column is 0-based to match the solver's `step 0:` print;
        // display as 1-based so it lines up with "Step 1/N" elsewhere in the UI.
        const xSteps = data.steps.map(s => s + 1);
        const traces = data.phiNames.map(name => {
            const color = fluxPhaseColor(name);
            const trace = {
                x: xSteps,
                y: data.phiSeries[name],
                type: 'scatter',
                mode: 'lines+markers',
                name,
                line: { width: 2 },
                marker: { size: 6 }
            };
            if (color) { trace.line.color = color; trace.marker.color = color; }
            return trace;
        });

        const legendConfig = traces.length <= 3
            ? { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
            : { x: 1.02, y: 1, xanchor: 'left' };

        await newPlotForRender(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 60, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', zeroline: false },
            yaxis: { title: 'Φ [Wb/m]', zeroline: true, tickformat: '.2e' },
            legend: legendConfig,
            showlegend: true,
            hovermode: 'closest'
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Flux linkage time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

async function renderBackEMFTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    try {
        const data = await loadFluxLinkageData(renderContext.resultPath, renderContext.controller.signal);
        assertRenderContextCurrent(renderContext, container);
        if (!data || data.steps.length < 2) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">'
                + 'Need at least 2 transient steps to compute dΦ/dstep.<br>'
                + 'Define <code>flux_linkage:</code> and run a transient analysis (total_steps ≥ 2).'
                + '</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        // Forward-difference EMF ∝ -ΔΦ/Δstep, plotted at the midpoint
        // between adjacent steps. We don't divide by physical Δt here —
        // the solver doesn't write it to the CSV — so this is "EMF in
        // Wb/m per step". For a uniformly-rotating rotor (Δstep ↔
        // constant electrical angle), the shape and amplitude of this
        // curve is exactly the back-EMF up to a known (RPM, pole-pair)
        // scale factor the user can apply downstream.
        const xCenters = [];
        const dPhi = {};
        data.phiNames.forEach(n => { dPhi[n] = []; });
        for (let i = 1; i < data.steps.length; i++) {
            // midpoint, in 1-based display coords
            xCenters.push(data.steps[i - 1] + 1.5);
            for (const name of data.phiNames) {
                const a = data.phiSeries[name][i - 1];
                const b = data.phiSeries[name][i];
                dPhi[name].push((a == null || b == null) ? null : -(b - a));
            }
        }

        const traces = data.phiNames.map(name => {
            const color = fluxPhaseColor(name);
            // Rename Φ_* → EMF_* in the legend so the user can tell at a
            // glance which palette item produced this trace.
            const legendName = name.replace(/^Phi/i, 'EMF').replace(/^Φ/i, 'EMF');
            const trace = {
                x: xCenters,
                y: dPhi[name],
                type: 'scatter',
                mode: 'lines+markers',
                name: legendName,
                line: { width: 2 },
                marker: { size: 6 }
            };
            if (color) { trace.line.color = color; trace.marker.color = color; }
            return trace;
        });

        const legendConfig = traces.length <= 3
            ? { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
            : { x: 1.02, y: 1, xanchor: 'left' };

        await newPlotForRender(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 60, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', zeroline: false },
            yaxis: { title: '-dΦ/dstep [Wb/m per step]', zeroline: true, tickformat: '.2e' },
            legend: legendConfig,
            showlegend: true,
            hovermode: 'closest'
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Back-EMF time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

async function renderEnergyTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const totalSteps = AppState.totalSteps;
    try {
        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i + 1, resultPath, renderContext.controller.signal);
            assertRenderContextCurrent(renderContext, container);
            allStepsData.push(data || null);
            if (data) hasData = true;
        }

        if (!hasData) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">No Energy data available</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

        const materialNames = new Set();
        allStepsData.forEach(data => {
            if (data && data.materials) {
                data.materials.forEach(mat => materialNames.add(mat.name));
            }
        });

        const traces = [];

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === renderContext.step)) ? highlightSize : baseSize;
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

            for (let i = 0; i < totalSteps; i++) {
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
        for (let i = 0; i < totalSteps; i++) {
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

        await newPlotForRender(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 45, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, totalSteps] },
            yaxis: { title: 'Energy [J/m]' },
            showlegend: true,
            legend: legendConfig,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Energy time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

async function renderSystemEnergyTime(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const totalSteps = AppState.totalSteps;
    try {
        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i + 1, resultPath, renderContext.controller.signal);
            assertRenderContextCurrent(renderContext, container);
            allStepsData.push(data || null);
            if (data && data.system_total_energy !== undefined) hasData = true;
        }

        if (!hasData) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">No System Energy data available</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

        // Full model multiplier for polar coordinates (energy is also multiplied for full model)
        const energyMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        // Extract system total energy for each step
        const energyData = [];
        for (let i = 0; i < totalSteps; i++) {
            const stepData = allStepsData[i];
            if (stepData && stepData.system_total_energy !== undefined) {
                energyData.push(stepData.system_total_energy * energyMultiplier);
            } else {
                energyData.push(0);
            }
        }

        const getMarkerSizes = (baseSize, highlightSize) => {
            return Array.from({ length: totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === renderContext.step)) ? highlightSize : baseSize;
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

        await newPlotForRender(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 55, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, totalSteps] },
            yaxis: { title: 'System Energy [J/m]' },
            showlegend: false,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('System energy time plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
    }
}

async function renderVirtualWork(containerId, step, renderContext = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    renderContext = ensureDashboardRenderContext(containerId, step, renderContext);
    const resultPath = renderContext.resultPath;
    const totalSteps = AppState.totalSteps;
    const analysisConditions = AppState.analysisConditions;
    try {
        // Load all steps data
        const allStepsData = [];
        let hasData = false;

        for (let i = 0; i < totalSteps; i++) {
            const data = await loadForceData(i + 1, resultPath, renderContext.controller.signal);
            assertRenderContextCurrent(renderContext, container);
            allStepsData.push(data || null);
            if (data && data.system_total_energy !== undefined) hasData = true;
        }

        if (!hasData) {
            showPlotMessage(container, '<div style="padding: 20px; text-align: center; color: #999;">No System Energy data available</div>', renderContext);
            return;
        }

        preparePlotlyContainer(container, renderContext);
        const size = getContainerSize(container);

        const xSteps = Array.from({ length: totalSteps }, (_, k) => k + 1);

        // Full model multiplier for polar coordinates
        const energyMultiplier = (AppState.isPolarCoordinates && AppState.polarFullModel && AppState.polarFullModelMultiplier > 1)
            ? AppState.polarFullModelMultiplier
            : 1;

        // Extract system total energy for each step
        const energyData = [];
        for (let i = 0; i < totalSteps; i++) {
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

        if (analysisConditions) {
            const transient = analysisConditions.transient;
            const slidePixelsPerStep = transient?.slide_pixels_per_step || 1;
            const slideDirection = transient?.slide_direction || 'horizontal';
            const coordSystem = analysisConditions.coordinate_system || 'cartesian';

            if (coordSystem === 'polar') {
                const polar = analysisConditions.polar;
                const rOrientation = polar?.r_orientation || 'horizontal';
                const dr = analysisConditions.dr || 0.001;
                const dtheta = analysisConditions.dtheta || 0.01;

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
                const dx = analysisConditions.dx || 0.001;
                const dy = analysisConditions.dy || 0.001;

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
        for (let i = 0; i < totalSteps; i++) {
            if (i === 0) {
                // Forward difference for first point
                if (totalSteps > 1) {
                    const dW = energyData[1] - energyData[0];
                    virtualWorkData.push(dW / displacementPerStep);
                } else {
                    virtualWorkData.push(0);
                }
            } else if (i === totalSteps - 1) {
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
            return Array.from({ length: totalSteps }, (_, i) => {
                return (AppState.isAnimating && (i + 1 === renderContext.step)) ? highlightSize : baseSize;
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

        await newPlotForRender(container, traces, {
            width: size.width,
            height: size.height,
            margin: { l: 55, r: 10, t: 10, b: 35 },
            xaxis: { title: 'Step', range: [1, totalSteps] },
            yaxis: { title: yAxisTitle },
            showlegend: false,
            dragmode: false
        }, { responsive: true, displayModeBar: AppState.showPlotlyModeBar }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') || !isRenderContextCurrent(renderContext, container)) {
            discardStaleRender(container, renderContext);
            return;
        }
        console.error('Virtual work plot error:', error);
        showPlotMessage(container, `<div style="padding: 20px; text-align: center; color: red;">Error: ${error.message}</div>`, renderContext);
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
function resolvePlotContainer(elementOrId) {
    return typeof elementOrId === 'string'
        ? document.getElementById(elementOrId)
        : elementOrId;
}

async function plotContour(elementOrId, data, title, usePhysicalAxes = false, renderContext = null) {
    const container = resolvePlotContainer(elementOrId);
    const elementId = container?.id || String(elementOrId);
    const analysisConditions = renderContext?.analysisConditions || AppState.analysisConditions;
    if (!container) {
        console.error(`plotContour: Container not found: ${elementId}`);
        return;
    }

    if (!data || data.length === 0) {
        showPlotMessage(container, '<p>No data available</p>', renderContext);
        return;
    }

    // Clear container before plotting
    preparePlotlyContainer(container, renderContext);

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
    if (usePhysicalAxes && analysisConditions) {
        const rows = data.length;
        const cols = data[0]?.length || 0;
        const coordSys = analysisConditions.coordinate_system || 'cartesian';

        if (coordSys === 'polar') {
            const theta_start = analysisConditions.theta_start || 0;
            const dr = analysisConditions.dr || 0.001;
            const dtheta = analysisConditions.dtheta || 0.001;
            const r_orientation = analysisConditions.polar?.r_orientation || 'horizontal';

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
            const dx = analysisConditions.dx || 0.001;
            const dy = analysisConditions.dy || 0.001;

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

    return await newPlotForRender(
        container,
        [trace],
        layout,
        { responsive: true, displayModeBar: AppState.showPlotlyModeBar },
        renderContext
    );
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
function setupZoomTracking(containerOrId) {
    const container = resolvePlotContainer(containerOrId);
    if (!container) return;
    const containerId = container.id;

    removeZoomTracking(container);
    const handler = (eventData) => {
        // Save zoom state when user zooms/pans
        if (eventData['xaxis.range[0]'] !== undefined || eventData['xaxis.range'] !== undefined) {
            const layout = container.layout;
            saveZoomState(containerId, layout);
        }
    };
    container._zoomTrackingHandler = handler;
    container.on('plotly_relayout', handler);
}

// ===== Plotting Functions =====
async function plotHeatmap(elementOrId, data, title, usePhysicalAxes = false, useHarmonicMean = false, renderContext = null) {
    const container = resolvePlotContainer(elementOrId);
    const elementId = container?.id || String(elementOrId);
    const analysisConditions = renderContext?.analysisConditions || AppState.analysisConditions;
    const polarView = renderContext?.polarView || snapshotPolarView(analysisConditions);
    if (!container) {
        console.error(`plotHeatmap: Container not found: ${elementId}`);
        return;
    }

    if (!data || data.length === 0) {
        showPlotMessage(container, '<p>No data available</p>', renderContext);
        return;
    }

    // Clear container before plotting
    preparePlotlyContainer(container, renderContext);

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

    // Get plot configuration if available
    // elementId might be "container-plot-1", but plotConfigs uses "plot-1"
    const plotId = elementId.replace('container-', '');
    const plotConfig = AppState.plotConfigs[plotId] || {};

    console.log(`plotHeatmap: elementId=${elementId}, plotId=${plotId}, config=`, plotConfig);

    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: plotConfig.colorscale || 'Viridis',
        colorbar: {
            title: colorbarTitle,
            thickness: colorbarThickness,
            len: 1.0
        }
    };

    // Apply z-range (colorscale range) if configured
    if (plotConfig.zRange && plotConfig.zRange !== 'auto') {
        trace.zmin = plotConfig.zRange[0];
        trace.zmax = plotConfig.zRange[1];
    }

    // Generate physical axes if requested and conditions are available
    let xaxis, yaxis;
    if (usePhysicalAxes && analysisConditions) {
        const rows = data.length;
        const cols = data[0]?.length || 0;
        const coordSys = analysisConditions.coordinate_system || 'cartesian';

        if (coordSys === 'polar' && polarView.cartesianTransform) {
            // Apply polar to cartesian transformation
            const transformedData = transformPolarToCartesian(
                data,
                analysisConditions,
                polarView.fullModel,
                useHarmonicMean,
                polarView.fullModelMultiplier
            );

            trace.x = transformedData.x;
            trace.y = transformedData.y;
            trace.z = transformedData.z;

            const r_o = analysisConditions.polar?.r_end || analysisConditions.r_o || 1;
            xaxis = {
                title: 'X [mm]',
                range: [-r_o * 1000, r_o * 1000],
                ...(polarView.fullModel && { scaleanchor: 'y', scaleratio: 1 })
            };
            yaxis = {
                title: 'Y [mm]',
                range: [-r_o * 1000, r_o * 1000]
            };
        } else if (coordSys === 'polar') {
            // Original polar view (r vs theta)
            const theta_start = analysisConditions.theta_start || 0;
            const dr = analysisConditions.dr || 0.001;
            const dtheta = analysisConditions.dtheta || 0.001;
            const r_orientation = analysisConditions.polar?.r_orientation || 'horizontal';

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
            const dx = analysisConditions.dx || 0.001;
            const dy = analysisConditions.dy || 0.001;

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
    if (polarView.isPolar && polarView.fullModel && polarView.fullModelMultiplier > 1) {
        graphTitle += ` (Full Model ×${polarView.fullModelMultiplier})`;
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

    // Apply plot configuration ranges if specified
    if (plotConfig.xRange && plotConfig.xRange !== 'auto') {
        layout.xaxis.range = plotConfig.xRange;
    }
    if (plotConfig.yRange && plotConfig.yRange !== 'auto') {
        layout.yaxis.range = plotConfig.yRange;
    }

    // Restore saved zoom state if exists (only if no fixed range configured)
    if (!plotConfig.xRange || plotConfig.xRange === 'auto') {
        layout = restoreZoomState(elementId, layout);
    }

    const plot = await newPlotForRender(
        container,
        [trace],
        layout,
        { responsive: true, displayModeBar: AppState.showPlotlyModeBar },
        renderContext
    );
    setupZoomTracking(container);
    return plot;
}

async function plotForceGraph(elementOrId, data, title, renderContext = null) {
    const container = resolvePlotContainer(elementOrId);
    const elementId = container?.id || String(elementOrId);
    if (!container) {
        console.error(`plotForceGraph: Container not found: ${elementId}`);
        return;
    }

    // Parse force data from CSV
    if (!data || !Array.isArray(data) || data.length < 2) {
        showPlotMessage(container, '<p>No force data available</p>', renderContext);
        return;
    }

    preparePlotlyContainer(container, renderContext);

    // Get plot configuration if available
    const plotId = elementId.replace('container-', '');
    const plotConfig = AppState.plotConfigs[plotId] || {};

    console.log(`plotForceGraph: elementId=${elementId}, plotId=${plotId}, config=`, plotConfig);

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

    // Apply plot configuration ranges if specified
    if (plotConfig.xRange && plotConfig.xRange !== 'auto') {
        layout.xaxis.range = plotConfig.xRange;
    }
    if (plotConfig.yRange && plotConfig.yRange !== 'auto') {
        layout.yaxis.range = plotConfig.yRange;
    }

    // Restore saved zoom state if exists (only if no fixed range configured)
    if (!plotConfig.xRange || plotConfig.xRange === 'auto') {
        layout = restoreZoomState(elementId, layout);
    }

    const plot = await newPlotForRender(
        container,
        [trace],
        layout,
        { responsive: true, displayModeBar: AppState.showPlotlyModeBar },
        renderContext
    );
    setupZoomTracking(container);
    return plot;
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

    // Load every user-owned file, not only completed analysis outputs.
    refreshUserFiles();
}

function fileManagerEscape(value) {
    return String(value == null ? '' : value)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function fileManagerStatus(message, type = 'info') {
    const target = document.getElementById('fileManagerStatus');
    if (target) {
        target.className = `status-${type}`;
        target.textContent = message;
    } else {
        showStatus('solverStatus', message, type);
    }
}

const USER_FILE_CATEGORY_LABEL = {
    configs: 'YAML configs',
    images: 'Input images',
    libraries: 'Material libraries',
    results: 'Analysis results',
};
const USER_FILE_CATEGORY_ICON = {
    configs: '📝', images: '🖼️', libraries: '🧱', results: '📊',
};

function userFileKey(file) {
    return `${file.category}\u0000${file.name}`;
}

function visibleUserFiles() {
    const files = AppState.userFiles || [];
    const filter = document.getElementById('fileManagerFilter')?.value || 'all';
    return filter === 'all' ? files : files.filter(file => file.category === filter);
}

function updateUserFileSelectionControls() {
    const visible = visibleUserFiles();
    const selected = AppState.userFileSelection;
    const visibleSelected = visible.filter(file => selected.has(userFileKey(file))).length;
    const totalSelected = (AppState.userFiles || [])
        .filter(file => selected.has(userFileKey(file))).length;
    const selectAll = document.getElementById('selectAllUserFiles');
    if (selectAll) {
        selectAll.checked = visible.length > 0 && visibleSelected === visible.length;
        selectAll.indeterminate = visibleSelected > 0 && visibleSelected < visible.length;
        selectAll.disabled = visible.length === 0;
    }
    const deleteButton = document.getElementById('deleteSelectedUserFilesBtn');
    if (deleteButton) deleteButton.disabled = totalSelected === 0;
    const count = document.getElementById('userFileSelectedCount');
    if (count) count.textContent = `${totalSelected} selected`;
}

function userFileSelectionChanged(checkbox) {
    const file = fileManagerRow(checkbox);
    if (!file) return;
    const key = userFileKey(file);
    if (checkbox.checked) AppState.userFileSelection.add(key);
    else AppState.userFileSelection.delete(key);
    updateUserFileSelectionControls();
}

function toggleAllUserFiles(checked) {
    for (const file of visibleUserFiles()) {
        const key = userFileKey(file);
        if (checked) AppState.userFileSelection.add(key);
        else AppState.userFileSelection.delete(key);
    }
    document.querySelectorAll('.user-file-checkbox').forEach(checkbox => {
        checkbox.checked = checked;
    });
    updateUserFileSelectionControls();
}

function userFileHtml(file) {
    const category = fileManagerEscape(file.category);
    const name = fileManagerEscape(file.name);
    const date = file.modified ? new Date(file.modified).toLocaleString('ja-JP') : '';
    const size = fmtBytes(Number(file.size) || 0);
    const detail = file.category === 'results'
        ? `${file.steps || 0} step(s) · ${size} · ${date}`
        : `${size} · ${date}`;
    const openLabel = file.category === 'results' ? 'Preview' : 'Open';
    const download = file.category === 'results' ? ''
        : `<button class="btn-secondary btn-small" onclick="fileManagerDownload(this)">Download</button>`;
    const checked = AppState.userFileSelection.has(userFileKey(file)) ? ' checked' : '';
    return `<div class="user-file-item" data-category="${category}" data-name="${name}">
        <input type="checkbox" class="user-file-checkbox"${checked}
               aria-label="Select ${name}" onchange="userFileSelectionChanged(this)">
        <div class="user-file-icon" aria-hidden="true">${USER_FILE_CATEGORY_ICON[file.category] || '📄'}</div>
        <div class="user-file-info">
            <div class="user-file-name">${name}</div>
            <div class="user-file-meta">${USER_FILE_CATEGORY_LABEL[file.category] || file.category} · ${detail}</div>
        </div>
        <div class="user-file-actions">
            <button class="btn-secondary btn-small" onclick="fileManagerOpen(this)">${openLabel}</button>
            ${download}
            <button class="btn-delete btn-small" onclick="fileManagerDelete(this)">Delete</button>
        </div>
    </div>`;
}

function renderUserFiles() {
    const list = document.getElementById('userFilesList');
    if (!list) return;
    const visible = visibleUserFiles();
    if (visible.length === 0) {
        list.innerHTML = '<p style="color:#666;">No files in this category.</p>';
        updateUserFileSelectionControls();
        return;
    }
    const order = ['configs', 'images', 'libraries', 'results'];
    let html = '';
    for (const category of order) {
        const group = visible.filter(file => file.category === category);
        if (!group.length) continue;
        html += `<h4 class="file-category-heading">${USER_FILE_CATEGORY_LABEL[category]} (${group.length})</h4>`;
        html += group.map(userFileHtml).join('');
    }
    list.innerHTML = html;
    updateUserFileSelectionControls();
}

async function refreshUserFiles() {
    const list = document.getElementById('userFilesList');
    if (!list) return;
    list.innerHTML = '<p style="color:#666;">Loading…</p>';
    try {
        const response = await fetch(`/api/user-files?userId=${encodeURIComponent(AppState.userId)}`);
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.error || 'Failed to load files');
        AppState.userFiles = result.files || [];
        const availableKeys = new Set(AppState.userFiles.map(userFileKey));
        AppState.userFileSelection = new Set(
            [...AppState.userFileSelection].filter(key => availableKeys.has(key))
        );
        renderUserFiles();
        fileManagerStatus(`${AppState.userFiles.length} file(s) available.`, 'info');
    } catch (error) {
        list.innerHTML = `<p style="color:#dc3545;">Error: ${fileManagerEscape(error.message)}</p>`;
        fileManagerStatus(error.message, 'error');
    }
}

function fileManagerRow(button) {
    const row = button && button.closest('.user-file-item');
    return row ? { category: row.dataset.category, name: row.dataset.name } : null;
}

async function fileManagerOpen(button) {
    const file = fileManagerRow(button);
    if (!file) return;
    try {
        if (file.category === 'configs') {
            switchTab('config');
            await refreshConfigList();
            const select = document.getElementById('configFileSelect');
            select.value = file.name;
            await loadSelectedConfig();
        } else if (file.category === 'images') {
            switchTab('run');
            await refreshImageList();
            const select = document.getElementById('imageSelect');
            select.value = file.name;
            loadSelectedImage();
        } else if (file.category === 'libraries') {
            await openLibraryManager();
            await selectLibraryFile(file.name);
        } else if (file.category === 'results') {
            await showOutputPreview(file.name);
        }
    } catch (error) {
        fileManagerStatus(`Open failed: ${error.message}`, 'error');
    }
}

async function fileManagerDownload(button) {
    const file = fileManagerRow(button);
    if (!file || file.category === 'results') return;
    let url;
    if (file.category === 'configs') {
        url = `/api/config?file=${encodeURIComponent(file.name)}&userId=${encodeURIComponent(AppState.userId)}`;
    } else if (file.category === 'libraries') {
        url = `/api/material-libraries/${encodeURIComponent(file.name)}?userId=${encodeURIComponent(AppState.userId)}`;
    } else {
        url = `/uploads/${encodeURIComponent(AppState.userId)}/${encodeURIComponent(file.name)}`;
    }
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = file.name;
        document.body.appendChild(link); link.click(); link.remove();
        URL.revokeObjectURL(link.href);
    } catch (error) {
        fileManagerStatus(`Download failed: ${error.message}`, 'error');
    }
}

async function fileManagerDelete(button) {
    const file = fileManagerRow(button);
    if (!file) return;
    if (!confirm(`Delete ${USER_FILE_CATEGORY_LABEL[file.category]} file "${file.name}"?\n\nThis action cannot be undone.`)) return;
    try {
        const response = await fetch('/api/user-files', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: AppState.userId, ...file }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.error || 'Delete failed');
        await refreshUserFiles();
        if (file.category === 'configs') await refreshConfigList();
        if (file.category === 'images') await refreshImageList();
        if (file.category === 'results') await refreshResultsList();
        if (file.category === 'results') clearFilePreview({ hide: true });
        fileManagerStatus(`Deleted ${file.name}`, 'success');
    } catch (error) {
        fileManagerStatus(`Delete failed: ${error.message}`, 'error');
    }
}

async function deleteSelectedUserFiles() {
    const selected = (AppState.userFiles || [])
        .filter(file => AppState.userFileSelection.has(userFileKey(file)));
    if (selected.length === 0) return;
    if (!confirm(`Delete ${selected.length} selected item(s)?\n\nThis action cannot be undone.`)) return;

    const deleteOne = async file => {
        const response = await fetch('/api/user-files', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: AppState.userId,
                category: file.category,
                name: file.name,
            }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(`${file.name}: ${result.error || 'Delete failed'}`);
        }
        return file;
    };

    const deleteButton = document.getElementById('deleteSelectedUserFilesBtn');
    if (deleteButton) deleteButton.disabled = true;
    fileManagerStatus(`Deleting ${selected.length} selected item(s)...`, 'info');
    const settled = [];
    const concurrency = 8;
    for (let offset = 0; offset < selected.length; offset += concurrency) {
        const batch = selected.slice(offset, offset + concurrency);
        settled.push(...await Promise.allSettled(batch.map(deleteOne)));
        fileManagerStatus(
            `Deleting selected items... ${Math.min(offset + batch.length, selected.length)}/${selected.length}`,
            'info'
        );
    }
    const deleted = settled
        .filter(result => result.status === 'fulfilled')
        .map(result => result.value);
    const failures = settled.filter(result => result.status === 'rejected');
    deleted.forEach(file => AppState.userFileSelection.delete(userFileKey(file)));

    await refreshUserFiles();
    const categories = new Set(deleted.map(file => file.category));
    const refreshes = [];
    if (categories.has('configs')) refreshes.push(refreshConfigList());
    if (categories.has('images')) refreshes.push(refreshImageList());
    if (categories.has('results')) refreshes.push(refreshResultsList());
    await Promise.allSettled(refreshes);
    if (categories.has('results')) clearFilePreview({ hide: true });
    updateUserFileSelectionControls();

    if (failures.length > 0) {
        const firstError = failures[0].reason?.message || 'Unknown error';
        fileManagerStatus(
            `Deleted ${deleted.length}; ${failures.length} failed. ${firstError}`,
            'error'
        );
    } else {
        fileManagerStatus(`Deleted ${deleted.length} selected item(s).`, 'success');
    }
}

/**
 * Refresh the list of output folders
 */
async function refreshOutputsList() {
    const outputsList = document.getElementById('outputsList');
    if (!outputsList) return refreshUserFiles();

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
            clearFilePreview({ hide: true });
            return;
        }

        // Render incrementally: keep the full list in state and append rows in
        // batches as the panel scrolls, so a user with hundreds of result
        // folders doesn't build (and lay out) every row at once.
        AppState.fileManager = { outputs: result.outputs, rendered: 0 };
        outputsList.innerHTML = '';
        fileManagerRenderMore();
        const panel = outputsList.closest('.file-list-panel');
        if (panel) {
            panel.onscroll = () => {
                if (panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 150) fileManagerRenderMore();
            };
        }

        // Reset selection count
        updateSelectedCount();

    } catch (error) {
        console.error('Error loading outputs:', error);
        outputsList.innerHTML = `<p style="color: #dc3545;">Error: ${error.message}</p>`;
    }
}

// One output-folder row's HTML (extracted so the incremental renderer and any
// future re-render share the same markup).
const FILE_MANAGER_BATCH = 60;
function fileManagerOutputHtml(output) {
    const date = new Date(output.created).toLocaleString('ja-JP');
    const escapedName = output.name.replace(/'/g, "\\'");
    return `
        <div class="output-item" data-folder="${escapedName}" onclick="selectOutput('${escapedName}', event)">
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

// Append the next batch of output rows (or all remaining) to #outputsList.
function fileManagerRenderMore() {
    const fm = AppState.fileManager;
    if (!fm) return;
    const list = document.getElementById('outputsList');
    if (!list) return;
    const end = Math.min(fm.rendered + FILE_MANAGER_BATCH, fm.outputs.length);
    if (end <= fm.rendered) return;
    let html = '';
    for (let i = fm.rendered; i < end; i++) html += fileManagerOutputHtml(fm.outputs[i]);
    list.insertAdjacentHTML('beforeend', html);
    fm.rendered = end;
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
        clearFilePreview({ hide: true });

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
    console.log('selectOutput called:', folderName);

    // Skip if clicking checkbox or button
    if (event && event.target) {
        const tagName = event.target.tagName.toLowerCase();
        if (event.target.type === 'checkbox' || tagName === 'button') {
            return;
        }
    }

    // Highlight selected item
    document.querySelectorAll('.output-item').forEach(el => el.classList.remove('active'));

    // Find the clicked item and highlight it
    const clickedItem = event && event.currentTarget ? event.currentTarget :
                        document.querySelector(`.output-item[data-folder="${folderName}"]`);
    if (clickedItem) {
        clickedItem.classList.add('active');
    }

    // Show preview panel
    await showOutputPreview(folderName);
}

function clearFilePreview({
    hide = false,
    descriptionHtml = '',
    plotHtml = '',
} = {}) {
    AppState.filePreviewController?.abort();
    AppState.filePreviewController = null;
    AppState.filePreviewResultPath = '';
    AppState.filePreviewGeneration++;

    const descDiv = document.getElementById('previewDescription');
    if (descDiv) descDiv.innerHTML = descriptionHtml;
    for (const id of ['filePreviewPlot1', 'filePreviewPlot2']) {
        const plot = document.getElementById(id);
        if (plot) showPlotMessage(plot, plotHtml);
    }

    const previewPanel = document.getElementById('filePreviewPanel');
    if (previewPanel) previewPanel.style.display = hide ? 'none' : 'block';
}

/**
 * Show output preview in right panel
 * @param {string} folderName - Folder name
 */
async function showOutputPreview(folderName) {
    console.log('showOutputPreview called:', folderName);

    const previewPanel = document.getElementById('filePreviewPanel');
    const descDiv = document.getElementById('previewDescription');

    if (!previewPanel || !descDiv) {
        console.error('Preview panel elements not found');
        return;
    }

    clearFilePreview({
        descriptionHtml: '<em>Loading preview...</em>',
        plotHtml: '<div style="padding:20px; text-align:center; color:#999;">Loading...</div>',
    });
    const controller = new AbortController();
    const resultPath = `outputs/${AppState.userId}/${folderName}`;
    const previewContext = {
        generation: ++AppState.filePreviewGeneration,
        controller,
        resultPath,
    };
    AppState.filePreviewController = controller;
    AppState.filePreviewResultPath = resultPath;
    const assertCurrent = () => {
        if (controller.signal.aborted ||
            AppState.filePreviewGeneration !== previewContext.generation ||
            AppState.filePreviewController !== controller ||
            AppState.filePreviewResultPath !== resultPath) {
            throw createAbortError('File preview superseded');
        }
    };

    try {
        // Fetch description
        const descResponse = await fetch(
            `/api/user-outputs/${encodeURIComponent(folderName)}/description?userId=${AppState.userId}`,
            { signal: controller.signal }
        );
        const descResult = await descResponse.json();
        assertCurrent();

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
        console.log('Loading preview from:', resultPath);

        // Load analysis conditions for this result
        let previewConditions = { coordinate_system: 'cartesian', dx: 0.001, dy: 0.001 };
        try {
            const conditionsResponse = await fetch(
                `/api/get-conditions?result=${encodeURIComponent(resultPath)}`,
                { signal: controller.signal }
            );
            if (conditionsResponse.ok) {
                const conditionsText = await conditionsResponse.text();
                previewConditions = JSON.parse(conditionsText);
                console.log('Loaded preview analysis conditions:', previewConditions);
            }
        } catch (error) {
            if (error && error.name === 'AbortError') throw error;
            console.warn('Could not load analysis conditions:', error);
        }
        assertCurrent();

        previewContext.analysisConditions = previewConditions;
        await renderFileManagerPreview(resultPath, previewConditions, previewContext);
        assertCurrent();
        console.log('Preview rendered successfully');

    } catch (error) {
        if ((error && error.name === 'AbortError') || controller.signal.aborted) return;
        console.error('Error loading preview:', error);
        if (AppState.filePreviewController === controller) {
            clearFilePreview({
                descriptionHtml: '<em style="color:#dc3545;">Error loading preview</em>',
                plotHtml: '<div style="padding:20px; text-align:center; color:#dc3545;">Preview not available</div>',
            });
        }
    }
}

/**
 * Render preview plots for File Manager
 * @param {string} resultPath - Result path
 */
async function renderFileManagerPreview(resultPath, analysisConditions, previewContext) {
    console.log('renderFileManagerPreview called:', resultPath);

    const plot1 = document.getElementById('filePreviewPlot1');
    const plot2 = document.getElementById('filePreviewPlot2');

    if (!plot1 || !plot2) {
        console.error('Preview plot elements not found');
        return;
    }

    // Load step 1 data for preview
    const step = 1;
    const assertCurrent = () => {
        if (previewContext.controller.signal.aborted ||
            AppState.filePreviewGeneration !== previewContext.generation ||
            AppState.filePreviewController !== previewContext.controller ||
            AppState.filePreviewResultPath !== resultPath) {
            throw createAbortError('File preview superseded');
        }
    };

    // Plot 1: Input image (use simple HTML img tag)
    try {
        const imgUrl = await loadImageDataUrl(
            `/api/get-step-input-image?result=${encodeURIComponent(resultPath)}&step=${step}&t=${Date.now()}`,
            previewContext.controller.signal
        );
        assertCurrent();
        console.log('Loaded file preview input image:', resultPath);

        purgePlotlyTree(plot1);
        plot1.innerHTML = `
            <h4 style="text-align: center; margin-bottom: 10px;">Input Image (Step ${step})</h4>
            <img src="${imgUrl}"
                 style="max-width: 100%; max-height: calc(100% - 50px); object-fit: contain; display: block; margin: 0 auto;"
                 onerror="this.parentElement.innerHTML='<div style=\\'padding: 20px; text-align: center; color: #999;\\'>Input image not available</div>'">
        `;
        console.log('Input image HTML set');
    } catch (error) {
        if (error && error.name === 'AbortError') throw error;
        console.error('Error loading input image:', error);
        assertCurrent();
        showPlotMessage(plot1, '<div style="padding: 20px; text-align: center; color: #999;">Input image not available</div>');
    }

    // Plot 2: B magnitude (calculated from Az and Mu)
    let azLease = null;
    let muLease = null;
    try {
        // Get grid spacing (handle both Cartesian and Polar coordinate systems)
        let dx = 0.001;
        let dy = 0.001;

        if (analysisConditions) {
            if (analysisConditions.coordinate_system === 'polar') {
                dx = analysisConditions.dr || 0.001;
                dy = analysisConditions.dtheta || 0.001;
            } else {
                dx = analysisConditions.dx || 0.001;
                dy = analysisConditions.dy || 0.001;
            }
        }

        console.log('Using dx:', dx, 'dy:', dy, 'coordinate_system:', analysisConditions?.coordinate_system);

        azLease = await acquireSharedFieldPayload(
            'Az', step, resultPath, previewContext.controller.signal
        );
        const { data: azData } = azLease.payload;
        assertCurrent();
        muLease = await acquireSharedFieldPayload(
            'Mu', step, resultPath, previewContext.controller.signal
        );
        const { data: muData } = muLease.payload;
        assertCurrent();
        console.log('Loaded Az and Mu data');

        // Flip data from analysis coordinate system (y-up) to image coordinate system (y-down)
        const azFlipped = flipVertical(azData);
        const muFlipped = flipVertical(muData);

        // Load coarsening mask for coarsening-aware B computation
        const maskResult = await getCoarseningMaskArray(
            resultPath,
            step,
            previewContext.controller.signal
        ).catch(error => {
            if ((error && error.name === 'AbortError') ||
                error?.code === 'FIELD_MEMORY_BUDGET') throw error;
            return null;
        });
        assertCurrent();
        const activeMask = maskResult ? maskResult.mask : null;

        const { B } = calculateMagneticField(
            azFlipped, muFlipped, dx, dy, activeMask, analysisConditions
        );
        console.log('Calculated B magnitude');

        purgePlotlyTree(plot2);
        plot2.innerHTML = '';
        const graphDiv = document.createElement('div');
        graphDiv.id = `filePreviewGraph-${previewContext.generation}`;
        graphDiv.style.cssText = 'width:100%; height:100%;';
        plot2.appendChild(graphDiv);
        const renderContext = createFilePreviewRenderContext(previewContext, graphDiv);
        await plotHeatmap(graphDiv, B, '|B| [T]', true, false, renderContext);
        console.log('B magnitude plot rendered');
    } catch (error) {
        if ((error && error.name === 'AbortError') || previewContext.controller.signal.aborted) return;
        console.error('Error calculating B magnitude:', error);
        assertCurrent();
        showPlotMessage(plot2, '<div style="padding: 20px; text-align: center; color: #999;">B magnitude not available</div>');
    } finally {
        muLease?.release();
        azLease?.release();
    }
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
    if (!selectAll) return;

    // When selecting all, first render every remaining row so unrendered
    // (lazily-loaded) folders are included in the selection / bulk delete.
    if (selectAll.checked && AppState.fileManager) {
        while (AppState.fileManager.rendered < AppState.fileManager.outputs.length) fileManagerRenderMore();
    }

    document.querySelectorAll('.output-checkbox').forEach(cb => {
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

        clearFilePreview({ hide: true });

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
        clearFilePreview({ hide: true });

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

// ===== Plot Configure Functions =====

/**
 * Open plot configuration modal
 */
function openPlotConfigModal() {
    const modal = document.getElementById('plotConfigModal');
    const listDiv = document.getElementById('plotConfigList');

    // Get dashboard plots
    const items = AppState.gridStack ? AppState.gridStack.getGridItems() : [];

    if (items.length === 0) {
        listDiv.innerHTML = '<p>No plots on dashboard</p>';
        modal.style.display = 'flex';
        return;
    }

    let html = '';
    items.forEach(item => {
        const plotId = item.getAttribute('data-plot-id');
        const plotType = item.getAttribute('data-plot-type');
        const title = plotDefinitions[plotType]?.name || plotType || 'Unknown Plot';

        // Debug: log if plotType or plotId is undefined
        if (!plotType || !plotId) {
            console.warn('Missing attributes for item:', {
                plotId: plotId,
                plotType: plotType,
                element: item,
                allAttributes: Array.from(item.attributes).map(a => `${a.name}="${a.value}"`)
            });
        }

        const config = AppState.plotConfigs[plotId] || {};
        const isHeatmap = ['az_heatmap', 'b_magnitude', 'h_magnitude', 'mu_distribution', 'energy_density', 'jz_distribution'].includes(plotType);
        const isTimeSeries = ['force_x_time', 'force_y_time', 'torque_time', 'energy_time', 'virtual_work'].includes(plotType);

        html += `
            <div class="plot-config-item" data-plot-id="${plotId}">
                <h4>${title}</h4>

                <!-- X-axis range -->
                <div class="config-row">
                    <label>X-axis:</label>
                    <input type="radio" name="xRange_${plotId}" value="auto" ${!config.xRange || config.xRange === 'auto' ? 'checked' : ''}
                           onchange="toggleRangeInputs('${plotId}', 'x', 'auto')">
                    <label>Auto</label>
                    <input type="radio" name="xRange_${plotId}" value="fixed" ${config.xRange && config.xRange !== 'auto' ? 'checked' : ''}
                           onchange="toggleRangeInputs('${plotId}', 'x', 'fixed')">
                    <label>Fixed:</label>
                    <input type="number" id="xMin_${plotId}" value="${config.xRange && config.xRange !== 'auto' ? config.xRange[0] : ''}"
                           placeholder="min" ${!config.xRange || config.xRange === 'auto' ? 'disabled' : ''}>
                    <span>to</span>
                    <input type="number" id="xMax_${plotId}" value="${config.xRange && config.xRange !== 'auto' ? config.xRange[1] : ''}"
                           placeholder="max" ${!config.xRange || config.xRange === 'auto' ? 'disabled' : ''}>
                </div>

                <!-- Y-axis range -->
                <div class="config-row">
                    <label>Y-axis:</label>
                    <input type="radio" name="yRange_${plotId}" value="auto" ${!config.yRange || config.yRange === 'auto' ? 'checked' : ''}
                           onchange="toggleRangeInputs('${plotId}', 'y', 'auto')">
                    <label>Auto</label>
                    <input type="radio" name="yRange_${plotId}" value="fixed" ${config.yRange && config.yRange !== 'auto' ? 'checked' : ''}
                           onchange="toggleRangeInputs('${plotId}', 'y', 'fixed')">
                    <label>Fixed:</label>
                    <input type="number" id="yMin_${plotId}" value="${config.yRange && config.yRange !== 'auto' ? config.yRange[0] : ''}"
                           placeholder="min" ${!config.yRange || config.yRange === 'auto' ? 'disabled' : ''}>
                    <span>to</span>
                    <input type="number" id="yMax_${plotId}" value="${config.yRange && config.yRange !== 'auto' ? config.yRange[1] : ''}"
                           placeholder="max" ${!config.yRange || config.yRange === 'auto' ? 'disabled' : ''}>
                </div>
        `;

        if (isHeatmap) {
            html += `
                <!-- Z-axis (colorscale) range -->
                <div class="config-row">
                    <label>Color scale:</label>
                    <input type="radio" name="zRange_${plotId}" value="auto" ${!config.zRange || config.zRange === 'auto' ? 'checked' : ''}
                           onchange="toggleRangeInputs('${plotId}', 'z', 'auto')">
                    <label>Auto</label>
                    <input type="radio" name="zRange_${plotId}" value="fixed" ${config.zRange && config.zRange !== 'auto' ? 'checked' : ''}
                           onchange="toggleRangeInputs('${plotId}', 'z', 'fixed')">
                    <label>Fixed:</label>
                    <input type="number" id="zMin_${plotId}" value="${config.zRange && config.zRange !== 'auto' ? config.zRange[0] : ''}"
                           placeholder="min" ${!config.zRange || config.zRange === 'auto' ? 'disabled' : ''}>
                    <span>to</span>
                    <input type="number" id="zMax_${plotId}" value="${config.zRange && config.zRange !== 'auto' ? config.zRange[1] : ''}"
                           placeholder="max" ${!config.zRange || config.zRange === 'auto' ? 'disabled' : ''}>
                </div>

                <!-- Colorscale theme -->
                <div class="config-row">
                    <label>Color theme:</label>
                    <select id="colorscale_${plotId}">
                        <option value="Viridis" ${config.colorscale === 'Viridis' || !config.colorscale ? 'selected' : ''}>Viridis</option>
                        <option value="Plasma" ${config.colorscale === 'Plasma' ? 'selected' : ''}>Plasma</option>
                        <option value="Inferno" ${config.colorscale === 'Inferno' ? 'selected' : ''}>Inferno</option>
                        <option value="Magma" ${config.colorscale === 'Magma' ? 'selected' : ''}>Magma</option>
                        <option value="Cividis" ${config.colorscale === 'Cividis' ? 'selected' : ''}>Cividis</option>
                        <option value="Turbo" ${config.colorscale === 'Turbo' ? 'selected' : ''}>Turbo</option>
                        <option value="Rainbow" ${config.colorscale === 'Rainbow' ? 'selected' : ''}>Rainbow</option>
                        <option value="Jet" ${config.colorscale === 'Jet' ? 'selected' : ''}>Jet</option>
                        <option value="Hot" ${config.colorscale === 'Hot' ? 'selected' : ''}>Hot</option>
                        <option value="Cool" ${config.colorscale === 'Cool' ? 'selected' : ''}>Cool</option>
                        <option value="Bluered" ${config.colorscale === 'Bluered' ? 'selected' : ''}>Blue-Red</option>
                        <option value="RdBu" ${config.colorscale === 'RdBu' ? 'selected' : ''}>RdBu</option>
                        <option value="Portland" ${config.colorscale === 'Portland' ? 'selected' : ''}>Portland</option>
                        <option value="Picnic" ${config.colorscale === 'Picnic' ? 'selected' : ''}>Picnic</option>
                        <option value="Electric" ${config.colorscale === 'Electric' ? 'selected' : ''}>Electric</option>
                        <option value="Blackbody" ${config.colorscale === 'Blackbody' ? 'selected' : ''}>Blackbody</option>
                    </select>
                </div>
            `;
        }

        html += `
            </div>
        `;
    });

    listDiv.innerHTML = html;
    modal.style.display = 'flex';
}

/**
 * Close plot configuration modal
 */
function closePlotConfigModal() {
    const modal = document.getElementById('plotConfigModal');
    modal.style.display = 'none';
}

/**
 * Toggle range input fields based on auto/fixed selection
 */
function toggleRangeInputs(plotId, axis, mode) {
    const minInput = document.getElementById(`${axis}Min_${plotId}`);
    const maxInput = document.getElementById(`${axis}Max_${plotId}`);

    if (mode === 'auto') {
        minInput.disabled = true;
        maxInput.disabled = true;
    } else {
        minInput.disabled = false;
        maxInput.disabled = false;
    }
}

/**
 * Apply plot configurations
 */
function applyPlotConfigs() {
    const items = AppState.gridStack ? AppState.gridStack.getGridItems() : [];

    items.forEach(item => {
        const plotId = item.getAttribute('data-plot-id');
        const plotType = item.getAttribute('data-plot-type');

        const config = {};

        // X-axis range
        const xRangeMode = document.querySelector(`input[name="xRange_${plotId}"]:checked`)?.value;
        if (xRangeMode === 'fixed') {
            const xMin = parseFloat(document.getElementById(`xMin_${plotId}`)?.value);
            const xMax = parseFloat(document.getElementById(`xMax_${plotId}`)?.value);
            if (!isNaN(xMin) && !isNaN(xMax)) {
                config.xRange = [xMin, xMax];
            }
        } else {
            config.xRange = 'auto';
        }

        // Y-axis range
        const yRangeMode = document.querySelector(`input[name="yRange_${plotId}"]:checked`)?.value;
        if (yRangeMode === 'fixed') {
            const yMin = parseFloat(document.getElementById(`yMin_${plotId}`)?.value);
            const yMax = parseFloat(document.getElementById(`yMax_${plotId}`)?.value);
            if (!isNaN(yMin) && !isNaN(yMax)) {
                config.yRange = [yMin, yMax];
            }
        } else {
            config.yRange = 'auto';
        }

        // Z-axis range (heatmaps only)
        const isHeatmap = ['az_heatmap', 'b_magnitude', 'h_magnitude', 'mu_distribution', 'energy_density', 'jz_distribution'].includes(plotType);
        if (isHeatmap) {
            const zRangeMode = document.querySelector(`input[name="zRange_${plotId}"]:checked`)?.value;
            if (zRangeMode === 'fixed') {
                const zMin = parseFloat(document.getElementById(`zMin_${plotId}`)?.value);
                const zMax = parseFloat(document.getElementById(`zMax_${plotId}`)?.value);
                if (!isNaN(zMin) && !isNaN(zMax)) {
                    config.zRange = [zMin, zMax];
                }
            } else {
                config.zRange = 'auto';
            }

            // Colorscale
            const colorscale = document.getElementById(`colorscale_${plotId}`)?.value;
            config.colorscale = colorscale || 'Viridis';
        }

        // Save config
        AppState.plotConfigs[plotId] = config;
    });

    // Refresh all plots with new configurations
    refreshAllPlots();

    // Close modal
    closePlotConfigModal();
}

/**
 * Reset all plot configurations to auto
 */
function resetAllPlotConfigs() {
    AppState.plotConfigs = {};
    refreshAllPlots();
    closePlotConfigModal();
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
}

/**
 * Transform polar coordinate data to cartesian (arc or full donut)
 * @param {Array<Array<number>>} polarData - 2D array in polar coordinates
 * @param {object} conditions - Analysis conditions containing r_i, r_o, theta_range, r_orientation, boundary_conditions
 * @param {boolean} fullModel - If true, replicate to full 360 degrees
 * @returns {object} - {x: Array, y: Array, z: Array} for Plotly heatmap
 */
function transformPolarToCartesian(
    polarData,
    conditions,
    fullModel = false,
    useHarmonicMean = false,
    fullModelMultiplier = 1
) {
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
    const N = fullModel ? Math.max(1, fullModelMultiplier) : 1;

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
            // Use harmonic mean for permeability (series magnetic circuit), arithmetic mean otherwise
            const value = useHarmonicMean
                ? bilinearInterpolateHarmonic(polarData, theta_idx, r_idx, thetaPeriodic, thetaAntiperiodic, r_orientation)
                : bilinearInterpolate(polarData, theta_idx, r_idx, thetaPeriodic, thetaAntiperiodic, r_orientation);
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
 * Bilinear interpolation using harmonic mean (for permeability in series magnetic circuits)
 * Formula: 1/μ_eff = (1-t)/μ_A + t/μ_B  =>  μ_eff = 1 / ((1-t)/μ_A + t/μ_B)
 * @param {Array<Array<number>>} data - 2D array [ntheta][nr] or [nr][ntheta] depending on r_orientation
 * @param {number} theta_idx - Theta index (fractional)
 * @param {number} r_idx - R index (fractional)
 * @param {boolean} thetaPeriodic - If true, wrap theta index periodically
 * @param {boolean} thetaAntiperiodic - If true, apply sign flip when crossing theta boundary
 * @param {string} r_orientation - 'horizontal' (data[theta][r]) or 'vertical' (data[r][theta])
 * @returns {number} - Interpolated value using harmonic mean
 */
function bilinearInterpolateHarmonic(data, theta_idx, r_idx, thetaPeriodic = false, thetaAntiperiodic = false, r_orientation = 'horizontal') {
    // Safety checks
    if (!data || data.length === 0 || !data[0] || data[0].length === 0) {
        return 1.0; // Default to mu_r = 1 (air) if no data
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
        v00 = (data[i0] && data[i0][j0] !== undefined) ? data[i0][j0] : 1.0;
        v01 = (data[i0] && data[i0][j1] !== undefined) ? data[i0][j1] : 1.0;
        v10 = (data[i1] && data[i1][j0] !== undefined) ? data[i1][j0] : 1.0;
        v11 = (data[i1] && data[i1][j1] !== undefined) ? data[i1][j1] : 1.0;
    } else {
        // data[r_idx][theta_idx]
        v00 = (data[j0] && data[j0][i0] !== undefined) ? data[j0][i0] : 1.0;
        v01 = (data[j1] && data[j1][i0] !== undefined) ? data[j1][i0] : 1.0;
        v10 = (data[j0] && data[j0][i1] !== undefined) ? data[j0][i1] : 1.0;
        v11 = (data[j1] && data[j1][i1] !== undefined) ? data[j1][i1] : 1.0;
    }

    // Apply sign flip for anti-periodic BC when crossing boundary
    // Note: Harmonic mean doesn't work with negative values, so we take absolute values
    // and restore sign at the end (though this is unusual for permeability)
    let signFlip = 1.0;
    if (thetaAntiperiodic && crossesBoundary) {
        signFlip = -1.0;
        v10 = Math.abs(v10);
        v11 = Math.abs(v11);
    }

    // Ensure no zero or negative values (permeability must be positive)
    const epsilon = 1e-10;
    v00 = Math.max(epsilon, Math.abs(v00));
    v01 = Math.max(epsilon, Math.abs(v01));
    v10 = Math.max(epsilon, Math.abs(v10));
    v11 = Math.max(epsilon, Math.abs(v11));

    // Harmonic mean interpolation in r direction
    // 1/v0 = (1-dr)/v00 + dr/v01
    const v0 = 1.0 / ((1.0 - dr) / v00 + dr / v01);  // at theta=i0
    // 1/v1 = (1-dr)/v10 + dr/v11
    const v1 = 1.0 / ((1.0 - dr) / v10 + dr / v11);  // at theta=i1

    // Harmonic mean interpolation in theta direction
    // 1/result = (1-dt)/v0 + dt/v1
    const result = 1.0 / ((1.0 - dt) / v0 + dt / v1);

    return result * signFlip;
}

/**
 * Refresh all plots in the dashboard (for when polar transform settings change)
 */
function refreshAllPlots() {
    void updateAllPlots().catch(error => {
        if (!error || error.name !== 'AbortError') {
            console.error('Dashboard refresh failed:', error);
        }
    });
}

// =====================================================
// Material Library Manager
// =====================================================

const LIB_TEMPLATE = `# OpenMagFDM Material Library
# Reference presets in analysis configs via:
#   materials:
#     iron_core:
#       rgb: [128, 128, 128]
#       preset: pure_iron_model
#
# -----------------------------------------------------------------------
# Key formats:
#   mu_r: table    mu_r: [[H1,H2,...],[mur1,mur2,...]]  (PCHIP interpolated)
#   mu_r: formula  mu_r: "5000 / (1 + ($H/200)^2)"
#   B-H:  table    B-H:  [[H1,H2,...],[B1,B2,...]]      (PCHIP interpolated)
#   B-H:  formula  B-H:  "expression with $H"
#
# Permanent magnets: specify Br [T] + magnetization block.
# Formula constants available: mu0 (4pi*1e-7 H/m), pi, exp, sin, cos, ...
# -----------------------------------------------------------------------

material_presets:

  # =========================================================================
  # Soft magnetic materials
  # =========================================================================

  # Silicon steel M19 — example using mu_r point table
  silicon_steel_m19:
    mu_r:
      - [0,    200,  500,  1000, 2000, 5000, 10000]  # H [A/m]
      - [5000, 4500, 3000, 2000, 1000,  500,   200]  # mu_r

  # Pure Iron — continuous function model (~2.1 T saturation)
  pure_iron_model:
    B-H: ($H/(40 + 0.52*$H) + 4*pi*1e-7*$H) * (1.0/(1.0 + exp(-0.1*($H - 40))))

  # Permendur (Fe-Co 49/49) — highest saturation of common soft magnetics (~2.4 T)
  permendur_model:
    B-H: ($H/(14.5 + 0.45*$H) + 4*pi*1e-7*$H) * (1.0/(1.0 + exp(-0.1*($H - 25))))

  # Structural steel (SS400 equivalent) — moderate permeability
  structural_steel_model:
    B-H: ($H/(97.0 + 0.625*$H) + 4*pi*1e-7*$H) * (1.0/(1.0 + exp(-0.1*($H - 20))))

  # =========================================================================
  # Permanent magnets — IEC 60404-8-1 / JIS C 2502 typical values
  #   Br: residual flux density [T]  mu_r: reversible permeability (~1.05 for NdFeB)
  #   angle: 0 = +X direction  (90 = +Y, 180 = -X, 270 = -Y)
  # =========================================================================

  # NdFeB sintered N35  Br=1.19T  Hcb>=764kA/m  Hcj>=955kA/m  BHmax~35MGOe
  NdFeB_N35:
    mu_r: 1.05
    magnetization:
      Br: 1.19
      pattern: parallel
      angle: 0

  # NdFeB sintered N40  Br=1.27T  Hcb>=836kA/m  Hcj>=955kA/m  BHmax~40MGOe
  NdFeB_N40:
    mu_r: 1.05
    magnetization:
      Br: 1.27
      pattern: parallel
      angle: 0

  # NdFeB sintered N45  Br=1.34T  Hcb>=836kA/m  Hcj>=876kA/m  BHmax~45MGOe
  NdFeB_N45:
    mu_r: 1.05
    magnetization:
      Br: 1.34
      pattern: parallel
      angle: 0

  # NdFeB sintered N50  Br=1.42T  Hcb>=796kA/m  Hcj>=876kA/m  BHmax~50MGOe
  NdFeB_N50:
    mu_r: 1.05
    magnetization:
      Br: 1.42
      pattern: parallel
      angle: 0

  # Ferrite sintered high-Br (FB9B equivalent)  Br=0.44T  Hcb>=255kA/m  Hcj>=280kA/m
  ferrite_high_B:
    mu_r: 1.05
    magnetization:
      Br: 0.44
      pattern: parallel
      angle: 0
`;

async function openLibraryManager() {
    if (!AppState.libraryAceEditor) {
        const editor = ace.edit('libraryAceEditor');
        editor.setTheme('ace/theme/monokai');
        editor.session.setMode('ace/mode/yaml');
        editor.setOptions({
            enableBasicAutocompletion: true,
            enableLiveAutocompletion: true,
            enableSnippets: false,
            fontSize: '13px',
            showPrintMargin: false,
            tabSize: 2,
            useSoftTabs: true
        });
        ace.require('ace/ext/language_tools');
        editor.completers = [createLibraryCompleter()];
        AppState.libraryAceEditor = editor;
    }
    await refreshLibraryList();
    document.getElementById('libraryManagerModal').style.display = 'flex';
    // Ace editor needs an explicit resize after the modal becomes visible,
    // otherwise it renders as only a few lines tall (container had 0 height while hidden).
    requestAnimationFrame(() => {
        if (AppState.libraryAceEditor) AppState.libraryAceEditor.resize();
    });
}

function closeLibraryManager() {
    document.getElementById('libraryManagerModal').style.display = 'none';
}

async function refreshLibraryList() {
    const listEl = document.getElementById('libraryFileList');
    const statusEl = document.getElementById('libraryListStatus');
    listEl.innerHTML = '';
    statusEl.textContent = 'Loading...';
    try {
        const response = await fetch(`/api/material-libraries?userId=${AppState.userId}`);
        if (!response.ok) throw new Error('Failed to load library list');
        const result = await response.json();

        statusEl.textContent = '';
        if (result.libraries.length === 0) {
            listEl.innerHTML = '<div style="color:#aaa; font-size:0.8rem; padding:5px;">No files yet</div>';
            return;
        }

        result.libraries.forEach(lib => {
            const item = document.createElement('div');
            item.className = 'lib-file-item' + (AppState.currentLibraryFile === lib.filename ? ' active' : '');
            item.textContent = lib.filename;
            item.title = `${(lib.size / 1024).toFixed(1)} KB  ${new Date(lib.modified).toLocaleString()}`;
            item.onclick = () => selectLibraryFile(lib.filename);
            listEl.appendChild(item);
        });
    } catch (e) {
        statusEl.textContent = `Error: ${e.message}`;
    }
}

async function selectLibraryFile(filename) {
    try {
        const response = await fetch(`/api/material-libraries/${encodeURIComponent(filename)}?userId=${AppState.userId}`);
        if (!response.ok) throw new Error('Failed to load file');
        const content = await response.text();

        AppState.currentLibraryFile = filename;
        if (AppState.libraryAceEditor) {
            AppState.libraryAceEditor.setValue(content, -1);
        }
        document.getElementById('libEditFilename').textContent = filename;
        document.getElementById('libDeleteBtn').style.display = 'inline-block';
        document.getElementById('libSaveBtn').style.display = 'inline-block';
        document.getElementById('libUseBtn').style.display = 'inline-block';

        document.querySelectorAll('.lib-file-item').forEach(el => {
            el.classList.toggle('active', el.textContent === filename);
        });

        // B-H Curves tab sources from the active library, not the editor — no update needed here
    } catch (e) {
        document.getElementById('libraryListStatus').textContent = `Error: ${e.message}`;
    }
}

function switchLibTab(tab) {
    const editPane  = document.getElementById('libPaneEdit');
    const bhPane    = document.getElementById('libPaneBH');
    const leftPanel = document.getElementById('libLeftPanel');
    const editBtn   = document.getElementById('libTabEdit');
    const bhBtn     = document.getElementById('libTabBH');

    if (tab === 'edit') {
        editPane.style.display  = 'flex';
        bhPane.style.display    = 'none';
        leftPanel.style.display = 'flex';
        editBtn.classList.add('lib-tab-active');
        bhBtn.classList.remove('lib-tab-active');
        if (AppState.libraryAceEditor) AppState.libraryAceEditor.resize();
    } else {
        editPane.style.display  = 'none';
        bhPane.style.display    = 'flex';
        leftPanel.style.display = 'none';
        editBtn.classList.remove('lib-tab-active');
        bhBtn.classList.add('lib-tab-active');
        // Defer so the pane is fully laid out before Plotly measures the container
        requestAnimationFrame(() => loadBHMaterialList());
    }
}

// Evaluate a tinyexpr-style mu_r formula string at a given H value.
// Replaces $H with the numeric value and maps tinyexpr names to JS Math.
function evaluateMuFormula(formula, H) {
    const expr = formula
        .replace(/\$H/g, `(${H})`)
        .replace(/\bmu0\b/g, '(4*Math.PI*1e-7)')  // vacuum permeability
        .replace(/\bpi\b/g, 'Math.PI')
        .replace(/\bexp\s*\(/g, 'Math.exp(')
        .replace(/\bsin\s*\(/g, 'Math.sin(')
        .replace(/\bcos\s*\(/g, 'Math.cos(')
        .replace(/\bsqrt\s*\(/g, 'Math.sqrt(')
        .replace(/\babs\s*\(/g, 'Math.abs(')
        .replace(/\bln\s*\(/g, 'Math.log(')      // ln = natural log
        .replace(/\blog\s*\(/g, 'Math.log10(')   // tinyexpr log = log10
        .replace(/\bpow\s*\(/g, 'Math.pow(')
        .replace(/\^/g, '**');                    // tinyexpr power operator
    try {
        // eslint-disable-next-line no-new-func
        const v = Function(`'use strict'; return (${expr})`)();
        return (isFinite(v) && v > 0) ? v : NaN;
    } catch (_) {
        return NaN;
    }
}

// Load the active library (AppState.selectedLibrary) and populate the
// material list in the B-H Curves pane left column.
async function loadBHMaterialList() {
    const listEl    = document.getElementById('libBHMaterialList');
    const plotEl    = document.getElementById('libBHPlotContainer');
    AppState.bhListController?.abort();
    const listController = new AbortController();
    const listGeneration = ++AppState.bhListGeneration;
    const selectedLibrary = AppState.selectedLibrary;
    AppState.bhListController = listController;
    const isListCurrent = () =>
        !listController.signal.aborted &&
        AppState.bhListGeneration === listGeneration &&
        AppState.bhListController === listController &&
        AppState.selectedLibrary === selectedLibrary;

    AppState.bhRenderController?.abort();
    AppState.bhRenderController = null;
    AppState.bhRenderGeneration++;
    AppState.currentBHMaterial = null;
    const toolbar = document.getElementById('libBHToolbar');
    if (toolbar) toolbar.style.display = 'none';
    listEl.innerHTML = '';
    preparePlotlyContainer(plotEl);

    if (!selectedLibrary) {
        listEl.innerHTML = `<div style="color:#888; font-size:0.82rem; padding:8px; line-height:1.5;">
            No active library.<br>
            Open a library file,<br>click <em>Use This Library</em>,<br>then return here.</div>`;
        if (AppState.bhListController === listController) {
            AppState.bhListController = null;
        }
        return;
    }

    try {
        const response = await fetch(
            `/api/material-libraries/${encodeURIComponent(selectedLibrary)}?userId=${AppState.userId}`,
            { signal: listController.signal }
        );
        if (!response.ok) throw new Error('Failed to load library');
        const content = await response.text();
        if (!isListCurrent()) return;

        const doc = jsyaml.load(content) || {};
        // Combine material_presets and materials sections
        const presets = Object.assign({}, doc.material_presets || {}, doc.materials || {});
        const names   = Object.keys(presets);

        if (names.length === 0) {
            listEl.innerHTML = '<div style="color:#888; font-size:0.82rem; padding:8px;">No materials in library.</div>';
            return;
        }

        // Header showing which library is active
        const hdr = document.createElement('div');
        hdr.style.cssText = 'font-weight:600; font-size:0.8rem; margin-bottom:8px; color:#495057; word-break:break-all;';
        hdr.textContent   = selectedLibrary;
        listEl.appendChild(hdr);

        // Build clickable material list
        let firstItem = null;
        for (const name of names) {
            const props = presets[name];
            const mur   = props && props.mu_r;
            const bh    = props && props['B-H'];
            const mag   = props && props.magnetization;
            // Classify: B-H array / mu_r array / formula / constant / magnet (Br)
            const isBH      = (Array.isArray(bh) && bh.length === 2 && Array.isArray(bh[0]))
                           || (typeof bh === 'string' && bh.trim() !== '');
            const isArray   = !isBH && Array.isArray(mur) && mur.length === 2 && Array.isArray(mur[0]);
            const isFormula = !isBH && !isArray && typeof mur === 'string' && mur.trim() !== '';
            // isMagnet: has constant Br (no explicit B-H), treated as linear demagnetization curve
            const isMagnet  = !isBH && mag && typeof mag.Br === 'number';
            const isConst   = !isBH && !isMagnet && typeof mur === 'number';
            const hasPlot   = isBH || isArray || isFormula || isConst || isMagnet;

            const item = document.createElement('div');
            item.className  = 'lib-file-item';
            item.dataset.matname = name;
            item.style.cssText = 'padding:5px 8px; cursor:pointer; border-radius:3px;'
                               + ' font-size:0.82rem; word-break:break-all; line-height:1.4;';
            item.title = isBH      ? 'B-H curve [[H],[B]]'
                       : isArray   ? 'Array μr(H)'
                       : isFormula ? 'Formula μr($H)'
                       : isMagnet  ? `Permanent magnet  Br = ${mag.Br} T`
                       : isConst   ? `Constant μr = ${mur}`
                       :             'No B-H / μr data';

            // Badge
            const badge = document.createElement('span');
            badge.style.cssText = 'float:right; font-size:0.68rem; border-radius:8px; padding:1px 5px; margin-left:4px;'
                                + (isBH      ? 'background:#fff3cd; color:#856404;'
                                  : isArray   ? 'background:#d4edda; color:#155724;'
                                  : isFormula ? 'background:#cce5ff; color:#004085;'
                                  : isMagnet  ? 'background:#fce4ec; color:#880e4f;'
                                  : isConst   ? 'background:#f8d7da; color:#721c24;'
                                  :             'background:#f0f0f0; color:#888;');
            badge.textContent = isBH      ? 'B-H'
                              : isArray   ? 'arr'
                              : isFormula ? 'fn'
                              : isMagnet  ? 'Br'
                              : isConst   ? 'const'
                              :             '—';
            item.appendChild(badge);
            item.appendChild(document.createTextNode(name));

            item.onclick = () => {
                listEl.querySelectorAll('.lib-file-item[data-matname]').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                // The renderer owns both plot and no-data states, so it can
                // abort/purge an older Plotly render and reset the toolbar.
                requestBHCurveRender(name, props);
            };

            listEl.appendChild(item);
            if (!firstItem) firstItem = { item, name, props, hasPlot };
        }

        // Auto-select the first material with plottable data (or just first)
        const autoSelect = names.reduce((found, name) => {
            if (found) return found;
            const p   = presets[name];
            const bh  = p && p['B-H'];
            const mur = p && p.mu_r;
            const mg  = p && p.magnetization;
            if ((Array.isArray(bh) && bh.length === 2 && Array.isArray(bh[0]))
                || (typeof bh === 'string' && bh.trim() !== '')
                || (Array.isArray(mur) && mur.length === 2 && Array.isArray(mur[0]))
                || typeof mur === 'string' || typeof mur === 'number'
                || (mg && typeof mg.Br === 'number')) {
                return { name, props: p };
            }
            return null;
        }, null) || (names.length > 0 ? { name: names[0], props: presets[names[0]] } : null);

        if (autoSelect) {
            if (!isListCurrent()) return;
            const target = listEl.querySelector(`[data-matname="${CSS.escape(autoSelect.name)}"]`);
            if (target) target.classList.add('active');
            requestBHCurveRender(autoSelect.name, autoSelect.props);
        }

    } catch (e) {
        if (e && e.name === 'AbortError') return;
        if (isListCurrent()) {
            listEl.innerHTML = `<div style="color:#c62828; font-size:0.82rem; padding:8px;">Error: ${e.message}</div>`;
        }
    } finally {
        if (AppState.bhListController === listController) {
            AppState.bhListController = null;
        }
    }
}

// Render dual-axis B-H / μr-H Plotly chart for a single material.
// Handles three data sources:
//   1. B-H: [[H,...],[B,...]]   — B values are direct; μr derived from B/μ₀H
//   2. mu_r: [[H,...],[μr,...]] — μr direct; B = μ₀·μr·H
//   3. mu_r: formula / constant
function requestBHCurveRender(name, props) {
    void renderBHCurveForMaterial(name, props).catch(error => {
        if (!error || error.name !== 'AbortError') console.error('B-H render failed:', error);
    });
}

async function renderBHCurveForMaterial(name, props) {
    const host = document.getElementById('libBHPlotContainer');
    const toolbar   = document.getElementById('libBHToolbar');
    if (!host) return;
    preparePlotlyContainer(host);
    const container = document.createElement('div');
    container.style.cssText = 'width:100%; height:100%;';
    host.appendChild(container);
    const renderContext = createBHRenderContext(container);
    preparePlotlyContainer(container, renderContext);

    AppState.currentBHMaterial = { name, props };

    const MU_0   = 4 * Math.PI * 1e-7;
    const bh     = props && props['B-H'];
    const mur    = props && props.mu_r;
    const bhType = (props && props['bh_type']) || 'auto';

    // Determine X-axis type from checkbox (default: log)
    const logCb = document.getElementById('libBHLogX');
    const xtype = (logCb && !logCb.checked) ? 'linear' : 'log';

    // Log-spaced H axis for formula / constant evaluation (1 to 1e5 A/m)
    const H_log = Array.from({ length: 200 }, (_, i) => Math.pow(10, i * 5 / 199));

    let H_arr = null, B_arr = null, mur_arr = null;
    let isDashed = false;
    let isDemag = false;  // demagnetization curve (H < 0)

    // --- Case 1a: B-H: formula string "expr($H)" ---
    if (typeof bh === 'string' && bh.trim() !== '') {
        // Evaluate over H range 1e-3..1e5 A/m (real materials saturate well below 1e5)
        const H_log = Array.from({ length: 200 }, (_, i) => Math.pow(10, -3 + i * 8 / 199));
        const bhFormula = bh.trim();
        const evalB = (H) => {
            const expr = bhFormula
                .replace(/\$H/g, `(${H})`)
                .replace(/\bmu0\b/g, '(4*Math.PI*1e-7)')  // vacuum permeability
                .replace(/\bpi\b/g, 'Math.PI')
                .replace(/\bexp\s*\(/g, 'Math.exp(')
                .replace(/\bsin\s*\(/g, 'Math.sin(')
                .replace(/\bcos\s*\(/g, 'Math.cos(')
                .replace(/\btanh\s*\(/g, 'Math.tanh(')
                .replace(/\bsqrt\s*\(/g, 'Math.sqrt(')
                .replace(/\babs\s*\(/g, 'Math.abs(')
                .replace(/\bln\s*\(/g, 'Math.log(')
                .replace(/\blog\s*\(/g, 'Math.log10(')
                .replace(/\bpow\s*\(/g, 'Math.pow(')
                .replace(/\^/g, '**');
            try { return Function(`'use strict'; return (${expr})`)(); }
            catch (_) { return NaN; }
        };
        const Br            = evalB(0);
        const treatAsMagnet = (bhType === 'magnet') || (bhType === 'auto' && isFinite(Br) && Math.abs(Br) > 1e-9);
        const treatAsSoft   = (bhType === 'soft');

        if (treatAsMagnet && !treatAsSoft) {
            // Demagnetization formula: sample H ∈ [-1e6, -1] A/m, log-spaced in |H|
            // Extended to 5 MA/m to cover strong demagnetizing fields
            const N = 200;
            const H_neg = Array.from({ length: N }, (_, i) =>
                -Math.pow(10, i * Math.log10(1e6) / (N - 1)));  // magnitude: 1 → 1e6
            const pairs = H_neg.map(H => [H, evalB(H)]).filter(([, B]) => isFinite(B));
            if (isFinite(Br)) pairs.push([0, Br]);   // remanence point at H=0
            pairs.sort((a, b) => a[0] - b[0]);        // ascending H (most-negative first)
            if (pairs.length > 1) {
                H_arr   = pairs.map(p => p[0]);
                B_arr   = pairs.map(p => p[1]);
                mur_arr = null;
                isDemag = true;
            }
        } else {
            // Soft magnet or no remanence: sample H ∈ [1e-3, 1e5] A/m
            const useBr = treatAsSoft ? 0 : (isFinite(Br) ? Br : 0);
            const H_ok  = H_log.filter(H => isFinite(evalB(H)));
            if (H_ok.length > 1) {
                H_arr   = H_ok;
                B_arr   = H_arr.map(H => evalB(H));
                mur_arr = H_arr.map((H, i) => Math.max(1, (B_arr[i] - useBr) / (MU_0 * H)));
            }
        }

    // --- Case 1b: B-H: [[H],[B]] ---
    } else if (Array.isArray(bh) && bh.length === 2 &&
        Array.isArray(bh[0]) && Array.isArray(bh[1]) &&
        bh[0].length === bh[1].length && bh[0].length >= 2) {

        if (bh[0].some(h => h < -1e-12)) {
            // --- Demagnetization curve (H < 0, second quadrant) ---
            // Sort by H ascending (most negative first → H≈0 at end)
            const pairs = bh[0].map((h, i) => [h, bh[1][i]]).sort((a, b) => a[0] - b[0]);
            H_arr   = pairs.map(p => p[0]);
            B_arr   = pairs.map(p => p[1]);
            mur_arr = null;
            isDemag = true;

        } else {
            // --- Normal magnetization curve (H >= 0) ---
            const H_raw = [...bh[0]];
            const B_raw = [...bh[1]];

            // Detect remanence: if H[0]=0 and B[0]≠0
            const Br = (Math.abs(H_raw[0]) < 1e-12 && Math.abs(B_raw[0]) > 1e-12) ? B_raw[0] : 0;

            // Prepend implicit (0,0) if first H > 0
            if (H_raw[0] > 1e-12) { H_raw.unshift(0); B_raw.unshift(0); }

            B_arr = B_raw;

            // Compute μr: μr(H=0) from initial slope; μr(H>0) = (B-Br)/(μ₀H)
            mur_arr = [];
            if (H_raw.length >= 2 && H_raw[1] > 0) {
                const mu0_val = (B_raw[1] - Br) / (MU_0 * H_raw[1]);
                mur_arr.push(Math.max(1, mu0_val));
            } else {
                mur_arr.push(1);
            }
            for (let i = 1; i < H_raw.length; i++) {
                if (H_raw[i] <= 1e-15) continue;
                mur_arr.push(Math.max(1, (B_raw[i] - Br) / (MU_0 * H_raw[i])));
            }

            H_arr = H_raw;
        }

    // --- Case 2: mu_r: [[H],[μr]] ---
    } else if (Array.isArray(mur) && mur.length === 2 &&
               Array.isArray(mur[0]) && Array.isArray(mur[1]) &&
               mur[0].length === mur[1].length && mur[0].length > 0) {
        H_arr   = mur[0];
        mur_arr = mur[1];
        B_arr   = H_arr.map((H, i) => MU_0 * mur_arr[i] * H);

    // --- Case 3: mu_r: formula string ---
    } else if (typeof mur === 'string' && mur.trim() !== '') {
        const pts = H_log
            .map(H => ({ H, mu: evaluateMuFormula(mur, H) }))
            .filter(p => !isNaN(p.mu) && p.mu > 0);
        if (pts.length > 0) {
            H_arr   = pts.map(p => p.H);
            mur_arr = pts.map(p => p.mu);
            B_arr   = H_arr.map((H, i) => MU_0 * mur_arr[i] * H);
        }

    // --- Case 4: magnetization.Br: constant (linear permanent magnet, no explicit B-H) ---
    // Must come before the "constant mu_r" case: NdFeB presets have both mu_r:1.05
    // and magnetization.Br, so magnet detection must take priority.
    // Models the linear second-quadrant demagnetization line: B = Br + μ₀·μr·H
    // Hcb = Br / (μ₀·μr)   (coercive field where B = 0)
    } else if (props && props.magnetization && typeof props.magnetization.Br === 'number') {
        const Br_val   = props.magnetization.Br;
        const mu_r_val = (typeof mur === 'number') ? mur : 1.05;  // default recoil μr for NdFeB
        const Hcb      = Br_val / (MU_0 * mu_r_val);
        const H_max_demag = 1e6;  // extend to 1 MA/m to cover strong demagnetizing fields
        const H_min = -Math.max(Hcb, H_max_demag);
        const N = 200;
        H_arr   = Array.from({ length: N + 1 }, (_, i) => H_min + i * (-H_min) / N);
        B_arr   = H_arr.map(H => Br_val + MU_0 * mu_r_val * H);
        mur_arr = null;
        isDemag = true;
        isDashed = true;  // straight line → dashed style

    // --- Case 5: mu_r: constant (non-magnet soft material) ---
    } else if (typeof mur === 'number') {
        H_arr   = [1, 1e5];
        mur_arr = [mur, mur];
        B_arr   = H_arr.map(H => MU_0 * mur * H);
        isDashed = true;
    }

    if (!H_arr || !B_arr) {
        showPlotMessage(
            container,
            '<div style="color:#888; text-align:center; padding:40px; font-size:0.9rem;">'
                + 'No plottable B-H / μr data for this material.</div>',
            renderContext
        );
        if (toolbar) toolbar.style.display = 'none';
        return;
    }

    const plotH = Math.max(350, host.offsetHeight || 0);

    // Keep checkbox label in sync: "Log |H| axis" for demagnetization, "Log X axis" otherwise
    const logLabelNode = logCb && logCb.nextSibling;
    if (logLabelNode) logLabelNode.textContent = isDemag ? ' Log |H| axis' : ' Log X axis';

    // ---- Demagnetization curve (H < 0): single B trace ----
    if (isDemag) {
        if (toolbar) toolbar.style.display = '';  // show log/linear checkbox

        const useLogH  = logCb && logCb.checked;
        const Br_demag = B_arr[B_arr.length - 1];  // B at H=0 (remanence)

        // Find Hcb: interpolate the H where B crosses zero.
        // H_arr is sorted ascending (most-negative first); data may extend beyond Hcb.
        let Hcb_demag = H_arr[0];  // fallback: leftmost H
        for (let i = 0; i < B_arr.length - 1; i++) {
            if (B_arr[i] <= 0 && B_arr[i + 1] > 0) {
                const t = -B_arr[i] / (B_arr[i + 1] - B_arr[i]);
                Hcb_demag = H_arr[i] + t * (H_arr[i + 1] - H_arr[i]);
                break;
            }
        }

        let x_data, y_data, x_title, x_type;
        if (useLogH) {
            // Show |H| on log scale (exclude H=0)
            const filtered = H_arr.map((h, i) => [Math.abs(h), B_arr[i]])
                                   .filter(([h]) => h > 1e-12)
                                   .sort((a, b) => a[0] - b[0]);  // ascending |H|
            x_data  = filtered.map(p => p[0]);
            y_data  = filtered.map(p => p[1]);
            x_title = '|H| [A/m]';
            x_type  = 'log';
        } else {
            x_data  = H_arr;
            y_data  = B_arr;
            x_title = 'H [A/m]';
            x_type  = 'linear';
        }

        // log|H| mode: axis is reversed (Hcb on left, Br on right).
        // ax offsets are flipped vs the non-reversed case so text stays inside the plot.
        const annotations = useLogH ? [
            { x: x_data[0], y: y_data[0], xref: 'x', yref: 'y',
              text: `Br ≈ ${Br_demag.toFixed(3)} T`,
              showarrow: true, arrowhead: 2, ax: -50, ay: -20,
              font: { color: '#e05252', size: 11 } },
            { x: Math.abs(Hcb_demag), y: 0,
              xref: 'x', yref: 'y',
              text: `Hcb = ${(Math.abs(Hcb_demag) / 1000).toFixed(0)} kA/m`,
              showarrow: true, arrowhead: 2, ax: 30, ay: -25,
              font: { color: '#555', size: 11 } }
        ] : [
            { x: 0, y: Br_demag, xref: 'x', yref: 'y',
              text: `Br = ${Br_demag.toFixed(3)} T`,
              showarrow: true, arrowhead: 2, ax: 40, ay: -20,
              font: { color: '#e05252', size: 11 } },
            { x: Hcb_demag, y: 0, xref: 'x', yref: 'y',
              text: `Hcb = ${(Math.abs(Hcb_demag) / 1000).toFixed(0)} kA/m`,
              showarrow: true, arrowhead: 2, ax: 30, ay: -25,
              font: { color: '#555', size: 11 } }
        ];

        try {
            await newPlotForRender(container, [{
            x: x_data, y: y_data,
            mode: isDashed ? 'lines' : 'lines+markers',
            name: 'B [T]',
            line:   { width: 2, color: '#e05252', ...(isDashed ? { dash: 'dash' } : {}) },
            ...(isDashed ? {} : { marker: { size: 4, color: '#e05252' } }),
            hovertemplate: `${useLogH ? '|H|' : 'H'}=%{x:.4g} A/m<br>B=%{y:.4g} T<extra></extra>`
        }], {
            title:  { text: `<b>${name}</b>`, font: { size: 14 } },
            xaxis:  { title: x_title, type: x_type, exponentformat: 'power',
                      ...(useLogH ? { autorange: 'reversed' } : {}) },
            yaxis:  { title: 'B [T]', rangemode: 'tozero', exponentformat: 'power',
                      titlefont: { color: '#e05252' }, tickfont: { color: '#e05252' } },
            shapes: useLogH ? [
                { type: 'line', x0: 0, x1: 1, y0: 0, y1: 0,
                  xref: 'paper', yref: 'y',
                  line: { color: '#bbb', dash: 'dot', width: 1 } }
            ] : [
                { type: 'line', x0: 0, x1: 0, y0: 0, y1: 1,
                  xref: 'x', yref: 'paper',
                  line: { color: '#bbb', dash: 'dot', width: 1 } },
                { type: 'line', x0: 0, x1: 1, y0: 0, y1: 0,
                  xref: 'paper', yref: 'y',
                  line: { color: '#bbb', dash: 'dot', width: 1 } }
            ],
            annotations,
            margin:     { t: 45, l: 65, r: 30, b: 55 },
            height:     plotH,
            showlegend: false
            }, { responsive: true, displayModeBar: false }, renderContext);
        } catch (error) {
            if ((error && error.name === 'AbortError') ||
                !isRenderContextCurrent(renderContext, container)) return;
            console.error('B-H demagnetization plot failed:', error);
            showPlotMessage(container, '<div style="color:#c62828; padding:40px; text-align:center;">Failed to render B-H plot</div>', renderContext);
        }
        return;
    }

    // ---- Normal B-H / μr curve ----
    const markerSize = (Array.isArray(bh) || Array.isArray(mur)) ? 4 : 0;
    const mode = isDashed ? 'lines' : (markerSize > 0 ? 'lines+markers' : 'lines');

    // B on left axis (y1), μr on right axis (y2)
    // For log X axis: skip H=0 point (Plotly handles gracefully but avoids log(0))
    const useLog = xtype === 'log';
    const hFilter = (_, i) => !useLog || H_arr[i] > 0;

    const H_plot    = H_arr.filter(hFilter);
    const B_plot    = B_arr.filter((_, i) => hFilter(null, i));
    const mur_plot  = mur_arr.filter((_, i) => hFilter(null, i));

    const traceB = {
        x: H_plot, y: B_plot,
        mode, name: 'B [T]', yaxis: 'y1',
        line: { width: 2, color: '#e05252', dash: isDashed ? 'dash' : 'solid' },
        ...(markerSize > 0 ? { marker: { size: markerSize, color: '#e05252' } } : {}),
        hovertemplate: 'H=%{x:.4g} A/m<br>B=%{y:.4g} T<extra></extra>'
    };

    const traceMur = {
        x: H_plot, y: mur_plot,
        mode, name: 'μr', yaxis: 'y2',
        line: { width: 2, color: '#667eea', dash: isDashed ? 'dash' : 'solid' },
        ...(markerSize > 0 ? { marker: { size: markerSize, color: '#667eea' } } : {}),
        hovertemplate: 'H=%{x:.4g} A/m<br>μr=%{y:.4g}<extra></extra>'
    };

    try {
        await newPlotForRender(container, [traceB, traceMur], {
        title:  { text: `<b>${name}</b>`, font: { size: 14 } },
        xaxis:  { title: 'H [A/m]', type: xtype, exponentformat: 'power' },
        yaxis:  { title: 'B [T]', exponentformat: 'power',
                  titlefont: { color: '#e05252' }, tickfont: { color: '#e05252' } },
        yaxis2: { title: 'μr', overlaying: 'y', side: 'right', exponentformat: 'power',
                  titlefont: { color: '#667eea' }, tickfont: { color: '#667eea' } },
        margin:     { t: 45, l: 65, r: 70, b: 55 },
        height:     plotH,
        legend:     { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: '#ccc', borderwidth: 1 },
        showlegend: true
        }, { responsive: true, displayModeBar: false }, renderContext);
    } catch (error) {
        if ((error && error.name === 'AbortError') ||
            !isRenderContextCurrent(renderContext, container)) return;
        console.error('B-H plot failed:', error);
        showPlotMessage(container, '<div style="color:#c62828; padding:40px; text-align:center;">Failed to render B-H plot</div>', renderContext);
        return;
    }

    if (toolbar && isRenderContextCurrent(renderContext, container)) toolbar.style.display = '';
}

// Called when the log/linear checkbox changes.
function onLibBHAxisChange() {
    if (AppState.currentBHMaterial) {
        requestBHCurveRender(AppState.currentBHMaterial.name, AppState.currentBHMaterial.props);
    }
}

function uploadLibrary() {
    document.getElementById('libraryUpload').click();
}

async function newLibrary() {
    const name = prompt('New library filename (must end with .yaml):');
    if (!name) return;
    const filename = (name.endsWith('.yaml') || name.endsWith('.yml')) ? name : name + '.yaml';

    AppState.currentLibraryFile = filename;
    if (AppState.libraryAceEditor) {
        AppState.libraryAceEditor.setValue(LIB_TEMPLATE, -1);
    }
    document.getElementById('libEditFilename').textContent = filename + '  (unsaved)';
    document.getElementById('libDeleteBtn').style.display = 'none';
    document.getElementById('libSaveBtn').style.display = 'inline-block';
    document.getElementById('libUseBtn').style.display = 'none';
}

async function handleLibraryUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = '';

    const formData = new FormData();
    formData.append('library', file);
    formData.append('userId', AppState.userId);

    try {
        const response = await fetch('/api/material-libraries', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (!result.success) throw new Error(result.error);
        await refreshLibraryList();
        await selectLibraryFile(result.filename);
        document.getElementById('libraryListStatus').textContent = 'Uploaded successfully';
    } catch (e) {
        document.getElementById('libraryListStatus').textContent = `Upload error: ${e.message}`;
    }
}

async function saveLibrary() {
    if (!AppState.libraryAceEditor) return;

    let filename = AppState.currentLibraryFile;
    if (!filename) {
        filename = prompt('Save as (must end with .yaml):');
        if (!filename) return;
        if (!filename.endsWith('.yaml') && !filename.endsWith('.yml')) filename += '.yaml';
        AppState.currentLibraryFile = filename;
    }

    const content = AppState.libraryAceEditor.getValue();
    try {
        const response = await fetch(`/api/material-libraries/${encodeURIComponent(filename)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, userId: AppState.userId })
        });
        const result = await response.json();
        if (!result.success) throw new Error(result.error);
        document.getElementById('libEditFilename').textContent = filename;
        document.getElementById('libDeleteBtn').style.display = 'inline-block';
        document.getElementById('libUseBtn').style.display = 'inline-block';
        document.getElementById('libraryListStatus').textContent = 'Saved';
        await refreshLibraryList();
    } catch (e) {
        document.getElementById('libraryListStatus').textContent = `Save error: ${e.message}`;
    }
}

async function deleteLibrary() {
    const filename = AppState.currentLibraryFile;
    if (!filename) return;
    if (!confirm(`Delete "${filename}"?`)) return;

    try {
        const response = await fetch(`/api/material-libraries/${encodeURIComponent(filename)}?userId=${AppState.userId}`, {
            method: 'DELETE'
        });
        const result = await response.json();
        if (!result.success) throw new Error(result.error);

        AppState.currentLibraryFile = null;
        if (AppState.libraryAceEditor) AppState.libraryAceEditor.setValue('', -1);
        document.getElementById('libEditFilename').textContent = 'No file selected';
        document.getElementById('libDeleteBtn').style.display = 'none';
        document.getElementById('libSaveBtn').style.display = 'none';
        document.getElementById('libUseBtn').style.display = 'none';

        if (AppState.selectedLibrary === filename) {
            clearActiveLibrary();
        }
        await refreshLibraryList();
    } catch (e) {
        document.getElementById('libraryListStatus').textContent = `Delete error: ${e.message}`;
    }
}

function setActiveLibrary() {
    const filename = AppState.currentLibraryFile;
    if (!filename) return;
    AppState.selectedLibrary = filename;
    document.getElementById('activeLibraryName').textContent = filename;
    document.getElementById('activeLibraryBadge').style.display = 'inline-flex';
    // Persist to cookie so it auto-loads on next visit
    setCookie('magfdm_last_library', filename, 365);
    showStatus('configStatus', `Material library set: ${filename}`, 'success');
}

function clearActiveLibrary() {
    AppState.selectedLibrary = null;
    document.getElementById('activeLibraryBadge').style.display = 'none';
    // Clear cookie
    setCookie('magfdm_last_library', '', -1);
}
