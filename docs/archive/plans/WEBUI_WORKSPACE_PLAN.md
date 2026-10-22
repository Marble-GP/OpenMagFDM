# WebUI workspace refresh

## Goal and boundaries

Make the workflow explicit: edit inputs → review the actual saved configuration →
run → inspect results → manage files. Preserve existing image preprocessing,
material libraries, plots, file operations and bounded-memory solver lifecycle.
No solver rebuild, long benchmark, deployment or release in this work.

## Results usability follow-up — 2026-10-17

- Result selection, load/reload, polar display options and format diagnostics
  now live in Results. Model & Run links directly to the loader. The empty
  canvas guides users to load data, run an analysis or import a backup.
- The loaded result is distinct from the selection awaiting Load. Load errors
  and list-refresh errors are shown locally in Results. A failed validation
  retains/re-finalizes the committed result; stale requests cannot overwrite
  the latest selection, loaded state or notice.
- Every plot identifies its result and field step, or its actually rendered
  timeline point count/range. Coordinates and polar display mode are included.
  During updates, old plots are hidden (without destroying layout dimensions)
  until the generation-checked render completes. Missing data/error messages
  are not labelled as successfully plotted data. No new field/trace cache.
- The palette supports click, Enter/Space and drag-and-drop. The result picker
  supports arrows, Home/End, Enter/Space and Escape. Removing the last plot
  restores the empty-state actions. Step/animation controls are unavailable
  until data is loaded.
- **Refresh list** only refreshes available result names. **Reload loaded data**
  reloads the committed result even if another name is selected, refreshes
  field caches and detected steps, pauses animation and preserves the current
  step within the new bounds. Timeline rendering continues to read fresh data.
- `node --test --test-concurrency=1 tests/*.test.js` from `webui`: **83/83 passed**
  (15 new Results tests), including real load/render lifecycle races, retained
  result on failed validation, cache-refresh targeting, timeline counts, safe
  text rendering and result-list refresh races. Syntax checks passed.
- Browser automation currently exposes no browser surfaces; desktop/narrow
  layout and real pointer/keyboard operation remain a manual acceptance gate.
  This follow-up does not change solver code, MCP permissions, release defaults,
  tags, or merge state.

## Implementation slices

1. Responsive workspace shell: persistent workflow navigation, restrained
   light surfaces, readable controls, focus indicators and workspace context.
2. Editing safety: loaded filename separate from dropdown selection, dirty
   indicator, guarded loading, Ctrl/Cmd+S and a save-before-run flow. Failed
   saves must prevent a run; asynchronous loads must not overwrite new edits.
3. Problems panel and execution review: jump-to-line diagnostics and a compact
   summary identifying config, image, material library and solver settings.
4. File discovery: search and sorting while preserving visible/all selection
   semantics and existing batch operations.
5. Automation foundation: shared pure validation/summary module, a versioned
   read-only inspection API and documented stable operation contracts. No
   browser-only validation rule required for an external agent to inspect YAML.
6. Automated safety/integration tests plus browser checks at desktop and narrow
   widths. Existing lifecycle and IntelliSense tests must continue to pass.

## MCP compatibility, not an unauthenticated MCP endpoint

Update (2026-10-15): embedded MCP implementation and remaining acceptance checks
are tracked separately in [MCP integration](../../MCP_INTEGRATION.md). The following
paragraph records the original UI foundation scope.

Existing user IDs are namespaces, not authentication credentials. Do not expose
the local REST server to remote agents as an authenticated multi-user service.
Future MCP adapter should use a local stdio transport initially, explicitly
bound workspace/user scope, versioned tool schemas and no arbitrary paths or
shell commands. Config inspection is read-only; starting a job, cancelling one,
writing a config and deleting files are separate capabilities. Destructive
actions require explicit target/confirmation semantics, not GUI dialog scraping.

Use existing `/api/jobs` lifecycle rather than introducing a second process
manager. Before a write-capable adapter is shipped: add optimistic revision
checks, idempotent start requests, capability restrictions, bounded result
pagination and audit events. These are follow-up work, not claimed implemented
by a visual refresh. No full remote-access security claim is made here.

## Follow-up gates

Full write-capable MCP adapter, result A/B comparison, recoverable deletion,
offline vendoring of remaining CDN libraries and full modal accessibility
are separate follow-ups. Preserve inputs and controls rather than replacing
working specialist tools with unfinished new views.

## Implemented foundation — 2026-10-13

- New responsive shell and restrained light/teal design, persistent workflow
  navigation, configuration/image/run/result context, problem list with line
  navigation, execution review, visible focus states and reduced-motion rules.
- Dirty indication, configuration-switch/unload guards, Ctrl/Cmd+S, explicit
  save-and-run / run-saved / cancel dialog. Loaded filename is independent of
  dropdown state. Failed saves block save-and-run. Concurrent saves are rejected;
  late file loads cannot overwrite newer loads or edits made in flight.
- File search and name/date/size sorting preserve existing visible selection
  and batch controls. No preload or additional field-result cache introduced.
- Shared `workspace-core.js` and read-only `/api/config/inspect` endpoint.
  See [automation contract](../../AUTOMATION_CONTRACT.md). Existing solver/job process
  management is retained, not replaced by a parallel implementation.
- `npm --prefix webui test`: **49/49 passed** (12 lifecycle, 22 IntelliSense,
  15 workspace/API). JavaScript syntax checks and `git diff --check` passed.
  Tests include save failure, edit/load races, duplicate saves, explicit run
  choices, missing inputs, nonmutating file sort and GUI/API inspection parity.

### Not yet verified / implemented

The browser-use tool returned no available browsers; desktop and narrow-width
visual checks could not run. Responsive CSS is implemented but **not visually
accepted** yet. Check all tabs (including the new Integrations page), dialogs, image preprocessing, plot resize,
keyboard focus, completion popup, and the save/run dialog before adoption.
No solver was started for UI verification. No executable rebuild was performed.
Computer-use skill guided the attempted browser verification and its limitation
is recorded rather than substituted with a claim of visual success.

The refresh is not a completed full-app accessibility audit. At the original
UI checkpoint MCP transport/authentication/write capabilities were future work;
their subsequent implementation and remaining acceptance checks are described
in [MCP integration](../../MCP_INTEGRATION.md). Result comparison and specialist modal
modernization remain follow-ups. On 2026-10-15 the combined source tests passed
68/68 before recording the UI/MCP checkpoint; browser acceptance is still pending.
The prior benchmark files and binaries were left intact.
