# WebUI workspace refresh

## Goal and boundaries

Make the workflow explicit: edit inputs → review the actual saved configuration →
run → inspect results → manage files. Preserve existing image preprocessing,
material libraries, plots, file operations and bounded-memory solver lifecycle.
No solver rebuild, long benchmark, deployment or release in this work.

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
are tracked separately in [MCP integration](MCP_INTEGRATION.md). The following
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
  See [automation contract](AUTOMATION_CONTRACT.md). Existing solver/job process
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
in [MCP integration](MCP_INTEGRATION.md). Result comparison and specialist modal
modernization remain follow-ups. On 2026-10-15 the combined source tests passed
68/68 before recording the UI/MCP checkpoint; browser acceptance is still pending.
The prior benchmark files and binaries were left intact.
