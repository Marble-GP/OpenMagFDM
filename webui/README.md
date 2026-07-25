# OpenMagFDM WebUI v1.6.1

The WebUI is the browser-based editor, preprocessor, solver launcher and result
viewer for OpenMagFDM. It runs on Node.js/Express and is also distributed as a
standalone executable built with `pkg`.

## Start

```bash
cd webui
npm ci
npm start
```

Open `http://localhost:3000`. The server locates `MagFDMsolver` beside the
package or in the configured development paths. Use `GET /api/health` and
`GET /api/solver/info` to diagnose the runtime.

## Main workflows

- Edit and validate YAML with Ace completion from `public/yaml-schema.json`.
- Upload an image, inspect physical/image properties and detect material RGBs.
- Uniformise anti-aliased CAD screenshots before analysis.
- Detect and interactively correct polar geometry, then warp and insert YAML.
- Generate a Cartesian/Linear template directly from an existing strip image.
- Preview permanent-magnet directions, including polar coordinate conversion.
- Launch/stop analyses and follow streaming logs.
- View TIFF or CSV fields, flux linkage, energy, force and torque timelines.
- Manage configurations, uploaded analysis images, material libraries and
  result folders together from the File Manager tab, including select-all and
  bulk deletion across file categories.
- Export/import selected user content as a standard ZIP backup from File
  Manager; the YAML editor remains focused on editing configuration content.

## v1.6.1 behaviours

- Polar `r_start/r_end` are physical metres; `r_start: 0` represents a full
  disc and requires a Dirichlet inner boundary.
- Decimal slide-region bounds are metres; integers are legacy pixel indices.
- Generated nonlinear templates enable Eisenstat-Walker and use
  `max_iterations: 100`, `tolerance: 1.0e-3`.
- Detected air-gap sliding is enabled by default when an air-gap candidate is
  available.
- Input Image and Magnetisation Preview support wheel zoom, middle-drag pan,
  reset and image-pixel rulers.
- Decoded field data uses a 256 MiB byte-budgeted LRU cache. Memory-heavy field
  fetch/decode work is serialized and cache, staging payloads and decode
  headroom share an estimated 384 MiB field-data working budget. In-flight
  loads are leased per viewer and the physical request is cancelled when its
  final viewer leaves. Preload stops at the budget instead of retaining every
  transient step.
- Coarsening masks use compact byte arrays and retain at most four completed
  result/step entries within a 32 MiB cache budget. Dashboard and File Manager
  consumers use independent cancellation, and deleted/redrawn plots release
  their Plotly state.
- Solver children are tracked until their `close` event and are stopped on an
  interrupted stream or server shutdown. The default process limits are two
  solver processes globally and one active solve per user.

Long-running server retention is bounded by default: at most 200 completed job
records are kept for 24 hours, with 256 KiB of log text per job. The principal
deployment overrides are `MAX_SOLVER_PROCESSES`, `MAX_CONCURRENT_JOBS`,
`MAX_CONCURRENT_JOBS_PER_USER`, `MAX_SOLVER_START_OPERATIONS`,
`MAX_SOLVER_START_OPERATIONS_PER_USER`, `MAX_JOB_RECORDS`,
`MAX_JOB_LOG_CHARS`, and `JOB_TTL_MS`. Solver preparation is reserved before
asynchronous YAML/file work begins. Legacy non-streaming responses retain at
most `LEGACY_RESPONSE_LOG_CHARS` characters per returned stream and are bounded
by `MAX_LEGACY_RESPONSES` until the response finishes or closes. Raise process
limits only after measuring solver peak memory on the target model.

Windows `Cached`/standby memory and the paged/nonpaged kernel pools are managed
by Windows, not by the WebUI cache. OpenMagFDM deliberately does not call
working-set or system-cache purge APIs: those counters may remain high while
their pages are reclaimable, and forced trimming causes avoidable rereads and
page faults. Diagnose application retention with each process's Private Bytes,
the number of live `MagFDMsolver` children, and whether those values return
after the run; the limits and lifecycle cleanup above address those owned
resources.

The authoritative YAML contract is in `../docs/CONFIGURATION.md`.

## Development rules

- Keep `public/yaml-schema.json`, generated templates in `public/app.js`,
  `sample_config.yaml`, and `docs/CONFIGURATION.md` synchronized.
- Add no server dependency without checking Node 18 `pkg` compatibility.
  ESM-only transitive dependencies have broken standalone builds before.
- After JavaScript edits run:

```bash
node --check server.js
node --check public/app.js
node -e "JSON.parse(require('fs').readFileSync('public/yaml-schema.json','utf8'))"
node --test tests/process_lifecycle.test.js
```

- Prefer browser-side decoding for TIFF; `public/lib/geotiff.js` is the shipped
  UMD bundle.
- User data lives under `configs/`, `uploads/`, `user-libs/` and `outputs/` and
  must never be committed or included unintentionally in packages.

## API groups

- Configuration: `/api/config*`, `/api/validate-config`
- Images/preprocess: `/api/images*`, `/api/materials/detect`,
  `/api/preprocess-filter/*`, `/api/preprocess-polar/*`
- Solver/jobs: `/api/solve*`, `/api/stop-solver`, `/api/jobs*`
- Results/files: `/api/results*`, `/api/user-files`, `/api/user-outputs*`, `/api/load-field`,
  `/api/get-flux-linkage`
- Libraries/backups: `/api/material-libraries*`, `/api/backup-manifest`,
  `/api/export`, `/api/import`

All user-supplied paths must stay basename/path-normalized and constrained to
the current user's server directories.
