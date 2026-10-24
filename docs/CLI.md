# CLI command manual

This manual covers the resident server console, startup flags and standalone 2D
solver. See [Operations](OPERATIONS.md) for deployment and [REST API](REST_API.md)
for HTTP automation. The release baseline is v1.7.3.

## 1. Startup

| Installation | Command |
|---|---|
| Packaged Windows | `webui-server.exe` |
| Packaged Linux/macOS | `./webui-server` |
| Source, repository root | `npm --prefix webui ci`, then `node webui/launcher.js` |
| Source, webui directory | `npm ci`, then `npm start` |

With a terminal attached, the server starts and opens `OpenMagFDM>`. Commands
below are entered at this prompt; do not type the prompt itself. The default
WebUI address is `http://localhost:3000`.

```text
help
status
menu
configure
```

`menu` offers arrow-key actions, including Storage and explicit job history;
`configure` guides input-file, result-folder and result-storage limits.
`storage` opens a guided storage menu. It can stop an idle
server, copy/verify data to a new directory and restart after confirmation.
It refuses to interrupt active analyses. Setting a path without migration
does not copy data; the menu states this before applying the change.
Tab completes commands; Up/Down recalls up to 100 commands from this session.
History stays in memory. Quote paths containing spaces, for example `"D:\Analysis Results"`.
Ctrl+C cancels a form, clears a partial command, or requests normal exit on an
empty command line. Ctrl+D/EOF closes the console and terminates active analyses.
Passwords use separate masked prompts and are excluded from command history.

## 2. Server control

| Command | Effect |
|---|---|
| `help`, `/help`, `?`, `--help`, `-h` | Show console help |
| `menu` | Choose an action |
| `configure` | Configure default or per-workspace input/result limits |
| `storage` | Guided storage location or verified migration |
| `status` | Show PID, address, readiness, activity and unfinished jobs as JSON |
| `start` | Start the stopped server |
| `stop [--force]` | Stop the server, keeping the console open |
| `restart [--force]` | Stop and start the server |
| `exit [--force]`, `quit [--force]` | Stop the server and close the console |
| `<group> --help` | Show administration arguments, e.g. `limits --help` |

Normal stop/restart/exit refuses active analyses or input preparation.
`--force` explicitly terminates analyses. Queued jobs are held after a restart;
use `jobs resume`. Startup failure leaves the console open to correct settings
and retry `start`.

## 3. Jobs (running server)

| Command | Effect |
|---|---|
| `jobs`, `jobs list` | Show up to 100 unfinished jobs across workspaces |
| `jobs list <workspace>` | Filter unfinished jobs by workspace |
| `jobs list [workspace] --all` | Include finished history |
| `jobs cancel <workspace> <jobId>` | Request cancellation |
| `jobs resume <workspace>` | Release jobs held after restart |

The console lists unfinished jobs by default. Held jobs remain visible for resumption;
completed, failed, cancelled and interrupted jobs require `--all`.

```text
jobs list user_example
jobs list user_example --all
jobs cancel user_example 01234567-89ab-cdef-0123-456789abcdef
jobs resume user_example
```

Submit new analyses through WebUI, REST or MCP. The local server console is an
administrator interface and can manage jobs across workspaces.

## 4. Accounts (live changes)

| Command | Effect |
|---|---|
| `users list` | List names, roles and disabled state |
| `users create <name> [--admin]` | Create an account with private password prompts |
| `users disable <name>` | Stop new operations and queued job starts |
| `users enable <name>` | Enable an account |
| `users reset-password <name>` | Set a new password with confirmation |

Passwords require at least 12 characters and at most 1,024 UTF-8 bytes, and use
salted scrypt. The last enabled administrator cannot be disabled. Disabling an
account retains its data and running analyses. Password changes and disabling
invalidate its Web sessions.

## 5. Workspaces (live changes)

| Command | Effect |
|---|---|
| `workspaces list` | Scan storage directories and list workspace owners; unassigned is `(unassigned)` |
| `workspaces assign <workspace> <account>` | Assign an enabled account |

Workspace IDs correspond to existing browser `userId` directories and use letters,
digits, `_` and `-`. This list contains workspaces, not one entry per job; it does
not use `jobs.json` to discover directories. Shared-mode users cannot access
unassigned workspaces. Assignment and disabling do not delete or move files.

## 6. File limits (live changes)

```text
limits show [workspace]
limits set-default <configs|images|libraries|results|result-mib> <value|unlimited>
limits set <workspace> <category> <value|unlimited|default>
```

Defaults are 20 configuration files, 20 input images and unlimited material
libraries per workspace. Counts include nested files in that input category,
not generated results or frozen run inputs. Limits apply to WebUI, MCP and
backup imports. A count is a nonnegative integer; `0` blocks creation.
`unlimited` removes the cap; `default` removes a workspace override and is only
accepted by `limits set`. Existing files remain and can be replaced at the cap.

Current source adds `results` (top-level result-folder count) and `result-mib`
(stored result bytes in integer MiB), both unlimited by default. Queued input
snapshots reserve a result folder. The byte budget includes snapshots and
field outputs and is measured from actual directories, without a quota index.
These limits apply to common WebUI/REST/MCP job acceptance, queued starts,
result writes and backup imports. MCP backup uploads reserve their declared
archive size against `result-mib` before any bytes are transferred; pending
uploads share the same budget. A queue held by a lowered quota can be
resumed after adjusting limits or removing unneeded results.

The storage budget is an **admission limit**: a running native solver may
grow beyond it, and is not automatically cancelled. Existing data is never
removed to satisfy a quota. Smaller replacements and deletion remain possible
above a lowered cap. A partially applied backup import reports applied IDs.

```text
limits set-default configs 100
limits set user_example images 200
limits set user_example libraries unlimited
limits set user_example images default
limits show user_example
limits set user_example results 100
limits set user_example result-mib 10240
```

## 7. Storage (stop before changes)

```text
storage show
storage set-workspace-root <absolute-directory>
storage migrate-workspaces <absolute-empty-directory>
storage set-results-root <absolute-directory>
storage migrate-results <absolute-empty-directory>
```

`storage show` prints the workspace root, each resolved category path and any results override.
The default workspace root is the application directory, with `configs`,
`uploads`, `user-libs` and `outputs` beneath it.

```text
stop
storage migrate-workspaces "E:\OpenMagFDM Data"
start
```

`set-workspace-root` changes settings only. It does not move existing data, and
preserves a separate results override. `migrate-workspaces` copies all four
categories from their currently resolved locations, including separately stored
results, verifies file counts/hashes, then switches to the new common root and
clears the results override. Original data is retained.

The two results commands affect only results. `set-results-root` changes settings
only; `migrate-results` copies and verifies before switching. A migration needs
an empty destination that does not overlap any source. Private server state and
public web assets cannot be used as data destinations. On failure, settings stay
unchanged and partial copies remain for inspection. Retry with another empty
destination after resolving the problem. Changes take effect at the next `start`.
See [Operations](OPERATIONS.md#upgrades-and-relocation).

## 8. Network (stop before changes)

```text
network local
network shared --cert <path> --key <path> [--host address]
```

Create an enabled administrator before enabling sharing. Provide a readable TLS
certificate/key and configure client trust. Shared mode defaults to `0.0.0.0`,
requires HTTPS and requires login even from localhost. `network local` restores
loopback-only HTTP. Set the WebUI port through `PORT` before server startup.

## 9. Startup flags and automation

These are OS-shell arguments, not console commands:

| Argument | Effect |
|---|---|
| None | Resident console with a TTY; server only otherwise |
| `--no-console` | Server only, including with a TTY |
| `--console` | Explicit console, including through pipes; forms still need a TTY |
| `help`, `--help`, `-h` | Print administration help and exit |
| `--runtime-info` | Print embedded Node version, OS, architecture and packaging information |
| `--mcp-admin-code` | Print the running installation's pairing code |
| `--mcp-stdio --connection <id>` | Bridge to an existing MCP server |
| `users`, `workspaces`, `limits`, `storage`, `network` | One-shot administration |

One-shot storage/network/limit mutations require the server to be stopped.
Use the resident console for live administration. `status`, `menu`, `jobs`,
`start` and `stop` are console commands. One-shot success exits 0; errors exit nonzero.

```powershell
.\webui-server.exe --runtime-info
.\webui-server.exe limits show
$env:PORT = '3100'
.\webui-server.exe --no-console
```

Environment variables are documented in [Operations](OPERATIONS.md#environment-and-resources).
MCP disconnection/OFF and WebUI logout do not cancel analyses. Server shutdown
terminates child solvers.

## 10. Standalone 2D solver

```text
MagFDMsolver[.exe] <config.yaml> <image.png> [output_folder]
MagFDMsolver[.exe] --version
```

```powershell
.\MagFDMsolver.exe .\sample_config.yaml .\model.png .\output_example
.\MagFDMsolver.exe --version
```

Linux/macOS use `./MagFDMsolver sample_config.yaml model.png output_example`.
Omitting the output directory creates a timestamped directory. Missing arguments
print usage and exit nonzero. Version output is `OpenMagFDM 1.7.3`.
The image defines RGB material regions; see [Configuration](CONFIGURATION.md).
Direct standalone runs do not enter the WebUI/MCP job ledger.
