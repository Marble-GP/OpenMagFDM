# OpenMagFDM v1.7.2

OpenMagFDM v1.7.2 adds a resident server administration console. You can enter
commands after startup, adjust file limits and manage accounts while the WebUI
continues running, and stop or restart the server from the same terminal.

## Highlights

- **Interactive server console:** `help`, arrow-key menus, guided file-limit
  forms, Tab completion, session history and masked password entry.
- **Live administration:** account, workspace-owner and file-limit changes
  apply immediately. File limits remain per workspace and preserve existing files.
- **Job and lifecycle controls:** inspect, cancel and resume durable jobs;
  use `stop`, `start`, `restart` and `exit` without leaving the console.
  Normal shutdown refuses active analysis work; `--force` explicitly terminates it.
- **Clean shutdown:** stdin EOF and supervisor loss trigger child-server cleanup.
  Headless startup, `--no-console`, existing one-shot commands and MCP stdio remain supported.
- **Current documentation:** refreshed README, complete CLI and REST API manuals,
  server operations and MCP setup. Historical plans, investigations, measurements
  and earlier implementation records are indexed under `docs/archive/`.

## Start using the console

Start `webui-server.exe` on Windows or `./webui-server` on Linux/macOS in a terminal.
For a source checkout, run `npm ci` and `npm start` in `webui`.

```text
OpenMagFDM> help
OpenMagFDM> menu
OpenMagFDM> configure
OpenMagFDM> limits set-default configs 100
OpenMagFDM> jobs list
```

Account and file-limit commands work while the server runs. To change storage
or HTTPS settings, use `stop`, enter the settings command, then `start`.
See the [CLI manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.2/docs/CLI.md),
[REST API manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.2/docs/REST_API.md)
and [operations manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.2/docs/OPERATIONS.md).

## Upgrading from v1.7.1

Stop the server before replacing binaries and public assets. Preserve your
configuration, image, material-library and result directories, as well as the
OS-protected account, job and MCP settings. Keep the same installation path to
continue using its existing management profile. Inspect held jobs before resuming them.

Existing 2D YAML files and result data require no conversion. The 2D numerical
method is unchanged. Experimental GPU rendering and 3D features are excluded
from this maintenance release.

## Downloads and runtime

- Windows x86_64: installer, or separate solver and WebUI ZIP archives.
- Linux x86_64: solver and standalone WebUI archives.
- macOS: arm64 solver archive; the standalone WebUI archive targets Intel x64
  and requires Rosetta 2 on Apple Silicon.
- WebUI-only archives require a separately installed solver and its native dependencies.
- Packaged WebUI embeds **Node.js 24.22.1**; a separate Node installation is unnecessary.
  Windows solver binaries require an AVX2-capable CPU.

## Validation

The console implementation passed 103 Node regression tests, dependency auditing
with no reported vulnerabilities, and source/packaged terminal, HTTP, MCP and
image-API checks on Windows, Linux and macOS before release preparation.
Terminal tests cover menus, forms, completion, history, password hashing and
cancellation, resizing, input recovery during logging and shutdown.
Windows native-solver checks cover active-analysis shutdown protection and forced cancellation.

The tagged release workflow reruns native-package checks and builds and tests the
C++ solver on all three platforms before assembling the downloadable artifacts.
These automated checks do not represent a complete manual review of every browser
screen or a long-duration memory/performance qualification.
