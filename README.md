# OpenMagFDM

OpenMagFDM is a 2D magnetic field analysis tool. RGB images define material
regions, a C++17 finite difference solver performs the calculation, and a WebUI
provides configuration editing, image preprocessing and result visualization.
It supports Cartesian and polar coordinates, nonlinear B-H materials,
permanent magnets, and transient analyses with prescribed motion.

**Latest published release: v1.7.2** · [Release notes](docs/releases/RELEASE_NOTES_v1.7.2.md) ·
[Changelog](CHANGELOG.md) · [Documentation](docs/README.md)

## Development changes (unreleased)

The current source adds whole-workspace storage configuration and verified
migration, clearer deletion failures, large-result deletion fixes, active-only
console job lists with optional history, and English public interfaces and manuals.
These changes are not included in the published v1.7.2 binaries.

## v1.7.2

Starting the server in a terminal opens a resident `OpenMagFDM>` console.
Use `help`, `menu` and `configure` to manage accounts, workspace ownership and
file limits while the server runs. Tab completion, session history, private
password prompts, job administration and stop/restart commands are included.
See the [CLI manual](docs/CLI.md) for all arguments.

Existing 2D YAML files and results remain compatible. Packaged WebUI binaries
embed **Node.js 24.22.1**. Experimental GPU rendering and the separate 3D solver
are excluded from the stable release packages.

## Download and start

Download packages from [GitHub Releases](https://github.com/Marble-GP/OpenMagFDM/releases).

| Platform | Installation |
|---|---|
| Windows x86_64 | Use the installer, or extract the Windows solver and WebUI ZIPs into the same directory. Start `webui-server.exe`. |
| Linux x86_64 | Install the Linux solver and WebUI together; run `./webui-server`. |
| macOS | Install the arm64 solver and WebUI together. The current WebUI archive is Intel x64 and requires Rosetta 2 on Apple Silicon. |

The packaged WebUI does not require a separate Node.js installation. WebUI-only
archives do not include the C++ solver. Retain the solver's runtime libraries;
the Windows solver also requires an AVX2-capable CPU.

Open `http://localhost:3000`. Local mode requires no WebUI login.
The server terminal accepts commands after startup:

```text
OpenMagFDM> help
OpenMagFDM> menu
OpenMagFDM> configure
OpenMagFDM> status
OpenMagFDM> limits set-default configs 100
OpenMagFDM> jobs list
```

Use `exit` to shut down. Normal exit refuses active analyses; `exit --force`
explicitly terminates them. Use `--no-console` for service deployments.

## First analysis

1. Save a material image and YAML configuration in the WebUI.
2. If needed, quantize image colors, detect material RGB values, or convert to polar coordinates.
3. Review dimensions, materials, current sources, magnetization and boundary conditions; submit from Run.
4. Inspect progress and logs, then view fields, field lines and time series in Results.
5. Save a description and export the required inputs/results as a ZIP backup.

Jobs continue on the server after a browser reload, tab close or logout.
Reconnect to restore controls. Stopping the server also stops its solver processes.
After a server restart, queued jobs are held until `jobs resume <workspace>`.

The solver is detected from standard installation layouts. If necessary, save
its absolute path in **Help & Integrations → Local startup settings** and restart.

## Manuals

| Topic | Manual |
|---|---|
| YAML, materials, formulas, boundaries and output | [Configuration](docs/CONFIGURATION.md) |
| Resident console, administration, startup flags and standalone solver | [CLI](docs/CLI.md) |
| HTTP authentication, files, jobs, fields, images and backups | [REST API](docs/REST_API.md) |
| HTTPS sharing, storage, ownership, upgrades and backups | [Operations](docs/OPERATIONS.md) |
| Local agent integration over HTTP or stdio | [MCP](docs/MCP_INTEGRATION.md) |
| Historical investigations, measurements and plans | [Archive](docs/archive/README.md) |

## Features

- RGB material images, shared material libraries, variables and expressions.
- Cartesian/polar domains, periodic/anti-periodic boundaries and prescribed magnetization.
- Nonlinear B-H materials, Newton-Krylov, AMGCL, OpenMP and optional domain decomposition.
- Transient prescribed rotation/motion, flux linkage, force and torque evaluation.
- TIFF/CSV field output, distribution/time-series plots and Az field lines without input images.
- Shared WebUI/REST/MCP job management, frozen inputs and revision-checked mutations.
- Per-workspace file limits, account ownership and explicit HTTPS sharing.

<a id="mcp-setup"></a>

## MCP setup

Local agents can access a fixed assigned workspace with selected permissions.
MCP is disabled initially.

1. Pair the browser in **Help & Integrations** using the code printed in the server terminal.
2. Check **Assigned workspace** and create a connection, initially with read permission.
3. Enable MCP, apply the setting, and run the connection's **Test**.
4. Register the displayed HTTP URL/Bearer token or stdio command/arguments in your client.

The default MCP URL is `http://127.0.0.1:3001/mcp`. Stdio bridges to the running
server; it must use the same OS account and installation. See the [MCP manual](docs/MCP_INTEGRATION.md)
for setup, sharing-mode limitations and troubleshooting.

## Build from source

Use the Node version in [.node-version](.node-version):

```powershell
cd webui
npm ci
npm start
```

The C++ solver depends on Eigen3, OpenCV, yaml-cpp and libtiff. Use Visual Studio
and vcpkg on Windows, or native development packages on Linux/macOS.
AMGCL and tinyexpr are included in the source tree.

```text
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Standalone usage is `MagFDMsolver <config.yaml> <image.png> [output_folder]`.
See [CLI](docs/CLI.md#10-standalone-2d-solver) and [Configuration](docs/CONFIGURATION.md).
Output defaults to double-precision TIFF with asynchronous writing; CSV remains available.

## Development and support

Run WebUI regressions with `npm --prefix webui test`. Release validation covers
native C++ tests, packaged Node HTTP/MCP checks and real terminal input on three OSes.
Public interfaces, current manuals and release notes are written in English;
archived historical records retain their original language.

Report problems through [GitHub Issues](https://github.com/Marble-GP/OpenMagFDM/issues)
with version, OS and reproduction steps. Remove credentials from shared logs/configurations.
Dependency licenses are included with their source and vendor files.
