# OpenMagFDM WebUI v1.7.3

The browser editor, image preprocessor, analysis manager and result viewer for
the OpenMagFDM 2D solver. The packaged executable embeds Node.js 24.22.1.
For a source checkout, use the version in [`.node-version`](../.node-version).

## Start

```bash
cd webui
npm ci
npm start
```

Open `http://localhost:3000`. With a terminal attached, the server opens the
resident `OpenMagFDM>` console. Enter `help`, `menu`, `configure` or `status`.
Use `--no-console` for service operation. The standalone package starts with
`webui-server.exe` on Windows or `./webui-server` on Linux/macOS.
Install the C++ solver and its native dependencies separately when using the
WebUI-only archive.

## Manuals

- [CLI commands and startup arguments](../docs/CLI.md)
- [REST API: authentication, revisions, jobs, results, images and backups](../docs/REST_API.md)
- [Server operation, sharing, storage and upgrades](../docs/OPERATIONS.md)
- [MCP HTTP/stdio setup](../docs/MCP_INTEGRATION.md)
- [YAML configuration](../docs/CONFIGURATION.md)
- [Documentation index](../docs/README.md)
- [Historical investigations and acceptance records](../docs/archive/README.md)

Account, owner and file-limit changes entered in the resident console apply
while the server runs. Storage and network changes use `stop`, settings, `start`.
Browser reloads and MCP disconnects do not cancel durable analysis jobs.
Existing one-shot administration commands remain available.

## Development checks

```bash
npm test
npm audit --audit-level=moderate
node tests/image_api_smoke.cjs
node tests/server_console_smoke.cjs
node tests/server_console_pty.cjs
node tools/build-mcp-package.cjs
```

The CI additionally tests packaged HTTP, MCP stdio, image handling and actual
terminal input on Windows, Linux and macOS. The PTY harness uses the development
dependency `node-pty`; on Linux it requires a native build toolchain.
Runtime prompt widgets use Inquirer; command editing uses Node's `readline`.

Keep `public/yaml-schema.json`, templates, sample YAML and the configuration
manual synchronized. Preserve workspace ownership, revision preconditions,
frozen job inputs and active-result protection when changing an API.
User data under `configs/`, `uploads/`, `user-libs/` and results directories must
not be committed or unintentionally packaged.
