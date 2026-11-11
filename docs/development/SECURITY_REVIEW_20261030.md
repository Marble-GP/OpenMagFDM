# Node.js and compression security review — 2026-10-30

## Scope and outcome

Reviewed the v1.7.3 release configuration and the current development tree for
Node.js compression advisories, npm dependency advisories, ZIP import boundaries,
and packaging/runtime regressions. Existing uncommitted WebUI work was retained.
This review did not create a commit, tag, release, or deployment.

The reported Brotli issue is already fixed by the configured Node.js 24.22.1.
The dependency audit found two additional moderate advisories; both were removed
from the development dependency tree. The final audit reports zero known
vulnerabilities, including development dependencies. This is a scoped review,
not a guarantee that the application has no vulnerabilities.

## Node.js and compression findings

| Finding | Evidence and disposition |
| --- | --- |
| CVE-2026-111497, Brotli decoder memory corruption | Node.js lists 24.22.1 as a fixed release. The installed runtime, `.node-version`, v1.7.3 tag, package engine minimum, and executable packaging target all use 24.22.1. No Node version change is needed for this advisory. |
| Brotli version still prints `1.2.0` | Node.js 24.22.1 cherry-picks the decoder fix without changing this embedded library version string. Do not classify this runtime from the Brotli version string alone. The local runtime also exposes the newly added `BROTLI_DECODER_ERROR_FORMAT_BLOCK_SWITCH` constant. |
| CVE-2026-106070, experimental `zlib.ZipFile` | The advisory affects Node.js 26.x. OpenMagFDM uses Node.js 24 and `yauzl`/`yazl`, not the affected API. |
| CVE-2026-87776, Express `compression` middleware | The dependency is absent from the application dependency tree; the response compression middleware is not installed. |
| HTTP request decompression | Express uses body-parser 1.20.9. Its JSON parser accepts gzip/deflate and rejects Brotli with HTTP 415. The configured 25 MiB limit applies to the expanded JSON body. |
| ZIP import | `yauzl` 3.4.0 uses lazy entries, strict filenames and entry-size validation. Transfers enforce 1 GiB expanded data and 10,000 entries, reject encrypted entries, links and case-insensitive duplicate targets, and publish streamed results only after successful extraction. Existing workspace authorization, revision and quota checks remain in place. |

Primary sources:

- [Node.js October 21 security release](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases)
- [Node.js 24.22.1 release and Brotli backport](https://nodejs.org/en/blog/release/v24.22.1)
- [Express compression advisory](https://github.com/expressjs/compression/security/advisories/GHSA-vc2v-76pw-4v95)

## Additional dependency updates

| Advisory | Previous dependency | Updated dependency | Application exposure assessment |
| --- | --- | --- | --- |
| GHSA-rmxm-3fg6-px4f / CVE-2026-118663 | `@hono/node-server` 1.19.17 | 2.1.5 | The affected `serveStatic` handler is not used. WebUI files use Express; the MCP adapter uses request conversion. No affected route was found, but the dependency was updated. |
| GHSA-6pq8-jhm2-2vcw / CVE-2026-110816 | `qs` 6.16.0 | 6.16.2 | The affected comma-format `qs.stringify` call was not found in the application paths inspected. Express/body-parser use parsing. The dependency was updated. |

The official MCP Node adapter 2.1.2 supports Hono node-server 2.x and requires
MCP server 2.4.x. Client/server/core were aligned to 2.4.0 and the Node adapter
to 2.1.2. No forced peer-dependency resolution or override was introduced.
The existing Zod override was preserved. Only these six resolved packages
changed in the lockfile.

`npm audit fix --dry-run` proposed downgrading Hono to 1.19.9. Instead, this
change uses the supported updated SDK combination and patched Hono 2.1.5.

Primary sources:

- [Hono node-server advisory](https://github.com/honojs/node-server/security/advisories/GHSA-rmxm-3fg6-px4f)
- [qs advisory](https://github.com/ljharb/qs/security/advisories/GHSA-6pq8-jhm2-2vcw)
- [Official MCP SDK 2.4.0 release](https://github.com/modelcontextprotocol/typescript-sdk/releases/tag/v2.4.0)

## Verification

- `npm audit --json`: zero known vulnerabilities across 331 dependencies.
- `npm ls`: supported MCP/Hono peer combination, no invalid dependency reported.
- `npm test`: 142 passed, zero failed or skipped.
- Six isolated compression probes passed: HTTP decompression limits/error
  handling, normal Brotli roundtrip, DEFLATE ZIP roundtrip, oversized declared
  expansion, understated expanded size, and case-folded duplicate entries.
- Windows SEA packaging succeeded with the existing official Node.js 24.22.1
  target. The new executable's `--runtime-info` confirmed Node.js 24.22.1 and
  packaged execution. The packaging implementation uses SHA-256 verification
  of downloaded Node archives and disables startup snapshots.
- Packaged MCP smoke passed: HTTP and stdio, setup, tool discovery, authorization,
  workspace isolation, revisions and revocation, with Node removed from the
  child PATH.
- Packaged image API smoke passed: PNG/JPEG/BMP upload and material detection,
  polar conversion, quantization, malformed input and foreign-path rejection.

The first sandboxed regression run could not perform localhost connections,
Windows link creation and private-storage writes. The suite passed when rerun
with those permissions. The temporary compression probe had a syntax typo
before its successful run; no production behavior was changed for either issue.

Internal probe scripts, test data, logs and the executable are under the sibling
`qa/security_20261030/` directory, which has an ignore-all `.gitignore` and is
outside this Git worktree. Existing user result fixtures were not changed.

## Deployment limits

The published v1.7.3 package and tag have not been replaced. Its build recipe
already pins the fixed Node.js 24.22.1; release archives were not downloaded and
individually revalidated in this review. The local prior packaged executable
and the newly built Windows executable both reported that runtime version.

The two npm dependency fixes are currently local development changes. A running
server must be restarted to load them; published standalone packages require a
new release build. Linux/macOS packaging and interactive browser testing were
not rerun here. No production exploit payload was sent to the user's server.
