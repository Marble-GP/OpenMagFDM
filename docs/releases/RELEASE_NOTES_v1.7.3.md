# OpenMagFDM v1.7.3

OpenMagFDM v1.7.3 improves workspace administration, job visibility, result
storage limits, and file operations in the existing 2D analysis workflow.

## Highlights

- **Workspace storage:** Set or migrate the workspace root from the resident
  console. Verified migration copies configurations, images, material libraries,
  and results, and keeps the original data.
- **Jobs and results:** A dedicated, paged Jobs screen shows queue progress and
  history without filling the Model & Run view. The console shows unfinished
  jobs by default; `jobs list --all` includes history. Completed results appear
  in the Results list without a page reload.
- **Storage limits:** Configure per-workspace result-folder counts and result
  storage admission limits in the console. Upload and import paths report quota
  rejection, and the file manager shows operation progress.
- **Reliable file operations:** Large-result deletion, partial failures,
  concurrent imports, input snapshots, and result-folder renames have additional
  guards and recovery checks. The WebUI reports deletion errors and preserves
  failed selections for retry.
- **English public interface:** First-party WebUI labels and current manuals
  use English. User-supplied filenames and descriptions remain unrestricted.

## Upgrade

Stop the server before replacing binaries and public assets. Preserve the
workspace directories and OS-private account, job, and MCP settings. Keep the
same installation path if you want to retain the existing management profile.
Existing 2D YAML and result files need no conversion. Check held jobs before
resuming them. See the [CLI manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.3/docs/CLI.md),
[REST API manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.3/docs/REST_API.md), and
[operations manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.3/docs/OPERATIONS.md).

## Packages and validation

Windows, Linux, and macOS solver packages, standalone WebUI packages, and the
Windows installer are built by the tagged release workflow. Packaged WebUI
embeds Node.js 24.22.1. WebUI-only packages require a separate native solver.
The macOS WebUI package targets Intel x64 and requires Rosetta 2 on Apple Silicon.

The source release passed 134 Node regression tests and the release staging
check before tagging. The tagged workflow builds and tests the native solver and
assembles packages on all supported platforms. Experimental GPU rendering and
3D solver features are excluded from the stable packages.
