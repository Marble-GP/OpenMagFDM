# OpenMagFDM v1.7.4

OpenMagFDM v1.7.4 improves file editing, result annotations, job-history cleanup
and plot configuration in the existing 2D analysis workflow. It also updates
the MCP dependencies following a security review.

## Highlights

- **Unified file editing:** Use **Files → Edit** to rename a file or result.
  Results include a multiline description in the same dialog. Revision checks,
  inline errors, saving indicators and discard confirmation protect edits.
- **Clearer result descriptions:** Results shows saved and unsaved states.
  Data reloads retain an unsaved description, and switching between results
  retains each draft for the current page session. A Files preview no longer
  changes the description or original-input link of the loaded result.
- **Job-history cleanup:** Jobs can clear finished history for a chosen time
  range. Active and held jobs, input files and result files are retained.
  Clearing history also removes its request-key deduplication record.
- **Consistent names:** Manual WebUI runs create `output_webui_…` directories;
  MCP runs retain `output_mcp_…`. Renaming results updates recorded job paths.
  Renaming an active material library preserves its selection, and supported
  special-character and Unicode library names remain editable.
- **Simpler plot settings:** A curated color-theme list includes gradient
  previews. The Coarsening Mask item has been removed from the plot palette.
  File lists and edit dialogs use available space more effectively.
- **Security maintenance:** MCP client/server dependencies are updated to
  2.4.0 and the Node adapter to 2.1.2, incorporating reviewed Hono and qs fixes.
  Standalone WebUI packages continue to embed Node.js **24.22.1**, which includes
  the October 21 Brotli security fix.

## Upgrade

Stop the server before replacing binaries and public assets. Preserve workspace
directories and OS-private account, job and MCP settings. Keep the same
installation path to retain the existing management profile. Existing 2D YAML
and result files need no conversion. Review held jobs before resuming them.
Description drafts survive result switching within the current page; save them
before closing or reloading the browser page.

See the [CLI manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.4/docs/CLI.md),
[REST API manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.4/docs/REST_API.md)
and [operations manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.4/docs/OPERATIONS.md).

## Packages and validation

The tagged workflow builds Windows, Linux and macOS solver packages, standalone
WebUI packages and the Windows installer. WebUI-only packages require a separate
native solver. The macOS WebUI package targets Intel x64 and requires Rosetta 2
on Apple Silicon.

Pre-release validation passed 159 Node regression tests and a dependency audit
with no known vulnerabilities. Additional checks cover
compressed-input and ZIP checks, real-solver REST/MCP acceptance, shared HTTPS
authorization, standalone Windows packaging, resident console and terminal
interaction, image APIs, release staging and visual WebUI checks.
Experimental GPU rendering and 3D solver features remain excluded from stable
packages. The 2D solver's numerical method is unchanged.
