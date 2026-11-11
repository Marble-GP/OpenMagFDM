# OpenMagFDM documentation

Latest release: **v1.7.5**. See the [Changelog](../CHANGELOG.md).

## User and administrator manuals

| Topic | Manual |
|---|---|
| Installation and first analysis | [README](../README.md) |
| YAML, materials, expressions, boundaries and output | [Configuration](CONFIGURATION.md) |
| Resident console, administration, startup flags and standalone solver | [CLI](CLI.md) |
| HTTP authentication, files, jobs, fields, images and ZIP backups | [REST API](REST_API.md) |
| Shared HTTPS, ownership, storage, metadata and upgrades | [Operations](OPERATIONS.md) |
| Local agent connections over HTTP/stdio | [MCP](MCP_INTEGRATION.md) |
| Experimental flux/EMF display estimates and repeat-state assumptions | [Flux and EMF filtering](EMF_FILTER.md) |

Public interfaces and current manuals use English. User-authored filenames,
descriptions and configurations may use other languages. Historical records
retain their original language.

## Current evaluation

- [Amperian force correction and verification](development/AMPERIAN_FORCE_VERIFICATION.md):
  implemented changes, measured comparisons, regression tests and limitations.
- [Single-step force and torque correction plan](development/FORCE_EVALUATION_PLAN.md):
  issue history, original implementation audit and broader method-selection gates.

- [v1.7.2 MCP practical trial — 2026-10-24](development/V1_7_2_MCP_TRIAL_20261024.md):
  connected baseline, release/source distinction and pending practical checks.

## Release and development history

- [v1.7.5 release notes](releases/RELEASE_NOTES_v1.7.5.md)
- [v1.7.4 release notes](releases/RELEASE_NOTES_v1.7.4.md)
- [v1.7.3 release notes](releases/RELEASE_NOTES_v1.7.3.md)
- [Release history](releases/README.md)
- [Changelog](../CHANGELOG.md)
- [Historical plans, investigations, measurements and old instructions](archive/README.md)

Use the manuals above for current procedures. Historical research, version-specific
operations, introduction contracts/acceptance records and the old README are
under `archive/`. Dependency documentation stays with `amgcl/`, `tinyexpr/` and
`webui/public/vendor/`.
