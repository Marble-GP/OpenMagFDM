# Remaining tasks — v1.6.1 release candidate

Updated: 2026-07-11

## Required before release

- [ ] Build the specification-refresh commit on Linux, macOS and Windows.
- [ ] Confirm `MagFDMsolver --version` reports 1.6.1 in packaged binaries.
- [ ] Run the committed Windows release-contract smoke: static Distributed
  Amperian Force CSV, export defaults, version, and slide-unit metadata.
- [ ] Solver regression: decimal metre bounds resolve to the expected radial
  or Cartesian pixel band; integer inputs remain bit-compatible.
- [ ] Polar-axis regression: `r_start=0` + Dirichlet succeeds; Robin fails with
  the documented message.
- [ ] Review README, configuration reference, sample config, WebUI schema and
  generated templates as one contract.
- [ ] Create and review PR into `main`; tag only after approval.

## Follow-up quality work

- Extend the committed release-contract smoke into cross-platform numerical
  regression coverage independent of the external `bench_results` workspace.
- Make the Windows AMGCL CI smoke step fail when AMGCL activation is missing
  instead of logging a warning only.
- Triage/close GitHub Issues #1 and #2 against current behaviour.
- Refresh the FEMM comparison benchmark with current AMGCL/DD output and
  quantitative accuracy/performance cases.

Historical v1.3/v1.5 plans and reports remain for traceability and are not the
active release checklist.
