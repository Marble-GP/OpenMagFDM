# FEM vs FDM benchmark

`run_benchmark.py` runs matched image/config cases with `solver.engine: fdm` and
`solver.engine: fem`, reporting median wall time and FEM relative L2 error against
FDM. FDM is a comparative reference, not an absolute accuracy certificate.

```powershell
$env:PATH="<dependency-bin>;$env:PATH"
py benchmarks/fem_vs_fdm/run_benchmark.py --sizes 128 256 512 --repeats 3
```

The generated `results/` directory is intentionally not committed. Use the same
solver binary, dependency DLLs, image generation, config, and repeat policy when
comparing revisions. Set `OMFDM_PROFILE=1` to print FEM phase timings.
