"""Bounded workspace ablation, not an adoption decision. No concurrent solvers.

Mask bits: 1 material preparation, 2 lazy fallback, 4 FGMRES storage,
8 accepted fields. Defaults screen priorities 1/2/3 separately and 2+3.
The immutable baseline is always run first. Resume verifies executable,
dependencies, inputs and settings and preserves incomplete results.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import yaml
from run_comparison import identity, run, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "baseline-repo", "trial", "trial-repo", "config", "image", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--masks", nargs="+", type=int, choices=range(16), default=[0, 1, 2, 4, 6])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-runs", type=int)
    a = ap.parse_args()
    if a.steps < 1 or a.threads < 1:
        ap.error("steps and threads must be positive")
    source, image, out = (p.resolve() for p in (a.config, a.image, a.output))
    cfg = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if cfg.get("omp", {}).get("threads", a.threads) != a.threads:
        ap.error("YAML omp.threads differs from --threads")
    if cfg["transient"]["parallel_chunks"] != 1:
        ap.error("workspace experiments require chunks=1")
    cfg = copy.deepcopy(cfg)
    cfg["transient"]["total_steps"] = a.steps
    cfg["nonlinear_solver"]["jacobian"] = "tangent"
    cfg["nonlinear_solver"]["line_search_objective"] = "energy"
    text = yaml.safe_dump(cfg, sort_keys=False)
    out.mkdir(parents=True, exist_ok=a.resume)
    meta = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                baseline=identity(a.baseline.resolve(), a.baseline_repo.resolve()),
                trial=identity(a.trial.resolve(), a.trial_repo.resolve()),
                source_config_sha256=sha256(source), config_text=text,
                image_sha256=sha256(image), image_path=str(image), threads=a.threads,
                steps=a.steps, masks=a.masks, preconditioner="tangent-amg",
                timeout_seconds=600*a.steps, memory_limit_bytes=8*2**30)
    manifest = out / "manifest.json"
    if a.resume:
        old = json.loads(manifest.read_text(encoding="utf-8"))
        for key in meta.keys() - {"timestamp", "baseline", "trial"}:
            if old[key] != meta[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        for name in ("baseline", "trial"):
            for key in ("executable_sha256", "dll_sha256"):
                if old[name][key] != meta[name][key]:
                    raise ValueError(f"Cannot resume: {name} {key} changed")
    else:
        manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out / "input.yaml").write_text(text, encoding="utf-8")
    if (out / "input.yaml").read_text(encoding="utf-8") != text:
        raise ValueError("Generated config changed")
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMFDM_") and k != "NK_PROF"}
    env.update(NK_PROF="1", OMP_NUM_THREADS=str(a.threads), OMFDM_TANGENT_PRECONDITIONER="tangent-amg")
    summaries = {}
    new_runs = 0
    for name, exe, mask in [("baseline", a.baseline, 0)] + [(f"mask_{m}", a.trial, m) for m in dict.fromkeys(a.masks)]:
        target = out / name
        measurement = target / "measurement.json"
        if a.resume and measurement.exists():
            result = json.loads(measurement.read_text(encoding="utf-8"))
        else:
            if target.exists():
                raise RuntimeError(f"Incomplete run retained at {target}; choose a new output directory")
            print(f"Starting {name}, {a.steps} steps", flush=True)
            result = run(exe.resolve(), out / "input.yaml", image, target,
                         dict(env, OMFDM_TANGENT_OPTIMIZATIONS=str(mask)),
                         timeout_seconds=meta["timeout_seconds"], memory_limit_bytes=meta["memory_limit_bytes"])
            new_runs += 1
        summaries[name] = {k: v for k, v in result.items() if k not in ("nk_profiles", "memory_samples_seconds_rss_private")}
        totals = {}
        for profile in result["nk_profiles"]:
            for key, value in profile.items():
                if key.endswith("_ms") or key in ("inner_iters", "fallbacks", "workspace_allocations", "workspace_reuses", "field_reuses", "field_invalidations", "field_updates"):
                    totals[key] = totals.get(key, 0.0) + float(value)
        summaries[name]["profile_totals"] = totals
        (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"{name}: {result['wall_seconds']:.3f}s converged={result['converged']} private={result['peak_private_bytes']/2**30:.3f}GiB", flush=True)
        if not result["converged"]:
            raise RuntimeError(f"{name} did not converge; screening stopped for review")
        if a.stop_after_runs and new_runs >= a.stop_after_runs:
            break


if __name__ == "__main__":
    main()
