"""Sequential large-grid cold-start diagnostic: secant, then tangent A/B/C.

Not the 12-step/three-repeat adoption benchmark. Preserve the physical model,
outer tolerance and iteration limit; change only sweep length and Jacobian.
Timeouts are inconclusive diagnostics, never converged timing results.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import time

import yaml

from run_comparison import identity, run, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "repo", "config", "image", "output"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--timeout-seconds", type=float, default=600)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--candidates", nargs="+", choices=("secant-amg", "tangent-amg", "tangent-ilut"),
                    default=["secant-amg", "tangent-amg", "tangent-ilut"])
    ap.add_argument("--trial-objective", choices=("residual", "energy"), default="residual",
                    help="Tangent-only merit comparison; secant always retains the input settings")
    a = ap.parse_args()
    if a.threads < 1 or a.timeout_seconds <= 0:
        ap.error("positive threads and timeout required")
    exe, repo, source, image, out = (p.resolve() for p in
                                    (a.exe, a.repo, a.config, a.image, a.output))
    cfg = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if cfg.get("omp", {}).get("threads", a.threads) != a.threads:
        ap.error("YAML omp.threads differs from --threads")
    if cfg["transient"]["parallel_chunks"] != 1:
        ap.error("diagnostic requires chunks=1")
    out.mkdir(parents=True, exist_ok=a.resume)
    manifest = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    source=identity(exe, repo), config_sha256=sha256(source),
                    config_text=source.read_text(encoding="utf-8-sig"),
                    image_sha256=sha256(image), image_path=str(image), threads=a.threads,
                    timeout_seconds=a.timeout_seconds, hard_memory_limit_bytes=8*2**30,
                    candidates=a.candidates, trial_objective=a.trial_objective,
                    purpose="one-step cold-start diagnostic; no adoption decision")
    manifest_path = out / "manifest.json"
    if a.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "image_sha256", "threads", "timeout_seconds", "hard_memory_limit_bytes"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        for key in ("executable_sha256", "dll_sha256"):
            if previous["source"][key] != manifest["source"][key]:
                raise ValueError(f"Cannot resume: {key} changed")
        if previous.get("candidates", ["secant-amg", "tangent-amg", "tangent-ilut"]) != a.candidates:
            raise ValueError("Cannot resume: candidate list changed")
        if previous.get("trial_objective", "residual") != a.trial_objective:
            raise ValueError("Cannot resume: trial objective changed")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("OMFDM_") or key == "NK_PROF":
            del env[key]
    env.update(NK_PROF="1", OMP_NUM_THREADS=str(a.threads))
    summaries = {}
    for name in ("secant", *dict.fromkeys(a.candidates)):
        config = copy.deepcopy(cfg)
        config["transient"]["total_steps"] = 1
        config["nonlinear_solver"]["jacobian"] = "secant" if name == "secant" else "tangent"
        if name != "secant":
            config["nonlinear_solver"]["line_search_objective"] = a.trial_objective
        config_path = out / f"{name}.yaml"
        config_text = yaml.safe_dump(config, sort_keys=False)
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != config_text:
                raise ValueError("Diagnostic config changed")
        else:
            config_path.write_text(config_text, encoding="utf-8")
        run_dir = out / name
        if (run_dir / "measurement.json").exists():
            result = json.loads((run_dir / "measurement.json").read_text(encoding="utf-8"))
        else:
            if run_dir.exists():
                raise RuntimeError(f"Incomplete run retained at {run_dir}; use a new output directory")
            run_env = env.copy()
            if name != "secant":
                run_env["OMFDM_TANGENT_PRECONDITIONER"] = name
            print(f"Starting {name}: one step, {a.threads} threads, diagnostic limit {a.timeout_seconds}s", flush=True)
            result = run(exe, config_path, image, run_dir, run_env,
                         timeout_seconds=a.timeout_seconds, memory_limit_bytes=8*2**30)
        log = (run_dir / "process.log").read_text(encoding="utf-8", errors="replace")
        tangent = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
                   if line.startswith("[TANGENT]")]
        summaries[name] = {key: result[key] for key in
                           ("exit_code", "converged", "wall_seconds", "peak_rss_bytes", "peak_private_bytes", "step_status")}
        summaries[name].update(termination_reason=result.get("termination_reason"),
                               tangent_attempts=len(tangent),
                               tangent_accepted=sum(p.get("accepted") == "1" for p in tangent),
                               tangent_inner_iterations=sum(int(p["iterations"]) for p in tangent),
                               secant_fallbacks=log.count("[one secant fallback]"),
                               completed_outer_profiles=len(result["nk_profiles"]))
        baseline = summaries["secant"]
        summaries[name]["cold_speed_reduction_fraction"] = (
            1-result["wall_seconds"]/baseline["wall_seconds"]
            if result["converged"] and baseline["converged"] else None)
        summaries[name]["within_twice_secant_peak"] = (
            result["peak_rss_bytes"] <= 2*baseline["peak_rss_bytes"] and
            result["peak_private_bytes"] <= 2*baseline["peak_private_bytes"])
        (out / "diagnostic_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps({name: summaries[name]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
