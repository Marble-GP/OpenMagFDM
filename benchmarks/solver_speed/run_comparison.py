"""Alternate baseline/trial processes; never compare unfinished solves as speedups.

Requires psutil. All outputs are placed in a NEW directory. This is intentionally
external to the solver: wall time includes initialization, output drain and exit.
Do not run performance measurements concurrently with builds or other solvers.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import psutil


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def identity(exe, repo):
    cache = next((p / "CMakeCache.txt" for p in exe.parents
                  if (p / "CMakeCache.txt").exists()), None)
    return dict(executable=str(exe), executable_sha256=sha256(exe),
                commit=git(repo, "rev-parse", "HEAD"),
                worktree_status=git(repo, "status", "--porcelain"),
                source_diff=git(repo, "diff", "HEAD"),
                cmake_cache=cache.read_text(encoding="utf-8") if cache else None,
                dll_sha256={p.name: sha256(p) for p in exe.parent.glob("*.dll")})


def run(exe, config, image, out, env):
    out.mkdir()
    start = time.perf_counter()
    rss_peak = private_peak = 0
    samples = []
    with (out / "process.log").open("w", encoding="utf-8") as log:
        p = subprocess.Popen([str(exe), str(config), str(image), str(out / "results")],
                             cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env)
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                mem = proc.memory_info()
                rss_peak = max(rss_peak, mem.rss, getattr(mem, "peak_wset", 0))
                private_peak = max(private_peak, getattr(mem, "private", 0),
                                   getattr(mem, "peak_pagefile", 0))
                samples.append([time.perf_counter() - start, mem.rss,
                                getattr(mem, "private", 0)])
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        code = p.wait()
    elapsed = time.perf_counter() - start
    statuses = list((out / "results").glob("**/*status*.csv"))
    status = []
    for path in statuses:
        with path.open(encoding="utf-8-sig", newline="") as f:
            status.extend(csv.DictReader(f))
    log_text = (out / "process.log").read_text(encoding="utf-8", errors="replace")
    profiles = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                for line in log_text.splitlines() if line.startswith("[NKPROF]")]
    result = dict(exit_code=code, converged=(code == 0), wall_seconds=elapsed,
                  peak_rss_bytes=rss_peak, peak_private_bytes=private_peak,
                  memory_method="50ms sampling; Windows also queries OS peak counters",
                  step_status=status, nk_profiles=profiles,
                  memory_samples_seconds_rss_private=samples)
    (out / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "trial", "baseline-repo", "trial-repo", "config", "image", "output"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=12,
                    help="OMP_NUM_THREADS; YAML omp.threads MUST agree or be absent")
    ap.add_argument("--trial-preconditioner", choices=("secant-amg", "tangent-amg", "tangent-ilut"))
    args = ap.parse_args()
    if args.repeats < 1 or args.threads < 1:
        ap.error("repeats and threads must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = {k: getattr(args, k).resolve() for k in
             ("baseline", "trial", "baseline_repo", "trial_repo", "config", "image")}
    env = os.environ.copy()
    # Do not accidentally inherit historical PoC switches into a benchmark.
    for k in list(env):
        if k.startswith("OMFDM_") or k == "NK_PROF":
            del env[k]
    env.update(NK_PROF="1", OMP_NUM_THREADS=str(args.threads))
    meta = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), platform=platform.platform(),
                processor=platform.processor(), physical_cores=psutil.cpu_count(logical=False),
                logical_cores=psutil.cpu_count(), threads=args.threads,
                config_sha256=sha256(paths["config"]), image_sha256=sha256(paths["image"]),
                config_text=paths["config"].read_text(encoding="utf-8-sig"),
                image_path=str(paths["image"]), repeats=args.repeats,
                trial_preconditioner=args.trial_preconditioner,
                baseline=identity(paths["baseline"], paths["baseline_repo"]),
                trial=identity(paths["trial"], paths["trial_repo"]))
    (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    results = {"baseline": [], "trial": []}
    for i in range(args.repeats):
        for name in ("baseline", "trial"):
            run_env = env.copy()
            if name == "trial" and args.trial_preconditioner:
                run_env["OMFDM_TANGENT_PRECONDITIONER"] = args.trial_preconditioner
            r = run(paths[name], paths["config"], paths["image"], out / f"{i+1}_{name}", run_env)
            results[name].append(r)
            print(f"{i+1} {name}: {r['wall_seconds']:.3f}s, exit={r['exit_code']}, "
                  f"peak RSS={r['peak_rss_bytes']/2**30:.3f}GiB", flush=True)
    all_converged = all(r["converged"] for group in results.values() for r in group)
    medians = {name: statistics.median(r["wall_seconds"] for r in group)
               for name, group in results.items()}
    summary = dict(all_runs_converged=all_converged, median_wall_seconds=medians,
                   speed_reduction_fraction=(1 - medians["trial"] / medians["baseline"])
                   if all_converged else None,
                   waveform_quality="not evaluated by this timing harness")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
