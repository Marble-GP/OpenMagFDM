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
import yaml


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args],
                                   encoding="utf-8", errors="replace").strip()


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
    cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    transient = cfg.get("transient", {})
    expected = transient.get("total_steps") if transient.get("enabled") else None
    status_ok = all(s.get("strict_converged") == "1" for s in status)
    if isinstance(expected, int) and expected > 0:
        status_ok &= len(status) == expected and [int(s["step"]) for s in status] == list(range(expected))
    result = dict(exit_code=code, converged=(code == 0 and status_ok), wall_seconds=elapsed,
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
    ap.add_argument("--trial-config", type=Path,
                    help="Only for algorithm comparisons; same physical model with jacobian: tangent")
    ap.add_argument("--resume", action="store_true", help="Reuse completed runs after verifying input/binary hashes")
    ap.add_argument("--stop-after-runs", type=int, help="Pause only BETWEEN completed runs (for build/test work)")
    args = ap.parse_args()
    if args.repeats < 1 or args.threads < 1:
        ap.error("repeats and threads must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    paths = {k: getattr(args, k).resolve() for k in
             ("baseline", "trial", "baseline_repo", "trial_repo", "config", "image")}
    trial_config = args.trial_config.resolve() if args.trial_config else paths["config"]
    for config in (paths["config"], trial_config):
        cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
        configured_threads = cfg.get("omp", {}).get("threads", args.threads)
        if configured_threads != args.threads:
            ap.error(f"{config}: YAML omp.threads does not match --threads")
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
                trial_config_sha256=sha256(trial_config),
                trial_config_text=trial_config.read_text(encoding="utf-8-sig"),
                image_path=str(paths["image"]), repeats=args.repeats,
                trial_preconditioner=args.trial_preconditioner,
                baseline=identity(paths["baseline"], paths["baseline_repo"]),
                trial=identity(paths["trial"], paths["trial_repo"]))
    manifest_path = out / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "image_sha256", "threads", "repeats", "trial_preconditioner"):
            if previous[key] != meta[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        if previous.get("trial_config_sha256", previous["config_sha256"]) != meta["trial_config_sha256"]:
            raise ValueError("Cannot resume: trial config changed")
        for name in ("baseline", "trial"):
            for key in ("executable_sha256", "dll_sha256"):
                if previous[name][key] != meta[name][key]:
                    raise ValueError(f"Cannot resume: {name} {key} changed")
        # Preserve the ORIGINAL source identity associated with these unchanged
        # binaries, even if development has since advanced in another build dir.
    else:
        manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    results = {"baseline": [], "trial": []}
    new_runs = 0
    for i in range(args.repeats):
        for name in ("baseline", "trial"):
            run_env = env.copy()
            if name == "trial" and args.trial_preconditioner:
                run_env["OMFDM_TANGENT_PRECONDITIONER"] = args.trial_preconditioner
            config = trial_config if name == "trial" else paths["config"]
            run_dir = out / f"{i+1}_{name}"
            retries = [run_dir] + sorted(out.glob(f"{i+1}_{name}_retry*"))
            completed = [p / "measurement.json" for p in retries if (p / "measurement.json").exists()]
            if args.resume and completed:
                r = json.loads(completed[-1].read_text(encoding="utf-8"))
            else:
                if args.resume and run_dir.exists():
                    # Preserve incomplete logs for diagnosis; never overwrite.
                    run_dir = out / f"{i+1}_{name}_retry{len(retries)}"
                r = run(paths[name], config, paths["image"], run_dir, run_env)
                new_runs += 1
            results[name].append(r)
            print(f"{i+1} {name}: {r['wall_seconds']:.3f}s, exit={r['exit_code']}, "
                  f"peak RSS={r['peak_rss_bytes']/2**30:.3f}GiB", flush=True)
            if args.stop_after_runs and new_runs >= args.stop_after_runs:
                print("Paused between runs; --resume verifies hashes before continuing.", flush=True)
                return
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
