#!/usr/bin/env python3
"""Run matched FDM/FEM benchmarks and report wall time and field error.

The FDM result is used as the reference field. This is a comparative benchmark,
not an accuracy certificate; use analytic/MMS tests for absolute accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLVER = ROOT / "build_fem_phase0" / "Release" / "MagFDMsolver.exe"


def make_image(path: Path, n: int, case: str) -> None:
    img = Image.new("RGB", (n, n), (255, 255, 255))
    px = img.load()
    if case == "uniform_current":
        pass
    elif case == "iron_block":
        for y in range(n // 3, 2 * n // 3):
            for x in range(n // 3, 2 * n // 3):
                px[x, y] = (255, 0, 0)
        for y in range(n // 4, 3 * n // 4):
            for x in range(n // 8, n // 8 + max(1, n // 32)):
                px[x, y] = (0, 0, 255)
            for x in range(7 * n // 8, 7 * n // 8 + max(1, n // 32)):
                px[x, y] = (0, 0, 255)
    elif case == "pm_pair":
        for y in range(n // 3, 2 * n // 3):
            for x in range(n // 6, n // 3):
                px[x, y] = (0, 255, 0)
            for x in range(2 * n // 3, 5 * n // 6):
                px[x, y] = (0, 255, 0)
    else:
        raise ValueError(case)
    img.save(path)


def make_config(path: Path, n: int, case: str, engine: str) -> None:
    dx = 0.02 / (n - 1)
    air_jz = 1.0e6 if case == "uniform_current" else 0.0
    left_jz = 1.0e6 if case == "iron_block" else 0.0
    right_jz = -1.0e6 if case == "iron_block" else 0.0
    path.write_text(
        f"""coordinate_system: cartesian
omp: {{ threads: 1 }}
mesh: {{ dx: {dx:.17g}, dy: {dx:.17g} }}
materials:
  air: {{ rgb: [255,255,255], mu_r: 1.0, jz: {air_jz:.17g} }}
  iron: {{ rgb: [255,0,0], mu_r: 100.0, jz: 0.0 }}
  coil_left: {{ rgb: [0,0,255], mu_r: 1.0, jz: {left_jz:.17g} }}
  coil_right: {{ rgb: [0,0,255], mu_r: 1.0, jz: {right_jz:.17g} }}
  magnet: {{ rgb: [0,255,0], mu_r: 1.05, jz: 0.0, magnetization: {{ Br: 1.0, pattern: parallel, angle: 90 }} }}
boundary_conditions:
  left: {{ type: dirichlet, value: 0.0 }}
  right: {{ type: dirichlet, value: 0.0 }}
  bottom: {{ type: dirichlet, value: 0.0 }}
  top: {{ type: dirichlet, value: 0.0 }}
solver:
  engine: {engine}
  fem:
    mesh: {{ min_spacing: {dx:.17g}, max_spacing: {4*dx:.17g}, min_angle_deg: 5.0 }}
export: {{ format: csv, async: false }}
""",
        encoding="utf-8",
    )


def read_matrix(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [[float(v) for v in row] for row in csv.reader(f)]


def relative_l2(a: list[list[float]], b: list[list[float]]) -> float:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("field shape mismatch")
    num = sum((x - y) ** 2 for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    den = sum(y * y for rb in b for y in rb)
    return (num / den) ** 0.5 if den > 0.0 else 0.0


def run_once(solver: Path, work: Path, n: int, case: str, engine: str) -> tuple[float, str]:
    case_dir = work / f"{case}_{n}_{engine}"
    if case_dir.exists():
        import shutil
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    image = case_dir / "model.png"
    config = case_dir / "config.yaml"
    out = case_dir / "out"
    make_image(image, n, case)
    make_config(config, n, case, engine)
    env = dict(os.environ)
    env["OMFDM_PROFILE"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(
        [str(solver), str(config), str(image), str(out)],
        cwd=case_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    (case_dir / "run.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{case} {n} {engine} failed ({proc.returncode}):\n{proc.stdout[-2000:]}")
    return elapsed, proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "fem_vs_fdm" / "results")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--cases", nargs="+", default=["uniform_current", "iron_block", "pm_pair"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in args.cases:
        for n in args.sizes:
            record = {"case": case, "n": n}
            fields = {}
            for engine in ("fdm", "fem"):
                times = []
                logs = []
                for _ in range(args.repeats):
                    elapsed, log = run_once(args.solver, args.out, n, case, engine)
                    times.append(elapsed)
                    logs.append(log)
                record[f"{engine}_median_s"] = statistics.median(times)
                record[f"{engine}_times_s"] = times
                fields[engine] = read_matrix(args.out / f"{case}_{n}_{engine}" / "out" / "Az" / "step_0001.csv")
                record[f"{engine}_log_tail"] = logs[-1][-1200:]
            record["relative_l2_vs_fdm"] = relative_l2(fields["fem"], fields["fdm"])
            record["speedup_fdm_over_fem"] = record["fdm_median_s"] / record["fem_median_s"]
            rows.append(record)
            print(f"{case:16s} n={n:4d} fdm={record['fdm_median_s']:.3f}s "
                  f"fem={record['fem_median_s']:.3f}s speedup={record['speedup_fdm_over_fem']:.2f}x "
                  f"relL2={record['relative_l2_vs_fdm']:.4f}")

    (args.out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# FEM vs FDM benchmark",
        "",
        "FDM is the reference field. Times are median wall-clock seconds.",
        "",
        "| case | n | FDM (s) | FEM (s) | speedup | relative L2 vs FDM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['case']} | {r['n']} | {r['fdm_median_s']:.3f} | "
                     f"{r['fem_median_s']:.3f} | {r['speedup_fdm_over_fem']:.2f}x | "
                     f"{r['relative_l2_vs_fdm']:.4f} |")
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
