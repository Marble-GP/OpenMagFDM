#!/usr/bin/env python3
"""FDM pathway determinism / checksum regression (design 10.6)."""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path
from PIL import Image

solver = Path(sys.argv[1])
artifacts = Path(sys.argv[2])
if artifacts.exists():
    shutil.rmtree(artifacts)
artifacts.mkdir(parents=True)

img = Image.new("RGB", (16, 16), (255, 255, 255))
for y in range(5, 11):
    for x in range(5, 11):
        img.putpixel((x, y), (255, 0, 0))
img_path = artifacts / "model.png"
img.save(img_path)

dx = 0.01 / 15
cfg = artifacts / "config.yaml"
cfg.write_text(
    "coordinate_system: cartesian\n"
    "omp: { threads: 1 }\n"
    "mesh: { dx: " + repr(dx) + ", dy: " + repr(dx) + " }\n"
    "materials:\n"
    "  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n"
    "  iron: { rgb: [255,0,0], mu_r: 100.0, jz: 0.0 }\n"
    "boundary_conditions:\n"
    "  left: { type: dirichlet, value: 0.0 }\n"
    "  right: { type: dirichlet, value: 0.0 }\n"
    "  bottom: { type: dirichlet, value: 0.0 }\n"
    "  top: { type: dirichlet, value: 0.0 }\n"
    "solver:\n"
    "  engine: fdm\n"
    "export: { format: csv, async: false }\n",
    encoding="utf-8",
)

env = dict(os.environ)
hashes = []
for i in (1, 2):
    out = artifacts / ("out%d" % i)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    proc = subprocess.run([str(solver), str(cfg), str(img_path), str(out)],
                          cwd=out, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        sys.exit(1)
    data = (out / "Az" / "step_0001.csv").read_bytes()
    hashes.append(hashlib.sha256(data).hexdigest())

if hashes[0] != hashes[1]:
    print("FDM checksum mismatch:", hashes[0], hashes[1])
    sys.exit(1)
print("FDM checksum stable:", hashes[0])
sys.exit(0)