"""End-to-end CLI and independent VTK XML/binary format checks (standard library only)."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


exe, fixture_exe, root = sys.argv[1:]
root = Path(root)
root.mkdir(parents=True, exist_ok=True)
# Each run has fresh outputs. Never remove another run's files.
work = Path(tempfile.mkdtemp(prefix="run_", dir=root))


def run(*args, ok=True):
    p = subprocess.run([exe, *map(str, args)], capture_output=True, encoding="utf-8", errors="replace")
    if (p.returncode == 0) != ok:
        raise AssertionError(f"{args}: exit {p.returncode}\n{p.stdout}\n{p.stderr}")
    return p


def vtk(path):
    raw = path.read_bytes()
    marker = b'<AppendedData encoding="raw">\n_'
    head, data = raw.split(marker, 1)
    tree = ET.fromstring(head + b"</VTKFile>")
    assert tree.attrib["byte_order"] == "LittleEndian"
    assert tree.attrib["header_type"] == "UInt64"
    arrays = {}
    ranges = []
    for item in tree.iter("DataArray"):
        start = int(item.attrib["offset"])
        size = struct.unpack_from("<Q", data, start)[0]
        fmt = {"UInt32": "I", "Int64": "q", "Float64": "d"}[item.attrib["type"]]
        unit = struct.calcsize(fmt)
        assert size % unit == 0
        arrays[item.attrib["Name"]] = struct.unpack_from("<" + fmt * (size // unit), data, start + 8)
        ranges.append((start, start + 8 + size))
    ranges.sort()
    assert ranges[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert data[ranges[-1][1]:] == b"\n</AppendedData>\n</VTKFile>\n"
    return tree, arrays


subprocess.run([fixture_exe, str(work)], check=True, capture_output=True)
box = work / "box.step"
inspect = json.loads(run("--input", box, "--inspect").stdout)
assert inspect["bounds_m"]["max"] == [0.002, 0.003, 0.004]
assert inspect["solids"][0]["solid_id"] == 1
out = work / "box_output"
run("--input", box, "--output", out, "--pitch-mm", 1)
manifest = json.loads((out / "manifest.json").read_text())
assert manifest["dimensions_cells"] == [4, 5, 6]
assert manifest["occupied_cells"] == 24 and manifest["surface_faces"] == 52
assert abs(manifest["solids"][0]["relative_volume_error"]) < 1e-12
tree, arrays = vtk(out / "model.vti")
image = tree.find("ImageData")
assert image.attrib["WholeExtent"] == "0 4 0 5 0 6"
assert list(map(float, image.attrib["Origin"].split())) == [-.001] * 3
assert len(arrays["material_id"]) == 120
assert arrays["solid_id"] == arrays["material_id"]
for z in range(6):
    for y in range(5):
        for x in range(4):
            expected = int(1 <= x < 3 and 1 <= y < 4 and 1 <= z < 5)
            assert arrays["material_id"][x + 4 * (y + 5*z)] == expected

tree, mesh = vtk(out / "surface.vtp")
assert int(tree.find("PolyData/Piece").attrib["NumberOfPolys"]) == 52
assert mesh["offsets"] == tuple(range(4, 209, 4))
assert mesh["connectivity"] == tuple(range(208))
assert set(mesh["material_id"]) == {1}
# Normals must point away from the box center. No triangle approximation.
points = list(zip(*[iter(mesh["Points"])] * 3))
for i in range(0, len(points), 4):
    a, b, c, d = points[i:i+4]
    u = [b[j]-a[j] for j in range(3)]
    v = [c[j]-a[j] for j in range(3)]
    normal = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    center = [(a[j]+b[j]+c[j]+d[j])/4 for j in range(3)]
    assert sum(normal[j] * (center[j] - [.001,.0015,.002][j]) for j in range(3)) > 0

# Existing outputs are immutable, including on failure.
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
run("--input", box, "--output", out, "--pitch-mm", 1, ok=False)
after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
assert before == after

material_map = work / "materials.json"
material_map.write_text('{"solid_material_ids":[17,4294967295]}')
mapped = work / "mapped"
run("--input", work / "assembly.step", "--output", mapped,
    "--pitch-mm", 1, "--material-map", material_map, "--no-surface")
_, ids = vtk(mapped / "model.vti")
assert set(ids["material_id"]) == {0, 17, 4294967295}
assert set(ids["solid_id"]) == {0, 1, 2}
assert not (mapped / "surface.vtp").exists()

# Non-ASCII and space-containing Windows paths.
unicode_input = work / "\u65e5\u672c\u8a9e model.step"
shutil.copyfile(box, unicode_input)
run("--input", unicode_input, "--output", work / "\u683c\u5b50 output", "--pitch-mm", 1)

invalid_args = [
    ("--pitch-mm", "0"), ("--pitch-mm", "-1"), ("--pitch-mm", "nan"),
    ("--pitch-mm", "inf"), ("--pitch-mm", "1x"), ("--pitch-mm", "1e-300"),
    ("--pitch-mm", "1", "--padding", "-1"),
    ("--pitch-mm", "1", "--padding", "18446744073709551615"),
    ("--pitch-mm", "1", "--max-cells", "10"),
    ("--pitch-mm", "1", "--max-cells", "0"),
    ("--pitch-mm", "1", "--max-surface-faces", "1"),
    ("--pitch-mm", "1", "--spacing-mm", "1", "1", "1"),
    ("--pitch-mm", "1", "--unknown"),
    ("--pitch-mm", "1", "--padding", "1", "--padding", "2"),
    ("--pitch-mm",),
]
for i, args in enumerate(invalid_args):
    target = work / f"bad_{i}"
    run("--input", box, "--output", target, *args, ok=False)
    assert not target.exists()
for name in ["sheet.step", "mixed.step", "overlap.step", "missing.step"]:
    run("--input", work / name, "--output", work / (name + "_bad"), "--pitch-mm", 1, ok=False)
for i, content in enumerate([
    '{"solid_material_ids":[0]}', '{"solid_material_ids":[1.0]}',
    '{"solid_material_ids":[4294967296]}', '{"solid_material_ids":[-1]}',
    '{"solid_material_ids":[true]}', '{"solid_material_ids":["1"]}',
    '{"solid_material_ids":[1,2]}', '{"wrong":[1]}', 'invalid json',
]):
    material_map.write_text(content)
    run("--input", box, "--inspect", "--material-map", material_map, ok=False)
bad = work / "broken.step"
bad.write_text("this is not STEP")
run("--input", bad, "--inspect", ok=False)
stl = work / "box.stl"
stl.write_text("solid box\nendsolid box\n")
run("--input", stl, "--inspect", ok=False)
print(f"CLI, binary formats, surface normals, IDs, Unicode, and rejection checks passed: {work}")
