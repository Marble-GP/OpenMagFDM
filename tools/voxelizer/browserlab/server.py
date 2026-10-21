"""Loopback-only, single-job bridge from the HTML lab to the native CAD tools."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import struct
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ID = re.compile(r"^[a-f0-9]{32}$")
MAX_CELLS = 5000000
MAX_UPLOAD = 50 * 1024 * 1024


def extract_ids(source, target, cells):
    """Extract the first UInt32 block from the native writer, without a full VTI copy."""
    marker = b'<AppendedData encoding="raw">\n_'
    with source.open("rb") as stream:
        header = stream.read(16384)
        start = header.find(marker)
        if start < 0 or b'Name="material_id"' not in header[:start]:
            raise ValueError("Unexpected native VTI layout")
        stream.seek(start + len(marker))
        size = struct.unpack("<Q", stream.read(8))[0]
        if size != cells * 4:
            raise ValueError("VTI material block length mismatch")
        with target.open("xb") as out:
            remaining = size
            while remaining:
                block = stream.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("Truncated VTI")
                out.write(block)
                remaining -= len(block)


class Lab:
    def __init__(self, root, native):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        suffix = ".exe" if os.name == "nt" else ""
        self.voxel = native / ("MagVoxelizer" + suffix)
        self.preview = native / ("MagCadPreview" + suffix)
        if not self.voxel.is_file() or not self.preview.is_file():
            raise ValueError("Build MagVoxelizer and MagCadPreview first")
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.jobs = {}
        self.models = {}
        self.active = None

    def start(self, work):
        with self.lock:
            if self.active is not None:
                raise ValueError("Another job is running; wait or cancel it")
            if len(self.jobs) >= 64:
                raise ValueError("Session limit reached (64 jobs); restart the lab for a new session")
            used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if used > 1024 * 1024 * 1024:
                raise ValueError("Session output exceeds 1 GiB; start a new session")
            key = uuid.uuid4().hex
            job = {"id": key, "state": "running", "stage": "準備中", "cancel": threading.Event(),
                   "process": None, "result": None, "error": None}
            self.jobs[key] = job
            self.active = key

        def execute():
            try:
                result = work(job)
                with self.lock:
                    if job["cancel"].is_set():
                        job["state"] = "cancelled"
                    else:
                        job.update(state="complete", result=result)
            except Exception as exc:
                with self.lock:
                    job.update(state="cancelled" if job["cancel"].is_set() else "failed",
                               error=str(exc)[-3000:])
            finally:
                with self.lock:
                    job["process"] = None
                    self.active = None
        threading.Thread(target=execute, daemon=True).start()
        return key

    def command(self, job, arguments, log):
        with self.lock:
            if job["cancel"].is_set():
                raise ValueError("Cancelled")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with log.open("wb") as output:
                process = subprocess.Popen([str(a) for a in arguments], stdout=output,
                                           stderr=subprocess.STDOUT, creationflags=flags)
            job["process"] = process
        try:
            code = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise ValueError("Native operation exceeded 180 seconds")
        finally:
            with self.lock:
                job["process"] = None
        if job["cancel"].is_set():
            raise ValueError("Cancelled")
        if code:
            raise ValueError(log.read_text(encoding="utf-8", errors="replace")[-3000:])

    def cancel(self, key):
        with self.lock:
            job = self.jobs[key]
            if job["state"] == "running":
                job["cancel"].set()
                if job["process"] is not None and job["process"].poll() is None:
                    job["process"].kill()

    def add_model(self, source, name, sample=False):
        def work(job):
            folder = self.root / job["id"]
            folder.mkdir()
            target = folder / "input.step"
            target.write_bytes(source)
            digest = hashlib.sha256(source).hexdigest()
            preview = folder / "preview.json"
            job["stage"] = "STEPを読み、表示用メッシュを作成中"
            started = time.perf_counter()
            self.command(job, [self.preview, target, preview, ".05"], folder / "preview.log")
            model = json.loads(preview.read_text(encoding="utf-8"))
            if len(model["solids"]) > 64:
                raise ValueError("Browser lab supports up to 64 solids")
            if sample:
                materials = json.loads((HERE.parent / "examples/materials.json").read_text())
                for body, material in zip(model["solids"], materials["solid_material_ids"]):
                    body["material_id"] = material
                (folder / "materials.json").write_text(json.dumps(materials))
                preview.write_text(json.dumps(model), encoding="utf-8")
            metadata = {
                "modelId": job["id"], "name": name, "sha256": digest,
                "bounds_m": model["bounds_m"], "solids": len(model["solids"]),
                "cadTriangles": model["preview_triangles"],
                "previewUrl": f"/api/files/{job['id']}/preview.json",
                "importMs": (time.perf_counter() - started) * 1000,
            }
            with self.lock:
                self.models[job["id"]] = metadata
            return metadata
        return self.start(work)

    def generate(self, body):
        model_id = body.get("modelId")
        if model_id not in self.models:
            raise ValueError("Unknown model; load a STEP file first")
        pitch = float(body.get("pitchMm", 0))
        margin = float(body.get("marginMm", -1))
        if not math.isfinite(pitch) or not .001 <= pitch <= 1000:
            raise ValueError("Pitch must be 0.001 to 1000 mm")
        if not math.isfinite(margin) or not 0 <= margin <= 1000:
            raise ValueError("Air margin must be 0 to 1000 mm")
        padding = math.ceil(margin / pitch - 1e-12)
        model = self.models[model_id]
        b = model["bounds_m"]
        dims = [math.ceil((b["max"][a] - b["min"][a]) * 1000 / pitch - 1e-10)
                + 2 * padding for a in range(3)]
        if math.prod(dims) > MAX_CELLS:
            raise ValueError(f"Grid estimate exceeds {MAX_CELLS:,} cells; choose a coarser pitch")

        def work(job):
            folder = self.root / job["id"]
            folder.mkdir()
            source = self.root / model_id / "input.step"
            output = folder / "voxel"
            job["stage"] = "C++でボクセル生成中"
            args = [self.voxel, "--input", source, "--output", output, "--pitch-mm", str(pitch),
                    "--padding", str(padding), "--no-surface", "--max-cells", str(MAX_CELLS)]
            material_map = source.parent / "materials.json"
            if material_map.exists():
                args += ["--material-map", material_map]
            started = time.perf_counter()
            self.command(job, args, folder / "voxel.log")
            native_ms = (time.perf_counter() - started) * 1000
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            extract_ids(output / "model.vti", folder / "grid.bin", manifest["cell_count"])
            return {
                "jobId": job["id"], "modelId": model_id, "modelSha256": model["sha256"],
                "pitchMm": pitch, "requestedMarginMm": margin,
                "nativeMs": native_ms, "nativeReportedMs": manifest["elapsed_seconds"] * 1000,
                "manifest": manifest, "gridUrl": f"/api/files/{job['id']}/grid.bin",
                "vtiUrl": f"/api/files/{job['id']}/model.vti",
                "manifestUrl": f"/api/files/{job['id']}/manifest.json",
            }
        return self.start(work)


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenMagFDM-Lab/0.1"

    def log_message(self, format, *args):
        # Do not echo posted content or upload names.
        pass

    def reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def local(self, mutation=False):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != origin[7:]:
            self.reply(403, {"error": "Use the printed 127.0.0.1 URL"})
            return False
        if self.headers.get("Origin") not in (None, origin):
            self.reply(403, {"error": "Cross-origin access is not allowed"})
            return False
        if mutation and not secrets.compare_digest(
                self.headers.get("X-Lab-Token", ""), self.server.lab.token):
            self.reply(403, {"error": "Invalid local session token"})
            return False
        return True

    def do_GET(self):
        if not self.local():
            return
        path = urlsplit(self.path).path
        try:
            lab = self.server.lab
            if path == "/api/status":
                return self.reply(200, {"token": lab.token, "maxCells": MAX_CELLS, "threeVersion": "0.186.1"})
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path)
            if match:
                with lab.lock:
                    job = lab.jobs[match[1]]
                    return self.reply(200, {k: job[k] for k in ("id", "state", "stage", "result", "error")})
            match = re.fullmatch(r"/api/files/([a-f0-9]{32})/(preview.json|grid.bin|model.vti|manifest.json)", path)
            if match:
                key, filename = match.groups()
                with lab.lock:
                    if lab.jobs[key]["state"] != "complete":
                        raise KeyError()
                base = lab.root / key
                if filename in ("model.vti", "manifest.json"):
                    base /= "voxel"
                return self.file(base / filename)
            if path == "/":
                path = "/index.html"
            # Only files shipped with this buildless page can be served.
            allowed = {"index.html", "app.js", "style.css", "mesh-worker.js", "mesh-core.js",
                       "metrics.js", "vendor/three.module.js", "vendor/three.core.js",
                       "vendor/OrbitControls.js", "vendor/THREE-LICENSE.txt"}
            if path[1:] not in allowed:
                raise KeyError()
            return self.file(HERE / "dist" / path[1:])
        except (KeyError, FileNotFoundError):
            self.reply(404, {"error": "Not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def file(self, path):
        size = path.stat().st_size
        self.send_response(200)
        mime = "text/javascript" if path.suffix == ".js" else mimetypes.guess_type(path)[0]
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self'; worker-src 'self'; connect-src 'self'; "
                         "img-src 'self' data: blob:; frame-ancestors 'none'")
        self.end_headers()
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, self.wfile, 1024 * 1024)

    def do_POST(self):
        if not self.local(mutation=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            path = urlsplit(self.path).path
            limit = MAX_UPLOAD if path == "/api/models" else 16384
            if not 0 <= length <= limit:
                raise ValueError("Request size limit exceeded")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request")
            lab = self.server.lab
            if path == "/api/sample":
                sample = HERE.parent / "examples/rotor_demo.step"
                job = lab.add_model(sample.read_bytes(), "rotor_demo.step", sample=True)
            elif path == "/api/models":
                name = parse_qs(urlsplit(self.path).query).get("name", ["model.step"])[0]
                if Path(name).suffix.lower() not in (".step", ".stp") or not raw:
                    raise ValueError("Choose a nonempty STEP/STP file")
                job = lab.add_model(raw, name[:160])
            elif path == "/api/generate":
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                job = lab.generate(body)
            else:
                match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", path)
                if not match:
                    return self.reply(404, {"error": "Not found"})
                lab.cancel(match[1])
                return self.reply(200, {"cancelled": True})
            self.reply(202, {"jobId": job})
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.reply(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--native", type=Path, default=REPO / "build_voxelizer/Release")
    parser.add_argument("--session-root", type=Path, default=REPO / "build_voxelizer/browserlab")
    args = parser.parse_args()
    if not (HERE / "dist/vendor/three.module.js").is_file():
        raise SystemExit("Prepare local assets first: npm ci && npm run vendor (in browserlab)")
    root = args.session_root / ("run_" + uuid.uuid4().hex)
    lab = Lab(root, args.native.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.lab = lab
    print(f"OpenMagFDM browser lab: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Session output: {root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with lab.lock:
            if lab.active:
                lab.cancel(lab.active)
        server.server_close()


if __name__ == "__main__":
    main()
