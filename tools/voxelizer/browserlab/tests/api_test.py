"""HTTP/native integration; no browser/GPU performance claims are made by this test."""
from collections import Counter
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import struct
import sys
import threading
import time
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Handler, Lab, REPO, ThreadingHTTPServer


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = Lab(REPO / 'build_voxelizer/browserlab' / ('test_' + uuid.uuid4().hex),
                      REPO / 'build_voxelizer/Release')
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.lab = cls.lab
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, data=None, extra=None, raw=False):
        connection = HTTPConnection('127.0.0.1', self.port, timeout=10)
        headers = {'X-Lab-Token': self.lab.token}
        if extra:
            headers.update(extra)
        body = data if raw else json.dumps(data or {}).encode() if method == 'POST' else None
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        payload = response.read()
        status = response.status
        connection.close()
        return status, payload

    def json_request(self, method, path, data=None, code=200, extra=None):
        status, raw = self.request(method, path, data, extra)
        self.assertEqual(status, code, raw[:1000])
        return json.loads(raw)

    def wait(self, key, expected='complete'):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = self.json_request('GET', '/api/jobs/' + key)
            if job['state'] != 'running':
                self.assertEqual(job['state'], expected, job)
                return job
            time.sleep(.03)
        self.fail('native job timed out')

    def test_sample_cad_and_three_resolutions(self):
        source = REPO / 'tools/voxelizer/examples/rotor_demo.step'
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        model = self.wait(self.json_request('POST', '/api/sample', code=202)['jobId'])['result']
        self.assertEqual(model['sha256'], source_hash)
        self.assertEqual(model['solids'], 7)
        preview = self.json_request('GET', model['previewUrl'])
        total = 0
        for body in preview['solids']:
            positions = body['positions']
            self.assertEqual(len(positions) % 9, 0)
            self.assertTrue(positions)
            total += len(positions) // 9
            for axis in range(3):
                self.assertGreaterEqual(min(positions[axis::3]), body['bounds_m']['min'][axis] - 1e-9)
                self.assertLessEqual(max(positions[axis::3]), body['bounds_m']['max'][axis] + 1e-9)
        self.assertEqual(total, model['cadTriangles'])
        report = {'kind': 'native-and-transfer-check-only', 'browser_verified': False,
                  'model': model, 'cases': []}
        for pitch, cells in [(2., 1372), (1., 10976), (.5, 87808)]:
            job = self.json_request('POST', '/api/generate',
                                    {'modelId': model['modelId'], 'pitchMm': pitch, 'marginMm': 2}, code=202)
            result = self.wait(job['jobId'])['result']
            manifest = result['manifest']
            self.assertEqual(manifest['cell_count'], cells)
            status, data = self.request('GET', result['gridUrl'])
            self.assertEqual(status, 200)
            self.assertEqual(len(data), cells * 4)
            counts = Counter(value[0] for value in struct.iter_unpack('<I', data))
            self.assertEqual(counts[0], manifest['air_cells'])
            self.assertEqual(cells - counts[0], manifest['occupied_cells'])
            self.assertTrue(set(counts).issubset({0, 1, 2, 3}))
            self.assertEqual(self.json_request('GET', result['manifestUrl'])['cell_count'], cells)
            status, vti = self.request('GET', result['vtiUrl'])
            self.assertEqual(status, 200)
            self.assertIn(b'ImageData', vti[:1000])
            report['cases'].append({'pitchMm': pitch, 'cells': cells,
                                    'occupied': manifest['occupied_cells'], 'nativeMs': result['nativeMs']})
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
        self.json_request('POST', '/api/generate',
                          {'modelId': model['modelId'], 'pitchMm': .001, 'marginMm': 2}, code=400)
        (self.lab.root / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Native/API validation report:', self.lab.root / 'validation.json')

    def test_access_guards_and_static_assets(self):
        self.json_request('POST', '/api/sample', extra={'X-Lab-Token': 'wrong'}, code=403)
        self.json_request('POST', '/api/sample', extra={'Origin': 'https://example.com'}, code=403)
        self.json_request('GET', '/api/status', extra={'Host': 'example.com'}, code=403)
        self.json_request('GET', '/../../README.md', code=404)
        self.json_request('GET', '/api/files/' + '0'*32 + '/grid.bin', code=404)
        for path in ('/', '/app.js', '/mesh-worker.js', '/mesh-core.js', '/metrics.js',
                     '/vendor/three.module.js', '/vendor/three.core.js', '/vendor/OrbitControls.js'):
            code, body = self.request('GET', path)
            self.assertEqual(code, 200, path)
            self.assertTrue(body)
        code, _ = self.request('POST', '/api/models?name=bad.stl', b'bad', raw=True)
        self.assertEqual(code, 400)
        code, raw = self.request('POST', '/api/models?name=bad.step', b'not STEP', raw=True)
        self.assertEqual(code, 202)
        self.wait(json.loads(raw)['jobId'], expected='failed')

    def test_single_job_and_cancellation_of_owned_process(self):
        def sleeper(job):
            folder = self.lab.root / job['id']
            folder.mkdir()
            self.lab.command(job, [sys.executable, '-c', 'import time; time.sleep(30)'], folder / 'test.log')
            return {}
        key = self.lab.start(sleeper)
        deadline = time.monotonic() + 5
        while self.lab.jobs[key]['process'] is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIsNotNone(self.lab.jobs[key]['process'])
        process = self.lab.jobs[key]['process']
        self.json_request('POST', '/api/sample', code=400)
        self.json_request('POST', f'/api/jobs/{key}/cancel')
        self.wait(key, expected='cancelled')
        self.assertIsNotNone(process.poll())


if __name__ == '__main__':
    unittest.main()
