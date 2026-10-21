import importlib.util
import pathlib
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage', ROOT / 'tools/stage-release.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class ReleaseStageTest(unittest.TestCase):
    def test_package_excludes_gpu_and_preserves_source(self):
        original = (ROOT / 'webui/public/index.html').read_bytes()
        for subdir in ('public', 'webui/public'):
            with tempfile.TemporaryDirectory(prefix='omfdm-release-') as tmp:
                public = pathlib.Path(tmp) / subdir
                public.mkdir(parents=True)
                shutil.copy(ROOT / 'webui/public/index.html', public)
                (public / 'field-gpu.js').write_text('// experimental fixture', encoding='utf-8')
                html = public / 'index.html'
                html.write_text(html.read_text(encoding='utf-8') + '<option value="gpu">GPU</option><script src="field-gpu.js"></script>', encoding='utf-8')
                stage.stage(tmp)
                self.assertFalse((public / 'field-gpu.js').exists())
                content = (public / 'index.html').read_text(encoding='utf-8')
                self.assertNotIn('value="gpu"', content)
                self.assertIn('value="legacy"', content)
        self.assertEqual(original, (ROOT / 'webui/public/index.html').read_bytes())
        with self.assertRaises(ValueError):
            stage.stage(ROOT)


if __name__ == '__main__':
    unittest.main()
