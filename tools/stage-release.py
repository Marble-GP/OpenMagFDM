"""Remove experimental GPU assets from an already copied v1.7.1 package.

Never run against the source tree. The 3D targets/tools are not copied by release.yml.
"""
import pathlib
import re
import sys


def stage(destination):
    root = pathlib.Path(destination).resolve(strict=True)
    source = pathlib.Path(__file__).resolve().parents[1]
    if root == source or root in source.parents:
        raise ValueError('Expected a copied package, not the source directory')
    public = root / 'webui' / 'public' if (root / 'webui' / 'public').is_dir() else root / 'public'
    if not public.is_dir() or public.resolve() == source / 'webui' / 'public':
        raise ValueError('Missing or source public directory')
    public.resolve().relative_to(root)
    html = public / 'index.html'
    if html.is_symlink():
        raise ValueError('Package index must not be a symlink')
    content = html.read_text(encoding='utf-8')
    content = re.sub(r'<option value="gpu">[^<]*</option>', '', content)
    content = re.sub(r'\s*<script src="field-gpu\.js"></script>', '', content)
    html.write_text(content, encoding='utf-8')
    gpu = public / 'field-gpu.js'
    if gpu.exists() or gpu.is_symlink():
        gpu.unlink()
    if 'field-gpu.js' in content or 'value="gpu"' in content:
        raise ValueError('Experimental renderer is still referenced')
    print(f'Staged v1.7.1 legacy-renderer package: {root}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: stage-release.py COPIED_PACKAGE_DIRECTORY')
    stage(sys.argv[1])
