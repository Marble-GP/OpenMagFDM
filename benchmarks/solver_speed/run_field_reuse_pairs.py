"""Bounded priority-4 experiment: masks 6/14, same pinned binary and inputs.

Reuse the previous idle experiment's manifest to pin executable/DLL/input hashes.
Resume skips completed, verified runs; incomplete attempts are never overwritten.
No adoption decision is made automatically. Run with the benchmark Python env.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import threading
import time

import psutil
import yaml

from run_comparison import identity, run, sha256


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def signature(result):
    keys = ('iter', 'residual', 'inner_iters', 'ls_trials', 'outcome')
    return [tuple(p[k] for k in keys) for p in result['nk_profiles']]


def fields(directory, steps):
    hashes = {}
    for name in ('Az', 'Mu'):
        paths = sorted(p for p in (directory/'results'/name).glob('*')
                       if p.suffix.lower() in ('.tif', '.tiff'))
        if len(paths) != steps:
            raise RuntimeError(f'{name}: expected {steps} fields, found {len(paths)}')
        hashes.update({str(p.relative_to(directory/'results')): sha256(p) for p in paths})
    return hashes


def summarize(records):
    accepted = [r for r in records if r['converged'] and r['equivalent'] and r['reuse_gate']]
    medians = {name: statistics.median(r['wall_seconds'] for r in accepted if r['case'] == name)
               for name in ('opt23', 'opt234') if any(r['case'] == name for r in accepted)}
    complete = (len(accepted) == 6 and all(
        sum(r['case'] == case for r in accepted) == 3 for case in ('opt23', 'opt234')))
    return dict(complete=complete, accepted_runs=len(accepted), runs=records,
                median_wall_seconds=medians,
                speed_reduction=1-medians['opt234']/medians['opt23'] if complete else None,
                note='Only converged, numerically equivalent runs enter medians. No full-waveform/adoption claim.')


def power_plan():
    if os.name != 'nt':
        return None
    raw = subprocess.check_output(['powercfg', '/getactivescheme']).decode(errors='replace')
    return re.search(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', raw)[0].lower()


def monitor(out, target, number, done):
    start, offset, tail, step, last_print = time.monotonic(), 0, '', 0, -60
    while not done.wait(5):
        log = target/'process.log'
        if log.exists():
            with log.open(encoding='utf-8', errors='replace') as f:
                f.seek(offset)
                text = tail + f.read()
                offset = f.tell()
            matches = re.findall(r'--- Step (\d+) / (\d+) ---', text)
            new_step = int(matches[-1][0]) if matches else step
            tail = text[-160:]
        else:
            new_step = step
        elapsed = time.monotonic()-start
        if new_step != step or elapsed-last_print >= 60:
            print(f'PROGRESS {number}/6 {target.name}: step {new_step}/12, {elapsed/60:.1f} min', flush=True)
            last_print = elapsed
        step = new_step
        write_json(out/'progress.json', dict(state='running', run_number=number,
                   total_runs=6, case=target.name, step=step, total_steps=12,
                   elapsed_seconds=elapsed, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    prior = json.loads(args.prior_manifest.read_text(encoding='utf-8'))
    source = prior['sources']['trial']
    exe, config, image = map(Path, (source['executable'], prior['config_path'], prior['image_path']))
    cfg = yaml.safe_load(config.read_text(encoding='utf-8-sig'))
    nl, transient = cfg['nonlinear_solver'], cfg['transient']
    if not (transient['total_steps'] == 12 and transient['parallel_chunks'] == 1
            and nl['tolerance'] == 1e-3 and nl['max_iterations'] == 300
            and nl['jacobian'] == 'tangent' and nl['line_search_objective'] == 'energy'):
        raise RuntimeError('Unexpected experiment configuration')
    pins = dict(executable_sha256=source['executable_sha256'], dll_sha256=source['dll_sha256'],
                config_sha256=prior['config_sha256'], image_sha256=prior['image_sha256'])

    def validate():
        actual = dict(executable_sha256=sha256(exe),
                      dll_sha256={p.name: sha256(p) for p in exe.parent.glob('*.dll')},
                      config_sha256=sha256(config), image_sha256=sha256(image))
        if actual != pins:
            raise RuntimeError('Prior executable/DLL/input hash mismatch')

    validate()
    engine_hash = sha256(Path(__file__).with_name('run_comparison.py'))
    if engine_hash != prior['measurement_engine_sha256']:
        raise RuntimeError('Measurement engine changed since idle retest')
    out = args.output.resolve()
    protocol = dict(pins=pins, prior_manifest_sha256=sha256(args.prior_manifest),
                    runner_sha256=sha256(Path(__file__)),
                    engine_sha256=engine_hash,
                    orders=[['opt23', 'opt234'], ['opt234', 'opt23'], ['opt23', 'opt234']],
                    masks=dict(opt23=6, opt234=14), threads=12,
                    timeout_seconds=7200, memory_limit_bytes=8*1024**3,
                    power_plan_guid=power_plan())
    if args.resume:
        manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if manifest['protocol'] != protocol:
            raise RuntimeError('Resume protocol/hash/power-plan mismatch')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'manifest.json', dict(protocol=protocol, source=identity(exe, repo),
                   config_path=str(config), image_path=str(image), platform=platform.platform(),
                   logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   binary_core_checkpoint='ca1027a (preserved executable, not rebuilt)'))
    if args.prepare_only:
        print('Validated masks 6/14, 3 pairs, 12 steps each; no solver launched.', flush=True)
        return
    env = {k: v for k, v in os.environ.items() if not k.startswith('OMFDM_') and k != 'NK_PROF'}
    env.update(NK_PROF='1', OMP_NUM_THREADS='12', OMFDM_TANGENT_PRECONDITIONER='tangent-amg')
    records, reference, reference_fields = [], None, None
    for cycle, order in enumerate(protocol['orders'], 1):
        for case in order:
            number = len(records)+1
            target = out/f'{cycle}_{case}'
            validate()
            if power_plan() != protocol['power_plan_guid']:
                raise RuntimeError('Power plan changed; stop for review')
            if target.exists():
                path = target/'measurement.json'
                if not args.resume or not path.exists():
                    raise RuntimeError(f'Preserving incomplete attempt {target}; review before retry')
                result = json.loads(path.read_text(encoding='utf-8'))
            else:
                if any(p.info['name'] and p.info['name'].lower() in ('magfdmsolver', 'magfdmsolver.exe')
                       for p in psutil.process_iter(['name'])):
                    raise RuntimeError('Another solver is running; do not overlap performance runs')
                busy = psutil.cpu_percent(interval=3)*psutil.cpu_count()/100
                write_json(out/f'{target.name}_idle.json', dict(busy_cores=busy,
                           available_memory_bytes=psutil.virtual_memory().available))
                print(f'START {number}/6 {case}, background={busy:.2f} cores', flush=True)
                done = threading.Event()
                watcher = threading.Thread(target=monitor, args=(out, target, number, done), daemon=True)
                watcher.start()
                try:
                    result = run(exe, config, image, target,
                                 dict(env, OMFDM_TANGENT_OPTIMIZATIONS=str(protocol['masks'][case])),
                                 timeout_seconds=protocol['timeout_seconds'],
                                 memory_limit_bytes=protocol['memory_limit_bytes'])
                finally:
                    done.set()
                    watcher.join()
            field_hashes = fields(target, 12) if result['converged'] else {}
            if reference is None:
                reference, reference_fields = result, field_hashes
            same = (bool(result['nk_profiles']) and signature(result) == signature(reference)
                    and result['step_status'] == reference['step_status'] and field_hashes == reference_fields)
            totals = {key: sum(int(p[key]) for p in result['nk_profiles'])
                      for key in ('field_reuses', 'field_invalidations', 'field_updates')}
            reuse_gate = totals['field_reuses'] > 0 if case == 'opt234' else totals['field_reuses'] == 0
            row = {k: result[k] for k in ('converged', 'wall_seconds', 'peak_rss_bytes',
                   'peak_private_bytes', 'estimated_other_busy_cores', 'termination_reason')}
            row.update(case=case, cycle=cycle, directory=target.name, equivalent=same,
                       reuse_gate=reuse_gate, field_counters=totals)
            records.append(row)
            write_json(out/f'{target.name}_field_hashes.json', field_hashes)
            write_json(out/'summary.json', summarize(records))
            print(f'FINISH {number}/6 {case}: {result["wall_seconds"]:.3f}s, '
                  f'converged={result["converged"]}, equivalent={same}, {totals}', flush=True)
            if not (result['converged'] and same and reuse_gate):
                raise RuntimeError('Numerical/field reuse gate failed; no further runs launched')
    write_json(out/'progress.json', dict(state='complete', completed_runs=6, total_runs=6))
    print('COMPLETE 6/6; see summary.json. Full adoption gates remain separate.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    try:
        execute(args)
    except Exception as error:
        print(f'STOPPED FOR REVIEW: {error}', flush=True)
        # Do not replace an existing run's state on a failed resume validation.
        if args.output.exists():
            write_json(args.output/'last_error.json', dict(error=str(error),
                       timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        raise


if __name__ == '__main__':
    main()
