"""Three secant/candidate pairs, then a gated 248-point waveform comparison.

Uses the preserved workspace binary. Priority 1/4 remain OFF. No solver changes.
Incomplete directories are never overwritten or automatically retried.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import threading
import time

import psutil
import yaml

from compare_waveforms import compare
from run_comparison import identity, run, sha256
from run_field_reuse_pairs import power_plan, signature, write_json

GIB = 1024**3
CASES = ('secant', 'candidate')


def strictly_converged(result, points):
    statuses = result['step_status']
    return (result['converged'] and result['exit_code'] == 0
            and result['termination_reason'] is None and len(statuses) == points
            and [int(s['step']) for s in statuses] == list(range(points))
            and all(s['strict_converged'] == '1' and math.isfinite(float(s['residual']))
                    and 0 <= float(s['residual']) < 1e-3 for s in statuses))


def timing_gate(rows):
    valid = [r for r in rows if r['strict_converged'] and r['repeat_history_equal']]
    complete = (len(rows) == len(valid) == 6 and all(
        sorted(r['cycle'] for r in valid if r['case'] == case) == [1, 2, 3] for case in CASES))
    medians = {c: statistics.median(r['wall_seconds'] for r in valid if r['case'] == c)
               for c in CASES if any(r['case'] == c for r in valid)}
    savings, memory_ok, reproducible = None, False, False
    pairs = []
    if complete:
        savings = 1-medians['candidate']/medians['secant']
        for i in (1, 2, 3):
            a, b = [next(r for r in valid if r['cycle'] == i and r['case'] == c) for c in CASES]
            pairs.append(dict(cycle=i, speed_reduction=1-b['wall_seconds']/a['wall_seconds'],
                              memory_pass=all(b[k] <= 2*a[k] and b[k] <= 8*GIB
                                              for k in ('peak_private_bytes', 'peak_rss_bytes'))))
        memory_ok = all(p['memory_pass'] for p in pairs)
        reproducible = all(p['speed_reduction'] > 0 for p in pairs)
    return dict(complete=complete, runs=rows, median_wall_seconds=medians,
                speed_reduction=savings, pairs=pairs, memory_pass=memory_ok,
                reproducible_shortening=reproducible,
                proceed_to_waveforms=bool(complete and savings >= .15 and memory_ok and reproducible),
                note='This selects a waveform candidate, not a release. Secondary/platform/retention gates remain.')


def monitor(out, target, label, points, done):
    start, offset, tail, step, last_print = time.monotonic(), 0, '', 0, -60
    import re
    while not done.wait(5):
        if (target/'process.log').exists():
            with (target/'process.log').open(encoding='utf-8', errors='replace') as f:
                f.seek(offset)
                text = tail+f.read()
                offset = f.tell()
            found = re.findall(r'--- Step (\d+) / (\d+) ---', text)
            current = int(found[-1][0]) if found else step
            tail = text[-160:]
        else:
            current = step
        elapsed = time.monotonic()-start
        if current != step or elapsed-last_print >= 60:
            print(f'PROGRESS {label}: step {current}/{points}, {elapsed/60:.1f} min', flush=True)
            last_print = elapsed
        step = current
        write_json(out/'progress.json', dict(state='running', run=label, step=step,
                   total_steps=points, elapsed_seconds=elapsed,
                   updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    prior = json.loads(args.prior_manifest.read_text(encoding='utf-8'))
    source = prior['sources']['trial']
    exe, image = Path(source['executable']), Path(prior['image_path'])
    configs = {c: yaml.safe_load((repo/'benchmarks/solver_speed'/n).read_text(encoding='utf-8'))
               for c, n in zip(CASES, ('ieej_12step.yaml', 'ieej_12step_tangent_energy.yaml'))}
    control = copy.deepcopy(configs['candidate'])
    del control['nonlinear_solver']['jacobian']
    del control['nonlinear_solver']['line_search_objective']
    if control != configs['secant']:
        raise RuntimeError('Inputs differ beyond the intended solver options')
    previous_input = yaml.safe_load(Path(prior['config_path']).read_text(encoding='utf-8-sig'))
    if configs['candidate'] != previous_input:
        raise RuntimeError('Candidate input changed since the controlled workspace experiment')
    # PNG dimensions verify that 248 * 12 pixels covers exactly one revolution.
    with image.open('rb') as f:
        header = f.read(24)
    width, height = struct.unpack('>II', header[16:24])
    if header[:8] != b'\x89PNG\r\n\x1a\n' or (width, height) != (450, 2976):
        raise RuntimeError('Unexpected full-revolution model dimensions')
    for cfg in configs.values():
        t, nl = cfg['transient'], cfg['nonlinear_solver']
        if not (t['total_steps'] == 12 and t['parallel_chunks'] == 1
                and t['slide_pixels_per_step']*248 == height and t['slide_direction'] == 'vertical'
                and nl['max_iterations'] == 300 and nl['tolerance'] == 1e-3
                and cfg.get('omp', {}).get('threads', 12) == 12):
            raise RuntimeError('Unexpected numerical/thread/sweep conditions')
    out = args.output.resolve()
    generated = {}
    for points in (12, 248):
        for case in CASES:
            cfg = copy.deepcopy(configs[case])
            cfg['transient']['total_steps'] = points
            generated[f'{case}_{points}.yaml'] = yaml.safe_dump(cfg, sort_keys=False)
    protocol = dict(binary_sha256=source['executable_sha256'], dll_sha256=source['dll_sha256'],
                    image_sha256=prior['image_sha256'], prior_manifest_sha256=sha256(args.prior_manifest),
                    scripts={p: sha256(Path(__file__).with_name(p)) for p in
                             ('run_selection_sweep.py', 'run_comparison.py',
                              'run_field_reuse_pairs.py', 'compare_waveforms.py')},
                    configs=generated, threads=12, masks=dict(secant=0, candidate=6),
                    orders=[['secant', 'candidate'], ['candidate', 'secant'], ['secant', 'candidate']],
                    power_plan_guid=power_plan(), run_waveforms=args.run_waveforms,
                    timeout_seconds_per_step=600, memory_limit_bytes=8*GIB)

    def validate():
        if (sha256(exe) != protocol['binary_sha256'] or sha256(image) != protocol['image_sha256']
                or {p.name: sha256(p) for p in exe.parent.glob('*.dll')} != protocol['dll_sha256']
                or power_plan() != protocol['power_plan_guid']):
            raise RuntimeError('Executable/input/DLL/power-plan identity mismatch')

    validate()
    if args.resume:
        saved = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if saved['protocol'] != protocol:
            raise RuntimeError('Resume protocol changed')
        for name, text in generated.items():
            if (out/name).read_text(encoding='utf-8') != text:
                raise RuntimeError('Generated configuration changed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        for name, text in generated.items():
            (out/name).write_text(text, encoding='utf-8')
        write_json(out/'manifest.json', dict(protocol=protocol, source=identity(exe, repo),
                   binary_core_checkpoint='ca1027a; preserved executable', image_path=str(image),
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if args.prepare_only:
        print('Validated: secant mask 0 vs candidate mask 6; priority 4 excluded.', flush=True)
        return
    env = {k: v for k, v in os.environ.items() if not k.startswith('OMFDM_') and k != 'NK_PROF'}
    env.update(NK_PROF='1', OMP_NUM_THREADS='12')

    def measure(label, case, points):
        validate()
        target = out/label
        if target.exists():
            if not args.resume or not (target/'measurement.json').exists():
                raise RuntimeError(f'Preserving incomplete attempt: {target}')
            return json.loads((target/'measurement.json').read_text(encoding='utf-8'))
        if any(p.info['name'] and p.info['name'].lower() in ('magfdmsolver', 'magfdmsolver.exe')
               for p in psutil.process_iter(['name'])):
            raise RuntimeError('Another solver is active')
        # Observed 12-step outputs were 0.45 GB; reserve generously for full output.
        if shutil.disk_usage(out).free < (30 if points == 248 else 5)*GIB:
            raise RuntimeError('Insufficient free disk space')
        busy = psutil.cpu_percent(interval=3)*psutil.cpu_count()/100
        write_json(out/f'{label}_idle.json', dict(busy_cores=busy,
                   available_memory_bytes=psutil.virtual_memory().available))
        run_env = dict(env, OMFDM_TANGENT_OPTIMIZATIONS=str(protocol['masks'][case]))
        if case == 'candidate':
            run_env['OMFDM_TANGENT_PRECONDITIONER'] = 'tangent-amg'
        print(f'START {label}: {points} steps, background={busy:.2f} cores', flush=True)
        done = threading.Event()
        watcher = threading.Thread(target=monitor, args=(out, target, label, points, done), daemon=True)
        watcher.start()
        try:
            result = run(exe, out/f'{case}_{points}.yaml', image, target, run_env,
                         timeout_seconds=600*points, memory_limit_bytes=8*GIB)
        finally:
            done.set()
            watcher.join()
        print(f'FINISH {label}: {result["wall_seconds"]:.3f}s, converged={result["converged"]}', flush=True)
        return result

    rows, references = [], {}
    for cycle, order in enumerate(protocol['orders'], 1):
        for case in order:
            r = measure(f'{cycle}_{case}', case, 12)
            reference = references.setdefault(case, r)
            same = (bool(r['nk_profiles']) and signature(r) == signature(reference)
                    and r['step_status'] == reference['step_status'])
            row = {k: r[k] for k in ('wall_seconds', 'peak_private_bytes', 'peak_rss_bytes',
                   'estimated_other_busy_cores')}
            row.update(cycle=cycle, case=case, strict_converged=strictly_converged(r, 12),
                       repeat_history_equal=same,
                       outer_iterations=sum(int(s['iterations']) for s in r['step_status']))
            rows.append(row)
            gate = timing_gate(rows)
            write_json(out/'summary.json', gate)
            if not row['strict_converged'] or not same:
                raise RuntimeError('Convergence/repeatability check failed; stop for review')
    if not gate['proceed_to_waveforms'] or not args.run_waveforms:
        write_json(out/'progress.json', dict(state='timing_complete', waveform_started=False,
                   selection_pass=gate['proceed_to_waveforms']))
        print('TIMING COMPLETE: '+json.dumps(gate, separators=(',', ':')), flush=True)
        return
    for case in CASES:
        r = measure(f'waveform_{case}', case, 248)
        if not strictly_converged(r, 248):
            raise RuntimeError('Full-revolution reference/candidate not converged; no waveform pass')
    quality = compare(out/'waveform_secant/results', out/'waveform_candidate/results',
                      points=248, fundamental=2)
    write_json(out/'waveform_quality.json', quality)
    write_json(out/'progress.json', dict(state='complete', timing_pass=True, waveform_pass=quality['passed']))
    print('COMPLETE: waveform_pass='+str(quality['passed'])+'; review remaining adoption gates.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--run-waveforms', action='store_true')
    args = ap.parse_args()
    try:
        execute(args)
    except Exception as error:
        print('STOPPED FOR REVIEW: '+str(error), flush=True)
        if args.output.exists():
            write_json(args.output/'last_error.json', dict(error=str(error),
                       timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        raise


if __name__ == '__main__':
    main()
