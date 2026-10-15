"""Pinned full-revolution and large-displacement gates for direct NK starts.

No predictor/Anderson/circuit changes. --prepare-only launches no solver.
Large-step probes check aligned flux, not periodic EMF or harmonic accuracy.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import threading
import time

import numpy as np
import psutil
import yaml

from compare_waveforms import compare, load
from run_comparison import identity, run, sha256
from run_field_reuse_pairs import power_plan, write_json
from run_initialization_pairs import parse_initialization
from run_selection_sweep import monitor, strictly_converged

GIB = 1024**3


def cases(suite):
    plans = [dict(name='full', pixels=12, points=248, periodic_quality=True, order=['linear','direct']),
             dict(name='large_48', pixels=48, points=12, periodic_quality=False, order=['direct','linear']),
             dict(name='large_124', pixels=124, points=12, periodic_quality=False, order=['linear','direct']),
             dict(name='large_248', pixels=248, points=12, periodic_quality=False, order=['direct','linear'])]
    if suite not in ('all', 'full', 'large'):
        raise ValueError('Unknown suite')
    return [p for p in plans if suite == 'all' or (p['periodic_quality'] == (suite == 'full'))]


def configuration(base, case):
    cfg = copy.deepcopy(base)
    cfg['transient']['total_steps'] = case['points']
    cfg['transient']['slide_pixels_per_step'] = case['pixels']
    return cfg


def initialization_gate(log, mode, points):
    records = parse_initialization(log)
    if mode not in ('linear', 'direct') or points < 1 or len(records) != points:
        return False, records
    for i, record in enumerate(records):
        used = mode == 'direct' and i > 0
        if (record.get('requested') != mode or record.get('direct_used') != str(int(used))
                or record.get('transient_eligible') != str(int(i > 0))):
            return False, records
        if used:
            try:
                residual = float(record['seed_residual'])
            except (KeyError, ValueError):
                return False, records
            if not math.isfinite(residual) or residual < 0:
                return False, records
    return True, records


def aligned_flux_arrays(names, base, other_names, trial):
    if (names != other_names or base.shape != trial.shape or not names
            or base.ndim != 2 or base.shape[0] < 2 or base.shape[1] != len(names)
            or len(set(names)) != len(names)):
        raise ValueError('Flux channel/shape mismatch')
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(trial))):
        raise ValueError('Nonfinite flux')
    channels = {}
    for i, name in enumerate(names):
        scale = np.linalg.norm(base[:, i])
        if scale <= np.finfo(float).tiny:
            raise ValueError('Zero reference waveform: relative gate undefined')
        error = float(np.linalg.norm(trial[:, i]-base[:, i])/scale)
        channels[name] = dict(normalized_rms=error, passed=error <= .001)
    return dict(passed=all(c['passed'] for c in channels.values()), channels=channels,
                periodic_emf_evaluated=False, note='Aligned-angle flux only; no sparse-sweep EMF/harmonic claim.')


def pair_report(case, results, quality):
    a, b = results['linear'], results['direct']
    memory = all(0 < b[k] <= 2*a[k] and max(a[k], b[k]) <= 8*GIB
                 for k in ('peak_rss_bytes','peak_private_bytes'))
    valid_runs = all(r['converged'] and r['initialization_gate']
                     and r['field_reuses'] == 0 and r['termination_reason'] is None
                     and math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0
                     for r in (a, b))
    passed = valid_runs and memory and quality['passed'] is True
    return dict(case=case, runs=results, quality=quality, memory_pass=memory,
                passed=passed,
                speed_reduction=1-b['wall_seconds']/a['wall_seconds'] if passed else None,
                note='One pair per displacement, not repeated timing evidence or automatic release acceptance.')


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    parent = args.reference.resolve()
    old = json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    previous = json.loads((parent/'summary.json').read_text(encoding='utf-8'))
    if not (previous['complete'] and len(previous['runs']) == 6
            and all(r['converged'] and r['initialization_gate'] and r['repeat_equal'] for r in previous['runs'])):
        raise RuntimeError('Prior three-pair ablation has not passed')
    exe, image, source_config = args.exe.resolve(), Path(old['image_path']), Path(old['config_path'])
    pins = old['protocol']
    if sha256(source_config) != pins['config_sha256']:
        raise RuntimeError('Prior config changed')
    base = yaml.safe_load(source_config.read_text(encoding='utf-8'))
    if not (base['transient']['parallel_chunks'] == 1 and base['transient']['enable_sliding']
            and base['transient']['slide_direction'] == 'vertical'
            and base['nonlinear_solver']['jacobian'] == 'tangent'
            and base['nonlinear_solver']['line_search_objective'] == 'energy'
            and base['nonlinear_solver']['tolerance'] == .001
            and base['nonlinear_solver']['max_iterations'] == 300
            and not base['nonlinear_solver']['anderson']['enabled']
            and base.get('omp', {}).get('threads', 12) == 12):
        raise RuntimeError('Unexpected reference configuration')
    with image.open('rb') as f:
        header = f.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (450,2976):
        raise RuntimeError('Model must be the pinned 450x2976 full-revolution image')
    plan = cases(args.suite)
    configs = {c['name']+'.yaml': yaml.safe_dump(configuration(base,c),sort_keys=False) for c in plan}
    scripts = ('run_initialization_validation.py','run_initialization_pairs.py',
               'run_selection_sweep.py','run_comparison.py','run_field_reuse_pairs.py','compare_waveforms.py')
    protocol = dict(executable_sha256=pins['executable_sha256'], dll_sha256=pins['dll_sha256'],
                    image_sha256=pins['image_sha256'], source_config_sha256=pins['config_sha256'],
                    previous_manifest_sha256=sha256(parent/'manifest.json'),
                    previous_summary_sha256=sha256(parent/'summary.json'),
                    scripts={p: sha256(Path(__file__).with_name(p)) for p in scripts},
                    configs=configs, cases=plan, threads=12, mask=6, anderson=False,
                    power_plan_guid=power_plan(), memory_limit_bytes=8*GIB, seconds_per_step=600)
    out = args.output.resolve()

    def validate():
        if (sha256(exe) != protocol['executable_sha256'] or sha256(image) != protocol['image_sha256']
                or sha256(source_config) != protocol['source_config_sha256']
                or {p.name: sha256(p) for p in exe.parent.glob('*.dll')} != protocol['dll_sha256']
                or power_plan() != protocol['power_plan_guid']):
            raise RuntimeError('Binary/DLL/input/power-plan identity changed')
        for name, text in configs.items():
            if out.exists() and (out/name).exists() and (out/name).read_text(encoding='utf-8') != text:
                raise RuntimeError('Generated configuration changed')

    validate()
    if args.resume:
        if json.loads((out/'manifest.json').read_text(encoding='utf-8'))['protocol'] != protocol:
            raise RuntimeError('Resume protocol changed')
        for name in configs:
            if not (out/name).exists():
                raise RuntimeError('Generated config missing')
    else:
        out.mkdir(parents=True,exist_ok=False)
        for name, text in configs.items():
            (out/name).write_text(text,encoding='utf-8')
        write_json(out/'manifest.json',dict(protocol=protocol,source=identity(exe,repo),
                   reference=str(parent),image_path=str(image),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if args.prepare_only:
        print(f'VALIDATED {len(plan)} pairs; no solver started.',flush=True)
        return
    env = {k:v for k,v in os.environ.items() if not k.startswith('OMFDM_') and k != 'NK_PROF'}
    env.update(NK_PROF='1',OMP_NUM_THREADS='12',OMFDM_TANGENT_OPTIMIZATIONS='6',
               OMFDM_TANGENT_PRECONDITIONER='tangent-amg')
    reports = []
    for case in plan:
        measurements = {}
        for mode in case['order']:
            validate()
            target = out/(case['name']+'_'+mode)
            if target.exists():
                if not args.resume or not (target/'measurement.json').exists():
                    raise RuntimeError('Preserving incomplete attempt: '+str(target))
                r = json.loads((target/'measurement.json').read_text(encoding='utf-8'))
                print('REUSE '+target.name,flush=True)
            else:
                if any(p.info['name'] and p.info['name'].lower() in ('magfdmsolver','magfdmsolver.exe')
                       for p in psutil.process_iter(['name'])):
                    raise RuntimeError('Another solver is running')
                if shutil.disk_usage(out).free < (30 if case['points']==248 else 5)*GIB:
                    raise RuntimeError('Insufficient disk space')
                busy = psutil.cpu_percent(interval=3)*psutil.cpu_count()/100
                write_json(out/(target.name+'_idle.json'),dict(busy_cores=busy,
                           available_memory_bytes=psutil.virtual_memory().available))
                print(f'START {target.name}: {case["points"]} steps, {case["pixels"]} px/step',flush=True)
                done = threading.Event()
                watcher = threading.Thread(target=monitor,args=(out,target,target.name,case['points'],done),daemon=True)
                watcher.start()
                try:
                    r = run(exe,out/(case['name']+'.yaml'),image,target,dict(env,OMFDM_NK_INITIALIZATION=mode),
                            timeout_seconds=600*case['points'],memory_limit_bytes=8*GIB)
                finally:
                    done.set(); watcher.join()
            gate, init = initialization_gate((target/'process.log').read_text(encoding='utf-8',errors='replace'),mode,case['points'])
            row = {k:r[k] for k in ('wall_seconds','peak_rss_bytes','peak_private_bytes','estimated_other_busy_cores','termination_reason')}
            row.update(converged=strictly_converged(r,case['points']),initialization_gate=gate,
                       outer_iterations=sum(int(s['iterations']) for s in r['step_status']),initialization=init,
                       field_reuses=sum(int(p['field_reuses']) for p in r['nk_profiles']))
            measurements[mode] = row
            write_json(out/(case['name']+'_runs.json'),measurements)
            print(f'FINISH {target.name}: {r["wall_seconds"]:.3f}s, converged={row["converged"]}',flush=True)
            if not (row['converged'] and gate and row['field_reuses']==0):
                raise RuntimeError('Convergence/initialization gate failed: '+target.name)
        linear, direct = (out/(case['name']+'_'+m)/'results' for m in ('linear','direct'))
        if case['periodic_quality']:
            quality = compare(linear,direct,points=248,fundamental=2)
        else:
            names, a = load(linear,case['points'])
            other_names, b = load(direct,case['points'])
            quality = aligned_flux_arrays(names,a,other_names,b)
        report = pair_report(case,measurements,quality)
        reports.append(report)
        write_json(out/'summary.json',dict(complete=len(reports)==len(plan),passed=all(p['passed'] for p in reports),cases=reports))
        if not report['passed']:
            raise RuntimeError('Quality/memory gate failed: '+case['name'])
    write_json(out/'progress.json',dict(state='complete',pairs=len(reports),passed=True))
    print('COMPLETE. Review before angle prediction; no automatic next algorithm.',flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('exe','reference','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--suite',choices=('all','full','large'),default='all')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    try:
        execute(args)
    except Exception as e:
        print('STOPPED FOR REVIEW: '+str(e),flush=True)
        if args.output.exists():
            write_json(args.output/'last_error.json',dict(error=str(e),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            write_json(args.output/'progress.json',dict(state='stopped_for_review',error=str(e)))
        raise


if __name__ == '__main__':
    main()
