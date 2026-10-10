"""Bounded direct-initialization ablation, with a preserved numerical control."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import statistics
import threading
import time

import psutil
import yaml

from run_comparison import identity, run, sha256
from run_field_reuse_pairs import fields, power_plan, signature, write_json
from run_selection_sweep import monitor, strictly_converged


def parse_initialization(log):
    # updateMu writes without a newline, so NKINIT may follow another record.
    # Scope key/value parsing to the marker, never to the preceding text.
    return [dict(re.findall(r'(\w+)=([^\s]+)', m.group(1)))
            for m in re.finditer(r'\[NKINIT\]([^\r\n\[]*)', log)]


def validate_resume(previous, current, accept_runner_change=False):
    expected = copy.deepcopy(previous)
    if accept_runner_change:
        expected['scripts']['run_initialization_pairs.py'] = current['scripts']['run_initialization_pairs.py']
    if expected != current:
        raise RuntimeError('Resume identity mismatch (only an explicitly accepted runner change is allowed)')


def summary(rows):
    valid = [r for r in rows if r['converged'] and r['initialization_gate'] and r['repeat_equal']]
    medians = {c: statistics.median(r['wall_seconds'] for r in valid if r['case'] == c)
               for c in ('linear', 'direct') if any(r['case'] == c for r in valid)}
    complete = (len(rows) == len(valid) == 6 and all(
        sum(r['case'] == c for r in valid) == 3 for c in ('linear', 'direct')))
    return dict(complete=complete, runs=rows, median_wall_seconds=medians,
                speed_reduction=1-medians['direct']/medians['linear'] if complete else None,
                quality='No full-revolution waveform gate in this short ablation; no automatic adoption.')


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    reference_root = args.reference.resolve()
    previous = json.loads((reference_root/'manifest.json').read_text(encoding='utf-8'))
    reference_dir = reference_root/'1_candidate'
    reference = json.loads((reference_dir/'measurement.json').read_text(encoding='utf-8'))
    if not strictly_converged(reference, 12):
        raise RuntimeError('Preserved control is not converged')
    config, image, exe = reference_root/'candidate_12.yaml', Path(previous['image_path']), args.exe.resolve()
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    if config.read_text(encoding='utf-8') != previous['protocol']['configs']['candidate_12.yaml']:
        raise RuntimeError('Preserved input changed')
    if (cfg['nonlinear_solver']['anderson']['enabled'] or cfg['transient']['total_steps'] != 12
            or cfg['transient']['parallel_chunks'] != 1):
        raise RuntimeError('Unexpected experiment settings')
    ident = identity(exe, repo)
    if ident['dll_sha256'] != previous['source']['dll_sha256']:
        raise RuntimeError('DLLs differ from preserved candidate')
    protocol = dict(executable_sha256=sha256(exe), dll_sha256=ident['dll_sha256'],
                    config_sha256=sha256(config), image_sha256=sha256(image),
                    reference_sha256=sha256(reference_dir/'measurement.json'),
                    scripts={p: sha256(Path(__file__).with_name(p)) for p in
                             ('run_initialization_pairs.py', 'run_comparison.py',
                              'run_field_reuse_pairs.py', 'run_selection_sweep.py')},
                    power_plan_guid=power_plan(), threads=12, mask=6,
                    orders=[['linear', 'direct'], ['direct', 'linear'], ['linear', 'direct']],
                    timeout_seconds=7200, memory_limit_bytes=8*1024**3)
    if protocol['image_sha256'] != previous['protocol']['image_sha256']:
        raise RuntimeError('Preserved image changed')
    out = args.output.resolve()
    runner_update = None
    if args.accept_runner_change and not args.resume:
        raise RuntimeError('--accept-runner-change requires --resume')
    if args.resume:
        old = json.loads((out/'manifest.json').read_text(encoding='utf-8'))['protocol']
        validate_resume(old, protocol, args.accept_runner_change)
        if old['scripts']['run_initialization_pairs.py'] != protocol['scripts']['run_initialization_pairs.py']:
            archive = out/'runner_before_parser_fix.py'
            if not archive.exists() or sha256(archive) != old['scripts']['run_initialization_pairs.py']:
                raise RuntimeError('Original runner archive missing or changed')
            runner_update = dict(parent_manifest_sha256=sha256(out/'manifest.json'),
                                 old_runner_sha256=old['scripts']['run_initialization_pairs.py'],
                                 new_runner_sha256=protocol['scripts']['run_initialization_pairs.py'],
                                 purpose='Fix inline NKINIT parsing and stopped-state reporting; solver, engine and inputs unchanged')
            audit = out/'parser_fix_resume.json'
            if audit.exists():
                saved = json.loads(audit.read_text(encoding='utf-8'))
                if any(saved[k] != v for k,v in runner_update.items()):
                    raise RuntimeError('Parser-fix audit mismatch')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'manifest.json', dict(protocol=protocol, source=ident,
                   config_path=str(config), image_path=str(image), reference_path=str(reference_dir),
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if args.prepare_only:
        print('Validated direct/linear ablation; no child launched.', flush=True)
        return
    if runner_update and not (out/'parser_fix_resume.json').exists():
        write_json(out/'parser_fix_resume.json', dict(runner_update,
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    env = {k: v for k, v in os.environ.items() if not k.startswith('OMFDM_') and k != 'NK_PROF'}
    env.update(NK_PROF='1', OMP_NUM_THREADS='12', OMFDM_TANGENT_PRECONDITIONER='tangent-amg',
               OMFDM_TANGENT_OPTIMIZATIONS='6')
    rows, controls = [], {}
    reference_fields = fields(reference_dir, 12)
    for cycle, order in enumerate(protocol['orders'], 1):
        for case in order:
            target = out/f'{cycle}_{case}'
            if (sha256(exe) != protocol['executable_sha256'] or sha256(config) != protocol['config_sha256']
                    or sha256(image) != protocol['image_sha256'] or power_plan() != protocol['power_plan_guid']
                    or {p.name: sha256(p) for p in exe.parent.glob('*.dll')} != protocol['dll_sha256']):
                raise RuntimeError('Experiment identity changed')
            if target.exists():
                if not args.resume or not (target/'measurement.json').exists():
                    raise RuntimeError(f'Preserving incomplete attempt: {target}')
                r = json.loads((target/'measurement.json').read_text(encoding='utf-8'))
                print(f'REUSE {len(rows)+1}/6 {target.name}: revalidate existing measurement, no recalculation', flush=True)
            else:
                if any(p.info['name'] and p.info['name'].lower() in ('magfdmsolver', 'magfdmsolver.exe')
                       for p in psutil.process_iter(['name'])):
                    raise RuntimeError('Another solver is running')
                busy = psutil.cpu_percent(interval=3)*psutil.cpu_count()/100
                write_json(out/f'{target.name}_idle.json', dict(busy_cores=busy,
                           available_memory_bytes=psutil.virtual_memory().available))
                print(f'START {len(rows)+1}/6 {target.name}, background={busy:.2f} cores', flush=True)
                done = threading.Event()
                watcher = threading.Thread(target=monitor, args=(out,target,target.name,12,done), daemon=True)
                watcher.start()
                try:
                    r = run(exe, config, image, target, dict(env, OMFDM_NK_INITIALIZATION=case),
                            timeout_seconds=protocol['timeout_seconds'], memory_limit_bytes=protocol['memory_limit_bytes'])
                finally:
                    done.set()
                    watcher.join()
            log = (target/'process.log').read_text(encoding='utf-8', errors='replace')
            init = parse_initialization(log)
            uses = sum(int(p['direct_used']) for p in init)
            control = controls.setdefault(case, r)
            same = (signature(r) == signature(control) and r['step_status'] == control['step_status'])
            if case == 'linear':
                same &= (signature(r) == signature(reference) and r['step_status'] == reference['step_status']
                         and strictly_converged(r,12) and fields(target,12) == reference_fields)
            row = {k: r[k] for k in ('wall_seconds','peak_private_bytes','peak_rss_bytes','estimated_other_busy_cores')}
            row.update(case=case, cycle=cycle, converged=strictly_converged(r,12), repeat_equal=same,
                       initialization_gate=len(init) == 12 and uses == (11 if case == 'direct' else 0),
                       direct_starts=uses, initialization=init,
                       outer_iterations=sum(int(s['iterations']) for s in r['step_status']))
            rows.append(row)
            write_json(out/'summary.json', summary(rows))
            print(f'FINISH {target.name}: {r["wall_seconds"]:.3f}s, outer={row["outer_iterations"]}, '
                  f'converged={row["converged"]}, direct_starts={uses}', flush=True)
            if not (row['converged'] and row['repeat_equal'] and row['initialization_gate']):
                raise RuntimeError('Convergence/control/initialization gate failed; review before continuing')
    write_json(out/'progress.json', dict(state='complete', completed_runs=6, total_runs=6))
    print('COMPLETE 6/6. Review before angle prediction; priority 4 stayed OFF.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('exe', 'reference', 'output'):
        ap.add_argument('--'+arg, type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--accept-runner-change', action='store_true',
                    help='Explicit audited parser-only migration; all other manifest pins remain mandatory')
    args = ap.parse_args()
    try:
        execute(args)
    except Exception as error:
        print('STOPPED FOR REVIEW: '+str(error), flush=True)
        if args.output.exists():
            write_json(args.output/'last_error.json', dict(error=str(error),
                       timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            write_json(args.output/'progress.json', dict(state='stopped_for_review', error=str(error),
                       updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        raise


if __name__ == '__main__':
    main()
