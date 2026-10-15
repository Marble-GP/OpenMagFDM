"""Separate, pinned AB/BA/AB trials: direct -> angle predictor -> safeguarded AA.

No automatic stage chaining or public-default changes. Preserve incomplete runs.
AA uses one fixed depth=3/beta=.3 and requires a reviewed predictor-stage result.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import threading
import time

import psutil
import yaml

from run_comparison import identity, run, sha256
from run_field_reuse_pairs import fields, power_plan, signature, write_json
from run_initialization_pairs import parse_initialization
from run_initialization_validation import aligned_flux_arrays
from compare_waveforms import load
from run_selection_sweep import monitor, strictly_converged

GIB = 1024**3
SCRIPTS = ('run_prediction_pairs.py', 'run_comparison.py', 'run_field_reuse_pairs.py',
           'run_initialization_pairs.py', 'run_initialization_validation.py',
           'run_selection_sweep.py', 'compare_waveforms.py')


def variants(stage, initializer='direct'):
    if stage == 'predictor':
        return [('direct_off', 'direct', False), ('predict_off', 'predict', False)]
    if stage == 'anderson' and initializer in ('direct', 'predict'):
        return [(initializer+'_off', initializer, False), (initializer+'_aa', initializer, True)]
    raise ValueError('Unknown experiment stage/initializer')


def configuration(base, aa):
    cfg = copy.deepcopy(base)
    if aa:
        cfg['nonlinear_solver']['anderson'] = dict(enabled=True, depth=3, beta=.3)
    return cfg


def initialization_gate(records, mode, points=12):
    if mode not in ('direct','predict') or len(records) != points:
        return False
    try:
        for i, record in enumerate(records):
            prediction = mode == 'predict' and i >= 2
            expected = dict(requested=mode, transient_eligible=str(int(i>0)),
                            direct_used=str(int(i>0)), prediction_eligible=str(int(prediction)),
                            prediction_attempted=str(int(prediction)))
            if any(record.get(k) != v for k,v in expected.items()):
                return False
            used = record['prediction_used']
            if used not in ('0','1') or (used == '1' and not prediction):
                return False
            seed = float(record['seed_residual'])
            if i > 0 and (not math.isfinite(seed) or seed < 0):
                return False
            if used == '1':
                candidate = float(record['prediction_residual'])
                if not math.isfinite(candidate) or candidate < 0 or candidate >= seed:
                    return False
        return True
    except (KeyError,ValueError,TypeError):
        return False


def valid_measurement(r):
    return (strictly_converged(r,12) and r['termination_reason'] is None
            and math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0
            and all(0 < r[k] <= 8*GIB for k in ('peak_rss_bytes','peak_private_bytes'))
            and sum(int(p['field_reuses']) for p in r['nk_profiles']) == 0)


def summarize(stage, cases, rows, pairs):
    good = [r for r in rows if r['passed']]
    medians = {c: statistics.median(r['wall_seconds'] for r in good if r['case']==c)
               for c in cases if any(r['case']==c for r in good)}
    complete = len(rows)==6 and len(pairs)==3
    passed = complete and len(good)==6 and all(p['passed'] for p in pairs)
    return dict(stage=stage, complete=complete, passed=passed, runs=rows, pairs=pairs,
                median_wall_seconds=medians,
                speed_reduction=1-medians[cases[1]]/medians[cases[0]] if passed else None,
                shortening_in_all_pairs=passed and all(p['speed_reduction']>0 for p in pairs),
                note='12-step aligned flux only; no periodic EMF, release acceptance or automatic stage chaining.')


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    root = args.reference.resolve()
    old = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    prior = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if not prior['complete'] or len(prior['runs'])!=6 or not all(r['converged'] and r['repeat_equal'] for r in prior['runs']):
        raise RuntimeError('Completed direct-start reference required')
    config, image = Path(old['config_path']), Path(old['image_path'])
    if sha256(config)!=old['protocol']['config_sha256'] or sha256(image)!=old['protocol']['image_sha256']:
        raise RuntimeError('Reference input changed')
    base = yaml.safe_load(config.read_text(encoding='utf-8'))
    nl, tr = base['nonlinear_solver'], base['transient']
    if not (tr['total_steps']==12 and tr['parallel_chunks']==1 and tr['slide_pixels_per_step']==12
            and nl['jacobian']=='tangent' and nl['line_search_objective']=='energy'
            and nl['max_iterations']==300 and nl['tolerance']==.001 and not nl['anderson']['enabled']
            and base.get('omp',{}).get('threads',12)==12):
        raise RuntimeError('Unexpected reference physical/solver settings')
    exe = args.exe.resolve()
    ident = identity(exe,repo)
    if ident['dll_sha256']!=old['protocol']['dll_sha256']:
        raise RuntimeError('Runtime DLLs differ from preserved reference')
    selected = variants(args.stage,args.initializer)
    names = [c[0] for c in selected]
    configs = {name+'.yaml':yaml.safe_dump(configuration(base,aa),sort_keys=False) for name,_,aa in selected}
    control_dir = root/'1_direct'
    predictor_pin = None
    if args.stage == 'anderson':
        if not args.predictor_reference:
            raise RuntimeError('Review predictor stage, then specify --predictor-reference and --initializer')
        prediction_root = args.predictor_reference.resolve()
        report = json.loads((prediction_root/'summary.json').read_text(encoding='utf-8'))
        pred = json.loads((prediction_root/'manifest.json').read_text(encoding='utf-8'))
        if not (report['stage']=='predictor' and report['passed'] and report['complete']
                and pred['protocol']['executable_sha256']==ident['executable_sha256']):
            raise RuntimeError('Predictor stage not complete/passed with this binary')
        control_dir = prediction_root/('1_'+args.initializer+'_off')
        predictor_pin = dict(manifest=sha256(prediction_root/'manifest.json'),summary=sha256(prediction_root/'summary.json'))
    reference = json.loads((control_dir/'measurement.json').read_text(encoding='utf-8'))
    if not valid_measurement(reference):
        raise RuntimeError('Preserved numerical control failed')
    reference_fields = fields(control_dir,12)
    protocol = dict(stage=args.stage, variants=selected, executable_sha256=ident['executable_sha256'],
                    dll_sha256=ident['dll_sha256'], source_config_sha256=sha256(config), image_sha256=sha256(image),
                    reference_manifest_sha256=sha256(root/'manifest.json'), reference_summary_sha256=sha256(root/'summary.json'),
                    reference_measurement_sha256=sha256(control_dir/'measurement.json'), reference_fields=reference_fields,
                    predictor_reference=predictor_pin, configs=configs,
                    scripts={s:sha256(Path(__file__).with_name(s)) for s in SCRIPTS},
                    threads=12, mask=6, orders=[names,list(reversed(names)),names],
                    timeout_seconds=7200, memory_limit_bytes=8*GIB, power_plan_guid=power_plan())
    # Record new/untracked implementation files too: git diff alone omits them.
    source_files = ('MagneticFieldAnalyzer.cpp','MagneticFieldAnalyzer.h','MagneticFieldAnalyzer_nonlinear_newton.cpp',
                    'SolverExperiments.h','TransientPredictionHistory.h','tests/solver_numerics_test.cpp',
                    'tests/nonlinear_prediction_sweep_test.cpp','CMakeLists.txt')
    ident['experiment_source_files'] = {name:dict(sha256=sha256(repo/name),text=(repo/name).read_text(encoding='utf-8'))
                                       for name in source_files if name=='TransientPredictionHistory.h'}
    ident['experiment_source_sha256'] = {name:sha256(repo/name) for name in source_files}
    out = args.output.resolve()
    if args.resume:
        saved = json.loads((out/'manifest.json').read_text(encoding='utf-8'))['protocol']
        if saved != json.loads(json.dumps(protocol)):
            raise RuntimeError('Resume protocol mismatch')
        for name,text in configs.items():
            if (out/name).read_text(encoding='utf-8')!=text:
                raise RuntimeError('Generated config changed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        for name,text in configs.items():
            (out/name).write_text(text,encoding='utf-8')
        write_json(out/'manifest.json',dict(protocol=protocol,source=ident,reference=str(root),
                   control_directory=str(control_dir),image_path=str(image),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if args.prepare_only:
        print('VALIDATED '+args.stage+'; no solver started.',flush=True)
        return
    env = {k:v for k,v in os.environ.items() if not k.startswith('OMFDM_') and k!='NK_PROF'}
    env.update(NK_PROF='1',OMP_NUM_THREADS='12',OMFDM_TANGENT_OPTIMIZATIONS='6',OMFDM_TANGENT_PRECONDITIONER='tangent-amg')
    rows, pairs, controls = [], [], {}
    for cycle,order in enumerate(protocol['orders'],1):
        cycle_rows = {}
        for case in order:
            _,mode,aa = next(c for c in selected if c[0]==case)
            target = out/f'{cycle}_{case}'
            if (sha256(exe)!=protocol['executable_sha256'] or sha256(image)!=protocol['image_sha256']
                    or sha256(config)!=protocol['source_config_sha256'] or power_plan()!=protocol['power_plan_guid']
                    or {p.name:sha256(p) for p in exe.parent.glob('*.dll')}!=protocol['dll_sha256']
                    or {s:sha256(Path(__file__).with_name(s)) for s in SCRIPTS}!=protocol['scripts']
                    or any((out/n).read_text(encoding='utf-8')!=t for n,t in configs.items())):
                raise RuntimeError('Experiment identity changed')
            if target.exists():
                if not args.resume or not (target/'measurement.json').exists():
                    raise RuntimeError('Preserving incomplete attempt: '+str(target))
                r = json.loads((target/'measurement.json').read_text(encoding='utf-8'))
                print('REUSE '+target.name,flush=True)
            else:
                if any(p.info['name'] and p.info['name'].lower() in ('magfdmsolver','magfdmsolver.exe') for p in psutil.process_iter(['name'])):
                    raise RuntimeError('Another solver is running')
                if shutil.disk_usage(out).free<5*GIB:
                    raise RuntimeError('Insufficient disk space')
                busy = psutil.cpu_percent(interval=3)*psutil.cpu_count()/100
                write_json(out/(target.name+'_idle.json'),dict(busy_cores=busy,available_memory_bytes=psutil.virtual_memory().available))
                print(f'START {len(rows)+1}/6 {target.name}; background={busy:.2f} cores',flush=True)
                done = threading.Event()
                watcher = threading.Thread(target=monitor,args=(out,target,target.name,12,done),daemon=True)
                watcher.start()
                try:
                    r = run(exe,out/(case+'.yaml'),image,target,dict(env,OMFDM_NK_INITIALIZATION=mode),
                            timeout_seconds=7200,memory_limit_bytes=8*GIB)
                finally:
                    done.set(); watcher.join()
            records = parse_initialization((target/'process.log').read_text(encoding='utf-8',errors='replace'))
            actual_fields = fields(target,12) if strictly_converged(r,12) else {}
            control,control_fields = controls.setdefault(case,(r,actual_fields))
            repeat = (signature(r)==signature(control) and r['step_status']==control['step_status'] and actual_fields==control_fields)
            equivalent = case!=names[0] or (signature(r)==signature(reference) and r['step_status']==reference['step_status'] and actual_fields==reference_fields)
            init_ok = initialization_gate(records,mode)
            aa_evals = sum(int(p['aa_evaluations']) for p in r['nk_profiles'])
            row = {k:r[k] for k in ('wall_seconds','peak_rss_bytes','peak_private_bytes','estimated_other_busy_cores','termination_reason')}
            row.update(case=case,cycle=cycle,converged=strictly_converged(r,12),initialization_gate=init_ok,
                       repeat_equal=repeat,control_equal=equivalent,initialization=records,
                       outer_iterations=sum(int(s['iterations']) for s in r['step_status']),
                       max_final_residual=max((float(s['residual']) for s in r['step_status']),default=None),
                       inner_iterations=sum(int(p['inner_iters']) for p in r['nk_profiles']),
                       prediction_used=sum(p.get('prediction_used')=='1' for p in records),
                       aa_evaluations=aa_evals,aa_accepted=sum(p['aa']=='accepted' for p in r['nk_profiles']),
                       aa_rejected=sum(p['aa']=='rejected' for p in r['nk_profiles']),
                       initialization_ms=sum(float(p['total_ms']) for p in records),
                       aa_ms=sum(float(p['aa_ms']) for p in r['nk_profiles']),
                       passed=valid_measurement(r) and init_ok and repeat and equivalent and ((aa_evals>0) if aa else aa_evals==0))
            rows.append(row); cycle_rows[case]=row
            write_json(out/'summary.json',summarize(args.stage,names,rows,pairs))
            print(f'FINISH {target.name}: {r["wall_seconds"]:.3f}s, outer={row["outer_iterations"]}, predictions={row["prediction_used"]}, passed={row["passed"]}',flush=True)
            if not row['passed']:
                raise RuntimeError('Convergence/initialization/control/repeat gate failed: '+target.name)
        reference_dir,trial_dir = (out/f'{cycle}_{name}'/'results' for name in names)
        a_names,a = load(reference_dir,12); b_names,b = load(trial_dir,12)
        quality = aligned_flux_arrays(a_names,a,b_names,b)
        baseline,candidate = (cycle_rows[name] for name in names)
        memory = all(candidate[k]<=2*baseline[k] for k in ('peak_rss_bytes','peak_private_bytes'))
        pair = dict(cycle=cycle,quality=quality,memory_pass=memory,passed=quality['passed'] and memory,
                    speed_reduction=1-candidate['wall_seconds']/baseline['wall_seconds'])
        pairs.append(pair)
        write_json(out/'summary.json',summarize(args.stage,names,rows,pairs))
        if not pair['passed']:
            raise RuntimeError('Aligned-flux/memory gate failed')
    write_json(out/'progress.json',dict(state='complete',stage=args.stage,completed_runs=6,total_runs=6))
    print('COMPLETE. Review before another stage or full-waveform validation.',flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('exe','reference','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--stage',choices=('predictor','anderson'),default='predictor')
    ap.add_argument('--initializer',choices=('direct','predict'),default='direct')
    ap.add_argument('--predictor-reference',type=Path)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    args = ap.parse_args()
    try:
        execute(args)
    except Exception as e:
        print('STOPPED FOR REVIEW: '+str(e),flush=True)
        if args.output.exists():
            write_json(args.output/'last_error.json',dict(error=str(e),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            write_json(args.output/'progress.json',dict(state='stopped_for_review',error=str(e)))
        raise


if __name__=='__main__':
    main()
