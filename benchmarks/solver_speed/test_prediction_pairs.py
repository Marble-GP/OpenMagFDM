import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import run_prediction_pairs as trial


def records(mode):
    return [dict(requested=mode, transient_eligible=str(int(i>0)),direct_used=str(int(i>0)),
                 seed_residual='1' if i else '-1', prediction_eligible=str(int(mode=='predict' and i>=2)),
                 prediction_attempted=str(int(mode=='predict' and i>=2)),
                 prediction_used=str(int(mode=='predict' and i>=2)),prediction_residual='.5',total_ms='2')
            for i in range(12)]


def measurement(aa=False):
    return dict(wall_seconds=100,peak_rss_bytes=trial.GIB,peak_private_bytes=trial.GIB,
                estimated_other_busy_cores=0,termination_reason=None,converged=True,exit_code=0,
                step_status=[dict(step=str(i),strict_converged='1',residual='.0009',iterations='5') for i in range(12)],
                nk_profiles=[dict(iter='1',residual='1',inner_iters='3',ls_trials='1',outcome='update',
                                  field_reuses='0',aa_evaluations='2' if aa else '0',
                                  aa='accepted' if aa else 'inactive',aa_ms='1')]*12)


class PredictionGateTests(unittest.TestCase):
    def test_valid_modes_and_rejected_proposals(self):
        self.assertTrue(trial.initialization_gate(records('direct'),'direct'))
        self.assertTrue(trial.initialization_gate(records('predict'),'predict'))
        rejected = records('predict')
        for record in rejected[2:]:
            record['prediction_used']='0'; record['prediction_residual']='nan'
        self.assertTrue(trial.initialization_gate(rejected,'predict'))

    def test_cold_history_missing_or_nonfinite_accepted_prediction_rejected(self):
        for key,value,index in [('prediction_used','1',0),('prediction_eligible','1',1),
                                ('prediction_attempted','0',2),('prediction_residual','nan',2),
                                ('prediction_residual','1',2),('seed_residual','inf',2),
                                ('prediction_used','2',2),('requested','direct',2)]:
            broken=records('predict'); broken[index][key]=value
            self.assertFalse(trial.initialization_gate(broken,'predict'),(key,value,index))
        self.assertFalse(trial.initialization_gate(records('predict')[:-1],'predict'))
        broken=records('predict'); del broken[2]['prediction_used']
        self.assertFalse(trial.initialization_gate(broken,'predict'))

    def test_fixed_anderson_and_no_changes_to_source(self):
        base=dict(nonlinear_solver=dict(anderson=dict(enabled=False,depth=5),tolerance=.001),mesh=dict(dx=.001))
        before=copy.deepcopy(base)
        on=trial.configuration(base,True)
        self.assertEqual(base,before)
        self.assertEqual(on['nonlinear_solver']['anderson'],dict(enabled=True,depth=3,beta=.3))
        self.assertEqual(trial.configuration(base,False),before)
        self.assertEqual(trial.variants('predictor'),[('direct_off','direct',False),('predict_off','predict',False)])
        self.assertEqual(trial.variants('anderson','predict'),[('predict_off','predict',False),('predict_aa','predict',True)])

    def test_memory_nonconvergence_and_priority4(self):
        self.assertTrue(trial.valid_measurement(measurement()))
        for k,v in [('wall_seconds',float('nan')),('peak_private_bytes',9*trial.GIB),
                    ('peak_rss_bytes',0),('termination_reason','timeout'),('exit_code',2)]:
            r=measurement();r[k]=v
            self.assertFalse(trial.valid_measurement(r))
        r=measurement();r['nk_profiles'][0]['field_reuses']='1'
        self.assertFalse(trial.valid_measurement(r))

    def test_partial_or_failed_summaries_have_no_speed_claim(self):
        self.assertFalse(trial.summarize('predictor',['a','b'],[],[])['passed'])
        self.assertIsNone(trial.summarize('predictor',['a','b'],[],[])['speed_reduction'])


class PredictionOrchestrationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.parent=self.root/'reference';self.parent.mkdir()
        self.exe=self.root/'solver.exe';self.exe.write_bytes(b'fixture only')
        image=self.root/'input.png';image.write_bytes(b'fixture only')
        config=self.root/'input.yaml'
        config.write_text(yaml.safe_dump(dict(transient=dict(total_steps=12,parallel_chunks=1,slide_pixels_per_step=12),
                  nonlinear_solver=dict(jacobian='tangent',line_search_objective='energy',max_iterations=300,
                                        tolerance=.001,anderson=dict(enabled=False)))),encoding='utf-8')
        (self.parent/'manifest.json').write_text(json.dumps(dict(config_path=str(config),image_path=str(image),
                    protocol=dict(config_sha256=trial.sha256(config),image_sha256=trial.sha256(image),dll_sha256={}))),encoding='utf-8')
        (self.parent/'summary.json').write_text(json.dumps(dict(complete=True,runs=[dict(converged=True,repeat_equal=True)]*6)),encoding='utf-8')
        control=self.parent/'1_direct';control.mkdir()
        (control/'measurement.json').write_text(json.dumps(measurement()),encoding='utf-8')
        self.args=SimpleNamespace(reference=self.parent,exe=self.exe,output=self.root/'out',stage='predictor',
                                  initializer='direct',predictor_reference=None,resume=False,prepare_only=False)
        for name,result in [('power_plan','plan'),('identity',dict(executable_sha256=trial.sha256(self.exe),dll_sha256={})),
                            ('fields',{'Az/step_1.tiff':'same'}),('monitor',None),('psutil.process_iter',[]),
                            ('psutil.cpu_percent',0),('psutil.cpu_count',12),
                            ('psutil.virtual_memory',SimpleNamespace(available=16*trial.GIB)),
                            ('shutil.disk_usage',SimpleNamespace(free=100*trial.GIB)),
                            ('load',(['U'],np.ones((12,1))))]:
            p=patch('run_prediction_pairs.'+name,return_value=result);p.start();self.addCleanup(p.stop)
        p=patch('run_prediction_pairs.run',side_effect=self.fake_run)
        self.run=p.start();self.addCleanup(p.stop)

    def fake_run(self,exe,config,image,target,env,**kwargs):
        target.mkdir()
        self.assertEqual(env['OMFDM_TANGENT_OPTIMIZATIONS'],'6')
        self.assertEqual(env['OMP_NUM_THREADS'],'12')
        self.assertEqual(kwargs['memory_limit_bytes'],8*trial.GIB)
        mode=env['OMFDM_NK_INITIALIZATION']
        (target/'process.log').write_text('\n'.join('[updateMu][NKINIT] '+' '.join(f'{k}={v}' for k,v in r.items())
                                        for r in records(mode)),encoding='utf-8')
        aa=yaml.safe_load(config.read_text(encoding='utf-8'))['nonlinear_solver']['anderson']['enabled']
        r=measurement(aa)
        (target/'measurement.json').write_text(json.dumps(r),encoding='utf-8')
        return r

    def test_prepare_resume_and_reuse_without_extra_solver(self):
        self.args.prepare_only=True;trial.execute(self.args);self.run.assert_not_called()
        self.args.prepare_only=False;self.args.resume=True;trial.execute(self.args)
        self.assertEqual(self.run.call_count,6)
        trial.execute(self.args);self.assertEqual(self.run.call_count,6)
        result=json.loads((self.args.output/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(result['complete'] and result['passed'])
        self.assertEqual(result['runs'][1]['prediction_used'],10)

    def test_anderson_requires_finished_gate_then_uses_selected_initializer(self):
        self.args.stage='anderson'
        with self.assertRaisesRegex(RuntimeError,'Review predictor'):trial.execute(self.args)
        self.run.assert_not_called()
        self.args.stage='predictor';trial.execute(self.args)
        original=self.args.output
        self.args.stage='anderson';self.args.initializer='predict'
        self.args.predictor_reference=original;self.args.output=self.root/'aa'
        trial.execute(self.args)
        result=json.loads((self.args.output/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(result['passed'])
        self.assertGreater(result['runs'][1]['aa_evaluations'],0)
        self.assertEqual(self.run.call_count,12)

    def test_failure_stops_without_starting_next_case(self):
        def fail(*a,**kw):
            r=self.fake_run(*a,**kw);r['exit_code']=2;return r
        self.run.side_effect=fail
        with self.assertRaisesRegex(RuntimeError,'gate failed'):trial.execute(self.args)
        self.assertEqual(self.run.call_count,1)
        report=json.loads((self.args.output/'summary.json').read_text(encoding='utf-8'))
        self.assertFalse(report['passed']);self.assertIsNone(report['speed_reduction'])

    def prepare_direct_anderson(self):
        # A rejected predictor is still a valid completed safety/numerical gate.
        # DIRECT, not PREDICT, is selected for the independent AA experiment.
        def rejected_predictor(*a, **kw):
            r = self.fake_run(*a, **kw)
            target = a[3]
            log = (target/'process.log').read_text(encoding='utf-8')
            (target/'process.log').write_text(log.replace('prediction_used=1', 'prediction_used=0'), encoding='utf-8')
            return r
        self.run.side_effect = rejected_predictor
        trial.execute(self.args)
        self.args.predictor_reference = self.args.output
        self.args.output = self.root/'direct_aa'
        self.args.stage = 'anderson'
        self.args.initializer = 'direct'
        self.run.reset_mock()
        self.run.side_effect = self.fake_run

    def test_direct_anderson_ablation_prepare_order_and_pinned_control(self):
        self.prepare_direct_anderson()
        self.args.prepare_only = True
        trial.execute(self.args)
        self.run.assert_not_called()
        off = yaml.safe_load((self.args.output/'direct_off.yaml').read_text(encoding='utf-8'))
        on = yaml.safe_load((self.args.output/'direct_aa.yaml').read_text(encoding='utf-8'))
        self.assertEqual(on['nonlinear_solver']['anderson'], dict(enabled=True, depth=3, beta=.3))
        on['nonlinear_solver']['anderson'] = off['nonlinear_solver']['anderson']
        self.assertEqual(on, off)
        manifest = json.loads((self.args.output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(Path(manifest['control_directory']).resolve(),
                         (self.args.predictor_reference/'1_direct_off').resolve())
        self.args.prepare_only = False
        self.args.resume = True
        trial.execute(self.args)
        self.assertEqual([c.args[3].name for c in self.run.call_args_list],
                         ['1_direct_off', '1_direct_aa', '2_direct_aa', '2_direct_off', '3_direct_off', '3_direct_aa'])
        self.assertTrue(all(c.args[4]['OMFDM_NK_INITIALIZATION'] == 'direct' for c in self.run.call_args_list))
        result = json.loads((self.args.output/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(result['complete'] and result['passed'])
        self.assertTrue(all(r['prediction_used'] == 0 for r in result['runs']))
        trial.execute(self.args)
        self.assertEqual(self.run.call_count, 6)  # no duplicate solver on resume

    def test_anderson_on_without_actual_evaluations_stops(self):
        self.prepare_direct_anderson()
        def no_evaluations(*a, **kw):
            r = self.fake_run(*a, **kw)
            for p in r['nk_profiles']:
                p['aa_evaluations'] = '0'
            return r
        self.run.side_effect = no_evaluations
        with self.assertRaisesRegex(RuntimeError, 'gate failed: 1_direct_aa'):
            trial.execute(self.args)
        self.assertEqual(self.run.call_count, 2)

    def test_anderson_off_with_unexpected_evaluations_stops(self):
        self.prepare_direct_anderson()
        def unexpected_evaluations(*a, **kw):
            r = self.fake_run(*a, **kw)
            r['nk_profiles'][0]['aa_evaluations'] = '1'
            return r
        self.run.side_effect = unexpected_evaluations
        with self.assertRaisesRegex(RuntimeError, 'gate failed: 1_direct_off'):
            trial.execute(self.args)
        self.assertEqual(self.run.call_count, 1)

    def test_anderson_final_residual_failure_stops_without_speed_claim(self):
        self.prepare_direct_anderson()
        def bad_final(*a, **kw):
            r = self.fake_run(*a, **kw)
            if a[3].name.endswith('_aa'):
                r['step_status'][-1]['residual'] = '.002'
            return r
        self.run.side_effect = bad_final
        with self.assertRaisesRegex(RuntimeError, 'gate failed: 1_direct_aa'):
            trial.execute(self.args)
        self.assertEqual(self.run.call_count, 2)
        result = json.loads((self.args.output/'summary.json').read_text(encoding='utf-8'))
        self.assertFalse(result['passed'])
        self.assertIsNone(result['speed_reduction'])

    def test_modified_config_and_incomplete_attempt_preserved(self):
        self.args.prepare_only=True;trial.execute(self.args)
        self.args.prepare_only=False;self.args.resume=True
        config=self.args.output/'direct_off.yaml';text=config.read_text(encoding='utf-8')
        config.write_text(text+'#changed',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Generated config changed'):trial.execute(self.args)
        config.write_text(text,encoding='utf-8')
        target=self.args.output/'1_direct_off';target.mkdir()
        (target/'partial').write_text('keep',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Preserving incomplete'):trial.execute(self.args)
        self.assertEqual((target/'partial').read_text(encoding='utf-8'),'keep');self.run.assert_not_called()

    def test_flux_failure_stops_after_one_pair(self):
        with patch.object(trial,'aligned_flux_arrays',return_value=dict(passed=False)):
            with self.assertRaisesRegex(RuntimeError,'Aligned-flux'):trial.execute(self.args)
        self.assertEqual(self.run.call_count,2)


if __name__=='__main__':
    unittest.main()
