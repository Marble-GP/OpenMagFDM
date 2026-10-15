import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import run_initialization_validation as validation
from run_initialization_validation import (
    GIB, aligned_flux_arrays, cases, configuration, initialization_gate, pair_report,
)


class InitializationValidationTests(unittest.TestCase):
    def log(self, mode, points):
        return '\n'.join(
            '[updateMu: changed=123][NKINIT] requested={} transient_eligible={} '
            'direct_used={} seed_residual={} total_ms=1'.format(
                mode, int(i > 0), int(mode == 'direct' and i > 0),
                '4.14e+00' if mode == 'direct' and i > 0 else '-1')
            for i in range(points))

    def test_suites_and_angles(self):
        full, = cases('full')
        self.assertEqual(full['pixels'] * full['points'], 2976)
        self.assertEqual(full['points'], 248)
        self.assertTrue(full['periodic_quality'])
        large = cases('large')
        self.assertEqual([c['pixels'] for c in large], [48, 124, 248])
        self.assertEqual([c['points'] for c in large], [12]*3)
        self.assertTrue(all(not c['periodic_quality'] for c in large))
        self.assertEqual(cases('all'), [full] + large)
        self.assertAlmostEqual(124*360/2976, 15)
        self.assertAlmostEqual(248*360/2976, 30)
        with self.assertRaises(ValueError):
            cases('unknown')

    def test_only_displacement_and_count_change(self):
        base = dict(transient=dict(total_steps=12, slide_pixels_per_step=12,
                                   parallel_chunks=1, unrelated=[1, 2]),
                    nonlinear_solver=dict(anderson=dict(enabled=False), tolerance=.001))
        original = copy.deepcopy(base)
        for case in cases('all'):
            actual = configuration(base, case)
            self.assertEqual(actual['transient']['total_steps'], case['points'])
            self.assertEqual(actual['transient']['slide_pixels_per_step'], case['pixels'])
            actual['transient']['total_steps'] = 12
            actual['transient']['slide_pixels_per_step'] = 12
            self.assertEqual(actual, original)
            actual['transient']['unrelated'].append(3)
            self.assertEqual(base, original)

    def test_inline_initialization_all_points(self):
        for mode in ('linear', 'direct'):
            for points in (12, 248):
                passed, records = initialization_gate(self.log(mode, points), mode, points)
                self.assertTrue(passed)
                self.assertEqual(len(records), points)

    def test_incomplete_wrong_or_unsafe_initialization(self):
        log = self.log('direct', 12)
        for invalid in (log.replace('transient_eligible=1', 'transient_eligible=0', 1),
                        log.replace('direct_used=0', 'direct_used=1', 1),
                        log.replace('direct_used=1', 'direct_used=0', 1),
                        log.replace('requested=direct', 'requested=linear', 1),
                        self.log('direct', 11), self.log('direct', 13)):
            self.assertFalse(initialization_gate(invalid, 'direct', 12)[0])
        for bad in ('nan', 'inf', '-1', 'bad'):
            self.assertFalse(initialization_gate(log.replace('4.14e+00', bad), 'direct', 12)[0])
        self.assertFalse(initialization_gate(log.replace('seed_residual=4.14e+00', ''), 'direct', 12)[0])
        self.assertFalse(initialization_gate('', 'direct', 0)[0])
        self.assertFalse(initialization_gate(log, 'invalid', 12)[0])

    def test_aligned_flux_no_periodic_claim(self):
        base = np.arange(1, 25, dtype=float).reshape(12, 2)
        good = aligned_flux_arrays(['U', 'V'], base, ['U', 'V'], base*1.0009)
        self.assertTrue(good['passed'])
        self.assertFalse(good['periodic_emf_evaluated'])
        self.assertAlmostEqual(good['channels']['U']['normalized_rms'], .0009)
        self.assertFalse(aligned_flux_arrays(['U', 'V'], base, ['U', 'V'], base*1.0011)['passed'])

    def test_flux_invalid_data_not_a_pass(self):
        base = np.ones((12, 2))
        for names, a, other, b in (
                (['U','V'], base, ['V','U'], base),
                (['U','V'], base, ['U','V'], base[:-1]),
                (['U'], base, ['U'], base),
                (['U','U'], base, ['U','U'], base),
                ([], base, [], base),
                (['U','V'], base[0], ['U','V'], base[0]),
                (['U','V'], base[:0], ['U','V'], base[:0]),
                (['U','V'], base*0, ['U','V'], base),
                (['U','V'], base, ['U','V'], base*np.nan),
                (['U','V'], base*np.inf, ['U','V'], base)):
            with self.assertRaises(ValueError):
                aligned_flux_arrays(names, a, other, b)

    def results(self):
        return {mode: dict(converged=True, initialization_gate=True, field_reuses=0,
                           termination_reason=None, wall_seconds=t,
                           peak_rss_bytes=GIB, peak_private_bytes=GIB)
                for mode, t in (('linear',100), ('direct',70))}

    def test_pair_valid(self):
        report = pair_report(cases('full')[0], self.results(), dict(passed=True))
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['speed_reduction'], .3)

    def test_failed_pair_no_speed_claim(self):
        for field, value in (('converged', False), ('initialization_gate', False),
                             ('field_reuses', 1), ('termination_reason', 'memory_limit'),
                             ('wall_seconds', 0), ('wall_seconds', float('nan')),
                             ('peak_private_bytes', 3*GIB), ('peak_rss_bytes', 0)):
            results = self.results()
            results['direct'][field] = value
            report = pair_report(cases('full')[0], results, dict(passed=True))
            self.assertFalse(report['passed'], field)
            self.assertIsNone(report['speed_reduction'], field)
        for quality in (False, None):
            report = pair_report(cases('full')[0], self.results(), dict(passed=quality))
            self.assertFalse(report['passed'])
            self.assertIsNone(report['speed_reduction'])

    def test_memory_absolute_limit_both_modes(self):
        for mode in ('linear', 'direct'):
            result = self.results()
            result[mode]['peak_private_bytes'] = 9*GIB
            self.assertFalse(pair_report(cases('full')[0], result, dict(passed=True))['memory_pass'])


class ValidationOrchestrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        exe = root/'solver.exe'
        exe.write_bytes(b'fixture, not an executable')
        image = root/'model.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\n' + bytes(8) + struct.pack('>II', 450, 2976))
        config = root/'source.yaml'
        config.write_text(yaml.safe_dump(dict(
            transient=dict(parallel_chunks=1, enable_sliding=True, slide_direction='vertical',
                           total_steps=12, slide_pixels_per_step=12),
            nonlinear_solver=dict(jacobian='tangent', line_search_objective='energy',
                                  tolerance=.001, max_iterations=300, anderson=dict(enabled=False)),
        )), encoding='utf-8')
        parent = root/'prior'
        parent.mkdir()
        (parent/'manifest.json').write_text(json.dumps(dict(
            image_path=str(image), config_path=str(config), protocol=dict(
                config_sha256=validation.sha256(config), image_sha256=validation.sha256(image),
                executable_sha256=validation.sha256(exe), dll_sha256={})
        )), encoding='utf-8')
        (parent/'summary.json').write_text(json.dumps(dict(complete=True, runs=[dict(
            converged=True, initialization_gate=True, repeat_equal=True)]*6)), encoding='utf-8')
        self.args = SimpleNamespace(exe=exe, reference=parent, output=root/'out',
                                    suite='all', prepare_only=False, resume=False)
        for name, value in (
                ('power_plan', 'fixture-guid'), ('identity', {}), ('monitor', None),
                ('psutil.process_iter', []), ('psutil.cpu_percent', 0), ('psutil.cpu_count', 12),
                ('psutil.virtual_memory', SimpleNamespace(available=16*GIB)),
                ('shutil.disk_usage', SimpleNamespace(free=100*GIB)),
                ('compare', dict(passed=True)),
                ('load', (['U'], np.ones((12, 1))))):
            p = patch('run_initialization_validation.'+name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = patch('run_initialization_validation.run', side_effect=self.fake_run)
        self.run = p.start()
        self.addCleanup(p.stop)

    def fake_run(self, exe, config, image, target, env, **kwargs):
        target.mkdir()
        points = yaml.safe_load(config.read_text(encoding='utf-8'))['transient']['total_steps']
        mode = env['OMFDM_NK_INITIALIZATION']
        self.assertEqual(env['OMFDM_TANGENT_OPTIMIZATIONS'], '6')
        self.assertEqual(env['OMP_NUM_THREADS'], '12')
        self.assertEqual(kwargs['memory_limit_bytes'], 8*GIB)
        (target/'process.log').write_text(InitializationValidationTests().log(mode, points), encoding='utf-8')
        result = dict(wall_seconds=100, peak_rss_bytes=GIB, peak_private_bytes=GIB,
                      estimated_other_busy_cores=0, termination_reason=None, converged=True,
                      exit_code=0, nk_profiles=[dict(field_reuses='0')]*points,
                      step_status=[dict(step=str(i), strict_converged='1', residual='0.0009',
                                        iterations='10') for i in range(points)])
        (target/'measurement.json').write_text(json.dumps(result), encoding='utf-8')
        return result

    def test_prepare_and_resume_complete_without_rerunning(self):
        self.args.prepare_only = True
        validation.execute(self.args)
        self.run.assert_not_called()
        self.assertEqual(len(list(self.args.output.glob('*.yaml'))), 4)
        self.args.prepare_only = False
        self.args.resume = True
        validation.execute(self.args)
        self.assertEqual(self.run.call_count, 8)
        validation.execute(self.args)
        self.assertEqual(self.run.call_count, 8)
        result = json.loads((self.args.output/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(result['complete'] and result['passed'])

    def test_convergence_failure_stops_next_launch(self):
        def failed(*a, **kw):
            result = self.fake_run(*a, **kw)
            result['converged'] = False
            return result
        self.run.side_effect = failed
        with self.assertRaisesRegex(RuntimeError, 'Convergence/initialization gate'):
            validation.execute(self.args)
        self.assertEqual(self.run.call_count, 1)

    def test_quality_failure_stops_next_case(self):
        with patch.object(validation, 'compare', return_value=dict(passed=False)):
            with self.assertRaisesRegex(RuntimeError, 'Quality/memory gate'):
                validation.execute(self.args)
        self.assertEqual(self.run.call_count, 2)

    def test_changed_input_and_incomplete_attempt_are_preserved(self):
        self.args.prepare_only = True
        validation.execute(self.args)
        self.args.prepare_only = False
        self.args.resume = True
        config = self.args.output/'full.yaml'
        original = config.read_text(encoding='utf-8')
        config.write_text(original+'# changed\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
            validation.execute(self.args)
        self.run.assert_not_called()
        config.write_text(original, encoding='utf-8')
        target = self.args.output/'full_linear'
        target.mkdir()
        (target/'partial.txt').write_text('keep me', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Preserving incomplete attempt'):
            validation.execute(self.args)
        self.run.assert_not_called()
        self.assertEqual((target/'partial.txt').read_text(encoding='utf-8'), 'keep me')


if __name__ == '__main__':
    unittest.main()
