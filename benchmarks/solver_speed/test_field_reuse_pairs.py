"""Runner regression checks; these do not launch a solver."""
import unittest

from run_field_reuse_pairs import signature, summarize


class FieldReusePairsTests(unittest.TestCase):
    def row(self, case, seconds, **extra):
        return dict(case=case, wall_seconds=seconds, converged=True,
                    equivalent=True, reuse_gate=True) | extra

    def test_incomplete_has_no_speed_claim(self):
        report = summarize([self.row('opt23', 100), self.row('opt234', 90)])
        self.assertFalse(report['complete'])
        self.assertIsNone(report['speed_reduction'])

    def test_failed_attempts_do_not_enter_medians(self):
        rows = [self.row('opt23', 100), self.row('opt234', 90)]
        for flag in ('converged', 'equivalent', 'reuse_gate'):
            rows.append(self.row('opt234', 1, **{flag: False}))
        report = summarize(rows)
        self.assertEqual(report['accepted_runs'], 2)
        self.assertEqual(report['median_wall_seconds']['opt234'], 90)

    def test_three_pairs(self):
        rows = [self.row('opt23', t) for t in (100, 110, 120)]
        rows += [self.row('opt234', t) for t in (90, 95, 100)]
        report = summarize(rows)
        self.assertTrue(report['complete'])
        self.assertAlmostEqual(report['speed_reduction'], 1-95/110)

    def test_signature_ignores_timings_but_not_numerics(self):
        p = dict(iter='1', residual='0.001', inner_iters='3', ls_trials='2', outcome='update')
        a = dict(nk_profiles=[dict(p, field_reuses='0', total_ms='100')])
        b = dict(nk_profiles=[dict(p, field_reuses='1', total_ms='90')])
        self.assertEqual(signature(a), signature(b))
        b['nk_profiles'][0]['residual'] = '0.002'
        self.assertNotEqual(signature(a), signature(b))


if __name__ == '__main__':
    unittest.main()
