import unittest
import copy
from run_initialization_pairs import summary, parse_initialization, validate_resume


class InitializationPairsTests(unittest.TestCase):
    def rows(self):
        return [dict(case=c, converged=True, initialization_gate=True, repeat_equal=True,
                     wall_seconds=t) for _ in range(3) for c,t in (('linear',100),('direct',90))]

    def test_complete(self):
        result = summary(self.rows())
        self.assertTrue(result['complete'])
        self.assertAlmostEqual(result['speed_reduction'], .1)

    def test_incomplete_or_failed_not_speed_claim(self):
        self.assertIsNone(summary(self.rows()[:2])['speed_reduction'])
        for flag in ('converged', 'initialization_gate', 'repeat_equal'):
            rows = self.rows()
            rows[-1][flag] = False
            self.assertIsNone(summary(rows)['speed_reduction'])

    def test_inline_records(self):
        cold = '[NKINIT] requested=direct transient_eligible=0 direct_used=0 seed_residual=-1 total_ms=2e+03'
        warm = '[NKINIT] requested=direct transient_eligible=1 direct_used=1 seed_residual=4.14e+00 total_ms=64.3'
        log = cold+'\r\n'+''.join(' [updateMu: changed=123]'+warm+'\n' for _ in range(11))
        parsed = parse_initialization(log)
        self.assertEqual(len(parsed), 12)
        self.assertEqual(sum(int(p['direct_used']) for p in parsed), 11)
        self.assertNotIn('changed', parsed[1])
        self.assertEqual(parsed[1]['seed_residual'], '4.14e+00')
        self.assertEqual(parse_initialization('unrelated log'), [])
        self.assertEqual(len(parse_initialization(cold+warm)), 2)

    def test_strict_resume_migration(self):
        old = dict(binary='same', inputs='same', scripts={'run_initialization_pairs.py':'old', 'engine':'same'})
        new = copy.deepcopy(old)
        new['scripts']['run_initialization_pairs.py'] = 'fixed'
        with self.assertRaises(RuntimeError):
            validate_resume(old, new)
        validate_resume(old, new, True)
        self.assertEqual(old['scripts']['run_initialization_pairs.py'], 'old')
        for key in ('binary', 'inputs'):
            changed = copy.deepcopy(new)
            changed[key] = 'different'
            with self.assertRaises(RuntimeError):
                validate_resume(old, changed, True)
        new['scripts']['engine'] = 'different'
        with self.assertRaises(RuntimeError):
            validate_resume(old, new, True)


if __name__ == '__main__':
    unittest.main()
