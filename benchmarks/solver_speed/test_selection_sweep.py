import copy
import unittest

from run_selection_sweep import GIB, strictly_converged, timing_gate


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(cycle=i, case=c, wall_seconds=t, strict_converged=True,
                     repeat_history_equal=True, peak_private_bytes=m, peak_rss_bytes=m)
                for i in (1, 2, 3) for c, t, m in
                (('secant', 100, GIB), ('candidate', 80, int(1.8*GIB)))]

    def test_success(self):
        self.assertTrue(timing_gate(self.rows())['proceed_to_waveforms'])

    def test_no_premature_selection(self):
        self.assertFalse(timing_gate(self.rows()[:2])['proceed_to_waveforms'])

    def test_failures_block_selection(self):
        for field, value in (('strict_converged', False), ('repeat_history_equal', False),
                             ('peak_private_bytes', 3*GIB), ('wall_seconds', 101)):
            rows = self.rows()
            rows[1][field] = value
            self.assertFalse(timing_gate(rows)['proceed_to_waveforms'], field)

    def test_insufficient_speedup(self):
        rows = self.rows()
        for row in rows:
            if row['case'] == 'candidate':
                row['wall_seconds'] = 86
        self.assertFalse(timing_gate(rows)['proceed_to_waveforms'])

    def test_duplicate_cycles_rejected(self):
        rows = self.rows()
        rows[-1]['cycle'] = 2
        self.assertFalse(timing_gate(rows)['proceed_to_waveforms'])

    def test_strict_status(self):
        r = dict(converged=True, exit_code=0, termination_reason=None,
                 step_status=[dict(step=str(i), strict_converged='1', residual='0.0009')
                              for i in range(12)])
        self.assertTrue(strictly_converged(r, 12))
        self.assertFalse(strictly_converged(r, 248))
        for bad in ('nan', 'inf', '0.0011', '-1'):
            changed = copy.deepcopy(r)
            changed['step_status'][0]['residual'] = bad
            self.assertFalse(strictly_converged(changed, 12))


if __name__ == '__main__':
    unittest.main()
