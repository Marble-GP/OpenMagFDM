import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from compare_waveforms import compare


def fixture(root, values, converged=True):
    d = root / "FluxLinkage"
    d.mkdir(parents=True)
    with (d / "flux_linkage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "coil"])
        w.writerows(enumerate(values))
    with (d / "flux_linkage_status.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "strict_converged", "residual"])
        w.writerows((i, int(converged), 1e-4 if converged else 1) for i in range(len(values)))


class WaveformsTest(unittest.TestCase):
    def test_gates(self):
        n = 248
        theta = np.arange(n) * (2*np.pi/n)
        reference = np.sin(2*theta)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root/"base", reference)
            fixture(root/"same", reference)
            fixture(root/"scaled", 1.01*reference)
            fixture(root/"noise", reference + 0.002*np.sin(37*theta))
            fixture(root/"unfinished", reference, converged=False)
            fixture(root/"short", reference[:-1])
            self.assertTrue(compare(root/"base",root/"same")["passed"])
            self.assertFalse(compare(root/"base",root/"scaled")["passed"])
            self.assertFalse(compare(root/"base",root/"noise")["passed"])
            with self.assertRaises(ValueError): compare(root/"unfinished",root/"same")
            with self.assertRaises(ValueError): compare(root/"base",root/"short")


if __name__ == "__main__":
    unittest.main()
