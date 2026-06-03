#!/bin/bash
# Anti-periodic BC benchmark driver (Phase B.4).
#
# Usage: bash run_test.sh /path/to/MagFDMsolver
#
# 1. Generates full_ring.png and half_ring.png.
# 2. Runs the solver on full_circle_ref.yaml and half_circle_ap.yaml.
# 3. Calls compare_az.py and reports PASS / FAIL.
#
# Exit codes: 0 = PASS, 1 = FAIL, 2 = setup error.
set -u
cd "$(dirname "$0")"

SOLVER="${1:-}"
if [ -z "$SOLVER" ] || [ ! -x "$SOLVER" ]; then
    echo "FAIL  solver binary not found / not executable: '$SOLVER'"
    echo "      Usage: bash run_test.sh /path/to/MagFDMsolver"
    exit 2
fi

echo "[setup]      writing full_ring.png (360x20) and half_ring.png (180x20)"
python3 make_test_image.py || exit 2

mkdir -p output_full output_half

echo "[full]       running $SOLVER on full_circle_ref.yaml ..."
"$SOLVER" full_circle_ref.yaml full_ring.png output_full > output_full.log 2>&1
if [ ! -d "output_full/Az" ]; then
    echo "FAIL  full-circle solve did not produce Az/. See output_full.log:"
    tail -20 output_full.log
    exit 2
fi

echo "[half]       running $SOLVER on half_circle_ap.yaml ..."
"$SOLVER" half_circle_ap.yaml half_ring.png output_half > output_half.log 2>&1
if [ ! -d "output_half/Az" ]; then
    echo "FAIL  half-circle AP solve did not produce Az/. See output_half.log:"
    tail -20 output_half.log
    exit 2
fi

python3 compare_az.py output_full output_half
exit $?
