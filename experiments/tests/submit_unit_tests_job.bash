set -euo pipefail
. experiments/include.bash

device=cpu
submit_job \
  unit-tests \
  "$device" \
  --time=60:00 \
  --mem-per-cpu=2G \
  -- \
  bash -c 'cd .. && bash scripts/run_tests.bash'
