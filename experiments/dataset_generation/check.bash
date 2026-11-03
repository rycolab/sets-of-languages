set -euo pipefail

. experiments/include.bash

for d in "$BASE_DIR"/languages/random-*/datasets/test; do
  if [[ ! -f $d/main.tok ]]; then
    echo missing $d/main.tok
  elif [[ $(wc -l < $d/main.tok) -ne 5010 ]]; then
    echo incomplete $d/main.tok
  fi
done
