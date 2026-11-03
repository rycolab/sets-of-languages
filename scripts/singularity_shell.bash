set -euo pipefail

. scripts/variables.bash

exec singularity shell --nv "$SINGULARITY_IMAGE".sif
