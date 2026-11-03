SINGULARITY_IMAGE_FILE=/cluster/project/cotterell/bdusell/container-images/private-nn-formal-languages-2026-04-29.sif
# BASE_DIR is a directory where all datasets, models, figures, etc. will be
# stored. By default it is results. You might want to change it to a directory
# of your own under /cluster/project/cotterell/$USER.
# The results for the ICLR 2025 submission are at
# /cluster/work/cotterell/bdusell/logs/2024-09-26/private-nn-formal-languages-results
BASE_DIR=$(cd "$(dirname "$BASH_SOURCE")"/.. && pwd)/results
# Extra arguments that will be passed to Singularity. You might need to add
# --bind arguments to give Singularity access to additional directories.
SINGULARITY_ARGS=()
if [[ -e ignore/config_override.bash ]]; then
  . ignore/config_override.bash
fi
