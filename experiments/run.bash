set -euo pipefail

usage() {
  echo "Usage: bash $0 <device> <command>...

  <device>    One of: cpu, gpu
"
}

device=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi

exec bash experiments/singularity_exec.bash "$device" bash -c 'cd src && exec bash ../scripts/poetry_run.bash "$@"' -- "$@"
