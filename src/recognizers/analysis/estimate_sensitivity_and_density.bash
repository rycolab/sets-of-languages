set -euo pipefail
. recognizers/functions.bash

STRING_SAMPLING_SEED=123456789

usage() {
  echo "Usage: bash $0 <base-directory> <language> <device>

Estimate the sensitivity and density of a language.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to prepare.
  <device>
    Either 'cpu' or 'gpu'.
"
}

base_dir=${1-}
language=${2-}
device=${3-}
if ! shift 3; then
  usage >&2
  exit 1
fi

case $device in
  cpu) device=cpu ;;
  gpu) device=cuda ;;
  *)
    usage >&2
    exit 1
    ;;
esac

language_dir=$(get_language_dir "$base_dir" "$language")
python recognizers/analysis/estimate_sensitivity_and_density.py \
  --language "$language_dir"/language.pt \
  --min-length 0 \
  --max-length 500 \
  --num-samples 1000 \
  --random-seed "$STRING_SAMPLING_SEED" \
  --device "$device" \
  --sensitivity-output "$language_dir"/sensitivity.txt \
  --density-output "$language_dir"/density.txt
