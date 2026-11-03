set -euo pipefail
. recognizers/functions.bash

usage() {
  echo "Usage: bash $0 <base-directory> <language>

Compute the baseline classifier accuracy for a language.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language.
"
}

base_dir=${1-}
language=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

language_dir=$(get_language_dir "$base_dir" "$language")
baseline_file=$language_dir/baseline.txt
echo "writing $baseline_file"
python recognizers/calibration/baseline_classifier.py "$language_dir" | tee "$baseline_file"
