set -euo pipefail

usage() {
  echo "Usage: bash $0 <base-dir> --language-classes ... --distributions ..."
}

base_dir=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi
language_classes=()
distributions=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --language-classes)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        language_classes+=("$1")
        shift
      done
      ;;
    --distributions)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        distributions+=("$1")
        shift
      done
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
done

for language_class in "${language_classes[@]}"; do
  for distribution_no in "${distributions[@]}"; do
    python recognizers/calibration/generate_random_language_distribution.py \
      --language-class "$language_class" \
      --distribution-no "$distribution_no" \
      --output "$base_dir"/calibration/"$language_class"/"$distribution_no"/hyperparameters.json
  done
done

