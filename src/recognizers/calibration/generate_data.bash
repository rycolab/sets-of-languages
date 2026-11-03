set -euo pipefail
. recognizers/functions.bash

usage() {
  echo "Usage: bash $0 <base-directory> <language-class> <distribution-no> <language-no> <device>

Perform all data generation and preprocessing for a randomly sampled language
and run the baseline classifier on it.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language-class>
    One of: podfa, star-free, regular, context-free
  <distribution-no>
    An integer distinguishing which random distribution of languages to sample
    from.
  <language-no>
    An index distinguishing a language in this class.
  <device>
    Either 'cpu' or 'gpu'.
"
}

base_dir=${1-}
language_class=${2-}
distribution_no=${3-}
language_no=${4-}
device=${5-}
if ! shift 5; then
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

distribution_dir=$base_dir/calibration/$language_class/$distribution_no
language_dir=$distribution_dir/$language_no
mkdir -p "$language_dir"
language_file=$language_dir/language.pt
sample_args=( \
  --hyperparameters "$distribution_dir"/hyperparameters.json \
  --random-seed "$(language_sampling_seed "$language_class" "$language_no")" \
  --output "$language_file" \
)
case $language_class in
  podfa)
    python recognizers/language_sampling/save_random_podfa_language.py "${sample_args[@]}"
    ;;
  star-free)
    python recognizers/language_sampling/save_random_star_free_language.py "${sample_args[@]}"
    ;;
  regular)
    python recognizers/language_sampling/save_random_regular_language.py "${sample_args[@]}"
    ;;
  context-free)
    python recognizers/language_sampling/save_random_context_free_language.py "${sample_args[@]}" --device "$device"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
sampler_file=$language_dir/sampler.pt
echo "writing $sampler_file"
python recognizers/string_sampling/prepare_sampler.py \
  --input "$language_file" \
  --output "$sampler_file" \
  --max-length 500 \
  --device "$device"
python recognizers/dataset_generation/generate_datasets.py \
  --sampler "$sampler_file" \
  --random-seed "$STRING_SAMPLING_SEED" \
  --output "$language_dir" \
  --datasets training test \
  --device "$device" \
  --num-training-examples 10000 \
  --num-test-examples 1000
baseline_file=$language_dir/baseline.txt
echo "writing $baseline_file"
python recognizers/calibration/baseline_classifier.py \
  "$language_dir" \
  > "$baseline_file"
cat "$baseline_file"
