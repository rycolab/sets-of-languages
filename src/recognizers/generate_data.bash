set -euo pipefail
. recognizers/functions.bash

usage() {
  echo "Usage: bash $0 <base-directory> <language> <device> [options...]

Perform all preprocessing and data generation from scratch for a particular
language. The name of the language determines whether the language is randomly
sampled and, if so, what kind of language class it is sampled from. The results
of this script are always deterministic; an integer in the language name can be
used to distinguish randomly sampled languages generated with different seeds.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to prepare.
    In the following, <n> is an integer that distinguishes different randomly
    sampled languages.
    Choices:
    - random-regular-<n>
      A randomly sampled regular language.
    - random-context-free-<n>
      A randomly sampled context-free language.
    - random-podfa-<n>
      A randomly sampled partially-ordered DFA language.
    - random-star-free-<n>
      A randomly sampled star-free language.
    - The name of a hand-picked language accepted by
      recognizers/hand_picked_languages/save_language.py.
    - The name of a hand-picked language accepted by
      recognizers/dataset_generation/generate_datasets.py.
  <device>
    Either 'cpu' or 'gpu'.
  options...
    Extra arguments to recognizers/dataset_generation/generate_datasets.py. This
    can include --max-test-length, which will also affect the maximum length
    that the sampler is prepared for.
"
}

base_dir=${1-}
language=${2-}
device=${3-}
if ! shift 3; then
  usage >&2
  exit 1
fi

generate_datasets_args=()
sample_random_language_args=()
max_test_length=500
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-test-length)
      generate_datasets_args+=("$1" "$2")
      shift
      max_test_length=$1
      ;;
    --num-density-samples|--density-max-length)
      sample_random_language_args+=("$1" "$2")
      shift
      ;;
    *)
      generate_datasets_args+=("$1")
      ;;
  esac
  shift
done

case $device in
  cpu) device=cpu ;;
  gpu) device=cuda ;;
  *)
    usage >&2
    exit 1
    ;;
esac

language_dir=$(get_language_dir "$base_dir" "$language")
mkdir -p "$language_dir"

if [[ $language =~ ^random-(podfa|star-free|regular|context-free)-([0-9]+)$ ]]; then
  language_class=${BASH_REMATCH[1]}
  language_no=${BASH_REMATCH[2]}
  is_random=true
  do_prepare_sampler=true
elif is_saveable_hand_picked_language "$language"; then
  is_random=false
  do_prepare_sampler=true
else
  do_prepare_sampler=false
fi

if $do_prepare_sampler; then
  language_file=$language_dir/language.pt
  sampler_file=$language_dir/sampler.pt
  if $is_random; then
    sample_args=( \
      --hyperparameters "$base_dir"/calibration/"$language_class"/hyperparameters.json \
      --random-seed "$(language_sampling_seed "$language_class" "$language_no")" \
      --output "$language_file" \
      "${sample_random_language_args[@]}" \
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
        echo "error: unknown language class $language_class" >&2
        exit 1
        ;;
    esac
  else
    python recognizers/hand_picked_languages/save_language.py \
      --name "$language" \
      --output "$language_file"
  fi
  echo "writing $sampler_file"
  python recognizers/string_sampling/prepare_sampler.py \
    --input "$language_file" \
    --output "$sampler_file" \
    --max-length "$max_test_length" \
    --device "$device"
  sample_dataset_input=(--sampler "$sampler_file")
else
  sample_dataset_input=(--language "$language")
fi

python recognizers/dataset_generation/generate_datasets.py \
  "${sample_dataset_input[@]}" \
  --random-seed "$STRING_SAMPLING_SEED" \
  --output "$language_dir" \
  --datasets training validation-long test \
  --device "$device" \
  "${generate_datasets_args[@]}"
bash recognizers/neural_networks/prepare_language.bash "$base_dir" "$language"
