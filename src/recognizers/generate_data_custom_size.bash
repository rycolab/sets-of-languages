set -euo pipefail
. recognizers/functions.bash

usage() {
  echo "Usage: bash $0 <base-directory> <language> <device> <num-examples>

For an existing language, generate an additional training set with a custom
number of examples.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to prepare. See generate_data.bash.
  <device>
    Either 'cpu' or 'gpu'.
  <num-examples>
    The number of examples in the new training set.
"
}

base_dir=${1-}
language=${2-}
device=${3-}
num_examples=${4-}
if ! shift 4; then
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

if [[ $language =~ ^random-(podfa|star-free|regular|context-free)-([0-9]+)$ ]] || \
   is_saveable_hand_picked_language "$language"; then
  sample_dataset_input=(--sampler "$language_dir"/sampler.pt)
else
  sample_dataset_input=(--language "$language")
fi

python recognizers/dataset_generation/generate_datasets.py \
  "${sample_dataset_input[@]}" \
  --random-seed "$STRING_SAMPLING_SEED" \
  --output "$language_dir" \
  --datasets training \
  --num-training-examples "$num_examples" \
  --training-dataset-name training-"$num_examples" \
  --device "$device"
python recognizers/neural_networks/prepare_data.py \
  --training-data "$language_dir" \
  --more-data training-"$num_examples" \
  --never-allow-unk \
  --only-more-data
