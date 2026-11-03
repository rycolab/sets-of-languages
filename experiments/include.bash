. experiments/config.bash

FIGURES_DIR=$BASE_DIR/figures

DFA_LANGUAGES=( \
  even-pairs \
  repeat-01 \
  parity \
  cycle-navigation \
  modular-arithmetic \
  dyck-2-3 \
  first \
)
BESPOKE_LANGUAGES=( \
  majority \
  stack-manipulation \
  marked-reversal \
  unmarked-reversal \
  marked-copy \
  missing-duplicate-string \
  odds-first \
  binary-addition \
  binary-multiplication \
  compute-sqrt \
  bucket-sort \
)
HAND_PICKED_LANGUAGES=("${DFA_LANGUAGES[@]}" "${BESPOKE_LANGUAGES[@]}")
CALIBRATION_NUM_DISTRIBUTIONS=30
CALIBRATION_DISTRIBUTIONS=($(seq 1 "$CALIBRATION_NUM_DISTRIBUTIONS"))
CALIBRATION_NUM_LANGUAGES=30
CALIBRATION_LANGUAGES=($(seq 1 "$CALIBRATION_NUM_LANGUAGES"))
LANGUAGE_CLASSES=(podfa star-free regular context-free)
NUM_LANGUAGES=30
LANGUAGES=()
for language_class in "${LANGUAGE_CLASSES[@]}"; do
  for ((language_no=1; language_no <= NUM_LANGUAGES; language_no++)); do
    LANGUAGES+=(random-"$language_class"-"$language_no")
  done
done
ARCHITECTURES=(transformer rnn lstm)
NUM_TRIALS=10
TRIALS=($(seq 1 "$NUM_TRIALS"))

submit_job() {
  bash experiments/submit_job.bash "$@"
}

to_latex_id() {
  sed '
    s/[-_]//g;
    s/0/zero/g;
    s/1/one/;
    s/2/two/;
    s/3/three/;
    s/4/four/;
    s/5/five/;
    s/6/six/;
    s/7/seven/;
    s/8/eight/;
    s/9/nine/;
  ' <<<"$1"
}

format_language_name() {
  echo "\\language$(to_latex_id "$1"){}"
}
