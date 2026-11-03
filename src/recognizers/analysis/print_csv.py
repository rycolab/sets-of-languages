import argparse
import pathlib

from recognizers.analysis.read_results_util import (
    read_test_accuracy,
    read_language_txt_float,
    read_logs,
    read_training_loop
)

def replace_none(x: float | None) -> float | str:
    return x if x is not None else ''

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=pathlib.Path, required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--language-classes', nargs='+', required=True)
    parser.add_argument('--num-languages', type=int, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    args = parser.parse_args()

    print('language_distribution,language,baseline_accuracy,sensitivity,density,architecture,num_parameters,initial_learning_rate,max_tokens_per_batch,accuracy')
    for language_class in args.language_classes:
        language_scores_list = []
        for language_no in range(1, args.num_languages+1):
            language = f'random-{language_class}-{language_no}'
            language_dir = args.base_dir / 'languages' / language
            language_scores = tuple(
                replace_none(read_language_txt_float(language_dir, name))
                for name in ('baseline', 'sensitivity', 'density')
            )
            language_scores_list.append((language, language_scores))
        for architecture in args.architectures:
            for language, (baseline_accuracy, sensitivity, density) in language_scores_list:
                for trial_no in range(1, args.num_trials+1):
                    model_dir = args.base_dir / 'models' / language / architecture / 'rec' / 'validation-long' / str(trial_no)
                    accuracy = replace_none(read_test_accuracy(model_dir, print_error=True))
                    logs_data = read_logs(model_dir, ['model_info'])
                    num_parameters = logs_data['model_info']['num_parameters'] if logs_data is not None and 'model_info' in logs_data else ''
                    training_loop_data = read_training_loop(model_dir)
                    initial_learning_rate = training_loop_data['initial_learning_rate'] if training_loop_data else ''
                    max_tokens_per_batch = training_loop_data['max_tokens_per_batch'] if training_loop_data else ''
                    print(f'{language_class},{language},{baseline_accuracy},{sensitivity},{density},{architecture},{num_parameters},{initial_learning_rate},{max_tokens_per_batch},{accuracy}')

if __name__ == '__main__':
    main()
