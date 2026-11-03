import argparse
import pathlib

from recognizers.analysis.read_results_util import (
    read_best_test_accuracy,
    read_language_txt_float
)

def replace_none(x: float | None) -> float | str:
    return x if x is not None else 'nan'

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=pathlib.Path, required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--language-classes', nargs='+', required=True)
    parser.add_argument('--num-languages', type=int, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    args = parser.parse_args()

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
            output_file_name = args.output_dir / language_class / f'{architecture}.dat'
            print(f'writing {output_file_name}')
            output_file_name.parent.mkdir(parents=True, exist_ok=True)
            with output_file_name.open('w') as fout:
                print('best_test_accuracy\tbaseline_test_accuracy\tsensitivity\tdensity', file=fout)
                for language, language_scores in language_scores_list:
                    best_test_accuracy = read_best_test_accuracy(
                        args.base_dir / 'models' / language / architecture / 'rec' / 'validation-long',
                        args.num_trials
                    )
                    if best_test_accuracy is not None:
                        print('\t'.join(map(str, [best_test_accuracy, *language_scores])), file=fout)

if __name__ == '__main__':
    main()
