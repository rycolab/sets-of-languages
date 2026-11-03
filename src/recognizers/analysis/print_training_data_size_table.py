import argparse
import functools
import json
import pathlib
import sys
from collections.abc import Callable

import numpy

from recognizers.analysis.read_results_util import read_test_accuracy
from recognizers.analysis.print_table_util import (
    print_table,
    Row,
    Column,
    format_text,
    format_mean_and_variance,
    format_float,
    Cache
)
from recognizers.analysis.print_class_summary_tables import (
    mean_and_accuracy,
    kebab_to_camel_case
)

TRAINING_DATA_SIZES = [10000, 100000]

def read_row(
    base_dir: pathlib.Path,
    architecture: str,
    language_classes: list[str],
    num_languages: int,
    num_trials: int
) -> Row:
    runs = 0
    value = {}
    messages = []
    for language_class in language_classes:
        for num_training_examples in TRAINING_DATA_SIZES:
            if num_training_examples == 10000:
                models_dir = 'models'
            else:
                models_dir = f'models-training-{num_training_examples}'
            language_class_results = []
            for language_no in range(1, num_languages+1):
                language_results = []
                for trial_no in range(1, num_trials+1):
                    model_path = (
                        base_dir /
                        models_dir /
                        f'random-{language_class}-{language_no}' /
                        architecture /
                        'rec' /
                        'validation-long' /
                        str(trial_no)
                    )
                    accuracy = read_test_accuracy(model_path)
                    if accuracy is not None:
                        language_results.append(accuracy)
                        runs += 1
                    else:
                        messages.append(f'missing: {model_path}')
                language_class_results.append(language_results)
            value[language_class, num_training_examples] = language_class_results
    return Row(
        label=f'\\{architecture}Label{{}}',
        value=value,
        runs=runs,
        target_runs=len(language_classes) * len(TRAINING_DATA_SIZES) * num_languages * num_trials,
        messages=messages
    )

def get_language_class_mean_test_accuracy(
    cache: Cache,
    language_class: str,
    num_training_examples: int
) -> tuple[float, float] | None:
    language_class_scores = []
    for language_results in cache['value'][language_class, num_training_examples]:
        if language_results:
            language_accuracy = max(language_results)
            language_class_scores.append(language_accuracy)
    return mean_and_accuracy(language_class_scores)

def get_language_class_perfect_rate(
    cache: Cache,
    language_class: str,
    num_training_examples: int
) -> float | None:
    num_perfect = 0
    total = 0
    for language_results in cache['value'][language_class, num_training_examples]:
        if language_results:
            num_perfect += int(any(accuracy == 1 for accuracy in language_results))
            total += 1
    if total > 0:
        return num_perfect / total
    else:
        return None

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=pathlib.Path, required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--language-classes', nargs='+', required=True)
    parser.add_argument('--num-languages', type=int, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    args = parser.parse_args()

    rows = [
        read_row(
            base_dir=args.base_dir,
            architecture=architecture,
            language_classes=args.language_classes,
            num_languages=args.num_languages,
            num_trials=args.num_trials
        )
        for architecture in args.architectures
    ]
    print_table(
        rows=rows,
        header=''.join([
            '& ',
            ' & '.join(
                r'\multicolumn{4}{c}{' f'\\{kebab_to_camel_case(c)}' r'Label{}}'
                for c in args.language_classes
            ),
            r' \\' '\n',
            ' '.join(r'\cmidrule(lr){' f'{i}-{i+3}' r'}' for i in range(2, 4*len(args.language_classes)+1, 4))
        ]),
        columns=[
            Column('Architecture', 'l', 'label', format_text()),
            *(
                Column(
                    f'{s // 1000}k',
                    'S',
                    f'test.accuracy.{c}.{s}',
                    format_mean_and_variance(places=(2, 2)),
                    bold_max=True
                )
                for c in args.language_classes
                for s in TRAINING_DATA_SIZES
            )
        ],
        callbacks={
            f'test.accuracy.{c}.{s}' : functools.partial(get_language_class_mean_test_accuracy, language_class=c, num_training_examples=s)
            for c in args.language_classes
            for s in TRAINING_DATA_SIZES
        }
    )
    print_table(
        rows=rows,
        header=''.join([
            '& ',
            ' & '.join(
                r'\multicolumn{2}{c}{' f'\\{kebab_to_camel_case(c)}' r'Label{}}'
                for c in args.language_classes
            ),
            r' \\' '\n',
            ' '.join(r'\cmidrule(lr){' f'{i}-{i+1}' r'}' for i in range(2, 2*len(args.language_classes)+1, 2))
        ]),
        columns=[
            Column('Architecture', 'l', 'label', format_text()),
            *(
                Column(
                    f'{s // 1000}k',
                    'S',
                    f'test.accuracy.{c}.{s}',
                    format_float(places=1),
                    bold_max=True
                )
                for c in args.language_classes
                for s in TRAINING_DATA_SIZES
            )
        ],
        callbacks={
            f'test.accuracy.{c}.{s}' : functools.partial(get_language_class_perfect_rate, language_class=c, num_training_examples=s)
            for c in args.language_classes
            for s in TRAINING_DATA_SIZES
        }
    )

if __name__ == '__main__':
    main()
