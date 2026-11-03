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
        language_class_results = []
        for language_no in range(1, num_languages+1):
            language_results = []
            for trial_no in range(1, num_trials+1):
                model_path = (
                    base_dir /
                    'models' /
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
        value[language_class] = language_class_results
    return Row(
        label=f'\\{architecture}Label{{}}',
        value=value,
        runs=runs,
        target_runs=len(language_classes) * num_languages * num_trials,
        messages=messages
    )

def get_language_class_mean_test_accuracy(
    cache: Cache,
    language_class: str
) -> tuple[float, float] | None:
    language_class_scores = []
    for language_results in cache['value'][language_class]:
        match language_results:
            case list():
                if language_results:
                    language_accuracy = max(language_results)
                    language_class_scores.append(language_accuracy)
            case float():
                language_class_scores.append(language_results)
            case _:
                raise TypeError
    return mean_and_accuracy(language_class_scores)

def mean_and_accuracy(values: list[float]) -> tuple[float, float] | None:
    array = numpy.array(values)
    if len(array) > 0:
        return numpy.mean(array).item(), numpy.std(array).item()
    else:
        return None

def read_baseline_row(
    base_dir: pathlib.Path,
    language_classes: list[str],
    num_languages: int
) -> Row:
    value = {}
    messages = []
    runs = 0
    for language_class in language_classes:
        language_class_results = []
        for language_no in range(1, num_languages+1):
            baseline_file = (
                base_dir /
                'languages' /
                f'random-{language_class}-{language_no}' /
                'baseline.txt'
            )
            try:
                accuracy = float(baseline_file.read_text().strip())
            except FileNotFoundError:
                messages.append(f'missing: {baseline_file}')
            else:
                language_class_results.append(accuracy)
                runs += 1
        value[language_class] = language_class_results
    return Row(
        label=f'3-Gram Classifier',
        value=value,
        runs=runs,
        target_runs=len(language_classes) * num_languages,
        messages=messages
    )

def get_language_class_perfect_rate(
    cache: Cache,
    language_class: str
) -> float | None:
    num_perfect = 0
    total = 0
    for language_results in cache['value'][language_class]:
        match language_results:
            case list():
                if language_results:
                    num_perfect += int(any(accuracy == 1 for accuracy in language_results))
                    total += 1
            case float():
                num_perfect += int(language_results == 1)
                total += 1
            case _:
                raise TypeError
    if total > 0:
        return num_perfect / total
    else:
        return None

def kebab_to_camel_case(s: str) -> str:
    parts = s.split('-')
    return parts[0] + ''.join(capitalize(x) for x in parts[1:])

def capitalize(s: str) -> str:
    if s:
        return s[0].upper() + s[1:]
    else:
        return s

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=pathlib.Path, required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--language-classes', nargs='+', required=True)
    parser.add_argument('--num-languages', type=int, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    args = parser.parse_args()

    rows = [
        *(
            read_row(
                base_dir=args.base_dir,
                architecture=architecture,
                language_classes=args.language_classes,
                num_languages=args.num_languages,
                num_trials=args.num_trials
            )
            for architecture in args.architectures
        ),
        read_baseline_row(
            base_dir=args.base_dir,
            language_classes=args.language_classes,
            num_languages=args.num_languages
        )
    ]
    print_table(
        rows=rows,
        columns=[
            Column('Architecture', 'l', 'label', format_text()),
            *(
                Column(
                    f'\\{kebab_to_camel_case(c)}Label{{}}',
                    'S',
                    f'test.accuracy.{c}',
                    format_mean_and_variance(places=(3, 2)),
                    bold_max=True
                )
                for c in args.language_classes
            )
        ],
        callbacks={
            f'test.accuracy.{c}' : functools.partial(get_language_class_mean_test_accuracy, language_class=c)
            for c in args.language_classes
        }
    )
    print_table(
        rows=rows,
        columns=[
            Column('Architecture', 'l', 'label', format_text()),
            *(
                Column(
                    f'\\{kebab_to_camel_case(c)}Label{{}}',
                    'S',
                    f'test.accuracy.{c}',
                    format_float(),
                    bold_max=True
                )
                for c in args.language_classes
            )
        ],
        callbacks={
            f'test.accuracy.{c}' : functools.partial(get_language_class_perfect_rate, language_class=c)
            for c in args.language_classes
        }
    )

if __name__ == '__main__':
    main()
