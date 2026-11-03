import argparse
import dataclasses
import json
import math
import pathlib
import statistics
import sys
from collections.abc import Iterable

@dataclasses.dataclass
class Distribution:
    directory: pathlib.Path
    mean_baseline_accuracy: float

def get_distributions(directory: pathlib.Path, num_languages: int) -> Iterable[Distribution]:
    for entity in directory.iterdir():
        if entity.is_dir():
            yield get_distribution(entity, num_languages)

def get_distribution(directory: pathlib.Path, num_languages: int) -> Distribution:
    return Distribution(
        directory=directory,
        mean_baseline_accuracy=statistics.fmean(
            l.baseline_accuracy
            for l in get_languages(directory, num_languages)
        )
    )

@dataclasses.dataclass
class Language:
    baseline_accuracy: float

def get_languages(directory: pathlib.Path, num_languages: int) -> Iterable[Language]:
    for language_no in range(1, num_languages+1):
        language_dir = directory / str(language_no)
        try:
            baseline_accuracy = float((language_dir / 'baseline.txt').read_text().strip())
        except FileNotFoundError:
            baseline_accuracy = math.inf
            print(f'warning: missing baseline score for {language_dir}', file=sys.stderr)
        yield Language(
            baseline_accuracy=baseline_accuracy
        )

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=pathlib.Path,
        help='Directory for the language class.')
    parser.add_argument('--num-languages', type=int, required=True,
        help='Number of languages per language distribution.')
    args = parser.parse_args()

    sorted_distributions = sorted(
        get_distributions(args.directory, args.num_languages),
        key=lambda x: x.mean_baseline_accuracy
    )
    for distribution in sorted_distributions:
        with (distribution.directory / 'hyperparameters.json').open() as fin:
            hyperparameters = json.dumps(json.load(fin))
        print(distribution.mean_baseline_accuracy, distribution.directory.name, hyperparameters)

if __name__ == '__main__':
    main()
