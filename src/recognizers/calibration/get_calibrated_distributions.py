import argparse
import dataclasses
import heapq
import json
import math
import pathlib
import shutil
import statistics
import sys
from collections.abc import Iterable

from recognizers.analysis.read_results_util import read_language_txt_float

@dataclasses.dataclass
class Distribution:
    directory: pathlib.Path
    mean_baseline_accuracy: float

@dataclasses.dataclass
class Language:
    baseline_accuracy: float

def get_languages(directory: pathlib.Path, num_languages: int) -> list[Language] | None:
    result = []
    for language_no in range(1, num_languages + 1):
        baseline_accuracy = read_language_txt_float(directory / str(language_no), 'baseline')
        if baseline_accuracy is not None:
            result.append(Language(baseline_accuracy))
        else:
            return None
    return result
 
def get_distribution(directory: pathlib.Path, num_languages: int) -> Distribution | None:
    languages = get_languages(directory, num_languages)
    if languages is not None:
        return Distribution(
            directory=directory,
            mean_baseline_accuracy=statistics.fmean(l.baseline_accuracy for l in languages)
        )
    else:
        print(f'discarding distribution {directory}', file=sys.stderr)
        return None

def get_distributions(directory: pathlib.Path, num_distributions: int, num_languages: int) -> Iterable[Distribution]:
    for distribution_no in range(1, num_distributions + 1):
        distribution = get_distribution(directory / str(distribution_no), num_languages)
        if distribution is not None:
            yield distribution

def get_sorted_distributions(directory: pathlib.Path, num_distributions: int, num_languages: int) -> list[Distribution]:
    return sorted(
        get_distributions(directory, num_distributions, num_languages),
        key=lambda d: d.mean_baseline_accuracy
    )

def smallest_range(nums: list[list[Distribution]]) -> list[Distribution]:
    """The classic min-heap solution to "smallest range covering elements from
    k lists", extended to also return which element of each list was chosen."""
    # Priority queue to store (value, list index, element index)
    pq: list[tuple[float, int, int]] = []
    max_val = float('-inf')
    range_start = 0.0
    range_end = float('inf')
    # Current element index per list, and the snapshot of the best selection
    cols = [0] * len(nums)
    best_cols = cols.copy()
    # Insert the first element from each list into the min-heap
    for i in range(len(nums)):
        heapq.heappush(pq, (nums[i][0].mean_baseline_accuracy, i, 0))
        max_val = max(max_val, nums[i][0].mean_baseline_accuracy)
    # Continue until we can't proceed further
    while len(pq) == len(nums):
        min_val, row, col = heapq.heappop(pq)
        # Update the smallest range
        if max_val - min_val < range_end - range_start:
            range_start = min_val
            range_end = max_val
            best_cols = cols.copy()
        # If possible, add the next element from the same row to the heap
        if col + 1 < len(nums[row]):
            next_val = nums[row][col + 1].mean_baseline_accuracy
            heapq.heappush(pq, (next_val, row, col + 1))
            cols[row] = col + 1
            max_val = max(max_val, next_val)
    return [nums[i][best_cols[i]] for i in range(len(nums))]

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Choose one hyperparameter setting per language class so that '
                    'the mean baseline accuracies are as close together as possible.'
    )
    parser.add_argument('directories', type=pathlib.Path, nargs='+',
        help='One directory per language class (e.g. regular podfa star-free '
             'context-free).')
    parser.add_argument('--num-distributions', type=int, required=True,
        help='Number of language distributions per language class.')
    parser.add_argument('--num-languages', type=int, required=True,
        help='Number of languages per language distribution.')
    parser.add_argument('--no-write', action='store_true', default=False)
    args = parser.parse_args()
 
    distribution_lists = [
        get_sorted_distributions(directory, args.num_distributions, args.num_languages)
        for directory in args.directories
    ]
    language_classes = []
    for language_class_dir, distribution_list in zip(args.directories, distribution_lists, strict=True):
        if distribution_list:
            language_classes.append((language_class_dir, distribution_list))
        else:
            print(f'discarding language class {language_class_dir}', file=sys.stderr)
    chosen = smallest_range([distribution_list for _, distribution_list in language_classes])
    accuracies = [d.mean_baseline_accuracy for d in chosen]
    print(f'range: [{min(accuracies)}, {max(accuracies)}] '
          f'(spread {max(accuracies) - min(accuracies)})')
    print()
    for (directory, distribution_list), distribution in zip(language_classes, chosen, strict=True):
        print(f'language class: {directory.name}')
        print(f'best distribution: {distribution.directory.name}')
        hyperparameter_file = distribution.directory / 'hyperparameters.json'
        try:
            with hyperparameter_file.open() as fin:
                print(json.dumps(json.load(fin)))
        except (FileNotFoundError, NotADirectoryError):
            print(f'warning: missing {hyperparameter_file}', file=sys.stderr)
        else:
            print(f'mean accuracy: {distribution.mean_baseline_accuracy}')
            if not args.no_write:
                output_file = directory / 'hyperparameters.json'
                print(f'writing {output_file}')
                shutil.copyfile(hyperparameter_file, output_file)
        print()

if __name__ == '__main__':
    main()
