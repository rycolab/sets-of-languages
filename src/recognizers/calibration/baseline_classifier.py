import argparse
import pathlib
import statistics
from collections.abc import Iterable

from recognizers.calibration.ngram_lm import NGramClassifier

def read_data(directory: pathlib.Path) -> Iterable[tuple[list[str], bool]]:
    return zip(
        read_strings(directory / 'main.tok'),
        read_labels(directory / 'labels.txt'),
        strict=True
    )

def read_strings(filename: pathlib.Path) -> Iterable[list[str]]:
    with filename.open() as fin:
        for line in fin:
            yield line.split()

def read_labels(filename: pathlib.Path) -> Iterable[bool]:
    with filename.open() as fin:
        for line in fin:
            match line.strip():
                case '0':
                    yield False
                case '1':
                    yield True
                case _:
                    raise ValueError

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('language_dir', type=pathlib.Path)
    parser.add_argument('-n', type=int, default=3)
    args = parser.parse_args()

    training_data = read_data(args.language_dir)
    classifier: NGramClassifier[str] = NGramClassifier(args.n, training_data)
    test_data = read_data(args.language_dir / 'datasets' / 'test')
    accuracy = statistics.fmean(
        classifier.classify(s) == label
        for s, label in test_data
    )
    print(accuracy)

if __name__ == '__main__':
    main()
