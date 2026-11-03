import argparse
import pathlib
from collections.abc import Iterable

from recognizers.analysis.read_results_util import read_test_accuracy
from recognizers.analysis.bubble_diagram import LanguageClass, HorizontalBubbles, print_language_class

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=pathlib.Path, required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--num-languages', type=int, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--num-columns', type=int, default=5)
    args = parser.parse_args()

    def get_trials(architecture: str, language: str) -> Iterable[float]:
        for trial_no in range(1, args.num_trials + 1):
            model_dir = args.base_dir / 'models' / language / architecture / 'rec' / 'validation-long' / str(trial_no)
            test_accuracy = read_test_accuracy(model_dir)
            if test_accuracy is not None:
                yield test_accuracy
            else:
                print(f'missing: {model_dir}')

    def get_language_result(architecture: str, language: str) -> bool:
        return any(x == 1 for x in get_trials(architecture, language))

    def get_language_results(architecture: str, languages: list[str]) -> list[bool]:
        return [get_language_result(architecture, language) for language in languages]

    def get_sorted_language_results(language_class: str) -> Iterable[tuple[str, list[bool]]]:
        languages = [f'random-{language_class}-{i}' for i in range(1, args.num_languages+1)]
        unsorted_results = { a : get_language_results(a, languages) for a in args.architectures }
        keys = [(-sum(map(int, x)), tuple(not xx for xx in x)) for x in zip(*unsorted_results.values())]
        indexes = sorted(range(len(keys)), key=lambda i: keys[i])
        return ((k, [v[i] for i in indexes]) for k, v in unsorted_results.items())

    sigma_star_and_finite_classes = [
        HorizontalBubbles(
            id='trivial',
            bubbles=[
                (r'$\Sigma^\ast$', 'sigmaStar'),
                ('Finite', 'finite')
            ]
        )
    ]
    print_language_class(
        LanguageClass(
            label=r'\contextFreeLabel{}',
            id='cf',
            subclasses=[LanguageClass(
                label=r'\regularLabel{}',
                id='r',
                subclasses=[LanguageClass(
                    label=r'\starFreeLabel{}',
                    id='sf',
                    subclasses=[LanguageClass(
                        label=r'\podfaLabel{}',
                        id='podfa',
                        subclasses=sigma_star_and_finite_classes
                    )]
                )]
            )]
        ),
        num_columns=args.num_columns,
        file_name=args.output_dir / 'language-classes.tex'
    )
    podfa_results = dict(get_sorted_language_results('podfa'))
    star_free_results = dict(get_sorted_language_results('star-free'))
    for architecture, regular_results in get_sorted_language_results('regular'):
        print_language_class(
            LanguageClass(
                label=r'\regularLabel{}',
                id='r',
                languages=regular_results,
                subclasses=[LanguageClass(
                    label=r'\starFreeLabel{}',
                    id='sf',
                    languages=star_free_results[architecture],
                    subclasses=[LanguageClass(
                        label=r'\podfaLabel{}',
                        id='podfa',
                        languages=podfa_results[architecture],
                        subclasses=sigma_star_and_finite_classes
                    )]
                )]
            ),
            num_columns=args.num_columns,
            file_name=args.output_dir / 'regular' / f'{architecture}.tex'
        )
    for architecture, results in get_sorted_language_results('context-free'):
        print_language_class(
            LanguageClass(
                label=r'\contextFreeLabel{}',
                id='cf',
                num_levels=4,
                languages=results,
                subclasses=sigma_star_and_finite_classes
            ),
            num_columns=args.num_columns,
            file_name=args.output_dir / 'context-free' / f'{architecture}.tex'
        )

if __name__ == '__main__':
    main()
