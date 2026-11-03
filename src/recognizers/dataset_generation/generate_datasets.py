import argparse
import contextlib
import enum
import math
import pathlib
import random
import shutil
import sys
from collections.abc import Set, MutableSet

import numpy
import torch

from rau.tools.torch.model_interface import parse_device
from recognizers.tools.jsonl import write_json_line
from recognizers.theory.symbol import Symbol, ReservedSymbol
from recognizers.dataset_generation.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    String,
    Parse,
    ValidNextSymbolList,
    EmptyLanguageError
)
from recognizers.string_sampling.dfa.weight_pushing import WeightedDeterministicFiniteAutomatonSampler
from recognizers.string_sampling.cnf_cfg.weight_pushing import (
    ChomskyNormalFormWeightedContextFreeGrammarSampler
)
from recognizers.string_sampling.dfa.language import DeterministicFiniteAutomatonLanguage
from recognizers.string_sampling.cnf_cfg.language import (
    ChomskyNormalFormContextFreeGrammarLanguage
)
from recognizers.tools.random_utils import python_to_numpy_generator
from recognizers.hand_picked_languages.bespoke.majority import Majority
from recognizers.hand_picked_languages.bespoke.stack_manipulation import StackManipulation
from recognizers.hand_picked_languages.bespoke.marked_reversal import MarkedReversal
from recognizers.hand_picked_languages.bespoke.unmarked_reversal import UnmarkedReversal
from recognizers.hand_picked_languages.bespoke.marked_copy import MarkedCopy
from recognizers.hand_picked_languages.bespoke.missing_duplicate_string import MissingDuplicateString
from recognizers.hand_picked_languages.bespoke.odds_first import OddsFirst
from recognizers.hand_picked_languages.bespoke.binary_addition import BinaryAddition
from recognizers.hand_picked_languages.bespoke.binary_multiplication import BinaryMultiplication
from recognizers.hand_picked_languages.bespoke.compute_sqrt import ComputeSqrt
from recognizers.hand_picked_languages.bespoke.bucket_sort import BucketSort

def generate_files(
    language: LengthRestrictedWeightedLanguage,
    length_range: tuple[int, int],
    alphabet: list[str],
    num_samples: int,
    only_negative: bool,
    perturbation_probability: float,
    strict_num_edits_distribution: bool,
    include_log_probability: bool,
    include_next_symbols: bool,
    include_edit_distance: bool,
    generator: random.Random,
    output_dir: pathlib.Path,
    generated_strings_output: MutableSet[String] | None,
    generated_strings_max_length: int | None,
    excluded_strings: Set[String] | None,
    max_positive_retries: int,
    max_negative_retries: int
) -> None:
    include_log_probability = include_log_probability and language.supports_log_probability()
    include_next_symbols = include_next_symbols and language.supports_next_symbols()
    include_edit_distance = include_edit_distance and language.supports_edit_distance()
    alphabet_size = len(alphabet)
    output_dir.mkdir(parents=True, exist_ok=True)
    strings_file = output_dir / 'main.tok'
    print(f'writing {strings_file}')
    labels_file = output_dir / 'labels.txt'
    print(f'writing {labels_file}')
    num_edits_file = output_dir / 'num-edits.txt'
    print(f'writing {num_edits_file}')
    with strings_file.open('w') as strings_fout, \
         labels_file.open('w') as labels_fout, \
         num_edits_file.open('w') as num_edits_fout, \
         contextlib.ExitStack() as exit_stack:
        if include_log_probability:
            log_probabilities_file = output_dir / 'log-probabilities.txt'
            print(f'writing {log_probabilities_file}')
            log_probabilities_fout = exit_stack.enter_context(log_probabilities_file.open('w'))
        if include_next_symbols:
            next_symbols_file = output_dir / 'next-symbols.jsonl'
            print(f'writing {next_symbols_file}')
            next_symbols_fout = exit_stack.enter_context(next_symbols_file.open('w'))
        if include_edit_distance:
            edit_distances_file = output_dir / 'edit-distances.txt'
            print(f'writing {edit_distances_file}')
            edit_distances_fout = exit_stack.enter_context(edit_distances_file.open('w'))
        for i in range(num_samples):
            s, label, parse, edit_distance, num_edits = generate_example(
                language,
                length_range,
                alphabet_size,
                only_negative,
                perturbation_probability,
                strict_num_edits_distribution,
                include_log_probability,
                include_next_symbols,
                include_edit_distance,
                generator,
                excluded_strings,
                max_positive_retries,
                max_negative_retries
            )
            print(' '.join(alphabet[c] for c in s), file=strings_fout)
            print(label, file=labels_fout)
            if label:
                assert parse is not None
                if include_log_probability:
                    print(parse.log_probability, file=log_probabilities_fout)
                if include_next_symbols:
                    assert parse.next_symbols is not None
                    check_next_symbols(s, parse.next_symbols)
                    next_symbols_data = []
                    for next_symbols_i in parse.next_symbols:
                        has_eos = ReservedSymbol.EOS in next_symbols_i
                        non_eos_symbols = sorted(alphabet[c] for c in next_symbols_i if c != ReservedSymbol.EOS)
                        next_symbols_data.append({
                            's' : ' '.join(non_eos_symbols),
                            'e' : has_eos
                        })
                    write_json_line(next_symbols_data, next_symbols_fout)
            else:
                print(num_edits if num_edits is not None else '', file=num_edits_fout)
                if include_edit_distance:
                    assert isinstance(edit_distance, int) and edit_distance > 0
                    print(edit_distance, file=edit_distances_fout)
            if (
                generated_strings_output is not None and
                (generated_strings_max_length is None or len(s) <= generated_strings_max_length)
            ):
                generated_strings_output.add(s)

def check_next_symbols(s: String, next_symbols: ValidNextSymbolList) -> None:
    if len(next_symbols) != len(s) + 1:
        raise ValueError(
            f'a list of next symbol sets is the wrong length '
            f'(expected {len(s) + 1}, got {len(next_symbols)}); '
            f'string: {s}; '
            f'next symbol sets: {next_symbols}'
        )
    for i, (s_i, next_symbols_i) in enumerate(zip(s, next_symbols[:-1], strict=True)):
        if s_i not in next_symbols_i:
            raise ValueError(
                f'a symbol in a string does not appear in the set '
                f'of valid next symbols; '
                f'at position {i}, symbol {s_i} is not in the set {next_symbols_i}; '
                f'string: {s}; '
                f'next symbol sets: {next_symbols}'
            )
    if ReservedSymbol.EOS not in next_symbols[-1]:
        raise ValueError(
            f'ReservedSymbol.EOS does not appear in the last set of valid next '
            f'symbols; '
            f'string: {s}; '
            f'next symbol sets: {next_symbols}'
        )

def generate_example(
    language: LengthRestrictedWeightedLanguage,
    length_range: tuple[int, int],
    alphabet_size: int,
    only_negative: bool,
    perturbation_probability: float,
    strict_num_edits_distribution: bool,
    include_log_probability: bool,
    include_next_symbols: bool,
    include_edit_distance: bool,
    generator: random.Random,
    excluded_strings: Set[String] | None,
    max_positive_retries: int,
    max_negative_retries: int
) -> tuple[String, int, Parse | None, int | None, int | None]:
    if only_negative:
        label = 0
    else:
        label = generator.randrange(2)
    if label:
        s, parse = generate_positive_example(
            language,
            include_log_probability,
            include_next_symbols,
            generator,
            excluded_strings,
            max_positive_retries
        )
        if not (length_range[0] <= len(s) <= length_range[1]):
            raise ValueError(
                f'a positive example of length {len(s)} was generated outside '
                f'the desired length range: {s}'
            )
        edit_distance = None
        num_edits = None
    else:
        s, edit_distance, num_edits = generate_negative_example(
            language,
            length_range,
            alphabet_size,
            perturbation_probability,
            strict_num_edits_distribution,
            include_edit_distance,
            generator,
            excluded_strings,
            max_negative_retries
        )
        parse = None
    return s, label, parse, edit_distance, num_edits

class CannotGenerateExampleError(ValueError):
    pass

def generate_positive_example(
    language: LengthRestrictedWeightedLanguage,
    include_log_probability: bool,
    include_next_symbols: bool,
    generator: random.Random,
    excluded_strings: Set[String] | None,
    max_tries: int
) -> tuple[String, Parse]:
    for _ in range(max_tries):
        s, parse = language.sample(
            generator=generator,
            include_log_probability=include_log_probability,
            include_next_symbols=include_next_symbols
        )
        if excluded_strings is None or s not in excluded_strings:
            return s, parse
    else:
        raise CannotGenerateExampleError(
            f'unable to generate a held-out positive example after {max_tries} '
            f'tries'
        )

def generate_negative_example(
    language: LengthRestrictedWeightedLanguage,
    length_range: tuple[int, int],
    alphabet_size: int,
    perturbation_probability: float,
    strict_num_edits_distribution: bool,
    include_edit_distance: bool,
    generator: random.Random,
    excluded_strings: Set[String] | None,
    max_tries: int
) -> tuple[String, int | None, int | None]:
    if strict_num_edits_distribution:
        # Decide whether to propose only random or perturbed strings.
        if generator.random() < perturbation_probability:
            # Sample a number of edits, and keep sampling strings with exactly
            # that many edits until it's negative.
            fixed_num_edits = generate_random_num_edits(generator)
            def propose_negative_example() -> tuple[String, int | None]:
                s = generate_perturbed_string_with_num_edits(
                    language,
                    length_range,
                    alphabet_size,
                    fixed_num_edits,
                    generator
                )
                return s, fixed_num_edits
        else:
            # Sample random strings.
            def propose_negative_example() -> tuple[String, int | None]:
                s = generate_random_string(length_range, alphabet_size, generator)
                return s, None
    else:
        # Decide which type of negative sampling to do on every retry.
        def propose_negative_example() -> tuple[String, int | None]:
            if generator.random() < perturbation_probability:
                return generate_perturbed_string(
                    language,
                    length_range,
                    alphabet_size,
                    generator
                )
            else:
                s = generate_random_string(length_range, alphabet_size, generator)
                return s, None
    for _ in range(max_tries):
        s, num_edits = propose_negative_example()
        if excluded_strings is None or s not in excluded_strings:
            is_negative, edit_distance = language.is_negative(
                s,
                include_edit_distance=include_edit_distance
            )
            if is_negative:
                return s, edit_distance, num_edits
    else:
        raise CannotGenerateExampleError(
            f'unable to generate a negative example after {max_tries} tries'
        )

def generate_random_string(
    length_range: tuple[int, int],
    alphabet_size: int,
    generator: random.Random
) -> String:
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(Symbol(generator.randrange(alphabet_size)) for _ in range(length))

def generate_perturbed_string(
    language: LengthRestrictedWeightedLanguage,
    length_range: tuple[int, int],
    alphabet_size: int,
    generator: random.Random
) -> tuple[String, int]:
    num_edits = generate_random_num_edits(generator)
    s = generate_perturbed_string_with_num_edits(
        language,
        length_range,
        alphabet_size,
        num_edits,
        generator
    )
    return s, num_edits

def generate_random_num_edits(generator: random.Random) -> int:
    return python_to_numpy_generator(generator).geometric(0.5)

def generate_perturbed_string_with_num_edits(
    language: LengthRestrictedWeightedLanguage,
    length_range: tuple[int, int],
    alphabet_size: int,
    num_edits: int,
    generator: random.Random
) -> String:
    s, _ = language.sample(
        generator=generator,
        include_log_probability=False,
        include_next_symbols=False
    )
    r = apply_random_edits(s, num_edits, length_range, alphabet_size, generator)
    return tuple(r)

def apply_random_edits(
    s: String,
    num_edits: int,
    length_range: tuple[int, int],
    alphabet_size: int,
    generator: random.Random
) -> list[Symbol]:
    r = list(s)
    for _ in range(num_edits):
        apply_random_edit(r, length_range, alphabet_size, generator)
    return r

class Edit(enum.IntEnum):
    INSERT = enum.auto()
    REPLACE = enum.auto()
    DELETE = enum.auto()

def apply_random_edit(
    s: list[Symbol],
    length_range: tuple[int, int],
    alphabet_size: int,
    generator: random.Random
) -> None:
    lo, hi = length_range
    choices = []
    if len(s) < hi:
        choices.append(Edit.INSERT)
    if len(s) > 0 and alphabet_size > 1:
        choices.append(Edit.REPLACE)
    if len(s) > lo:
        choices.append(Edit.DELETE)
    match generator.choice(choices):
        case Edit.INSERT:
            apply_random_insertion(s, alphabet_size, generator)
        case Edit.REPLACE:
            apply_random_replacement(s, alphabet_size, generator)
        case Edit.DELETE:
            apply_random_deletion(s, generator)
        case _:
            raise ValueError

def apply_random_insertion(
    s: list[Symbol],
    alphabet_size: int,
    generator: random.Random
) -> None:
    i = generator.randrange(len(s) + 1)
    symbol = Symbol(generator.randrange(alphabet_size))
    s.insert(i, symbol)

def apply_random_replacement(
    s: list[Symbol],
    alphabet_size: int,
    generator: random.Random
) -> None:
    i = generator.randrange(len(s))
    old_symbol = s[i]
    new_symbol = generator.randrange(alphabet_size - 1)
    new_symbol += int(new_symbol >= old_symbol)
    s[i] = Symbol(new_symbol)

def apply_random_deletion(s: list[Symbol], generator: random.Random) -> None:
    i = generator.randrange(len(s))
    del s[i]

def get_hand_coded_language(name: str) -> WeightedLanguage:
    match name:
        case 'majority':
            return Majority()
        case 'stack-manipulation':
            return StackManipulation()
        case 'marked-reversal':
            return MarkedReversal()
        case 'unmarked-reversal':
            return UnmarkedReversal()
        case 'marked-copy':
            return MarkedCopy()
        case 'missing-duplicate-string':
            return MissingDuplicateString()
        case 'odds-first':
            return OddsFirst()
        case 'binary-addition':
            return BinaryAddition()
        case 'binary-multiplication':
            return BinaryMultiplication()
        case 'compute-sqrt':
            return ComputeSqrt()
        case 'bucket-sort':
            return BucketSort()
        case _:
            raise ValueError(f'invalid hand-coded language name: {name}')

def get_saved_language(
    filename: pathlib.Path,
    dtype: torch.dtype,
    device: torch.device
) -> WeightedLanguage:
    data = torch.load(filename, weights_only=False, map_location=torch.device('cpu'))
    sampler = data['sampler']
    alphabet = data.pop('alphabet', None)
    match sampler:
        case WeightedDeterministicFiniteAutomatonSampler():
            return DeterministicFiniteAutomatonLanguage(sampler, alphabet, dtype, device)
        case ChomskyNormalFormWeightedContextFreeGrammarSampler():
            sampler.set_cky_device(device)
            return ChomskyNormalFormContextFreeGrammarLanguage(sampler, alphabet, dtype, device)
        case _:
            raise ValueError

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True,
        help='Output directory where the data files will be written.')
    parser.add_argument('--random-seed', type=int, required=True,
        help='Random seed used for random sampling.')
    parser.add_argument('--language',
        help='Name of a hand-coded language to sample from.')
    parser.add_argument('--sampler', type=pathlib.Path,
        help='A .pt file containing a sampler object.')
    parser.add_argument('--perturbation-probability', type=float, default=0.5,
        help='The probability that a proposed negative example (before '
             'rejection sampling) will be a perturbed positive example.')
    parser.add_argument('--strict-num-edits-distribution', action='store_true', default=False,
        help='Make sure the distribution of number of edits on negative '
             'examples is followed even after rejection sampling.')
    parser.add_argument('--include-log-probability', action='store_true', default=False,
        help='Output the log probability of each positive example.')
    parser.add_argument('--include-next-symbols', action='store_true', default=False,
        help='Output the set of valid next symbols at each position for each '
             'positive example.')
    dataset_names = [
        'training',
        'validation-short',
        'validation-long',
        'test',
        'test-short-held-out',
        'test-edit-distance'
    ]
    parser.add_argument('--datasets', nargs='*', choices=dataset_names,
        default=dataset_names,
        help='Names of datasets to generate. By default, all datasets will be '
             'generated.')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16',
        help='dtype to use for PyTorch operations.')
    parser.add_argument('--device',
        help='Device to use for PyTorch operations.')
    parser.add_argument('--max-training-length', type=int, default=40,
        help='Maximum length of strings in the training set.')
    parser.add_argument('--max-validation-long-length', type=int,
        help='Maximum length of strings in the long validation set. The '
             'default is twice the maximum length of the training set.')
    parser.add_argument('--max-test-length', type=int, default=500,
        help='Maximum length of strings in the test set.')
    parser.add_argument('--num-training-examples', type=int, default=10000,
        help='Number of examples in the training set.')
    parser.add_argument('--training-dataset-name',
        help='Optional, alternate place to write the training set.')
    parser.add_argument('--num-test-examples', type=int, default=5010,
        help='Number of examples in the test set.')
    parser.add_argument('--max-positive-retries', type=int, default=1000,
        help='Maximum number of retries when generating held-out positive '
             'examples.')
    parser.add_argument('--max-negative-retries', type=int, default=1000,
        help='Maximum number of retries when generating negative examples.')
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = parse_device(args.device)

    match (args.language is not None, args.sampler is not None):
        case (True, False):
            language = get_hand_coded_language(args.language)
        case (False, True):
            language = get_saved_language(args.sampler, dtype, device)
        case _:
            parser.error('exactly one of --language or --sampler must be given')

    test_short_held_out_dependencies = [
        'training',
        'validation-short',
        'validation-long'
    ]
    if (
        'test-short-held-out' in args.datasets and
        not all(x in args.datasets for x in test_short_held_out_dependencies)
    ):
        parser.error(f'generating test-short-held-out requires generating {", ".join(test_short_held_out_dependencies)}')

    generator = random.Random(args.random_seed)
    training_length_range = (0, args.max_training_length)
    max_validation_long_length = (
        args.max_validation_long_length
        if args.max_validation_long_length is not None
        else 2 * training_length_range[1]
    )
    validation_long_length_range = (0, max_validation_long_length)
    test_length_range = (0, args.max_test_length)
    alphabet = [language.symbol_to_str(Symbol(c)) for c in range(language.alphabet_size())]
    try:
        if any(x in args.datasets for x in ['training', 'validation-short', 'test-short-held-out']):
            training_language = language.with_length_range(training_length_range)
        if 'validation-long' in args.datasets:
            validation_long_language = language.with_length_range(validation_long_length_range)
        if any(x in args.datasets for x in ['test', 'test-edit-distance']):
            test_language = language.with_length_range(test_length_range)
    except EmptyLanguageError as e:
        print(f'error: cannot sample positive examples: {e}', file=sys.stderr)
        sys.exit(1)

    def generate_split(
        language: LengthRestrictedWeightedLanguage,
        length_range: tuple[int, int],
        num_samples: int,
        output_dir: pathlib.Path,
        only_negative: bool = False,
        include_edit_distance: bool = False,
        generated_strings_output: MutableSet[String] | None = None,
        generated_strings_max_length: int | None = None,
        excluded_strings: Set[String] | None = None
    ):
        generate_files(
            language=language,
            length_range=length_range,
            alphabet=alphabet,
            num_samples=num_samples,
            only_negative=only_negative,
            perturbation_probability=args.perturbation_probability,
            strict_num_edits_distribution=args.strict_num_edits_distribution,
            include_log_probability=args.include_log_probability,
            include_next_symbols=args.include_next_symbols,
            include_edit_distance=include_edit_distance,
            generator=generator,
            output_dir=output_dir,
            generated_strings_output=generated_strings_output,
            generated_strings_max_length=generated_strings_max_length,
            excluded_strings=excluded_strings,
            max_positive_retries=args.max_positive_retries,
            max_negative_retries=args.max_negative_retries
        )

    excluded_strings: set[String] | None = set() if 'test-short-held-out' in args.datasets else None
    if 'training' in args.datasets:
        generate_split(
            language=training_language,
            length_range=training_length_range,
            num_samples=args.num_training_examples,
            output_dir=args.output if args.training_dataset_name is None else args.output / 'datasets' / args.training_dataset_name,
            generated_strings_output=excluded_strings
        )
    if 'validation-short' in args.datasets:
        generate_split(
            language=training_language,
            length_range=training_length_range,
            num_samples=1000,
            output_dir=args.output / 'datasets' / 'validation-short',
            generated_strings_output=excluded_strings
        )
    if 'validation-long' in args.datasets:
        generate_split(
            language=validation_long_language,
            length_range=validation_long_length_range,
            num_samples=1000,
            output_dir=args.output / 'datasets' / 'validation-long',
            generated_strings_output=excluded_strings,
            generated_strings_max_length=training_length_range[1]
        )
    if 'test' in args.datasets:
        generate_split(
            language=test_language,
            length_range=test_length_range,
            num_samples=args.num_test_examples,
            output_dir=args.output / 'datasets' / 'test'
        )
    if 'test-short-held-out' in args.datasets:
        test_short_held_out_dir = args.output / 'datasets' / 'test-short-held-out'
        try:
            generate_split(
                language=training_language,
                length_range=training_length_range,
                num_samples=1000,
                output_dir=test_short_held_out_dir,
                excluded_strings=excluded_strings
            )
        except CannotGenerateExampleError as e:
            print(e)
            print('skipping test-short-held-out')
            print(f'removing {test_short_held_out_dir}')
            shutil.rmtree(test_short_held_out_dir)
    if 'test-edit-distance' in args.datasets:
        if test_language.supports_edit_distance():
            generate_split(
                language=test_language,
                length_range=test_length_range,
                num_samples=50,
                output_dir=args.output / 'datasets' / 'test-edit-distance',
                only_negative=True,
                include_edit_distance=True
            )
        else:
            print('skipping test-edit-distance (edit distance is not supported)')

if __name__ == '__main__':
    main()
