import argparse
import dataclasses
import itertools
import pathlib
from collections.abc import Iterable

import torch

from rau.tasks.common.data import load_prepared_data_file
from rau.tools.torch.model_interface import ModelInterface
from rau.vocab import ToStringVocabulary, VocabularyBuilder

from .vocabulary import VocabularyData, load_vocabulary_data_from_file
from .model_interface import RecognitionModelInterface, Example

type PreparedData = list[Example]

@dataclasses.dataclass
class VocabularyContainer:
    input_vocab: ToStringVocabulary
    output_vocab: ToStringVocabulary

def add_data_arguments(
    parser: argparse.ArgumentParser,
    validation: bool = True
) -> None:
    group = parser.add_argument_group('Dataset options')
    group.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing prepared training data. The file '
             '<training-data>/main.prepared will be used as the training '
             'data, and the file <training-data>/main.vocab will be used as '
             'the vocabulary.')
    group.add_argument('--training-data-name',
        help='Optional alternative dataset that will be used as the training '
             'data. If provided, the file '
             '<training-data>/datasets/<training-data-name>/main.prepared will '
             'be used as the training data.')
    if validation:
        group.add_argument('--validation-data', default='validation',
            help='Name of the dataset in the prepared training data directory '
                 'that will be used as validation data. The file '
                 '<training-data>/datasets/<validation-data>/main.prepared '
                 'will be used as the validation data. The default name is '
                 '"validation".')

def load_prepared_data_from_directory(
    directory: pathlib.Path,
    model_interface: RecognitionModelInterface
) -> PreparedData:
    strings_data = load_prepared_data_file(directory / 'main.prepared')
    labels_data = load_prepared_labels_file(directory / 'labels.prepared')
    if model_interface.use_next_symbols_head:
        next_symbols_data = load_prepared_next_symbols_file(
            directory / 'next-symbols.prepared',
            labels_data
        )
    else:
        next_symbols_data = itertools.repeat(None, len(labels_data))
    return list(zip(strings_data, zip(labels_data, next_symbols_data, strict=True), strict=True))

def load_prepared_labels_file(path: pathlib.Path) -> list[bool]:
    return torch.load(path)

def load_prepared_next_symbols_file(
    path: pathlib.Path,
    labels: list[bool]
) -> Iterable[list[list[int]] | None]:
    data = torch.load(path)
    data_it = iter(data)
    for label in labels:
        if label:
            yield next(data_it)
        else:
            yield None
    try:
        next(data_it)
    except StopIteration:
        pass
    else:
        raise ValueError('there are more lists of valid token sets than positive labels')

def load_vocabulary_data(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser
) -> VocabularyData:
    return load_vocabulary_data_from_file(args.training_data / 'main.vocab')

def load_prepared_data(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    vocabulary_data: VocabularyData,
    model_interface: ModelInterface,
    builder: VocabularyBuilder = None
) -> tuple[PreparedData, PreparedData | None, VocabularyContainer]:
    training_data = load_prepared_data_from_directory(
        args.training_data if args.training_data_name is None else args.training_data / 'datasets' / args.training_data_name,
        model_interface
    )
    if hasattr(args, 'validation_data'):
        validation_data = load_prepared_data_from_directory(
            args.training_data / 'datasets' / args.validation_data,
            model_interface
        )
    else:
        validation_data = None
    input_vocab, output_vocab = model_interface.get_vocabularies(
        vocabulary_data,
        builder
    )
    return (
        training_data,
        validation_data,
        VocabularyContainer(input_vocab, output_vocab)
    )

def load_vocabularies(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    model_interface: ModelInterface,
    builder: VocabularyBuilder = None
) -> VocabularyContainer:
    input_vocab, output_vocab = model_interface.get_vocabularies(
        load_vocabulary_data(args, parser),
        builder
    )
    return VocabularyContainer(input_vocab, output_vocab)
