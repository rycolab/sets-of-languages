import argparse
import json
import pathlib
import random
import statistics
import sys
from collections.abc import Iterable

import torch

from rau.tools.torch.model_interface import parse_device

from recognizers.theory.symbol import Symbol
from recognizers.theory.dfa import DeterministicFiniteAutomaton
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammar
from recognizers.theory.cnf_cfg_generates import PyTorchChomskyNormalFormContextFreeGrammarMembershipTester
from recognizers.dataset_generation.generate_datasets import (
    apply_random_replacement,
    generate_random_string
)

class Language:

    def is_member(self, string: Iterable[Symbol]) -> bool:
        raise NotImplementedError

    def alphabet_size(self) -> int:
        raise NotImplementedError

class DeterministicFiniteAutomatonLanguage(Language):

    def __init__(self, dfa: DeterministicFiniteAutomaton) -> None:
        super().__init__()
        self._dfa = dfa

    def is_member(self, string: Iterable[Symbol]) -> bool:
        return dfa_accepts(self._dfa, string)

    def alphabet_size(self) -> int:
        return self._dfa.alphabet_size()

class ChomskyNormalFormContextFreeGrammarLanguage(Language):

    def __init__(self, cnf_cfg: ChomskyNormalFormContextFreeGrammar, device: torch.device) -> None:
        super().__init__()
        self._indexed_cnf_cfg = PyTorchChomskyNormalFormContextFreeGrammarMembershipTester(cnf_cfg, device)

    def is_member(self, string: Iterable[Symbol]) -> bool:
        return self._indexed_cnf_cfg.generates(string)

    def alphabet_size(self) -> int:
        return self._indexed_cnf_cfg.num_terminals()

def estimated_sensitivity_and_density(
    language: Language,
    length_range: tuple[int, int],
    num_samples: int,
    generator: random.Random
) -> tuple[float, float]:
    alphabet_size = language.alphabet_size()
    sensitivity = []
    density = []
    for _ in range(num_samples):
        string = random_string(length_range, alphabet_size, generator)
        is_member = language.is_member(string)
        if string:
            apply_random_replacement(string, alphabet_size, generator)
            new_is_member = language.is_member(string)
            changed = new_is_member != is_member
        else:
            changed = False
        sensitivity.append(changed)
        density.append(is_member)
    return statistics.fmean(sensitivity), statistics.fmean(density)

def random_string(
    length_range: tuple[int, int],
    alphabet_size: int,
    generator: random.Random
) -> list[Symbol]:
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return [Symbol(generator.randrange(alphabet_size)) for _ in range(length)]

def save_output(file_name: pathlib.Path | None, x: float) -> None:
    if file_name is not None:
        print(f'writing {file_name}')
        with file_name.open('w') as fout:
            print(x, file=fout)

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', required=True)
    parser.add_argument('--min-length', type=int, default=0)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--random-seed', type=int, default=123)
    parser.add_argument('--sensitivity-output', type=pathlib.Path)
    parser.add_argument('--density-output', type=pathlib.Path)
    parser.add_argument('--device')
    args = parser.parse_args()

    data = torch.load(args.language, weights_only=False)
    data_structure = data['language']
    language: Language
    match data_structure:
        case DeterministicFiniteAutomaton():
            language = DeterministicFiniteAutomatonLanguage(data_structure)
        case ChomskyNormalFormContextFreeGrammar():
            device = parse_device(args.device)
            language = ChomskyNormalFormContextFreeGrammarLanguage(data_structure, device)
        case _:
            raise ValueError
    generator = random.Random(args.random_seed)
    sensitivity, density = estimated_sensitivity_and_density(
        language,
        (args.min_length, args.max_length),
        args.num_samples,
        generator
    )
    save_output(args.sensitivity_output, sensitivity)
    save_output(args.density_output, density)

if __name__ == '__main__':
    main()
