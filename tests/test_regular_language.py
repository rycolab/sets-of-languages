import random

import torch

from recognizers.theory.symbol import Symbol
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.string_sampling.dfa.weight_lifting import lift_dfa_weights
from recognizers.string_sampling.dfa.weight_pushing import WeightedDeterministicFiniteAutomatonSampler
from recognizers.string_sampling.dfa.language import DeterministicFiniteAutomatonLanguage
from recognizers.dataset_generation.generate_datasets import generate_example

from random_string import random_string

def matches_repeat_00(w: tuple[Symbol, ...]) -> bool:
    return len(w) % 2 == 0 and all(wi == 0 for wi in w)

def test_repeat_00() -> None:
    generator = random.Random(123)
    alphabet_size = 2
    dfa = DeterministicFiniteAutomatonContainer.from_parts(
        num_states=2,
        alphabet_size=alphabet_size,
        start_state=0,
        transitions=[
            (0, 0, 1),
            (1, 0, 0)
        ],
        accept_states=[0]
    )
    for _ in range(100):
        w = tuple(random_string(alphabet_size, 0, 100, generator))
        assert dfa_accepts(dfa, w) == matches_repeat_00(w)
    dtype = torch.float32
    device = torch.device('cpu')
    wdfa = lift_dfa_weights(
        dfa,
        max_length=100,
        dtype=dtype,
        device=device
    )
    sampler = WeightedDeterministicFiniteAutomatonSampler(
        dfa,
        wdfa,
        dtype=dtype,
        device=device
    )
    language = DeterministicFiniteAutomatonLanguage(sampler, None, dtype, device)
    for _ in range(100):
        w = tuple(random_string(alphabet_size, 0, 100, generator))
        is_negative, edit_distance = language.is_negative(w, False)
        assert is_negative == (not matches_repeat_00(w))
    language_with_length = language.with_length_range((0, 100))
    for _ in range(100):
        w = tuple(random_string(alphabet_size, 0, 100, generator))
        is_negative, edit_distance = language_with_length.is_negative(w, False)
        assert is_negative == (not matches_repeat_00(w))
    dataset = []
    for _ in range(100):
        w, label, parse, edit_distance, num_edits = generate_example(
            language=language_with_length,
            length_range=(0, 100),
            alphabet_size=alphabet_size,
            only_negative=False,
            perturbation_probability=0.5,
            strict_num_edits_distribution=False,
            include_log_probability=False,
            include_next_symbols=False,
            include_edit_distance=False,
            generator=generator,
            excluded_strings=None,
            max_positive_retries=1000,
            max_negative_retries=1000
        )
        dataset.append((w, label))
    for w, label in dataset:
        assert label == matches_repeat_00(w)
    assert any(len(w) % 2 == 1 and all(wi == 0 for wi in w) for w, label in dataset)
