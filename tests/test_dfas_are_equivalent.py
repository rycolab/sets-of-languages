import random

from recognizers.theory.dfas_are_equivalent import (
    dfas_are_equivalent_symmetric_difference,
    dfas_are_equivalent_minimization
)
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from random_dfa import random_dfa, permute_dfa

def test_simple_dfas() -> None:
    A = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        2,
        0,
        [
            (0, 0, 0),
            (0, 1, 0)
        ],
        [0]
    )
    B = DeterministicFiniteAutomatonContainer.from_parts(
        3,
        5, # Bigger alphabet
        1,
        [
            (0, 0, 1),
            (0, 1, 2),
            (1, 0, 2),
            (1, 1, 1),
            (2, 0, 0),
            (2, 1, 2)
        ],
        [0, 1, 2]
    )
    assert dfas_are_equivalent_symmetric_difference(A, B)
    assert dfas_are_equivalent_symmetric_difference(B, A)
    # assert dfas_are_equivalent_minimization(A, B)
    # assert dfas_are_equivalent_minimization(B, A)

def test_simple_non_equivalent_dfas() -> None:
    A = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [
            (0, 0, 1),
            (1, 1, 0)
        ],
        [1]
    )
    B = DeterministicFiniteAutomatonContainer.from_parts(
        3,
        2,
        0,
        [
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 2)
        ],
        [1, 2]
    )
    assert not dfas_are_equivalent_symmetric_difference(A, B)
    assert not dfas_are_equivalent_symmetric_difference(B, A)
    # assert not dfas_are_equivalent_minimization(A, B)
    # assert not dfas_are_equivalent_minimization(B, A)

def test_random_dfas_equivalent_to_self() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        permuted_M = permute_dfa(M, generator)
        assert dfas_are_equivalent_symmetric_difference(M, permuted_M)
        assert dfas_are_equivalent_symmetric_difference(permuted_M, M)
        # assert dfas_are_equivalent_minimization(M, permnuted_M)
        # assert dfas_are_equivalent_minimization(permnuted_M, M)
