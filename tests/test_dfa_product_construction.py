import random

from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.theory.dfa_product_construction import dfa_union, dfa_intersection
from recognizers.theory.nfa_union import nfa_union
from recognizers.theory.determinize_nfa import determinize_nfa
from recognizers.theory.dfas_are_equivalent import dfas_are_equivalent
from recognizers.theory.nfa_is_trim import nfa_is_trim, nfa_is_accessible
from recognizers.theory.trim_nfa import trim_nfa
from random_dfa import random_dfa

def test_simple_dfa_union() -> None:
    A = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1)],
        [1]
    )
    B = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 1, 1)],
        [1]
    )
    result = dfa_union(A, B)
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1), (0, 1, 1)],
        [1]
    )
    assert dfas_are_equivalent(result, expected)

def test_random_dfa_union() -> None:
    generator = random.Random(123)
    for i in range(100):
        A = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        B = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        result = dfa_union(A, B)
        nfa_result = determinize_nfa(nfa_union(A, B))
        assert dfas_are_equivalent(result, nfa_result)
        assert nfa_is_accessible(result)
        trim_result = dfa_union(trim_nfa(A), trim_nfa(B))
        assert dfas_are_equivalent(trim_result, nfa_result)
        assert nfa_is_trim(trim_result)
