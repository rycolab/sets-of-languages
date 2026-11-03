import random

from fsa import FSA as VieiraFSA

from recognizers.theory.symbol import ReservedSymbol
from recognizers.theory.nfa import NondeterministicFiniteAutomatonContainer
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.theory.dfas_are_isomorphic import dfas_are_isomorphic
from recognizers.theory.determinize_nfa import determinize_nfa
from recognizers.theory.nfa_is_trim import nfa_is_trim, nfa_is_accessible
from recognizers.theory.trim_nfa import trim_nfa
from recognizers.theory.dfas_are_equivalent import dfas_are_equivalent_symmetric_difference
from recognizers.theory.nfa_is_empty import nfa_is_empty
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.theory.nfa_accepts import nfa_accepts
from recognizers.theory.reverse_nfa import reverse_multi_start_nfa
from recognizers.theory.vieira_util import nfa_to_vieira_fsa, vieira_fsa_to_dfa, vieira_fsa_remove_state
from random_nfa import random_nfa, permute_nfa
from random_string import random_string

def test_example() -> None:
    # From Sipser Example 1.41.
    M = NondeterministicFiniteAutomatonContainer.from_parts(
        3,
        2,
        0,
        [
            (0, 1, 1),
            (0, ReservedSymbol.EPSILON, 2),
            (1, 0, 1),
            (1, 0, 2),
            (1, 1, 2),
            (2, 0, 0)
        ],
        [0]
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        5,
        2,
        0,
        [
            (0, 0, 0),
            (0, 1, 2),
            (1, 0, 0),
            (2, 0, 3),
            (2, 1, 1),
            (3, 0, 4),
            (3, 1, 1),
            (4, 0, 4),
            (4, 1, 3)
        ],
        [0, 4]
    )
    result = determinize_nfa(M)
    assert dfas_are_isomorphic(result, expected)

def test_empty() -> None:
    M = NondeterministicFiniteAutomatonContainer.from_parts(
        1,
        2,
        0,
        [],
        []
    )
    result = determinize_nfa(M)
    assert nfa_is_empty(result)

def test_random_nfas_sanity_check() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_nfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.4,
            accept_probability=0.4,
            generator=generator
        )
        det_M = determinize_nfa(M)
        # Check that the automata are equivalent on a bunch of random strings.
        for _ in range(20):
            w = random_string(M.alphabet_size(), 0, 20, generator)
            assert dfa_accepts(det_M, w) == nfa_accepts(M, w)
        # Check that all states are reachable from the start state.
        assert nfa_is_accessible(det_M)
        # Check that trimming first results in an equivalent DFA.
        trim_M = trim_nfa(M)
        det_trim_M = determinize_nfa(trim_M)
        assert dfas_are_equivalent_symmetric_difference(det_M, det_trim_M)
        # Check that determinizing a trim NFA results in a trim DFA.
        assert nfa_is_trim(det_trim_M)
        # Check that determinizing a permuted NFA results in an isomorphic DFA.
        perm_M = permute_nfa(M, generator)
        det_perm_M = determinize_nfa(perm_M)
        assert dfas_are_isomorphic(det_perm_M, det_M)

def vieira_det(M: VieiraFSA) -> VieiraFSA:
    # Remove the empty set state, if it exists.
    return vieira_fsa_remove_state(M.det(), frozenset())

def test_random_nfas_against_vieira() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_nfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.4,
            accept_probability=0.4,
            generator=generator
        )
        # Test that both implementations produce the same DFA.
        det_M = determinize_nfa(M)
        vieira_det_M = vieira_fsa_to_dfa(vieira_det(nfa_to_vieira_fsa(M)), M.alphabet_size())
        assert dfas_are_isomorphic(det_M, vieira_det_M)
        # Test that both implementations produce the same DFA when the input is
        # reversed.
        rev_det_M = determinize_nfa(reverse_multi_start_nfa(trim_nfa(M)))
        vieira_rev_det_M = vieira_fsa_to_dfa(vieira_det(nfa_to_vieira_fsa(trim_nfa(M)).reverse()), M.alphabet_size())
        assert dfas_are_isomorphic(rev_det_M, vieira_rev_det_M)
