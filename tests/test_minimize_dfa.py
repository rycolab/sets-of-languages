import random

from fsa import FSA as VieiraFSA
import pytest

from recognizers.theory.symbol import Symbol
from recognizers.theory.automaton import State
from recognizers.theory.dfa import (
    DeterministicFiniteAutomaton,
    DeterministicFiniteAutomatonContainer
)
from recognizers.theory.minimize_dfa import minimize_dfa, minimize_nfa_brzozowski
from recognizers.theory.dfas_are_isomorphic import dfas_are_isomorphic
from recognizers.theory.dfas_are_equivalent import dfas_are_equivalent_symmetric_difference
from recognizers.theory.trim_nfa import trim_nfa
from recognizers.theory.nfa_is_trim import nfa_is_trim
from recognizers.theory.vieira_util import nfa_to_vieira_fsa, vieira_fsa_to_dfa
from test_dfa import dfa_to_tuple
from random_dfa import random_dfa

def test_example() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(
        8,
        2,
        0,
        [
            (0, 0, 1),
            (0, 1, 3),
            (1, 0, 4),
            (1, 1, 2),
            (2, 0, 1),
            (2, 1, 5),
            (3, 0, 0),
            (3, 1, 4),
            (4, 0, 4),
            (4, 1, 4),
            (5, 0, 2),
            (5, 1, 4),
            (6, 0, 7),
            (6, 1, 2),
            (7, 0, 5),
            (7, 1, 6)
        ],
        [0, 2, 7]
    )
    min_M = minimize_dfa(M)
    assert dfas_are_isomorphic(
        min_M,
        DeterministicFiniteAutomatonContainer.from_parts(
            3,
            2,
            0,
            [
                (0, 0, 1),
                (0, 1, 2),
                (1, 1, 0),
                (2, 0, 0)
            ],
            [0]
        )
    )

def test_accept_all_already_minimal() -> None:
    A = 5
    M = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        A,
        0,
        [(0, a, 0) for a in range(A)],
        [0]
    )
    min_M = minimize_dfa(M)
    assert dfa_to_tuple(min_M) == (
        1,
        A,
        0,
        {(0, a, 0) for a in range(A)},
        {0}
    )

def test_reject_all_already_minimal() -> None:
    A = 5
    M = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        A,
        0,
        [],
        []
    )
    min_M = minimize_dfa(M)
    assert dfa_to_tuple(min_M) == (
        1,
        A,
        0,
        set(),
        set()
    )

def test_accept_all_not_minimal() -> None:
    Q = 4
    A = 5
    M = DeterministicFiniteAutomatonContainer.from_parts(
        Q,
        A,
        0,
        [(q, a, (q+1) % Q) for q in range(Q) for a in range(A)],
        list(range(Q))
    )
    min_M = minimize_dfa(M)
    assert dfa_to_tuple(min_M) == (
        1,
        A,
        0,
        {(0, a, 0) for a in range(A)},
        {0}
    )

def test_reject_all_not_minimal() -> None:
    Q = 4
    A = 5
    M = DeterministicFiniteAutomatonContainer.from_parts(
        Q,
        A,
        0,
        [(q, a, (q+1) % Q) for q in range(Q) for a in range(A)],
        []
    )
    min_M = minimize_dfa(M)
    assert dfa_to_tuple(min_M) == (
        1,
        A,
        0,
        set(),
        set()
    )

def test_chain() -> None:
    Q = 10
    M = DeterministicFiniteAutomatonContainer.from_parts(
        Q,
        1,
        0,
        [(q, 0, q+1) for q in range(Q-1)],
        [Q-1]
    )
    min_M = minimize_dfa(M)
    assert dfas_are_isomorphic(min_M, M)

def test_cycle() -> None:
    Q = 20
    M = DeterministicFiniteAutomatonContainer.from_parts(
        Q,
        1,
        0,
        [(q, 0, (q+1) % Q) for q in range(Q)],
        [0]
    )
    min_M = minimize_dfa(M)
    assert dfas_are_isomorphic(min_M, M)

def test_repeated_cycle() -> None:
    Q = 20
    N = 3
    M = DeterministicFiniteAutomatonContainer.from_parts(
        N * Q,
        1,
        0,
        [(q, 0, (q+1) % (N * Q)) for q in range(N * Q)],
        [i * Q for i in range(N)]
    )
    min_M = minimize_dfa(M)
    assert dfas_are_isomorphic(min_M, DeterministicFiniteAutomatonContainer.from_parts(
        Q,
        1,
        0,
        [(q, 0, (q+1) % Q) for q in range(Q)],
        [0]
    ))

def test_random_dfas_sanity_check() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        min_M = minimize_dfa(M)
        # Test that the DFA didn't get bigger and is trim.
        assert min_M.num_states() <= M.num_states()
        assert min_M.alphabet_size() <= M.alphabet_size()
        assert min_M.num_transitions() <= M.num_transitions()
        assert nfa_is_trim(min_M)
        # Test that the minimized DFA is equivalent.
        assert dfas_are_equivalent_symmetric_difference(M, min_M)
        # Test that trimming before minimizing results in the same DFA.
        trim_M = trim_nfa(M)
        min_trim_M = minimize_dfa(trim_M)
        assert dfas_are_isomorphic(min_M, min_trim_M)
        # Test that minimizing twice results in the same DFA.
        min_min_M = minimize_dfa(min_M)
        assert dfas_are_isomorphic(min_min_M, min_M)

def test_random_dfas_brzozowski_sanity_check() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        min_M = minimize_nfa_brzozowski(M)
        # Test that the DFA didn't get bigger and is trim.
        # assert min_M.num_states() <= M.num_states()
        assert min_M.alphabet_size() == M.alphabet_size()
        # assert min_M.num_transitions() <= M.num_transitions()
        assert nfa_is_trim(min_M)
        # Test that the minimized DFA is equivalent.
        assert dfas_are_equivalent_symmetric_difference(M, min_M)
        # Test that minimizing twice results in the same DFA.
        min_min_M = minimize_nfa_brzozowski(min_M)
        assert dfas_are_isomorphic(min_min_M, min_M)

def test_random_dfas_matches_brzozowski() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        min_M = minimize_dfa(M)
        b_min_M = minimize_nfa_brzozowski(M)
        assert dfas_are_isomorphic(min_M, b_min_M)

def ensure_start_state(N: VieiraFSA) -> None:
    if not N.start:
        start_state = 0
        assert start_state not in N.start
        N.add_start(start_state)

def minimize_dfa_with_vieira(M: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonContainer:
    N = nfa_to_vieira_fsa(M).min()
    ensure_start_state(N)
    return vieira_fsa_to_dfa(N, M.alphabet_size())

def minimize_dfa_brzozowski_with_vieira(M: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonContainer:
    N = nfa_to_vieira_fsa(M).min_brzozowski()
    ensure_start_state(N)
    return vieira_fsa_to_dfa(N, M.alphabet_size())

def test_random_dfas_vieira_brzozowski_sanity_check() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        min_M = minimize_dfa_brzozowski_with_vieira(M)
        # Test that the DFA didn't get bigger and is trim.
        assert min_M.num_states() <= M.num_states()
        assert min_M.alphabet_size() <= M.alphabet_size()
        assert min_M.num_transitions() <= M.num_transitions()
        is_trim = nfa_is_trim(min_M)
        are_equivalent = dfas_are_equivalent_symmetric_difference(M, min_M)
        assert nfa_is_trim(min_M)
        # Test that the minimized DFA is equivalent.
        assert dfas_are_equivalent_symmetric_difference(M, min_M)
        # Test that minimizing twice results in the same DFA.
        min_min_M = minimize_dfa_brzozowski_with_vieira(min_M)
        assert dfas_are_isomorphic(min_min_M, min_M)

def test_random_dfas_vieira_against_brzozowski() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        min_M = minimize_dfa_with_vieira(M)
        b_min_M = minimize_dfa_brzozowski_with_vieira(M)
        assert dfas_are_isomorphic(min_M, b_min_M)

def test_random_dfas_brzozowski_against_vieira() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        )
        min_M = minimize_nfa_brzozowski(M)
        vieira_min_M = minimize_dfa_brzozowski_with_vieira(M)
        assert dfas_are_isomorphic(min_M, vieira_min_M)
