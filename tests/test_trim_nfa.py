import random

from recognizers.theory.symbol import Symbol
from recognizers.theory.automaton import State
from recognizers.theory.nfa import (
    SymbolOrEpsilon,
    NondeterministicFiniteAutomatonContainer
)
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.tools.random_utils import (
    sample_from_negative_binomial,
    sample_from_bernoulli
)
from recognizers.theory.trim_nfa import trim_nfa
from recognizers.theory.nfa_is_trim import nfa_is_trim
from recognizers.theory.dfas_are_equivalent import dfas_are_equivalent_symmetric_difference
from random_dfa import random_dfa

def result(
    num_states: int,
    alphabet_size: int,
    transitions: list[tuple[int, int, int]],
    start_state: int,
    accept_states: list[int],
    trim_alphabet: bool = False
) -> tuple[int, int, set[tuple[State, SymbolOrEpsilon, State]], State, set[State]]:
    M = NondeterministicFiniteAutomatonContainer(
        num_states=num_states,
        alphabet_size=alphabet_size,
        start_state=State(start_state)
    )
    for q, a, r in transitions:
        M.add_transition(State(q), Symbol(a), State(r))
    for f in accept_states:
        M.add_accept_state(State(f))
    trim_M = trim_nfa(M, trim_alphabet=trim_alphabet)
    return (
        trim_M.num_states(),
        trim_M.alphabet_size(),
        {(t.state_from, t.symbol, t.state_to) for t in trim_M.transitions()},
        trim_M.start_state(),
        set(trim_M.accept_states())
    )

def test_reachable_from_start() -> None:
    assert result(
        6,
        1,
        [
            (0, 0, 1),
            (1, 0, 4),
            (4, 0, 0),
            (2, 0, 3),
            (3, 0, 5),
            (5, 0, 2)
        ],
        0,
        [1, 5]
    ) == (
        3,
        1,
        {
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 0)
        },
        0,
        {1}
    )

def test_reachable_from_accept() -> None:
    assert result(
        8,
        1,
        [
            (0, 0, 1),
            (1, 0, 3),
            (3, 0, 4),
            (4, 0, 1),
            (0, 0, 2),
            (2, 0, 5),
            (5, 0, 4),
            (2, 0, 6),
            (6, 0, 7),
            (5, 0, 7)
        ],
        0,
        [7]
    ) == (
        5,
        1,
        {
            (0, 0, 1),
            (1, 0, 2),
            (1, 0, 3),
            (2, 0, 4),
            (3, 0, 4)
        },
        0,
        {4}
    )

def test_reachable_from_multiple_accept() -> None:
    assert result(
        5,
        1,
        [
            (3, 0, 4),
            (3, 0, 2),
            (0, 0, 2),
            (0, 0, 1)
        ],
        3,
        [1, 2, 4]
    ) == (
        3,
        1,
        {
            (1, 0, 2),
            (1, 0, 0)
        },
        1,
        {0, 2}
    )

def test_reject_after_accept() -> None:
    assert result(
        3,
        1,
        [
            (0, 0, 1),
            (1, 0, 2)
        ],
        0,
        [1]
    ) == (
        2,
        1,
        {
            (0, 0, 1)
        },
        0,
        {1}
    )

def test_extra_states() -> None:
    assert result(
        3,
        1,
        [],
        0,
        []
    ) == (
        1,
        1,
        set(),
        0,
        set()
    )

def test_empty_cycle() -> None:
    assert result(
        3,
        1,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 0)
        ],
        0,
        []
    ) == (
        1,
        1,
        set(),
        0,
        set()
    )

def test_empty_loop() -> None:
    assert result(
        1,
        1,
        [
            (0, 0, 0)
        ],
        0,
        []
    ) == (
        1,
        1,
        set(),
        0,
        set()
    )

def test_empty_cycle_and_loop() -> None:
    assert result(
        3,
        1,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 0),
            (0, 0, 0)
        ],
        0,
        []
    ) == (
        1,
        1,
        set(),
        0,
        set()
    )

def test_empty_one_state() -> None:
    assert result(
        1,
        1,
        [],
        0,
        []
    ) == (
        1,
        1,
        set(),
        0,
        set()
    )

def test_accept_one_state() -> None:
    assert result(
        1,
        1,
        [],
        0,
        [0]
    ) == (
        1,
        1,
        set(),
        0,
        {0}
    )

def test_remove_symbol_1() -> None:
    assert result(
        3,
        2,
        [
            (0, 0, 1),
            (1, 1, 2)
        ],
        0,
        [1],
        trim_alphabet=True
    ) == (
        2,
        1,
        {
            (0, 0, 1)
        },
        0,
        {1}
    )

def test_remove_symbol_0() -> None:
    assert result(
        3,
        2,
        [
            (0, 1, 1),
            (1, 0, 2)
        ],
        0,
        [1],
        trim_alphabet=True
    ) == (
        2,
        1,
        {
            (0, 0, 1)
        },
        0,
        {1}
    )

def test_remap_symbol_and_state() -> None:
    assert result(
        3,
        2,
        [
            (0, 1, 2),
            (2, 0, 1)
        ],
        0,
        [2],
        trim_alphabet=True
    ) == (
        2,
        1,
        {
            (0, 0, 1)
        },
        0,
        {1}
    )

def test_random_dfas() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.4,
            accept_probability=0.4,
            generator=generator
        )
        trim_M = trim_nfa(M)
        assert nfa_is_trim(trim_M)
        assert dfas_are_equivalent_symmetric_difference(M, trim_M)
        trim_M = trim_nfa(M, trim_alphabet=True)
        assert nfa_is_trim(trim_M)

def test_tricky_podfa() -> None:
    M = DeterministicFiniteAutomatonContainer(num_states=26, alphabet_size=6, start_state=State(0))
    M.add_transition(State(0), Symbol(1), State(2))
    M.add_transition(State(0), Symbol(3), State(19))
    M.add_transition(State(0), Symbol(4), State(17))
    M.add_transition(State(0), Symbol(5), State(10))
    M.add_transition(State(1), Symbol(2), State(2))
    M.add_transition(State(1), Symbol(3), State(11))
    M.add_transition(State(2), Symbol(0), State(13))
    M.add_transition(State(2), Symbol(2), State(8))
    M.add_transition(State(2), Symbol(3), State(20))
    M.add_transition(State(2), Symbol(4), State(23))
    M.add_transition(State(3), Symbol(0), State(8))
    M.add_transition(State(3), Symbol(1), State(3))
    M.add_transition(State(3), Symbol(3), State(7))
    M.add_transition(State(3), Symbol(5), State(21))
    M.add_transition(State(4), Symbol(3), State(6))
    M.add_transition(State(4), Symbol(4), State(23))
    M.add_transition(State(4), Symbol(5), State(7))
    M.add_transition(State(5), Symbol(0), State(17))
    M.add_transition(State(5), Symbol(1), State(9))
    M.add_transition(State(5), Symbol(2), State(19))
    M.add_transition(State(5), Symbol(3), State(5))
    M.add_transition(State(5), Symbol(4), State(25))
    M.add_transition(State(5), Symbol(5), State(16))
    M.add_transition(State(6), Symbol(0), State(13))
    M.add_transition(State(6), Symbol(4), State(11))
    M.add_transition(State(7), Symbol(4), State(11))
    M.add_transition(State(7), Symbol(5), State(9))
    M.add_transition(State(8), Symbol(3), State(8))
    M.add_transition(State(8), Symbol(4), State(13))
    M.add_transition(State(9), Symbol(1), State(24))
    M.add_transition(State(9), Symbol(3), State(24))
    M.add_transition(State(10), Symbol(3), State(22))
    M.add_transition(State(11), Symbol(0), State(15))
    M.add_transition(State(11), Symbol(1), State(17))
    M.add_transition(State(11), Symbol(2), State(22))
    M.add_transition(State(11), Symbol(3), State(22))
    M.add_transition(State(12), Symbol(0), State(15))
    M.add_transition(State(12), Symbol(4), State(17))
    M.add_transition(State(13), Symbol(1), State(24))
    M.add_transition(State(13), Symbol(2), State(20))
    M.add_transition(State(13), Symbol(4), State(13))
    M.add_transition(State(14), Symbol(0), State(23))
    M.add_transition(State(14), Symbol(1), State(14))
    M.add_transition(State(14), Symbol(2), State(20))
    M.add_transition(State(15), Symbol(0), State(25))
    M.add_transition(State(15), Symbol(1), State(24))
    M.add_transition(State(15), Symbol(3), State(20))
    M.add_transition(State(15), Symbol(4), State(24))
    M.add_transition(State(15), Symbol(5), State(23))
    M.add_transition(State(16), Symbol(0), State(20))
    M.add_transition(State(16), Symbol(1), State(18))
    M.add_transition(State(16), Symbol(3), State(22))
    M.add_transition(State(16), Symbol(4), State(25))
    M.add_transition(State(17), Symbol(2), State(21))
    M.add_transition(State(17), Symbol(3), State(25))
    M.add_transition(State(17), Symbol(5), State(24))
    M.add_transition(State(18), Symbol(0), State(25))
    M.add_transition(State(18), Symbol(1), State(23))
    M.add_transition(State(18), Symbol(2), State(18))
    M.add_transition(State(18), Symbol(5), State(25))
    M.add_transition(State(19), Symbol(1), State(24))
    M.add_transition(State(19), Symbol(2), State(24))
    M.add_transition(State(19), Symbol(3), State(20))
    M.add_transition(State(20), Symbol(1), State(20))
    M.add_transition(State(20), Symbol(3), State(20))
    M.add_transition(State(21), Symbol(0), State(21))
    M.add_transition(State(21), Symbol(2), State(23))
    M.add_transition(State(21), Symbol(3), State(24))
    M.add_transition(State(21), Symbol(4), State(21))
    M.add_transition(State(21), Symbol(5), State(23))
    M.add_transition(State(22), Symbol(0), State(23))
    M.add_transition(State(22), Symbol(4), State(25))
    M.add_transition(State(23), Symbol(0), State(23))
    M.add_transition(State(23), Symbol(2), State(24))
    M.add_transition(State(23), Symbol(3), State(24))
    M.add_transition(State(23), Symbol(4), State(25))
    M.add_transition(State(24), Symbol(2), State(24))
    M.add_transition(State(24), Symbol(5), State(25))
    M.add_transition(State(25), Symbol(1), State(25))
    M.add_transition(State(25), Symbol(4), State(25))
    M.add_transition(State(25), Symbol(5), State(25))
    M.add_accept_state(State(1))
    M.add_accept_state(State(6))
    M.add_accept_state(State(9))
    M.add_accept_state(State(11))
    M.add_accept_state(State(15))
    M.add_accept_state(State(20))
    M.add_accept_state(State(21))
    M.add_accept_state(State(22))
    M.add_accept_state(State(23))
    M.add_accept_state(State(24))
    trim_M = trim_nfa(M)
    assert nfa_is_trim(trim_M)
