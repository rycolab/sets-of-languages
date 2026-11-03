from recognizers.theory.symbol import Symbol
from recognizers.theory.automaton import State
from recognizers.theory.nfa import NondeterministicFiniteAutomatonContainer
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.theory.nfa_is_trim import nfa_is_trim

def make_nfa(
    num_states: int,
    alphabet_size: int,
    start_state: int,
    transitions: list[tuple[int, int, int]],
    accept_states: list[int]
) -> NondeterministicFiniteAutomatonContainer:
    M = NondeterministicFiniteAutomatonContainer(
        num_states=num_states,
        alphabet_size=alphabet_size,
        start_state=State(start_state)
    )
    for q, a, r in transitions:
        M.add_transition(State(q), Symbol(a), State(r))
    for f in accept_states:
        M.add_accept_state(State(f))
    return M

def test_empty_nfa() -> None:
    assert nfa_is_trim(make_nfa(1, 1, 0, [], []))

def test_untrim_empty_nfa() -> None:
    assert not nfa_is_trim(make_nfa(2, 1, 0, [(0, 0, 1)], []))

def test_empty_nfa_with_self_loop() -> None:
    assert not nfa_is_trim(make_nfa(1, 1, 0, [(0, 0, 0)], []))

def test_accepts_all_strings() -> None:
    assert nfa_is_trim(make_nfa(1, 2, 0, [(0, 0, 0), (0, 1, 0)], [0]))

def test_empty_string() -> None:
    assert nfa_is_trim(make_nfa(1, 1, 0, [], [0]))

def test_unused_symbols() -> None:
    assert nfa_is_trim(make_nfa(2, 10, 0, [(0, 2, 1), (1, 6, 0)], [0]))

def test_disconnected_states() -> None:
    assert not nfa_is_trim(make_nfa(10, 2, 0, [], [2, 3, 6]))

def test_long_connected_chain() -> None:
    n = 50
    assert nfa_is_trim(make_nfa(
        n,
        1,
        0,
        [
            (i, 0, i+1)
            for i in range(n-1)
        ],
        [n-1]
    ))

def test_simple_cycle() -> None:
    assert nfa_is_trim(make_nfa(
        3,
        1,
        0,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 1)
        ],
        [1]
    ))

def test_disconnected_cycles() -> None:
    assert not nfa_is_trim(make_nfa(
        6,
        1,
        0,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 1),
            (3, 0, 4),
            (4, 0, 5),
            (5, 0, 3)
        ],
        [1, 3, 5]
    ))

def test_non_accessible_cycle() -> None:
    assert not nfa_is_trim(make_nfa(
        8,
        1,
        0,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 1),
            (3, 0, 4),
            (4, 0, 5),
            (5, 0, 6),
            (2, 0, 6),
            (5, 0, 3),
            (6, 0, 7),
            (7, 0, 6)
        ],
        [1, 3, 5, 6]
    ))

def test_accessible_cycle() -> None:
    assert nfa_is_trim(make_nfa(
        8,
        1,
        0,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 1),
            (3, 0, 4),
            (4, 0, 5),
            (5, 0, 6),
            (2, 0, 6),
            (5, 0, 3),
            (6, 0, 7),
            (7, 0, 6),
            (1, 0, 3)
        ],
        [1, 3, 5, 6]
    ))

def test_non_coaccessible_cycle() -> None:
    assert not nfa_is_trim(make_nfa(
        6,
        1,
        0,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 1),
            (3, 0, 4),
            (4, 0, 5),
            (5, 0, 3),
            (2, 0, 3)
        ],
        [1, 2]
    ))

def test_tricky_podfa() -> None:
    M = DeterministicFiniteAutomatonContainer(num_states=12, alphabet_size=6, start_state=State(0))
    M.add_transition(State(0), Symbol(1), State(1))
    M.add_transition(State(0), Symbol(3), State(6))
    M.add_transition(State(0), Symbol(4), State(5))
    M.add_transition(State(0), Symbol(5), State(3))
    M.add_transition(State(1), Symbol(0), State(4))
    M.add_transition(State(1), Symbol(2), State(1))
    M.add_transition(State(1), Symbol(3), State(7))
    M.add_transition(State(1), Symbol(4), State(10))
    M.add_transition(State(2), Symbol(3), State(2))
    M.add_transition(State(2), Symbol(4), State(4))
    M.add_transition(State(3), Symbol(3), State(9))
    M.add_transition(State(4), Symbol(1), State(11))
    M.add_transition(State(4), Symbol(2), State(7))
    M.add_transition(State(4), Symbol(4), State(4))
    M.add_transition(State(5), Symbol(2), State(8))
    M.add_transition(State(5), Symbol(5), State(11))
    M.add_transition(State(6), Symbol(1), State(11))
    M.add_transition(State(6), Symbol(2), State(11))
    M.add_transition(State(6), Symbol(3), State(7))
    M.add_transition(State(7), Symbol(1), State(7))
    M.add_transition(State(7), Symbol(3), State(7))
    M.add_transition(State(8), Symbol(0), State(8))
    M.add_transition(State(8), Symbol(2), State(10))
    M.add_transition(State(8), Symbol(3), State(11))
    M.add_transition(State(8), Symbol(4), State(8))
    M.add_transition(State(8), Symbol(5), State(10))
    M.add_transition(State(9), Symbol(0), State(10))
    M.add_transition(State(10), Symbol(0), State(10))
    M.add_transition(State(10), Symbol(2), State(11))
    M.add_transition(State(10), Symbol(3), State(11))
    M.add_transition(State(11), Symbol(2), State(11))
    M.add_accept_state(State(7))
    M.add_accept_state(State(8))
    M.add_accept_state(State(9))
    M.add_accept_state(State(10))
    M.add_accept_state(State(11))
    assert not nfa_is_trim(M)
