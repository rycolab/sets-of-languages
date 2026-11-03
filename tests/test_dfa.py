from recognizers.theory.symbol import Symbol
from recognizers.theory.automaton import State
from recognizers.theory.dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer

def dfa_to_tuple(
    M: DeterministicFiniteAutomaton
) -> tuple[int, int, State, set[tuple[State, Symbol, State]], set[State]]:
    return (
        M.num_states(),
        M.alphabet_size(),
        M.start_state(),
        {(t.state_from, t.symbol, t.state_to) for t in M.transitions()},
        set(M.accept_states())
    )

def test_remove_state() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(
        6,
        2,
        4,
        [
            (0, 0, 1),
            (0, 1, 2),
            (1, 0, 3),
            (1, 1, 2),
            (2, 0, 5),
            (2, 1, 2),
            (3, 0, 3),
            (3, 1, 3),
            (4, 0, 0),
            (4, 1, 1),
            (5, 0, 2),
            (5, 1, 4)
        ],
        [1, 2, 4]
    )
    M.remove_state(State(2))
    assert dfa_to_tuple(M) == (
        5,
        2,
        3,
        {
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 2),
            (2, 1, 2),
            (3, 0, 0),
            (3, 1, 1),
            (4, 1, 3)
        },
        {1, 3}
    )
