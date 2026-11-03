from collections.abc import Iterable
from typing import Literal

from orderedsets import OrderedSet

from .symbol import Symbol, ReservedSymbol
from .automaton import State, AutomatonContainer, StartStateMixin, MultiAcceptStateMixin
from .multi_start_nfa import (
    MultiStartNondeterministicFiniteAutomaton,
    Transition,
    SymbolOrEpsilon,
    TransitionMixin
)

class NondeterministicFiniteAutomaton(MultiStartNondeterministicFiniteAutomaton):
    r"""A nondeterministic finite automaton (NFA)."""

    def start_state(self) -> State:
        r"""Get the start state."""
        raise NotImplementedError

    def is_start_state(self, state: State) -> bool:
        return state == self.start_state()

    def num_start_states(self) -> int:
        return 1

    def start_states(self) -> Iterable[State]:
        return [self.start_state()]

    def is_deterministic(self) -> bool:
        return self.has_deterministic_transitions()

class NondeterministicFiniteAutomatonContainer(
    StartStateMixin,
    TransitionMixin,
    MultiAcceptStateMixin,
    AutomatonContainer,
    NondeterministicFiniteAutomaton
):

    def __init__(self,
        *,
        num_states: int = 1,
        alphabet_size: int,
        start_state: State = State(0)
    ) -> None:
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            start_state=start_state
        )

    @staticmethod
    def from_parts(
        num_states: int,
        alphabet_size: int,
        start_state: int,
        transitions: Iterable[tuple[int, int | Literal[ReservedSymbol.EPSILON], int]],
        accept_states: Iterable[int]
    ) -> 'NondeterministicFiniteAutomatonContainer':
        M = NondeterministicFiniteAutomatonContainer(
            num_states=num_states,
            alphabet_size=alphabet_size,
            start_state=State(start_state)
        )
        for q, a, r in transitions:
            M.add_transition(
                State(q),
                Symbol(a) if a != ReservedSymbol.EPSILON else a,
                State(r)
            )
        for f in accept_states:
            M.add_accept_state(State(f))
        return M

    def __repr__(self) -> str:
        return f'''\
NondeterministicFiniteAutomatonContainer.from_parts(
    {self.num_states()},
    {self.alphabet_size()},
    {self.start_state()},
    [{','.join(f'\n        ({t.state_from}, {t.symbol}, {t.state_to})' for t in self.transitions())}
    ],
    [{', '.join(map(str, self.accept_states()))}]
)'''
