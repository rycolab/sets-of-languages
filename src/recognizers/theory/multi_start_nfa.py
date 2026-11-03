import dataclasses
from collections.abc import Iterable
from typing import Any, Literal

from orderedsets import OrderedSet

from .symbol import Symbol, ReservedSymbol
from .automaton import (
    State,
    Automaton,
    AutomatonContainer,
    MultiStartStateMixin,
    MultiAcceptStateMixin
)

SymbolOrEpsilon = Symbol | Literal[ReservedSymbol.EPSILON]

@dataclasses.dataclass(frozen=True)
class Transition:
    r"""A transition in an NFA."""
    state_from: State
    symbol: SymbolOrEpsilon
    state_to: State

class MultiStartNondeterministicFiniteAutomaton(Automaton):
    r"""A nondeterministic finite automaton (NFA) that allows multiple start
    states.
    """

    def is_start_state(self, state: State) -> bool:
        r"""Check whether a state is a start state."""
        raise NotImplementedError

    def num_start_states(self) -> int:
        r"""Get the number of start states."""
        return sum(1 for _ in self.start_states())

    def start_states(self) -> Iterable[State]:
        r"""Get the set of start states."""
        return filter(self.is_start_state, self.states())

    def num_transitions(self) -> int:
        r"""Get the number of transitions."""
        return sum(1 for _ in self.transitions())

    def transitions(self) -> Iterable[Transition]:
        r"""Get a list of all transitions."""
        raise NotImplementedError

    def is_accept_state(self, state: State) -> bool:
        r"""Check whether a state is an accept state."""
        raise NotImplementedError

    def num_accept_states(self) -> int:
        r"""Get the number of accept states."""
        return sum(1 for _ in self.accept_states())

    def accept_states(self) -> Iterable[State]:
        r"""Get the set of accept states."""
        return filter(self.is_accept_state, self.states())

    def is_deterministic(self) -> bool:
        r"""Check whether this NFA is a partial DFA.

        The automaton does not need to be complete (not all symbols need to have
        an outgoing transition from each state).
        """
        return self.num_start_states() == 1 and self.has_deterministic_transitions()

    def has_deterministic_transitions(self) -> bool:
        seen = set()
        for transition in self.transitions():
            key = (transition.state_from, transition.symbol)
            if key in seen:
                return False
            seen.add(key)
        return True

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MultiStartNondeterministicFiniteAutomaton) and
            self.num_states() == other.num_states() and
            self.alphabet_size() == other.alphabet_size() and
            set(self.start_states()) == set(other.start_states()) and
            set(self.transitions()) == set(other.transitions()) and
            set(self.accept_states()) == set(other.accept_states())
        )

class TransitionMixin(AutomatonContainer):

    _transitions: OrderedSet[Transition]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._transitions = OrderedSet()

    def num_transitions(self) -> int:
        return len(self._transitions)

    def transitions(self) -> Iterable[Transition]:
        return self._transitions

    def add_transition(self, state_from: State, symbol: SymbolOrEpsilon, state_to: State) -> None:
        r"""Add a transition."""
        if not (0 <= state_from < self._num_states):
            raise ValueError(f'invalid state_from {state_from}')
        if not (symbol == ReservedSymbol.EPSILON or 0 <= symbol < self._alphabet_size):
            raise ValueError(f'invalid symbol {symbol}')
        if not (0 <= state_to < self._num_states):
            raise ValueError(f'invalid state_to {state_to}')
        self._transitions.add(Transition(state_from, symbol, state_to))

class MultiStartNondeterministicFiniteAutomatonContainer(
    MultiStartStateMixin,
    TransitionMixin,
    MultiAcceptStateMixin,
    AutomatonContainer,
    MultiStartNondeterministicFiniteAutomaton
):

    def __init__(self,
        *,
        num_states: int = 0,
        alphabet_size: int
    ) -> None:
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size
        )

    @staticmethod
    def from_parts(
        num_states: int,
        alphabet_size: int,
        start_states: Iterable[int],
        transitions: Iterable[tuple[int, int | Literal[ReservedSymbol.EPSILON], int]],
        accept_states: Iterable[int]
    ) -> 'MultiStartNondeterministicFiniteAutomatonContainer':
        M = MultiStartNondeterministicFiniteAutomatonContainer(
            num_states=num_states,
            alphabet_size=alphabet_size
        )
        for s in start_states:
            M.add_start_state(State(s))
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
MultiStartNondeterministicFiniteAutomatonContainer.from_parts(
    {self.num_states()},
    {self.alphabet_size()},
    [{', '.join(map(str, self.start_states()))}],
    [{','.join(f'\n        ({t.state_from}, {t.symbol}, {t.state_to})' for t in self.transitions())}
    ],
    [{', '.join(map(str, self.accept_states()))}]
)'''
