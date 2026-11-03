from collections.abc import Iterable

from orderedsets import OrderedSet

from .symbol import Symbol
from .automaton import State, Automaton, AutomatonContainer, StartStateMixin, MultiAcceptStateMixin
from .nfa import NondeterministicFiniteAutomaton, Transition as NFATransition

class Transition(NFATransition):
    symbol: Symbol

class DeterministicFiniteAutomaton(NondeterministicFiniteAutomaton):
    r"""A (partial) deterministic finite automaton.

    The automaton does not need to be complete (not all symbols need to have an
    outgoing transition from each state).
    """

    def transitions(self) -> Iterable[Transition]:
        for state_from in self.states():
            for symbol in self.alphabet():
                state_to = self.transition(state_from, symbol)
                if state_to is not None:
                    yield Transition(state_from, symbol, state_to)

    def is_deterministic(self) -> bool:
        return True

    def transition(self, state: State, symbol: Symbol) -> State | None:
        r"""Given a state and a symbol, get the next state that the transition
        function maps them to. If there is no transition, return ``None``."""
        raise NotImplementedError

    def is_complete(self) -> bool:
        r"""Return whether this DFA is complete."""
        return self.num_transitions() == self.num_states() * self.alphabet_size()

class DeterministicFiniteAutomatonContainer(
    StartStateMixin,
    MultiAcceptStateMixin,
    AutomatonContainer,
    DeterministicFiniteAutomaton
):

    _transitions: dict[tuple[State, Symbol], State]

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
        self._transitions = {}

    @staticmethod
    def from_parts(
        num_states: int,
        alphabet_size: int,
        start_state: int,
        transitions: Iterable[tuple[int, int, int]],
        accept_states: Iterable[int]
    ) -> DeterministicFiniteAutomatonContainer:
        M = DeterministicFiniteAutomatonContainer(
            num_states=num_states,
            alphabet_size=alphabet_size,
            start_state=State(start_state)
        )
        for q, a, r in transitions:
            M.add_transition(State(q), Symbol(a), State(r))
        for f in accept_states:
            M.add_accept_state(State(f))
        return M

    def __repr__(self) -> str:
        return f'''\
DeterministicFiniteAutomatonContainer.from_parts(
    {self.num_states()},
    {self.alphabet_size()},
    {self.start_state()},
    [{','.join(f'\n        ({t.state_from}, {t.symbol}, {t.state_to})' for t in self.transitions())}
    ],
    [{', '.join(map(str, self.accept_states()))}]
)'''

    def remove_state(self, state: State) -> None:
        r"""Remove a state and any transitions that reference it. Renumber all
        states. Removing the start state is not allowed. Note that this is
        *not* O(1) time.
        """
        if not (0 <= state < self._num_states):
            raise ValueError(f'invalid state {state}')
        if state == self.start_state():
            raise ValueError('unable to remove start state')
        self._num_states -= 1
        self._start_state = State(self._start_state - (self._start_state > state))
        new_transitions = {}
        for (state_from, symbol), state_to in self._transitions.items():
            if state_from != state and state_to != state:
                new_state_from = State(state_from - (state_from > state))
                new_state_to = State(state_to - (state_to > state))
                new_transitions[new_state_from, symbol] = new_state_to
        self._transitions = new_transitions
        self._accept_states.discard(state)
        self._accept_states = OrderedSet(State(f - (f > state)) for f in self._accept_states)

    def num_transitions(self) -> int:
        return len(self._transitions)

    def transitions(self) -> Iterable[Transition]:
        for (state_from, symbol), state_to in self._transitions.items():
            yield Transition(state_from, symbol, state_to)

    def transition(self, state: State, symbol: Symbol) -> State | None:
        return self._transitions.get((state, symbol))

    def add_transition(self, state_from: State, symbol: Symbol, state_to: State) -> None:
        r"""Add a transition.

        :raises ValueError: if a transition from ``state_from`` on ``symbol``
            already exists."""
        if not (0 <= state_from < self._num_states):
            raise ValueError(f'invalid state_from {state_from}')
        if not (0 <= symbol < self._alphabet_size):
            raise ValueError(f'invalid symbol {symbol}')
        if not (0 <= state_to < self._num_states):
            raise ValueError(f'invalid state_to {state_to}')
        key = (state_from, symbol)
        if key in self._transitions:
            raise ValueError(
                f'a transition from state {state_from} on symbol {symbol} '
                f'already exists'
            )
        self._transitions[key] = state_to

    def set_transition(self, state_from: State, symbol: Symbol, state_to: State) -> None:
        r"""Set the destination state for a transition. It doesn't matter if a
        transition from ``state_from`` on ``symbol`` already exists or not.
        """
        self._transitions[state_from, symbol] = state_to

    def remove_transition(self, state_from: State, symbol: Symbol) -> None:
        r"""Remove a transition."""
        del self._transitions[state_from, symbol]

    def copy(self) -> DeterministicFiniteAutomatonContainer:
        cls = self.__class__
        copy = cls.__new__(cls)
        copy._num_states = self._num_states
        copy._alphabet_size = self._alphabet_size
        copy._start_state = self._start_state
        copy._transitions = self._transitions.copy()
        copy._accept_states = self._accept_states.copy()
        return copy
