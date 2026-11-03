from collections.abc import Iterable

from .symbol import Symbol, ReservedSymbol
from .automaton import State, AutomatonContainer, StartStateMixin
from .semiring import Semiring
from .wfa import (
    WeightedFiniteAutomaton,
    SemiringMixin,
    AcceptWeightsMixin,
    Transition as WFATransition
)
from .dfa import Transition as DFATransition
from .nfa import SymbolOrEpsilon

class Transition(DFATransition, WFATransition):
    symbol: Symbol

class WeightedDeterministicFiniteAutomaton[Weight](WeightedFiniteAutomaton[Weight]):
    r"""A (partial) weighted deterministic finite automaton.

    There is only one start state with start weight 1.
    """

    def start_state(self) -> State:
        r"""Get the start state."""
        raise NotImplementedError

    def start_weights(self) -> Iterable[tuple[State, Weight]]:
        dtype, device = self.dtype_and_device()
        return [(self.start_state(), self.semiring().ones((), dtype, device))]

    def transitions(self) -> Iterable[Transition]:
        for state_from in self.states():
            for symbol in self.alphabet():
                result = self.transition(state_from, symbol)
                if result is not None:
                    state_to, weight = result
                    yield Transition(state_from, symbol, state_to, weight)

    def transition_weight(self, state_from: State, symbol: SymbolOrEpsilon, state_to: State) -> Weight | None:
        if symbol != ReservedSymbol.EPSILON:
            pair = self.transition(state_from, symbol)
            if pair is not None:
                actual_state_to, weight = pair
                if actual_state_to == state_to:
                    return weight
        return None

    def transition(self, state: State, symbol: Symbol) -> tuple[State, Weight] | None:
        r"""Given a state and a symbol, get the next state and transition
        weight that the transition function maps them to. If there is no
        transition, return ``None``. A missing transition is assumed to have a
        weight of zero."""
        raise NotImplementedError

    def _weights(self) -> Iterable[Weight]:
        for t in self.transitions():
            yield t.weight
        for _, w in self.accept_weights():
            yield w

class WeightedDeterministicFiniteAutomatonContainer[Weight](
    SemiringMixin[Weight],
    StartStateMixin,
    AcceptWeightsMixin[Weight],
    AutomatonContainer,
    WeightedDeterministicFiniteAutomaton[Weight]
):

    _transitions: dict[tuple[State, Symbol], tuple[State, Weight]]

    def __init__(self,
        *,
        num_states: int = 1,
        alphabet_size: int,
        start_state: State = State(0),
        semiring: Semiring[Weight]
    ) -> None:
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            start_state=start_state,
            semiring=semiring
        )
        self._transitions = {}

    def transitions(self) -> Iterable[Transition]:
        for (state_from, symbol), (state_to, weight) in self._transitions.items():
            yield Transition(state_from, symbol, state_to, weight)

    def transition(self, state: State, symbol: Symbol) -> tuple[State, Weight] | None:
        return self._transitions.get((state, symbol))

    def set_transition_weight(self,
        state_from: State,
        symbol: Symbol,
        state_to: State,
        weight: Weight
    ) -> None:
        if not (0 <= state_from < self._num_states):
            raise ValueError(f'invalid state_from {state_from}')
        if not (0 <= symbol < self._alphabet_size):
            raise ValueError(f'invalid symbol {symbol}')
        if not (0 <= state_to < self._num_states):
            raise ValueError(f'invalid state_to {state_to}')
        self._transitions[state_from, symbol] = (state_to, weight)
