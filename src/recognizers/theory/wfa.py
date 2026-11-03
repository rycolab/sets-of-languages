import dataclasses
from collections.abc import Iterable

import torch

from .symbol import Symbol, ReservedSymbol
from .automaton import Automaton, AutomatonContainer, State
from .nfa import Transition as NFATransition, SymbolOrEpsilon
from .semiring import Semiring

@dataclasses.dataclass(frozen=True)
class Transition[Weight](NFATransition):
    weight: Weight

class WeightedFiniteAutomaton[Weight](Automaton):
    r"""A (nondeterministic) weighted finite automaton (WFA).

    The WFA can have different start weights for states, nondeterministic
    transitions, and accept weights.

    The weights come from a :py:class:`Semiring`.
    """

    def semiring(self) -> Semiring[Weight]:
        r"""Get the semiring which the weights of this WFA come from."""
        raise NotImplementedError

    def start_weights(self) -> Iterable[tuple[State, Weight]]:
        r"""Get all start states and their weights. Missing states are assumed
        to have a weight of zero."""
        raise NotImplementedError

    def start_weight(self, state: State) -> Weight | None:
        r"""Get the start weight of a state. A weight of ``None`` should be
        interpreted as zero."""
        raise NotImplementedError

    def transitions(self) -> Iterable[Transition]:
        r"""Get all transitions. Missing transitions are assumed to have a
        weight of zero."""
        raise NotImplementedError

    def transition_weight(self, state_from: State, symbol: SymbolOrEpsilon, state_to: State) -> Weight | None:
        r"""Get the weight of a transition. A weight of ``None`` should be
        interpreted as zero."""
        raise NotImplementedError

    def accept_weights(self) -> Iterable[tuple[State, Weight]]:
        r"""Get all accept states and their weights. Missing states are assumed
        to have a weight of zero."""
        raise NotImplementedError

    def accept_weight(self, state: State) -> Weight | None:
        r"""Get the accept weight of a state. A weight of ``None`` should be
        interpreted as zero."""
        raise NotImplementedError

    def dtype_and_device(self) -> tuple[torch.dtype, torch.device]:
        semiring = self.semiring()
        for w in self._weights():
            return semiring.get_dtype(w), semiring.get_device(w)
        return torch.float32, torch.device('cpu')

    def _weights(self) -> Iterable[Weight]:
        for _, w in self.start_weights():
            yield w
        for t in self.transitions():
            yield t.weight
        for _, w in self.accept_weights():
            yield w

class SemiringMixin[Weight](AutomatonContainer):

    _semiring: Semiring[Weight]

    def __init__(self, *, semiring: Semiring, **kwargs) -> None:
        super().__init__(**kwargs)
        self._semiring = semiring

    def semiring(self) -> Semiring[Weight]:
        return self._semiring

class StartWeightsMixin[Weight](AutomatonContainer):

    _start_weights: dict[State, Weight]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._start_weights = {}

    def start_weights(self) -> Iterable[tuple[State, Weight]]:
        return self._start_weights.items()

    def start_weight(self, state: State) -> Weight | None:
        return self._start_weights.get(state)

    def set_start_weight(self, state: State, weight: Weight) -> None:
        if not 0 <= state < self._num_states:
            raise ValueError(f'invalid state {state}')
        self._start_weights[state] = weight

class AcceptWeightsMixin[Weight](AutomatonContainer):

    _accept_weights: dict[State, Weight]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._accept_weights = {}

    def accept_weights(self) -> Iterable[tuple[State, Weight]]:
        return self._accept_weights.items()

    def accept_weight(self, state: State) -> Weight | None:
        return self._accept_weights.get(state)

    def set_accept_weight(self, state: State, weight: Weight) -> None:
        if not 0 <= state < self._num_states:
            raise ValueError(f'invalid state {state}')
        self._accept_weights[state] = weight

class WeightedFiniteAutomatonContainer[Weight](
    SemiringMixin[Weight],
    StartWeightsMixin[Weight],
    AcceptWeightsMixin[Weight],
    AutomatonContainer,
    WeightedFiniteAutomaton[Weight]
):

    _transitions: dict[tuple[State, SymbolOrEpsilon, State], Weight]

    def __init__(self,
        *,
        num_states: int = 0,
        alphabet_size: int,
        semiring: Semiring
    ) -> None:
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            semiring=semiring
        )
        self._transitions = {}

    def transitions(self) -> Iterable[Transition]:
        for (state_from, symbol, state_to), weight in self._transitions.items():
            yield Transition(state_from, symbol, state_to, weight)

    def transition_weight(self, state_from: State, symbol: SymbolOrEpsilon, state_to: State) -> Weight | None:
        return self._transitions.get((state_from, symbol, state_to))

    def set_transition_weight(self,
        state_from: State,
        symbol: Symbol,
        state_to: State,
        weight: Weight
    ) -> None:
        if not (0 <= state_from < self._num_states):
            raise ValueError(f'invalid state_from {state_from}')
        if not (symbol == ReservedSymbol.EPSILON or 0 <= symbol < self._alphabet_size):
            raise ValueError(f'invalid symbol {symbol}')
        if not (0 <= state_to < self._num_states):
            raise ValueError(f'invalid state_to {state_to}')
        self._transitions[state_from, symbol, state_to] = weight
