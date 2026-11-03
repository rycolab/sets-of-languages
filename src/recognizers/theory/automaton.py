from collections.abc import Iterable, Sequence
from typing import cast, NewType

from orderedsets import OrderedSet

from .symbol import Symbol

State = NewType('State', int)

class Automaton:

    def states(self) -> Sequence[State]:
        r"""Get the set of all states."""
        return cast(Sequence[State], range(self.num_states()))

    def num_states(self) -> int:
        r"""Get the number of states."""
        raise NotImplementedError

    def alphabet(self) -> Sequence[Symbol]:
        r"""Get the alphabet."""
        return cast(Sequence[Symbol], range(self.alphabet_size()))

    def alphabet_size(self) -> int:
        r"""Get the size of the alphabet."""
        raise NotImplementedError

class AutomatonContainer(Automaton):

    _num_states: int
    _alphabet_size: int

    def __init__(self,
        *,
        num_states: int=1,
        alphabet_size: int
    ):
        super().__init__()
        if not num_states >= 0:
            raise ValueError(f'num_states ({num_states}) must be at least 0')
        self._num_states = num_states
        if not alphabet_size >= 1:
            raise ValueError(f'alphabet_size ({alphabet_size}) must be at least 1')
        self._alphabet_size = alphabet_size

    def num_states(self) -> int:
        return self._num_states

    def new_state(self) -> State:
        r"""Add a new state and return it."""
        state = self._num_states
        self._num_states += 1
        return State(state)

    def alphabet_size(self) -> int:
        return self._alphabet_size

class StartStateMixin(AutomatonContainer):

    _start_state: State

    def __init__(self,
        *,
        start_state: State=State(0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if not (0 <= start_state < self._num_states):
            raise ValueError(f'invalid start state {start_state}')
        self._start_state = start_state

    def start_state(self) -> State:
        return self._start_state

class MultiStartStateMixin(AutomatonContainer):

    _start_states: OrderedSet[State]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._start_states = OrderedSet()

    def is_start_state(self, state: State) -> bool:
        return state in self._start_states

    def num_start_states(self) -> int:
        return len(self._start_states)

    def start_states(self) -> Iterable[State]:
        return self._start_states

    def add_start_state(self, state: State) -> None:
        if not (0 <= state < self._num_states):
            raise ValueError(f'invalid start state {state}')
        self._start_states.add(state)

class MultiAcceptStateMixin(AutomatonContainer):

    _accept_states: OrderedSet[State]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._accept_states = OrderedSet()

    def is_accept_state(self, state: State) -> bool:
        return state in self._accept_states

    def num_accept_states(self) -> int:
        return len(self._accept_states)

    def accept_states(self) -> Iterable[State]:
        return self._accept_states

    def add_accept_state(self, state: State) -> None:
        if not (0 <= state < self._num_states):
            raise ValueError(f'invalid accept state {state}')
        self._accept_states.add(state)
