from collections.abc import Iterable
from typing import cast

from .symbol import Symbol
from .automaton import State
from .dfa import DeterministicFiniteAutomaton, Transition

class CompleteDeterministicFiniteAutomaton(DeterministicFiniteAutomaton):

    def transitions(self) -> Iterable[Transition]:
        for state_from in self.states():
            for symbol in self.alphabet():
                state_to = self.transition(state_from, symbol)
                yield Transition(state_from, symbol, state_to)

    def transition(self, state: State, symbol: Symbol) -> State:
        r"""The resulting state is guaranteed not to be ``None``."""
        raise NotImplementedError

    def is_complete(self) -> bool:
        return True

class CompleteDeterministicFiniteAutomatonNoStateWrapper(CompleteDeterministicFiniteAutomaton):

    _dfa: DeterministicFiniteAutomaton

    def __init__(self, dfa: DeterministicFiniteAutomaton) -> None:
        super().__init__()
        self._dfa = dfa

    def num_states(self) -> int:
        return self._dfa.num_states()

    def alphabet_size(self) -> int:
        return self._dfa.alphabet_size()

    def start_state(self) -> State:
        return self._dfa.start_state()

    def num_transitions(self) -> int:
        return self._dfa.num_transitions()

    def is_accept_state(self, state: State) -> bool:
        return self._dfa.is_accept_state(state)

    def num_accept_states(self) -> int:
        return self._dfa.num_accept_states()

    def accept_states(self) -> Iterable[State]:
        return self._dfa.accept_states()

    def transition(self, state: State, symbol: Symbol) -> State:
        return cast(State, self._dfa.transition(state, symbol))

class CompleteDeterministicFiniteAutomatonTrapStateWrapper(CompleteDeterministicFiniteAutomaton):

    _dfa: DeterministicFiniteAutomaton

    def __init__(self, dfa: DeterministicFiniteAutomaton) -> None:
        super().__init__()
        self._dfa = dfa

    def num_states(self) -> int:
        return self._dfa.num_states() + 1

    def alphabet_size(self) -> int:
        return self._dfa.alphabet_size()

    def start_state(self) -> State:
        return self._dfa.start_state()

    def num_transitions(self) -> int:
        return self.num_states() * self.alphabet_size()

    def is_accept_state(self, state: State) -> bool:
        return state != self.trap_state() and self._dfa.is_accept_state(state)

    def trap_state(self) -> State:
        return State(self._dfa.num_states())

    def num_accept_states(self) -> int:
        return self._dfa.num_accept_states()

    def accept_states(self) -> Iterable[State]:
        return self._dfa.accept_states()

    def transition(self, state: State, symbol: Symbol) -> State:
        trap_state = self.trap_state()
        if state == trap_state:
            return trap_state
        else:
            result = self._dfa.transition(state, symbol)
            return result if result is not None else trap_state

def make_dfa_complete(M: DeterministicFiniteAutomaton) -> CompleteDeterministicFiniteAutomaton:
    if isinstance(M, CompleteDeterministicFiniteAutomaton):
        return M
    elif M.is_complete():
        return CompleteDeterministicFiniteAutomatonNoStateWrapper(M)
    else:
        return CompleteDeterministicFiniteAutomatonTrapStateWrapper(M)
