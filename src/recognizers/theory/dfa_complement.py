from collections.abc import Iterable

from .symbol import Symbol
from .automaton import State
from .dfa import DeterministicFiniteAutomaton
from .complete_dfa import CompleteDeterministicFiniteAutomaton, make_dfa_complete

class DeterministicFiniteAutomatonComplementWrapper(CompleteDeterministicFiniteAutomaton):

    _dfa: CompleteDeterministicFiniteAutomaton

    def __init__(self, dfa: CompleteDeterministicFiniteAutomaton) -> None:
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
        # Crucially, flip the accept states of the wrapped DFA.
        return not self._dfa.is_accept_state(state)

    def num_accept_states(self) -> int:
        # Return the number of non-accept states in the wrapped DFA.
        return self._dfa.num_states() - self._dfa.num_accept_states()

    def accept_states(self) -> Iterable[State]:
        # Return the new accept states.
        return filter(self.is_accept_state, self.states())

    def transition(self, state: State, symbol: Symbol) -> State:
        return self._dfa.transition(state, symbol)

def dfa_complement(M: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonComplementWrapper:
    return DeterministicFiniteAutomatonComplementWrapper(make_dfa_complete(M))
