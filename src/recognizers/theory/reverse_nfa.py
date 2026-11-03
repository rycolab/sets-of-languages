from .symbol import ReservedSymbol
from .automaton import State
from .nfa import NondeterministicFiniteAutomaton, NondeterministicFiniteAutomatonContainer
from .multi_start_nfa import MultiStartNondeterministicFiniteAutomaton, MultiStartNondeterministicFiniteAutomatonContainer

def reverse_nfa(M: MultiStartNondeterministicFiniteAutomaton) -> NondeterministicFiniteAutomatonContainer:
    new_start_state = State(M.num_states())
    result = NondeterministicFiniteAutomatonContainer(
        num_states=M.num_states() + 1,
        alphabet_size=M.alphabet_size(),
        start_state=new_start_state
    )
    for f in M.accept_states():
        result.add_transition(new_start_state, ReservedSymbol.EPSILON, f)
    for t in M.transitions():
        result.add_transition(t.state_to, t.symbol, t.state_from)
    for s in M.start_states():
        result.add_accept_state(s)
    return result

def reverse_multi_start_nfa(M: MultiStartNondeterministicFiniteAutomaton) -> MultiStartNondeterministicFiniteAutomaton:
    result = MultiStartNondeterministicFiniteAutomatonContainer(
        num_states=M.num_states(),
        alphabet_size=M.alphabet_size()
    )
    for f in M.accept_states():
        result.add_start_state(f)
    for t in M.transitions():
        result.add_transition(t.state_to, t.symbol, t.state_from)
    for s in M.start_states():
        result.add_accept_state(s)
    return result
