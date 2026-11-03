from recognizers.tools.graph_traversal import graph_find
from .automaton import State
from .nfa import NondeterministicFiniteAutomaton

def nfa_is_empty(M: NondeterministicFiniteAutomaton) -> bool:
    transitions: dict[State, set[State]] = {}
    for t in M.transitions():
        transitions.setdefault(t.state_from, set()).add(t.state_to)
    return not graph_find([M.start_state()], lambda q: transitions.get(q, []), M.is_accept_state)
