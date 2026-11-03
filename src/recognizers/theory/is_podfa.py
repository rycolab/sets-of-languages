from recognizers.tools.graph_traversal import graph_has_cycle
from .automaton import State
from .dfa import DeterministicFiniteAutomaton

def is_podfa(M: DeterministicFiniteAutomaton) -> bool:
    edges: dict[State, set[State]] = {}
    for t in M.transitions():
        if t.state_from != t.state_to:
            edges.setdefault(t.state_from, set()).add(t.state_to)
    return not graph_has_cycle([M.start_state()], lambda q: edges.get(q, []))
