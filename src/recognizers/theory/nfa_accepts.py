from collections.abc import Iterable

from recognizers.tools.graph_traversal import graph_vertexes

from .symbol import Symbol, ReservedSymbol
from .automaton import State
from .nfa import NondeterministicFiniteAutomaton

def nfa_accepts(M: NondeterministicFiniteAutomaton, w: Iterable[Symbol]) -> bool:
    epsilon_transitions: dict[State, list[State]] = {}
    scanning_transitions: dict[tuple[State, Symbol], list[State]] = {}
    for t in M.transitions():
        if t.symbol == ReservedSymbol.EPSILON:
            epsilon_transitions.setdefault(t.state_from, []).append(t.state_to)
        else:
            scanning_transitions.setdefault((t.state_from, t.symbol), []).append(t.state_to)
    def epsilon_reachable(subset: Iterable[State]) -> set[State]:
        return graph_vertexes(subset, lambda q: epsilon_transitions.get(q, []))
    subset = epsilon_reachable(M.start_states())
    for wi in w:
        subset = epsilon_reachable({
            r
            for q in subset
            for r in scanning_transitions.get((q, wi), [])
        })
        if not subset:
            return False
    return any(M.is_accept_state(q) for q in subset)
