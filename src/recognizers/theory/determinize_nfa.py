from collections.abc import Iterable

from orderedsets import OrderedSet, FrozenOrderedSet

from recognizers.tools.graph_traversal import graph_vertexes_ordered
from .symbol import Symbol, ReservedSymbol
from .automaton import State
from .nfa import MultiStartNondeterministicFiniteAutomaton
from .dfa import DeterministicFiniteAutomatonContainer

def determinize_nfa(M: MultiStartNondeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonContainer:
    r"""Convert an NFA to an equivalent partial DFA using DFS over the subset
    construction.

    All of the states in the output DFA will be accessible (reachable from the
    start state). If all the states in the input NFA are co-accessible (can lead
    to an accept state), then the output DFA will be trim (so a trim NFA will
    result in a trim DFA).
    """
    # We don't need to use sets as the values here because there can't be
    # duplicate destination states.
    epsilon_transitions: dict[State, list[State]] = {}
    scanning_transitions: dict[State, dict[Symbol, list[State]]] = {}
    for t in M.transitions():
        if t.symbol == ReservedSymbol.EPSILON:
            epsilon_transitions.setdefault(t.state_from, []).append(t.state_to)
        else:
            scanning_transitions.setdefault(t.state_from, {}).setdefault(t.symbol, []).append(t.state_to)
    def epsilon_reachable(subset: Iterable[State]) -> OrderedSet[State]:
        return graph_vertexes_ordered(subset, lambda q: epsilon_transitions.get(q, []))
    new_transitions = []
    def successor_func(subset_from: FrozenOrderedSet[State]) -> Iterable[FrozenOrderedSet[State]]:
        symbols = OrderedSet(
            a
            for q in subset_from
            for a in scanning_transitions.get(q, {}).keys()
        )
        for a in symbols:
            subset_to = FrozenOrderedSet(epsilon_reachable(OrderedSet(
                r
                for q in subset_from
                for r in scanning_transitions.get(q, {}).get(a, [])
            )))
            new_transitions.append((subset_from, a, subset_to))
            yield subset_to
    start_subset = FrozenOrderedSet(epsilon_reachable(M.start_states()))
    subsets = graph_vertexes_ordered([start_subset], successor_func)
    subset_to_new_state = { subset : State(i) for i, subset in enumerate(subsets) }
    result = DeterministicFiniteAutomatonContainer(
        num_states=len(subsets),
        alphabet_size=M.alphabet_size(),
        start_state=subset_to_new_state[start_subset]
    )
    for subset_from, symbol, subset_to in new_transitions:
        result.add_transition(subset_to_new_state[subset_from], symbol, subset_to_new_state[subset_to])
    for subset in subsets:
        if any(M.is_accept_state(q) for q in subset):
            result.add_accept_state(subset_to_new_state[subset])
    return result
