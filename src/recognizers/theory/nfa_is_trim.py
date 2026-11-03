from recognizers.tools.graph_traversal import graph_vertexes
from .automaton import State
from .nfa import NondeterministicFiniteAutomaton

def nfa_is_trim(M: NondeterministicFiniteAutomaton) -> bool:
    # If the NFA consists of a single non-accept state with no transitions,
    # consider it trim, since this is the only way to represent an NFA that
    # accepts no strings.
    # Otherwise, check that all states are reachable from the start state and
    # that all states can lead to an accept state.
    return nfa_is_empty_and_trim(M) or (nfa_is_accessible(M) and nfa_is_coaccessible(M))

def nfa_is_empty_and_trim(M: NondeterministicFiniteAutomaton) -> bool:
    return (
        M.num_states() == 1 and
        not M.is_accept_state(State(0))
        and not any(True for _ in M.transitions())
    )

def nfa_is_accessible(M: NondeterministicFiniteAutomaton) -> bool:
    # Check that all states are reachable from the start state.
    transitions_by_state_from: dict[State, set[State]] = {}
    for t in M.transitions():
        transitions_by_state_from.setdefault(t.state_from, set()).add(t.state_to)
    reachable_states = graph_vertexes([M.start_state()], lambda q: transitions_by_state_from.get(q, []))
    return len(reachable_states) == M.num_states()

def nfa_is_coaccessible(M: NondeterministicFiniteAutomaton) -> bool:
    # Check that all states can lead to an accept state.
    transitions_by_state_to: dict[State, set[State]] = {}
    for t in M.transitions():
        transitions_by_state_to.setdefault(t.state_to, set()).add(t.state_from)
    reachable_states = graph_vertexes(M.accept_states(), lambda q: transitions_by_state_to.get(q, []))
    return len(reachable_states) == M.num_states()
