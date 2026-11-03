import collections

from .symbol import Symbol
from .automaton import State
from .dfa import DeterministicFiniteAutomaton

def dfas_are_isomorphic(A: DeterministicFiniteAutomaton, B: DeterministicFiniteAutomaton) -> bool:
    r"""Check whether two DFAs are isomorphic with respect to state numbering,
    assuming that all states in both DFAs are accessible from the start state.
    """
    if not (
        A.num_states() == B.num_states() and
        A.alphabet_size() == B.alphabet_size() and
        A.num_transitions() == B.num_transitions() and
        A.num_accept_states() == B.num_accept_states()
    ):
        return False
    # Run DFS starting from the start state, filling in the state mapping as
    # needed and checking that the state mapping is consistent at later states.
    A_transitions: dict[State, list[tuple[Symbol, State]]] = {}
    for t in A.transitions():
        A_transitions.setdefault(t.state_from, []).append((t.symbol, t.state_to))
    B_num_transitions: collections.Counter[State] = collections.Counter()
    for t in B.transitions():
        B_num_transitions[t.state_from] += 1
    state_mapping = { A.start_state() : B.start_state() }
    agenda = [A.start_state()]
    discovered = set(agenda)
    while agenda:
        A_state_from = agenda.pop()
        B_state_from = state_mapping[A_state_from]
        A_outgoing_transitions = A_transitions.get(A_state_from) or []
        if len(A_outgoing_transitions) != B_num_transitions[B_state_from]:
            return False
        for symbol, A_state_to in A_outgoing_transitions:
            B_state_to = B.transition(B_state_from, symbol)
            if B_state_to is None:
                return False
            if A_state_to in state_mapping:
                if state_mapping[A_state_to] != B_state_to:
                    return False
            else:
                state_mapping[A_state_to] = B_state_to
        for _, v in A_outgoing_transitions:
            if v not in discovered:
                discovered.add(v)
                agenda.append(v)
    if len(state_mapping) != A.num_states():
        raise ValueError(f'DFAs have non-accessible states')
    return all(A.is_accept_state(q) == B.is_accept_state(state_mapping[q]) for q in A.states())
