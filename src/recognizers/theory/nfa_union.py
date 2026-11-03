from .symbol import ReservedSymbol
from .automaton import State
from .nfa import NondeterministicFiniteAutomaton, NondeterministicFiniteAutomatonContainer

def nfa_union(
    A: NondeterministicFiniteAutomaton,
    B: NondeterministicFiniteAutomaton
) -> NondeterministicFiniteAutomatonContainer:
    A_num_states = A.num_states()
    B_num_states = B.num_states()
    new_start_state = State(A_num_states + B_num_states)
    M = NondeterministicFiniteAutomatonContainer(
        num_states=A_num_states + B_num_states + 1,
        alphabet_size=max(A.alphabet_size(), B.alphabet_size()),
        start_state=new_start_state
    )
    M.add_transition(new_start_state, ReservedSymbol.EPSILON, A.start_state())
    M.add_transition(new_start_state, ReservedSymbol.EPSILON, State(B.start_state() + A_num_states))
    for t in A.transitions():
        M.add_transition(t.state_from, t.symbol, t.state_to)
    for t in B.transitions():
        M.add_transition(State(t.state_from + A_num_states), t.symbol, State(t.state_to + A_num_states))
    for f in A.accept_states():
        M.add_accept_state(f)
    for f in B.accept_states():
        M.add_accept_state(State(f + A_num_states))
    return M
