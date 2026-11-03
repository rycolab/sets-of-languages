from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def repeat_01_dfa() -> tuple[DeterministicFiniteAutomatonContainer, None]:
    M = DeterministicFiniteAutomatonContainer(
        num_states=2,
        alphabet_size=2
    )
    q0, q1 = M.states()
    _0, _1 = M.alphabet()
    M.add_transition(q0, _0, q1)
    M.add_transition(q1, _1, q0)
    M.add_accept_state(q0)
    return M, None
