from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def all_strings_dfa() -> tuple[DeterministicFiniteAutomatonContainer, None]:
    M = DeterministicFiniteAutomatonContainer(
        num_states=1,
        alphabet_size=2
    )
    q, = M.states()
    _0, _1 = M.alphabet()
    M.add_transition(q, _0, q)
    M.add_transition(q, _1, q)
    M.add_accept_state(M.start_state())
    return M, None
