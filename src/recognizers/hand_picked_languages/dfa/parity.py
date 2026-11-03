from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def parity_dfa() -> tuple[DeterministicFiniteAutomatonContainer, None]:
    M = DeterministicFiniteAutomatonContainer(
        num_states=2,
        alphabet_size=2
    )
    even, odd = M.states()
    _0, _1 = M.alphabet()
    M.add_transition(even, _0, even)
    M.add_transition(even, _1, odd)
    M.add_transition(odd, _0, odd)
    M.add_transition(odd, _1, even)
    M.add_accept_state(odd)
    return M, None
