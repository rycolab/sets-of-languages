from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def even_pairs_dfa() -> tuple[DeterministicFiniteAutomatonContainer, None]:
    M = DeterministicFiniteAutomatonContainer(
        num_states=5,
        alphabet_size=2
    )
    start, _0_even, _1_even, _1_odd, _0_odd = M.states()
    _0, _1 = M.alphabet()
    M.add_transition(start, _0, _0_even)
    M.add_transition(start, _1, _1_even)
    M.add_transition(_0_even, _0, _0_even)
    M.add_transition(_0_even, _1, _1_odd)
    M.add_transition(_1_odd, _0, _0_even)
    M.add_transition(_1_odd, _1, _1_odd)
    M.add_transition(_1_even, _0, _0_odd)
    M.add_transition(_1_even, _1, _1_even)
    M.add_transition(_0_odd, _0, _0_odd)
    M.add_transition(_0_odd, _1, _1_even)
    M.add_accept_state(start)
    M.add_accept_state(_0_even)
    M.add_accept_state(_1_even)
    return M, None
