from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def cycle_navigation_dfa() -> tuple[DeterministicFiniteAutomatonContainer, list[str]]:
    size = 5
    M = DeterministicFiniteAutomatonContainer(
        num_states=size + 1,
        alphabet_size=3 + size
    )
    *cycle, accept = M.states()
    left, stay, right, *labels = M.alphabet()
    for i in range(size):
        M.add_transition(cycle[i], left, cycle[(i-1) % size])
        M.add_transition(cycle[i], stay, cycle[i])
        M.add_transition(cycle[i], right, cycle[(i+1) % size])
        M.add_transition(cycle[i], labels[i], accept)
    M.add_accept_state(accept)
    return M, ['<', '=', '>', *(str(i) for i in range(size))]
