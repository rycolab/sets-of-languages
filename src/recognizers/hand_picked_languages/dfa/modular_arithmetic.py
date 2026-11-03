import itertools

from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def modular_arithmetic_dfa() -> tuple[DeterministicFiniteAutomatonContainer, list[str]]:
    size = 5
    M = DeterministicFiniteAutomatonContainer(
        num_states=2 + 5 * size,
        alphabet_size=size + 4
    )
    start, *rest, accept = M.states()
    number, number_plus, number_minus, number_times, number_equals = itertools.batched(rest, size, strict=True)
    *number_symbol, plus, minus, times, equals = M.alphabet()
    for i in range(size):
        M.add_transition(start, number_symbol[i], number[i])
        M.add_transition(number[i], plus, number_plus[i])
        M.add_transition(number[i], minus, number_minus[i])
        M.add_transition(number[i], times, number_times[i])
        M.add_transition(number[i], equals, number_equals[i])
        for j in range(size):
            M.add_transition(number_plus[i], number_symbol[j], number[(i+j) % size])
            M.add_transition(number_minus[i], number_symbol[j], number[(i-j) % size])
            M.add_transition(number_times[i], number_symbol[j], number[(i*j) % size])
        M.add_transition(number_equals[i], number_symbol[i], accept)
    M.add_accept_state(accept)
    return M, [*(str(i) for i in range(size)), '+', '-', '×', '=']
