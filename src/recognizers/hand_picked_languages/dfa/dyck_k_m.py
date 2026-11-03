from recognizers.theory.symbol import Symbol
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def dyck_k_m_dfa(num_brackets: int, max_depth: int) -> tuple[DeterministicFiniteAutomatonContainer, list[str]]:
    M = DeterministicFiniteAutomatonContainer(
        alphabet_size=2 * num_brackets
    )
    prev_states = [M.start_state()]
    for depth in range(max_depth):
        next_states = []
        for i in range(num_brackets):
            for prev_state in prev_states:
                new_state = M.new_state()
                open_bracket = Symbol(2 * i)
                close_bracket = Symbol(open_bracket + 1)
                M.add_transition(prev_state, open_bracket, new_state)
                M.add_transition(new_state, close_bracket, prev_state)
                next_states.append(new_state)
        prev_states = next_states
    M.add_accept_state(M.start_state())
    return M, [f'{b}{i}' for i in range(num_brackets) for b in ['(', ')']]
