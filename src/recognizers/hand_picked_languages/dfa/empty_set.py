from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer

def empty_set_dfa() -> tuple[DeterministicFiniteAutomatonContainer, None]:
    return DeterministicFiniteAutomatonContainer(alphabet_size=2), None
