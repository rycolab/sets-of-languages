from typing import overload

from recognizers.tools.graph_traversal import graph_vertexes
from .symbol import Symbol, ReservedSymbol
from .automaton import State
from .nfa import (
    NondeterministicFiniteAutomaton,
    NondeterministicFiniteAutomatonContainer
)
from .dfa import (
    DeterministicFiniteAutomaton,
    DeterministicFiniteAutomatonContainer
)

@overload
def trim_nfa(M: DeterministicFiniteAutomaton, trim_alphabet: bool = False) -> DeterministicFiniteAutomaton:
    ...
@overload
def trim_nfa(M: NondeterministicFiniteAutomaton, trim_alphabet: bool = False) -> NondeterministicFiniteAutomaton:
    ...
def trim_nfa(M: NondeterministicFiniteAutomaton, trim_alphabet: bool = False) -> NondeterministicFiniteAutomaton:
    # Get the set of states reachable from the start state.
    transitions_by_state_from: dict[State, set[State]] = {}
    for t in M.transitions():
        transitions_by_state_from.setdefault(t.state_from, set()).add(t.state_to)
    new_states = graph_vertexes([M.start_state()], lambda q: transitions_by_state_from.get(q, []))
    transitions_by_state_to: dict[State, set[State]] = {}
    for t in M.transitions():
        if t.state_from in new_states and t.state_to in new_states:
            transitions_by_state_to.setdefault(t.state_to, set()).add(t.state_from)
    sorted_new_accept_states = [q for q in M.accept_states() if q in new_states]
    return _trim_nfa_non_coaccessible(
        M,
        trim_alphabet,
        sorted_new_accept_states,
        transitions_by_state_to
    )

@overload
def trim_nfa_non_coaccessible(M: DeterministicFiniteAutomaton, trim_alphabet: bool = False) -> DeterministicFiniteAutomaton:
    ...
@overload
def trim_nfa_non_coaccessible(M: NondeterministicFiniteAutomaton, trim_alphabet: bool = False) -> NondeterministicFiniteAutomaton:
    ...
def trim_nfa_non_coaccessible(M: NondeterministicFiniteAutomaton, trim_alphabet: bool = False) -> NondeterministicFiniteAutomaton:
    # Get the set of states that can reach an accept state.
    transitions_by_state_to: dict[State, set[State]] = {}
    for t in M.transitions():
        transitions_by_state_to.setdefault(t.state_to, set()).add(t.state_from)
    sorted_new_accept_states = list(M.accept_states())
    return _trim_nfa_non_coaccessible(
        M,
        trim_alphabet,
        sorted_new_accept_states,
        transitions_by_state_to
    )

def _trim_nfa_non_coaccessible(
    M: NondeterministicFiniteAutomaton,
    trim_alphabet: bool,
    sorted_new_accept_states: list[State],
    transitions_by_state_to: dict[State, set[State]]
) -> NondeterministicFiniteAutomaton:
    # Get the set of remaining states that can reach an accept state.
    new_states = graph_vertexes(sorted_new_accept_states, lambda q: transitions_by_state_to.get(q, []))
    new_transitions = [t for t in M.transitions() if t.state_from in new_states and t.state_to in new_states]
    if trim_alphabet:
        # Optionally remove unused symbols from the alphabet.
        new_alphabet = { t.symbol for t in new_transitions }
        sorted_new_alphabet = [a for a in M.alphabet() if a in new_alphabet]
        alphabet_map = { a : Symbol(i) for i, a in enumerate(sorted_new_alphabet) }
        def remap_symbol(a: Symbol) -> Symbol:
            return alphabet_map[a]
        new_alphabet_size = max(len(alphabet_map), 1)
    else:
        def remap_symbol(a: Symbol) -> Symbol:
            return a
        new_alphabet_size = M.alphabet_size()
    new_states.add(M.start_state())
    sorted_new_states = [q for q in M.states() if q in new_states]
    states_map = { q : State(i) for i, q in enumerate(sorted_new_states) }
    new_num_states = len(states_map)
    new_start_state = states_map[M.start_state()]
    if isinstance(M, DeterministicFiniteAutomaton):
        new_dfa = DeterministicFiniteAutomatonContainer(
            num_states=new_num_states,
            alphabet_size=new_alphabet_size,
            start_state=new_start_state
        )
        for t in new_transitions:
            assert t.symbol != ReservedSymbol.EPSILON
            new_dfa.add_transition(
                states_map[t.state_from],
                remap_symbol(t.symbol),
                states_map[t.state_to]
            )
        for f in sorted_new_accept_states:
            new_dfa.add_accept_state(states_map[f])
        return new_dfa
    else:
        new_nfa = NondeterministicFiniteAutomatonContainer(
            num_states=new_num_states,
            alphabet_size=new_alphabet_size,
            start_state=new_start_state
        )
        for t in new_transitions:
            new_nfa.add_transition(
                states_map[t.state_from],
                ReservedSymbol.EPSILON if t.symbol == ReservedSymbol.EPSILON else remap_symbol(t.symbol),
                states_map[t.state_to]
            )
        for f in sorted_new_accept_states:
            new_nfa.add_accept_state(states_map[f])
        return new_nfa
