from typing import Any

from fsa import FSA as VieiraFSA, eps as VIEIRA_EPSILON

from .symbol import Symbol, ReservedSymbol
from .automaton import State
from .dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer
from .nfa import NondeterministicFiniteAutomaton

def nfa_to_vieira_fsa(M: NondeterministicFiniteAutomaton) -> VieiraFSA:
    vieira_fsa = VieiraFSA()
    vieira_fsa.nodes.update(M.states())
    vieira_fsa.syms.update(M.alphabet())
    vieira_fsa.add_start(M.start_state())
    for t in M.transitions():
        vieira_fsa.add(
            t.state_from,
            t.symbol if t.symbol != ReservedSymbol.EPSILON else VIEIRA_EPSILON,
            t.state_to
        )
    for f in M.accept_states():
        vieira_fsa.add_stop(f)
    return vieira_fsa

def vieira_fsa_to_dfa(vieira_fsa: VieiraFSA, alphabet_size: int) -> DeterministicFiniteAutomatonContainer:
    assert all(isinstance(x, int) and 0 <= x < alphabet_size for x in vieira_fsa.syms)
    sorted_states = sorted(vieira_fsa.nodes)
    state_to_int = { q : State(i) for i, q in enumerate(sorted_states) }
    q0, = vieira_fsa.start
    M = DeterministicFiniteAutomatonContainer(
        num_states=len(state_to_int),
        alphabet_size=alphabet_size,
        start_state=state_to_int[q0]
    )
    for q, a, r in sorted(vieira_fsa.arcs()):
        M.add_transition(state_to_int[q], Symbol(a), state_to_int[r])
    for f in sorted(vieira_fsa.stop):
        M.add_accept_state(state_to_int[f])
    return M

def vieira_fsa_remove_state(M: VieiraFSA, removed_state: Any) -> VieiraFSA:
    if removed_state in M.start:
        return M
    else:
        result = VieiraFSA()
        result.nodes.update(M.nodes - {removed_state})
        result.syms.update(M.syms)
        for q in M.start:
            if q != removed_state:
                result.add_start(q)
        for q, a, r in M.arcs():
            if q != removed_state and r != removed_state:
                result.add(q, a, r)
        for f in M.stop:
            if f != removed_state:
                result.add_stop(f)
        return result
