from .sfre import (
    StarFreeRegularExpression,
    EmptySetRegularExpression,
    EmptyStringRegularExpression,
    SymbolRegularExpression,
    ConcatenationRegularExpression,
    UnionRegularExpression,
    ComplementRegularExpression
)
from .symbol import Symbol
from .automaton import State
from .dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer
from .minimize_dfa import minimize_trim_nfa_brzozowski, minimize_trim_dfa_hopcroft
from .nfa_concatenation import nfa_concatenation
from .dfa_product_construction import dfa_union
from .dfa_complement import dfa_complement
from .trim_nfa import trim_nfa_non_coaccessible

def sfre_to_min_dfa(
    R: StarFreeRegularExpression,
    alphabet_size: int
) -> DeterministicFiniteAutomaton:
    match R:
        case EmptySetRegularExpression():
            return DeterministicFiniteAutomatonContainer(
                num_states=1,
                alphabet_size=alphabet_size
            )
        case EmptyStringRegularExpression():
            M = DeterministicFiniteAutomatonContainer(
                num_states=1,
                alphabet_size=alphabet_size
            )
            M.add_accept_state(State(0))
            return M
        case SymbolRegularExpression(symbol):
            M = DeterministicFiniteAutomatonContainer(
                num_states=2,
                alphabet_size=alphabet_size
            )
            M.add_transition(State(0), symbol, State(1))
            M.add_accept_state(State(1))
            return M
        case ConcatenationRegularExpression(first, second):
            return minimize_trim_nfa_brzozowski(nfa_concatenation(
                sfre_to_min_dfa(first, alphabet_size),
                sfre_to_min_dfa(second, alphabet_size)
            ))
        case UnionRegularExpression(first, second):
            return minimize_trim_dfa_hopcroft(dfa_union(
                sfre_to_min_dfa(first, alphabet_size),
                sfre_to_min_dfa(second, alphabet_size)
            ))
        case ComplementRegularExpression(argument):
            argument_dfa = sfre_to_min_dfa(argument, alphabet_size)
            # In order to keep the DFA minimal, treat the empty set as a special
            # case.
            if argument_dfa.num_accept_states() == 0:
                M = DeterministicFiniteAutomatonContainer(
                    num_states=1,
                    alphabet_size=alphabet_size
                )
                for a in range(alphabet_size):
                    M.add_transition(State(0), Symbol(a), State(0))
                M.add_accept_state(State(0))
                return M
            else:
                return trim_nfa_non_coaccessible(dfa_complement(argument_dfa))
        case _:
            raise TypeError
