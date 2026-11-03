import random

from recognizers.theory.sfre_to_min_dfa import sfre_to_min_dfa
from recognizers.theory.sfre import (
    StarFreeRegularExpression,
    EmptySetRegularExpression,
    EmptyStringRegularExpression,
    SymbolRegularExpression,
    ConcatenationRegularExpression,
    UnionRegularExpression,
    ComplementRegularExpression
)
from recognizers.theory.symbol import Symbol
from recognizers.theory.automaton import State
from recognizers.theory.dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer
from recognizers.theory.nfa_concatenation import nfa_concatenation
from recognizers.theory.nfa_union import nfa_union
from recognizers.theory.dfa_complement import dfa_complement
from recognizers.theory.determinize_nfa import determinize_nfa
from recognizers.theory.minimize_dfa import minimize_dfa
from recognizers.theory.dfas_are_isomorphic import dfas_are_isomorphic
from random_sfre import random_sfre

def is_minimal_and_equivalent(
    result: DeterministicFiniteAutomaton,
    expected: DeterministicFiniteAutomaton
) -> bool:
    min_expected = minimize_dfa(expected)
    return dfas_are_isomorphic(result, min_expected)

def test_empty_set() -> None:
    result = sfre_to_min_dfa(EmptySetRegularExpression(), 2)
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        2,
        0,
        [],
        []
    )
    assert is_minimal_and_equivalent(result, expected)

def test_empty_string() -> None:
    result = sfre_to_min_dfa(EmptyStringRegularExpression(), 2)
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        2,
        0,
        [],
        [0]
    )
    assert is_minimal_and_equivalent(result, expected)

def test_symbol() -> None:
    for i in range(2):
        result = sfre_to_min_dfa(SymbolRegularExpression(Symbol(i)), 2)
        expected = DeterministicFiniteAutomatonContainer.from_parts(
            2,
            2,
            0,
            [(0, i, 1)],
            [1]
        )
        assert is_minimal_and_equivalent(result, expected)

def test_concatenation() -> None:
    result = sfre_to_min_dfa(
        ConcatenationRegularExpression(
            SymbolRegularExpression(Symbol(0)),
            SymbolRegularExpression(Symbol(1))
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        3,
        2,
        0,
        [(0, 0, 1), (1, 1, 2)],
        [2]
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        ConcatenationRegularExpression(
            SymbolRegularExpression(Symbol(0)),
            EmptySetRegularExpression()
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        2,
        0,
        [],
        []
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        ConcatenationRegularExpression(
            EmptyStringRegularExpression(),
            SymbolRegularExpression(Symbol(0))
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1)],
        [1]
    )
    assert is_minimal_and_equivalent(result, expected)

def test_union() -> None:
    result = sfre_to_min_dfa(
        UnionRegularExpression(
            SymbolRegularExpression(Symbol(0)),
            SymbolRegularExpression(Symbol(1))
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1), (0, 1, 1)],
        [1]
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        UnionRegularExpression(
            SymbolRegularExpression(Symbol(0)),
            SymbolRegularExpression(Symbol(0))
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1)],
        [1]
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        UnionRegularExpression(
            SymbolRegularExpression(Symbol(0)),
            EmptySetRegularExpression()
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1)],
        [1]
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        UnionRegularExpression(
            SymbolRegularExpression(Symbol(0)),
            EmptyStringRegularExpression()
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1)],
        [0, 1]
    )
    assert is_minimal_and_equivalent(result, expected)

def test_complement() -> None:
    result = sfre_to_min_dfa(
        ComplementRegularExpression(
            SymbolRegularExpression(Symbol(0))
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        3,
        2,
        0,
        [(0, 1, 2), (0, 0, 1), (1, 0, 2), (1, 1, 2), (2, 0, 2), (2, 1, 2)],
        [0, 2]
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        ComplementRegularExpression(
            EmptySetRegularExpression()
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        1,
        2,
        0,
        [(0, 0, 0), (0, 1, 0)],
        [0]
    )
    assert is_minimal_and_equivalent(result, expected)
    result = sfre_to_min_dfa(
        ComplementRegularExpression(
            EmptyStringRegularExpression()
        ),
        2
    )
    expected = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        [1]
    )
    assert is_minimal_and_equivalent(result, expected)

def naive_sfre_to_min_dfa(
    R: StarFreeRegularExpression,
    alphabet_size: int
) -> DeterministicFiniteAutomatonContainer:
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
            return minimize_dfa(determinize_nfa(nfa_concatenation(
                naive_sfre_to_min_dfa(first, alphabet_size),
                naive_sfre_to_min_dfa(second, alphabet_size)
            )))
        case UnionRegularExpression(first, second):
            return minimize_dfa(determinize_nfa(nfa_union(
                naive_sfre_to_min_dfa(first, alphabet_size),
                naive_sfre_to_min_dfa(second, alphabet_size)
            )))
        case ComplementRegularExpression(argument):
            return minimize_dfa(dfa_complement(
                naive_sfre_to_min_dfa(argument, alphabet_size)
            ))
        case _:
            raise TypeError

def test_random_sfres() -> None:
    generator = random.Random(123)
    for _ in range(100):
        R, alphabet_size = random_sfre(
            mean_alphabet_size=5,
            empty_set_weight=1,
            empty_string_weight=1,
            symbol_weight=1,
            concatenation_weight=200,
            union_weight=200,
            complement_weight=200,
            sharpness=25,
            generator=generator
        )
        result = sfre_to_min_dfa(R, alphabet_size)
        expected = naive_sfre_to_min_dfa(R, alphabet_size)
        assert dfas_are_isomorphic(result, expected)
