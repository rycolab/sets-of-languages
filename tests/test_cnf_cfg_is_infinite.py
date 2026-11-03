from collections.abc import Callable, Iterable

from recognizers.theory.cfg import Variable, Terminal
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammarContainer
from recognizers.theory.cnf_cfg_is_infinite import trim_cnf_cfg_is_infinite

def make_cfg(
    num_variables: int,
    num_terminals: int,
    get_rules: Callable[
        [Iterable[Variable], Iterable[Terminal]],
        tuple[
            bool,
            list[tuple[Variable, Variable, Variable]],
            list[tuple[Variable, Terminal]]
        ]
    ]
) -> ChomskyNormalFormContextFreeGrammarContainer:
    result = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=num_variables,
        num_terminals=num_terminals
    )
    has_empty_rule, binary_rules, terminal_rules = get_rules(
        result.variables(),
        result.terminals()
    )
    if has_empty_rule:
        result.add_empty_rule()
    for A, B, C in binary_rules:
        result.add_binary_rule(A, B, C)
    for A, a in terminal_rules:
        result.add_terminal_rule(A, a)
    return result

def test_empty() -> None:
    def get_rules(V, T):
        return (False, [], [])
    G = make_cfg(1, 2, get_rules)
    assert not trim_cnf_cfg_is_infinite(G)

def test_empty_string() -> None:
    def get_rules(V, T):
        return (True, [], [])
    G = make_cfg(1, 2, get_rules)
    assert not trim_cnf_cfg_is_infinite(G)

def test_simple_tree() -> None:
    def get_rules(V, T):
        S, A, B, C, D = V
        a, b, c = T
        return (
            False,
            [
                (S, A, B),
                (A, C, D)
            ],
            [
                (C, a),
                (D, b),
                (B, c)
            ]
        )
    G = make_cfg(5, 3, get_rules)
    assert not trim_cnf_cfg_is_infinite(G)

def test_simple_cycle() -> None:
    def get_rules(V, T):
        S, A, B, C, D, E, F = V
        a, b, c = T
        return (
            False,
            [
                (S, A, B),
                (A, C, D),
                (D, E, F),
                (F, A, B)
            ],
            [
                (C, a),
                (D, b),
                (B, c),
                (E, a)
            ]
        )
    G = make_cfg(7, 3, get_rules)
    assert trim_cnf_cfg_is_infinite(G)

def test_infinite_example() -> None:
    def get_rules(V, T):
        S, A, B = V
        a, b = T
        return (
            True,
            [
                (S, A, A),
                (A, A, B),
                (A, B, B),
            ],
            [
                (A, a),
                (B, b)
            ]
        )
    G = make_cfg(3, 2, get_rules)
    assert trim_cnf_cfg_is_infinite(G)

def test_finite_example() -> None:
    def get_rules(V, T):
        S, A, B = V
        a, b = T
        return (
            True,
            [
                (S, A, B),
            ],
            [
                (S, b),
                (A, a),
                (B, b)
            ]
        )
    G = make_cfg(3, 2, get_rules)
    assert not trim_cnf_cfg_is_infinite(G)
