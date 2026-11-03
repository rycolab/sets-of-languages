import random
from collections.abc import Callable, Iterable
from typing import cast

from recognizers.theory.cfg import ContextFreeGrammar, Variable, Terminal, Rule
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammarContainer
from recognizers.theory.trim_cnf_cfg import trim_cnf_cfg
from recognizers.theory.cnf_cfg_is_trim import cnf_cfg_is_trim
from random_cnf_cfg import random_cnf_cfg

def make_cfg(
    num_variables: int,
    num_terminals: int,
    get_rules: Callable[[Iterable[Variable], Iterable[Terminal]], list[tuple[Variable, list[Terminal | Variable]]]]
) -> ChomskyNormalFormContextFreeGrammarContainer:
    G = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=num_variables,
        num_terminals=num_terminals
    )
    for left, right in get_rules(G.variables(), G.terminals()):
        match len(right):
            case 0:
                assert left == G.start_variable()
                G.add_empty_rule()
            case 1:
                G.add_terminal_rule(left, cast(Terminal, right[0]))
            case 2:
                G.add_binary_rule(left, cast(Variable, right[0]), cast(Variable, right[1]))
            case _:
                assert False
    return G

def cfg_to_result(G: ContextFreeGrammar) -> tuple[
    int,
    int,
    set[tuple[Variable, tuple[Terminal | Variable, ...]]],
    Variable
]:
    return (
        G.num_variables(),
        G.num_terminals(),
        {(r.left, r.right) for r in G.rules()},
        G.start_variable()
    )

def result(
    num_variables: int,
    num_terminals: int,
    get_rules: Callable[[Iterable[Variable], Iterable[Terminal]], list[tuple[Variable, list[Terminal | Variable]]]]
) -> tuple[
    int,
    int,
    set[tuple[Variable, tuple[Terminal | Variable, ...]]],
    Variable
]:
    G = make_cfg(num_variables, num_terminals, get_rules)
    G = trim_cnf_cfg(G)
    assert cnf_cfg_is_trim(G)
    return cfg_to_result(G)

def expected(
    num_variables: int,
    num_terminals: int,
    get_rules: Callable[[Iterable[Variable], Iterable[Terminal]], list[tuple[Variable, list[Terminal | Variable]]]]
) -> tuple[
    int,
    int,
    set[tuple[Variable, tuple[Terminal | Variable, ...]]],
    Variable
]:
    G = make_cfg(num_variables, num_terminals, get_rules)
    assert cnf_cfg_is_trim(G)
    return cfg_to_result(G)

def test_simple_already_trim() -> None:
    def rules(V, T):
        S, A1, A2 = V
        a, b = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [b])
        ]
    assert result(3, 2, rules) == expected(3, 2, rules)

def test_unreachable() -> None:
    def rules(V, T):
        S, A1, A2, A3, A4, A5, A6, A7 = V
        a, = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [A5, A7]),
            (A5, [a]),
            (A7, [a]),
            (A4, [A6, A3]),
            (A6, [a]),
            (A3, [A6, A4])
        ]
    def new_rules(V, T):
        S, A1, A2, A5, A7 = V
        a, = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [A5, A7]),
            (A5, [a]),
            (A7, [a])
        ]
    assert result(8, 1, rules) == expected(5, 1, new_rules)

def test_just_epsilon() -> None:
    def rules(V, T):
        S, = V
        return [
            (S, ())
        ]
    assert result(1, 0, rules) == expected(1, 0, rules)

def test_just_epsilon_extra_terminals() -> None:
    def rules(V, T):
        S, = V
        return [
            (S, ())
        ]
    assert result(1, 10, rules) == expected(1, 10, rules)

def test_no_rules() -> None:
    def rules(V, T):
        S, = V
        return []
    assert result(1, 0, rules) == expected(1, 0, rules)

def test_simple_has_empty_variables() -> None:
    def rules(V, T):
        S, A1, A2, A3, A4 = V
        a, = T
        return [
            (S, [A1, A2]),
            (S, [A2, A2]),
            (A1, [A3, A4]),
            (A2, [a])
        ]
    def new_rules(V, T):
        S, A2 = V
        a, = T
        return [
            (S, [A2, A2]),
            (A2, [a])
        ]
    assert result(5, 1, rules) == expected(2, 1, new_rules)

def test_has_unreachable_and_empty() -> None:
    def rules(V, T):
        S, A1, A5, A2, A6, A3, A7, A4 = V
        a, = T
        return [
            (S, [A1, A2]),
            (S, [A2, A2]),
            (A1, [A3, A4]),
            (A2, [a]),
            (A5, [A6, A7]),
            (A6, [a]),
            (A7, [a])
        ]
    def new_rules(V, T):
        S, A2 = V
        a, = T
        return [
            (S, [A2, A2]),
            (A2, [a])
        ]
    assert result(8, 1, rules) == expected(2, 1, new_rules)

def test_extra_variables() -> None:
    def rules(V, T):
        S, _, A1, _, _, A2, *_ = V
        a, = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [a])
        ]
    def new_rules(V, T):
        S, A1, A2 = V
        a, = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [a])
        ]
    assert result(10, 1, rules) == expected(3, 1, new_rules)

def test_extra_variables_and_terminals() -> None:
    def rules(V, T):
        S, _, A1, _, _, A2, *_ = V
        _, _, a, _, b, *_ = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [b])
        ]
    def new_rules(V, T):
        S, A1, A2 = V
        _, _, a, _, b, *_ = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [b])
        ]
    assert result(13, 17, rules) == expected(3, 17, new_rules)

def test_a_star() -> None:
    def rules(V, T):
        S, A, B = V
        a, = T
        return [
            (S, []),
            (S, [a]),
            (S, [A, B]),
            (A, [a]),
            (B, [A, B]),
            (B, [a])
        ]
    assert result(3, 1, rules) == expected(3, 1, rules)

def test_a_star_missing_rule() -> None:
    def rules(V, T):
        S, A, B = V
        a, = T
        return [
            (S, []),
            (S, [a]),
            (S, [A, B]),
            (A, [a]),
            (B, [A, B])
        ]
    def new_rules(V, T):
        S, = V
        a, = T
        return [
            (S, []),
            (S, [a])
        ]
    assert result(3, 1, rules) == expected(1, 1, new_rules)

def test_empty_2() -> None:
    def rules(V, T):
        S, A1, A2, A3, A4, A5, A6 = V
        a, = T
        return [
            (S, [A1, A2]),
            (S, [A2, A3]),
            (A1, [A3, A4]),
            (A3, [a]),
            (A2, [A5, A6]),
            (A5, [a]),
            (A6, [a])
        ]
    def new_rules(V, T):
        S, A2, A3, A5, A6 = V
        a, = T
        return [
            (S, [A2, A3]),
            (A3, [a]),
            (A2, [A5, A6]),
            (A5, [a]),
            (A6, [a])
        ]
    assert result(7, 1, rules) == expected(5, 1, new_rules)

def test_unreachable_and_empty_2() -> None:
    def rules(V, T):
        S, A1, A2, A3, A4, A5, A6 = V
        a, = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [a]),
            (A3, [A2, A4]),
            (A4, [a]),
            (S, [A5, A6]),
            (A6, [a])
        ]
    def new_rules(V, T):
        S, A1, A2 = V
        a, = T
        return [
            (S, [A1, A2]),
            (A1, [a]),
            (A2, [a])
        ]
    assert result(7, 1, rules) == expected(3, 1, new_rules)

def test_empty_and_unreachable_2() -> None:
    def rules(V, T):
        S, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 = V
        a, = T
        return [
            (S, [A1, A2]),
            (S, [A8, A9]),
            (A1, [A3, A4]),
            (A3, [a]),
            (A2, [A5, A6]),
            (A6, [a]),
            (A8, [a]),
            (A9, [a]),
            (A7, [a]),
            (A6, [A10, A11]),
            (A10, [a]),
            (A11, [a])
        ]
    def new_rules(V, T):
        S, A8, A9 = V
        a, = T
        return [
            (S, [A8, A9]),
            (A8, [a]),
            (A9, [a])
        ]
    assert result(12, 1, rules) == expected(3, 1, new_rules)

def test_random_cnf_cfgs() -> None:
    generator = random.Random(123)
    for i in range(100):
        G = random_cnf_cfg(
            mean_num_variables=10,
            mean_num_terminals=5,
            rule_probability=0.5,
            generator=generator
        )
        trim_G = trim_cnf_cfg(G)
        assert cnf_cfg_is_trim(trim_G)
        trim_trim_G = trim_cnf_cfg(trim_G)
        assert trim_trim_G == trim_G
