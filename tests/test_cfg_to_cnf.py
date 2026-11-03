from orderedsets import OrderedSet

from recognizers.theory.grammar import Variable, Terminal
from recognizers.theory.cfg import ContextFreeGrammarContainer, Rule
from recognizers.theory.cfg_to_cnf import (
    cfg_remove_epsilon_rules,
    cfg_remove_unit_rules,
    cfg_replace_terminals,
    cfg_binarize,
    cfg_to_cnf
)
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammarContainer

def test_sipser_remove_epsilon_rules() -> None:
    G = ContextFreeGrammarContainer(num_variables=4, num_terminals=2)
    S0, S, A, B = G.variables()
    a, b = G.terminals()
    G.add_rules_from_parts([
        (S0, (S,)),
        (S, (A, S, A)),
        (S, (a, B)),
        (A, (B,)),
        (A, (S,)),
        (B, (b,)),
        (B, ())
    ])
    S_is_empty, new_rules = cfg_remove_epsilon_rules(S, G.rules())
    assert not S_is_empty
    assert set(new_rules) == {
        Rule(S0, (S,)),
        Rule(S, (A, S, A)),
        Rule(S, (a, B)),
        Rule(S, (a,)),
        Rule(S, (S, A)),
        Rule(S, (A, S)),
        Rule(S, (S,)),
        Rule(A, (B,)),
        Rule(A, (S,)),
        Rule(B, (b,))
    }

def test_sipser_remove_unit_rules() -> None:
    G = ContextFreeGrammarContainer(num_variables=4, num_terminals=2)
    S0, S, A, B = G.variables()
    a, b = G.terminals()
    rules = [
        Rule(S0, (S,)),
        Rule(S, (A, S, A)),
        Rule(S, (a, B)),
        Rule(S, (a,)),
        Rule(S, (S, A)),
        Rule(S, (A, S)),
        Rule(S, (S,)),
        Rule(A, (B,)),
        Rule(A, (S,)),
        Rule(B, (b,))
    ]
    new_rules = cfg_remove_unit_rules(G.num_terminals(), rules)
    assert new_rules == {
        S0 : {(A, S, A), (a, B), (a,), (S, A), (A, S)},
        S : {(A, S, A), (a, B), (a,), (S, A), (A, S)},
        A : {(b,), (A, S, A), (a, B), (a,), (S, A), (A, S)},
        B : {(b,)}
    }

def test_sipser_replace_terminals() -> None:
    G = ContextFreeGrammarContainer(num_variables=4, num_terminals=2)
    S0, S, A, B = G.variables()
    a, b = G.terminals()
    rules_by_left: dict[Variable, OrderedSet[tuple[Variable | Terminal, ...]]] = {
        S0 : OrderedSet([(A, S, A), (a, B), (a,), (S, A), (A, S)]),
        S : OrderedSet([(A, S, A), (a, B), (a,), (S, A), (A, S)]),
        A : OrderedSet([(b,), (A, S, A), (a, B), (a,), (S, A), (A, S)]),
        B : OrderedSet([(b,)])
    }
    num_variables, non_terminal_rules, terminal_rules = cfg_replace_terminals(
        G.num_terminals(),
        G.num_variables(),
        S0,
        rules_by_left
    )
    U_a = Variable(G.num_terminals() + 4)
    assert num_variables == 5
    assert set(non_terminal_rules) == {
        (S0, (A, S, A)),
        (S0, (U_a, B)),
        (S0, (S, A)),
        (S0, (A, S)),
        (S, (A, S, A)),
        (S, (U_a, B)),
        (S, (S, A)),
        (S, (A, S)),
        (A, (A, S, A)),
        (A, (U_a, B)),
        (A, (S, A)),
        (A, (A, S))
    }
    assert set(terminal_rules) == {
        (S0, a),
        (S, a),
        (A, b),
        (A, a),
        (B, b),
        (U_a, a)
    }

def test_sipser_binarize() -> None:
    G = ContextFreeGrammarContainer(num_variables=5, num_terminals=2)
    S0, S, A, B, U = G.variables()
    a, b = G.terminals()
    rules = [
        (S0, (A, S, A)),
        (S0, (U, B)),
        (S0, (S, A)),
        (S0, (A, S)),
        (S, (A, S, A)),
        (S, (U, B)),
        (S, (S, A)),
        (S, (A, S)),
        (A, (A, S, A)),
        (A, (U, B)),
        (A, (S, A)),
        (A, (A, S))
    ]
    num_variables, binary_rules = cfg_binarize(
        G.num_terminals(),
        G.num_variables(),
        S0,
        rules
    )
    assert num_variables == 6
    SA = Variable(G.num_terminals() + 5)
    assert set(binary_rules) == {
        (S0, A, SA),
        (S0, U, B),
        (S0, S, A),
        (S0, A, S),
        (S, A, SA),
        (S, U, B),
        (S, S, A),
        (S, A, S),
        (A, A, SA),
        (A, U, B),
        (A, S, A),
        (A, A, S),
        (SA, S, A)
    }

def test_sipser_full() -> None:
    G = ContextFreeGrammarContainer(num_variables=3, num_terminals=2)
    S, A, B = G.variables()
    a, b = G.terminals()
    G.add_rules_from_parts([
        (S, (A, S, A)),
        (S, (a, B)),
        (A, (B,)),
        (A, (S,)),
        (B, (b,)),
        (B, ())
    ])
    result = cfg_to_cnf(G)
    expected = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=6,
        num_terminals=2,
        start_variable=Variable(G.num_terminals() + G.num_variables())
    )
    S, A, B, S0, U, A1 = expected.variables()
    expected.add_rules_from_parts([
        (S0, A, A1),
        (S0, U, B),
        (S0, a),
        (S0, S, A),
        (S0, A, S),
        (S, A, A1),
        (S, U, B),
        (S, a),
        (S, S, A),
        (S, A, S),
        (A, b),
        (A, A, A1),
        (A, U, B),
        (A, a),
        (A, S, A),
        (A, A, S),
        (A1, S, A),
        (U, a),
        (B, b)
    ])
    assert result == expected
