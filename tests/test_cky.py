from collections.abc import Sequence
import random

import torch

from recognizers.theory.grammar import Terminal
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammarContainer
from recognizers.theory.cnf_cfg_generates import (
    ChomskyNormalFormContextFreeGrammarMembershipTester,
    PyTorchChomskyNormalFormContextFreeGrammarMembershipTester
)
from random_cnf_cfg import random_cnf_cfg

def random_string(
    min_length: int,
    max_length: int,
    alphabet: Sequence[Terminal],
    generator: random.Random
) -> list[Terminal]:
    length = generator.randint(min_length, max_length)
    return generator.choices(alphabet, k=length)

def test_marked_reversal() -> None:
    G = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=6,
        num_terminals=3
    )
    S, T, TA, TB, A, B = G.variables()
    a, b, c = G.terminals()
    G.add_rules_from_parts([
        (S, A, TA),
        (S, B, TB),
        (S, c),
        (T, A, TA),
        (T, B, TB),
        (T, c),
        (TA, T, A),
        (TB, T, B),
        (A, a),
        (B, b)
    ])
    I = PyTorchChomskyNormalFormContextFreeGrammarMembershipTester(G, torch.device('cpu'))
    assert not I.generates([])
    assert I.generates([c])
    assert not I.generates([a])
    assert not I.generates([b])
    assert I.generates([a, c, a])
    assert I.generates([b, c, b])
    assert I.generates([a, b, a, a, b, c, b, a, a, b, a])
    assert not I.generates([a, b, a, a, b, c, b, a, b, b, a])

    def is_in_language(s: list[Terminal]) -> bool:
        n, r = divmod(len(s), 2)
        return (
            r == 1 and
            s[n] == c and
            all(si in (a, b) for si in s[:n]) and
            s[:n] == list(reversed(s[n+1:]))
        )

    generator = random.Random(123)
    for _ in range(100):
        w = random_string(2, 10, [a, b], generator)
        assert I.generates([*w, c, *reversed(w)])
    for _ in range(100):
        s = random_string(2, 20, [a, b, c], generator)
        assert I.generates(s) == is_in_language(s)

def test_unmarked_reversal() -> None:
    G = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=6,
        num_terminals=2
    )
    S, T, TA, TB, A, B = G.variables()
    a, b = G.terminals()
    G.add_rules_from_parts([
        (S,),
        (S, A, TA),
        (S, B, TB),
        (S, A, A),
        (S, B, B),
        (T, A, TA),
        (T, B, TB),
        (T, A, A),
        (T, B, B),
        (TA, T, A),
        (TB, T, B),
        (A, a),
        (B, b)
    ])
    I = PyTorchChomskyNormalFormContextFreeGrammarMembershipTester(G, torch.device('cpu'))
    assert I.generates([])
    assert not I.generates([a])
    assert not I.generates([b])
    assert I.generates([a, a])
    assert I.generates([b, b])
    assert I.generates([a, b, a, a, b, b, a, a, b, a])
    assert not I.generates([a, b, a, a, b, b, a, b, b, a])

    def is_in_language(w: list[Terminal]) -> bool:
        n, r = divmod(len(w), 2)
        return r == 0 and w[:n] == list(reversed(w[n:]))

    generator = random.Random(123)
    for _ in range(100):
        w = random_string(2, 10, [a, b], generator)
        assert I.generates([*w, *reversed(w)])
    for _ in range(100):
        s = random_string(2, 20, [a, b], generator)
        assert I.generates(s) == is_in_language(s)

def test_equal_number() -> None:
    G = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=6,
        num_terminals=2
    )
    S, T, TA, TB, A, B = G.variables()
    a, b = G.terminals()
    G.add_rules_from_parts([
        (S,),
        (S, A, TB),
        (S, B, TA),
        (S, T, T),
        (S, A, B),
        (S, B, A),
        (T, A, TB),
        (T, B, TA),
        (T, T, T),
        (T, A, B),
        (T, B, A),
        (TA, T, A),
        (TB, T, B),
        (A, a),
        (B, b)
    ])
    I = PyTorchChomskyNormalFormContextFreeGrammarMembershipTester(G, torch.device('cpu'))
    assert I.generates([])
    assert not I.generates([a])
    assert not I.generates([b])
    assert I.generates([a, b])
    assert I.generates([b, a])
    assert not I.generates([a, a])
    assert not I.generates([b, b])
    assert I.generates([a, b, b, a, b, a])
    assert not I.generates([a, b, b, a, b, b, a])

    def is_in_language(w: list[Terminal]) -> bool:
        counts = [0, 0]
        for wi in w:
            counts[wi] += 1
        return counts[0] == counts[1]

    generator = random.Random(123)
    for _ in range(100):
        n = generator.randint(1, 10)
        s = [*(a for _ in range(n)), *(b for _ in range(n))]
        generator.shuffle(s)
        assert I.generates(s)
    for _ in range(100):
        s = random_string(2, 20, [a, b], generator)
        assert I.generates(s) == is_in_language(s)

def test_pytorch_matches_python_on_random_cnf_cfgs() -> None:
    generator = random.Random(123)
    for _ in range(50):
        G = random_cnf_cfg(
            mean_num_variables=10,
            mean_num_terminals=5,
            rule_probability=0.5,
            generator=generator
        )
        python_tester = ChomskyNormalFormContextFreeGrammarMembershipTester(G)
        pytorch_tester = PyTorchChomskyNormalFormContextFreeGrammarMembershipTester(G, torch.device('cpu'))
        terminals = G.terminals()
        for _ in range(10):
            s = random_string(0, 30, terminals, generator)
            assert pytorch_tester.generates(s) == python_tester.generates(s)
