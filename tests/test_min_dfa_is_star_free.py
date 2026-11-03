import random

import torch

from recognizers.theory.automaton import State
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.theory.complete_dfa import make_dfa_complete
from recognizers.theory.sfre_to_min_dfa import sfre_to_min_dfa
from recognizers.theory.min_dfa_is_star_free import (
    compose_tuples,
    syntactic_monoid,
    min_dfa_is_star_free
)
from random_sfre import random_sfre

def test_compose_tuples() -> None:
    q0, q1, q2 = map(State, range(3))
    assert compose_tuples((q0, q1, q2), (q0, q1, q2)) == (q0, q1, q2)
    assert compose_tuples((q0, q1, q2), (q2, q0, q1)) == (q2, q0, q1)
    assert compose_tuples((q2, q0, q1), (q0, q1, q2)) == (q2, q0, q1)
    assert compose_tuples((q1, q0, q1), (q1, q2, q1)) == (q2, q1, q2)

def test_empty_set() -> None:
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=1,
        alphabet_size=2,
        start_state=0,
        transitions=[
            (0, 0, 0),
            (0, 1, 0)
        ],
        accept_states=[]
    ))
    assert syntactic_monoid(M) == { (0,) }
    assert min_dfa_is_star_free(M)

def test_sigma_star() -> None:
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=1,
        alphabet_size=2,
        start_state=0,
        transitions=[
            (0, 0, 0),
            (0, 1, 0)
        ],
        accept_states=[0]
    ))
    assert syntactic_monoid(M) == { (0,) }
    assert min_dfa_is_star_free(M)

def test_first() -> None:
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=3,
        alphabet_size=2,
        start_state=0,
        transitions=[
            (0, 0, 1),
            (0, 1, 2),
            (1, 0, 1),
            (1, 1, 1),
            (2, 0, 2),
            (2, 1, 2)
        ],
        accept_states=[1]
    ))
    assert syntactic_monoid(M) == {
        (0, 1, 2),
        (1, 1, 2),
        (2, 1, 2)
    }
    assert min_dfa_is_star_free(M)

def test_parity() -> None:
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=2,
        alphabet_size=2,
        start_state=0,
        transitions=[
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 1)
        ],
        accept_states=[1]
    ))
    assert syntactic_monoid(M) == {
        (0, 1),
        (1, 0)
    }
    assert not min_dfa_is_star_free(M)

def test_fork() -> None:
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=4,
        alphabet_size=2,
        start_state=0,
        transitions=[
            (0, 0, 1),
            (0, 1, 2),
            (1, 0, 1),
            (1, 1, 1),
            (2, 0, 3),
            (2, 1, 2),
            (3, 0, 2),
            (3, 1, 3)
        ],
        accept_states=[1, 3]
    ))
    assert syntactic_monoid(M) == {
        (0, 1, 2, 3),
        (1, 1, 3, 2),
        (2, 1, 2, 3),
        (1, 1, 2, 3),
        (3, 1, 3, 2)
    }
    assert not min_dfa_is_star_free(M)

def test_chain() -> None:
    n = 10
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=n+1,
        alphabet_size=2,
        start_state=0,
        transitions=[(i, 0, i+1) for i in range(n)],
        accept_states=[n]
    ))
    assert min_dfa_is_star_free(M)

def test_cycle() -> None:
    n = 10
    M = make_dfa_complete(DeterministicFiniteAutomatonContainer.from_parts(
        num_states=n,
        alphabet_size=2,
        start_state=0,
        transitions=[(i, 0, (i+1) % n) for i in range(n)],
        accept_states=[n-1]
    ))
    assert not min_dfa_is_star_free(M)

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
        M = sfre_to_min_dfa(R, alphabet_size)
        assert min_dfa_is_star_free(M)
