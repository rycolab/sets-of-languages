import random

from recognizers.theory.is_podfa import is_podfa
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.language_sampling.sample_podfa import sample_podfa

def test_example_1() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(
        2,
        2,
        0,
        [
            (0, 0, 1),
            (1, 1, 0)
        ],
        [0]
    )
    assert not is_podfa(M)

def test_example_2() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(
        3,
        2,
        0,
        [
            (0, 0, 1),
            (0, 1, 2),
            (1, 1, 1)
        ],
        [0]
    )
    assert is_podfa(M)

def test_example_3() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(
        4,
        2,
        0,
        [
            (0, 0, 1),
            (1, 0, 2),
            (2, 1, 3),
            (3, 0, 1)
        ],
        [0]
    )

def test_random_podfas() -> None:
    for i in range(100):
        generator = random.Random(i)
        M = sample_podfa(
            mean_num_states=10,
            mean_alphabet_size=10,
            transition_probability=0.5,
            accept_probability=0.4,
            max_density=0.8,
            num_density_samples=50,
            density_length_range=(0, 500),
            generator=generator
        )
        assert is_podfa(M)
