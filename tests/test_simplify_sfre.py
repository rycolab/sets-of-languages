import random

from recognizers.theory.simplify_sfre import simplify_sfre
from recognizers.theory.sfre_to_min_dfa import sfre_to_min_dfa
from recognizers.theory.dfas_are_isomorphic import dfas_are_isomorphic
from random_sfre import random_sfre

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
        result = sfre_to_min_dfa(simplify_sfre(R), alphabet_size)
        expected = sfre_to_min_dfa(R, alphabet_size)
        assert dfas_are_isomorphic(result, expected)
