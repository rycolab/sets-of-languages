import random
from collections.abc import Callable

from recognizers.dataset_generation.generate_datasets import generate_random_string
from recognizers.theory.symbol import Symbol

def has_max_density(
    is_member: Callable[[tuple[Symbol, ...]], bool],
    density: float,
    num_samples: int,
    length_range: tuple[int, int],
    alphabet_size: int,
    generator: random.Random
) -> bool:
    num_in_language = 0
    for sample_no in range(num_samples):
        if num_in_language / num_samples > density:
            # Even if all the remaining samples are negative, the density will
            # be too high.
            return False
        elif (num_in_language + num_samples - sample_no) / num_samples <= density:
            # Even if all the remaining samples are positive, the density will
            # be low enough.
            return True
        num_in_language += int(is_member(generate_random_string(length_range, alphabet_size, generator)))
    return num_in_language / num_samples <= density
