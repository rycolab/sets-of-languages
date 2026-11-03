import random

import numpy

def python_to_numpy_generator(generator: random.Random) -> numpy.random.Generator:
    return numpy.random.default_rng(generator.getrandbits(32))

def sample_from_negative_binomial(
    mean: float,
    min_value: int,
    generator: random.Random
) -> int:
    return python_to_numpy_generator(generator).negative_binomial(mean - min_value, 0.5) + min_value

def sample_from_bernoulli(
    success_probability: float,
    generator: random.Random
) -> bool:
    return generator.random() < success_probability
