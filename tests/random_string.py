import random

from recognizers.theory.symbol import Symbol

def random_string(alphabet_size: int, min_length: int, max_length: int, generator: random.Random) -> list[Symbol]:
    n = generator.randint(min_length, max_length)
    return [Symbol(generator.randrange(alphabet_size)) for _ in range(n)]
