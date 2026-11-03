import argparse
import pathlib
import random

from recognizers.language_sampling.sample_podfa import sample_podfa
from recognizers.theory.draw_nfa import draw_nfa

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    for i in range(1, args.n + 1):
        generator = random.Random(args.seed + i)
        M = sample_podfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.6,
            max_density=0.8,
            num_density_samples=50,
            density_length_range=(0, 500),
            generator=generator
        )
        draw_nfa(M, args.output / f'random-podfa-{i}.png')

if __name__ == '__main__':
    main()
