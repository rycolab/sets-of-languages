import argparse
import pathlib
import random

from recognizers.language_sampling.sample_sfre import sample_sfre
from recognizers.theory.sfre_dot_depth import sfre_dot_depth
from recognizers.theory.draw_nfa import draw_nfa

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    for i in range(1, args.n + 1):
        generator = random.Random(args.seed + i)
        base_case_weight = 1
        recursive_case_weight = 200
        R, M = sample_sfre(
            mean_alphabet_size=5,
            empty_set_weight=base_case_weight,
            empty_string_weight=base_case_weight,
            symbol_weight=base_case_weight,
            concatenation_weight=recursive_case_weight,
            union_weight=recursive_case_weight,
            complement_weight=recursive_case_weight,
            sharpness=15,
            max_density=0.8,
            num_density_samples=50,
            density_length_range=(0, 500),
            generator=generator
        )
        print(f'random-star-free-{i}')
        print(R)
        print(f'dot-depth: {sfre_dot_depth(R)}')
        draw_nfa(M, args.output / f'random-star-free-{i}.png')
        print()

if __name__ == '__main__':
    main()
