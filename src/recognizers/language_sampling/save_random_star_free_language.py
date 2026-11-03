import argparse
import json
import pathlib
import random

import torch

from recognizers.language_sampling.sample_sfre import sample_sfre

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=pathlib.Path, required=True)
    parser.add_argument('--random-seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with args.hyperparameters.open() as fin:
        data = json.load(fin)
    generator = random.Random(args.random_seed)
    R, language = sample_sfre(
        mean_alphabet_size=data['mean_alphabet_size'],
        empty_set_weight=data['empty_set_weight'],
        empty_string_weight=data['empty_string_weight'],
        symbol_weight=data['symbol_weight'],
        concatenation_weight=data['concatenation_weight'],
        union_weight=data['union_weight'],
        complement_weight=data['complement_weight'],
        sharpness=data['sharpness'],
        max_density=data['max_density'],
        num_density_samples=data['num_density_samples'],
        density_length_range=tuple(data['density_length_range']),
        generator=generator
    )
    torch.save(dict(
        random_seed=args.random_seed,
        language_class='star-free',
        language=language,
        regular_expression=R
    ), args.output)

if __name__ == '__main__':
    main()
