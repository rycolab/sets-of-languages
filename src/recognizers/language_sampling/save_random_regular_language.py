import argparse
import json
import pathlib
import random

import torch

from recognizers.language_sampling.sample_dfa import sample_dfa

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=pathlib.Path, required=True)
    parser.add_argument('--random-seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with args.hyperparameters.open() as fin:
        data = json.load(fin)
    generator = random.Random(args.random_seed)
    language = sample_dfa(
        mean_num_states=data['mean_num_states'],
        mean_alphabet_size=data['mean_alphabet_size'],
        transition_probability=data['transition_probability'],
        accept_probability=data['accept_probability'],
        max_density=data['max_density'],
        num_density_samples=data['num_density_samples'],
        density_length_range=tuple(data['density_length_range']),
        generator=generator
    )
    torch.save(dict(
        random_seed=args.random_seed,
        language_class='regular',
        language=language
    ), args.output)

if __name__ == '__main__':
    main()
