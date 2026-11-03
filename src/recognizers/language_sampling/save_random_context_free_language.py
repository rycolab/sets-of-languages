import argparse
import json
import pathlib
import random

import torch

from rau.tools.torch.model_interface import parse_device

from recognizers.language_sampling.sample_cnf_cfg import sample_cnf_cfg

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=pathlib.Path, required=True)
    parser.add_argument('--random-seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device')
    args = parser.parse_args()

    with args.hyperparameters.open() as fin:
        data = json.load(fin)
    generator = random.Random(args.random_seed)
    language = sample_cnf_cfg(
        mean_num_variables=data['mean_num_variables'],
        mean_num_terminals=data['mean_num_terminals'],
        empty_rule_probability=data['empty_rule_probability'],
        mean_num_binary_rules_per_variable=data['mean_num_binary_rules_per_variable'],
        mean_num_terminal_rules_per_variable=data['mean_num_terminal_rules_per_variable'],
        mean_num_self_embedding_chains=data['mean_num_self_embedding_chains'],
        mean_self_embedding_chain_length=data['mean_self_embedding_chain_length'],
        max_density=data['max_density'],
        num_density_samples=data['num_density_samples'],
        density_length_range=tuple(data['density_length_range']),
        generator=generator,
        device=parse_device(args.device)
    )
    torch.save(dict(
        random_seed=args.random_seed,
        language_class='context-free',
        language=language
    ), args.output)

if __name__ == '__main__':
    main()
