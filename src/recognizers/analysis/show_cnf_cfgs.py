import argparse
import pathlib
import random

from rau.tools.torch.model_interface import parse_device

from recognizers.language_sampling.sample_cnf_cfg import sample_cnf_cfg
from recognizers.theory.print_cfg import cfg_to_text

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--device')
    args = parser.parse_args()

    device = parse_device(args.device)
    for i in range(1, args.n + 1):
        print(f'random-context-free-{i}')
        generator = random.Random(args.seed + i)
        G = sample_cnf_cfg(
            mean_num_variables=10,
            mean_num_terminals=5,
            empty_rule_probability=0.25,
            mean_num_binary_rules_per_variable=5,
            mean_num_terminal_rules_per_variable=0.5,
            mean_num_self_embedding_chains=1,
            mean_self_embedding_chain_length=3,
            max_density=0.8,
            num_density_samples=5,
            density_length_range=(0, 500),
            generator=generator,
            device=device
        )
        print(cfg_to_text(G))
        print()

if __name__ == '__main__':
    main()
