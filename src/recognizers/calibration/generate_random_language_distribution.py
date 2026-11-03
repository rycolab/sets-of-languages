import argparse
import hashlib
import json
import numpy
import pathlib
from typing import Any

def string_to_seed(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode('ascii')).digest()[:4], 'big')

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--language-class',
        choices=['podfa', 'star-free', 'regular', 'context-free'],
        required=True
    )
    parser.add_argument('--distribution-no', type=int, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    seed = string_to_seed(f'{args.language_class} {args.distribution_no}')
    generator = numpy.random.default_rng(seed)
    mean_alphabet_size = 5
    data: dict[str, Any]
    match args.language_class:
        case 'podfa' | 'regular':
            data = dict(
                mean_num_states=generator.uniform(5, 15),
                mean_alphabet_size=mean_alphabet_size,
                transition_probability=generator.uniform(0.2, 0.8),
                accept_probability=generator.uniform(0.2, 0.8)
            )
        case 'star-free':
            data = dict(
                mean_alphabet_size=mean_alphabet_size,
                empty_set_weight=generator.uniform(1, 10),
                empty_string_weight=generator.uniform(1, 10),
                symbol_weight=generator.uniform(1, 10),
                concatenation_weight=generator.uniform(50, 1000),
                union_weight=generator.uniform(50, 1000),
                complement_weight=generator.uniform(50, 1000),
                sharpness=generator.uniform(1, 1000)
            )
        case 'context-free':
            data = dict(
                mean_num_variables=generator.uniform(5, 15),
                mean_num_terminals=mean_alphabet_size,
                empty_rule_probability=generator.uniform(0.2, 0.8),
                mean_num_binary_rules_per_variable=generator.uniform(3, 7),
                mean_num_terminal_rules_per_variable=generator.uniform(0.5, 1.5),
                mean_num_self_embedding_chains=generator.uniform(0.5, 2),
                mean_self_embedding_chain_length=generator.uniform(2, 5)
            )
        case _:
            raise ValueError
    data.update(dict(
        max_density=0.8,
        num_density_samples=50,
        density_length_range=(0, 500)
    ))
    print(f'writing {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as fout:
        json.dump(data, fout, indent=2)

if __name__ == '__main__':
    main()
