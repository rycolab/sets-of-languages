import argparse
import pathlib

import torch

from recognizers.theory.dfa import DeterministicFiniteAutomaton
from recognizers.theory.draw_nfa import draw_nfa
from recognizers.theory.sfre import StarFreeRegularExpression
from recognizers.theory.sfre_dot_depth import sfre_dot_depth
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammar
from recognizers.theory.print_cfg import cfg_to_text

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('language_directory', type=pathlib.Path)
    args = parser.parse_args()

    data = torch.load(args.language_directory / 'language.pt', weights_only=False)
    language = data['language']
    match language:
        case DeterministicFiniteAutomaton():
            draw_nfa(language, args.language_directory / 'dfa.png', alphabet=data.get('alphabet'))
            R = data.get('regular_expression')
            if isinstance(R, StarFreeRegularExpression):
                print(R)
                print(f'dot-depth: {sfre_dot_depth(R)}')
        case ChomskyNormalFormContextFreeGrammar():
            print(cfg_to_text(language))
        case _:
            raise ValueError

if __name__ == '__main__':
    main()
