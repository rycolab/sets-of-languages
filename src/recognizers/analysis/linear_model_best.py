import argparse
import re
import pathlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=pathlib.Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    # Use a fixed ordering for categories.
    df['language_distribution'] = pd.Categorical(
        df['language_distribution'],
        categories=['podfa', 'star-free', 'regular', 'context-free'],
        ordered=True
    )
    df['architecture'] = pd.Categorical(
        df['architecture'],
        categories=['rnn', 'lstm', 'transformer'],
        ordered=True
    )
    # Use only the runs with maximum test accuracy.
    df = df.loc[df.groupby(['language', 'architecture'])['accuracy'].idxmax()]

    result = smf.mixedlm(
        formula=r'''
            accuracy ~
                C(language_distribution) * C(architecture) +
                sensitivity * C(architecture) +
                baseline_accuracy
        ''',
        data=df,
        groups=df['language'],
    ).fit()
    print(result.summary())
    table = pd.DataFrame({
        "Value": result.params,
        "Std.\\ Error": result.bse,
        # "z": result.tvalues,
        "$p$-value": result.pvalues,
    })
    name_map = {
        'C(language_distribution)[T.star-free]' : 'star-free',
        'C(language_distribution)[T.regular]' : 'regular',
        'C(language_distribution)[T.context-free]' : 'context-free',
        'C(architecture)[T.lstm]' : 'LSTM',
        'C(architecture)[T.transformer]' : 'transformer',
        'sensitivity' : 's',
        'baseline_accuracy' : 'b'
    }
    def mapper(s: str) -> str:
        match s:
            case 'Intercept':
                return r'$\beta_0$'
            case 'Group Var':
                return r'$u_{ci}$'
            case _:
                parts = s.split(':')
                subscript = r' \times '.join(r'\text{' f'{name_map.get(p, p)}' '}' for p in parts)
                return r'$\beta_{' f'{subscript}' r'}$'
    table = table.rename(index=mapper)
    print(table.to_latex(
        float_format="%.3f",
        escape=False
    ))

if __name__ == '__main__':
    main()
