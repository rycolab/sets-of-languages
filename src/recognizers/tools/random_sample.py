import argparse
import math

import numpy
import scipy.stats

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--int', action='store_true', default=False)
    parser.add_argument('min', type=float)
    parser.add_argument('max', type=float)
    args = parser.parse_args()

    generator = numpy.random.default_rng(args.seed)
    if args.log:
        result = scipy.stats.loguniform(args.min, args.max).rvs(random_state=generator)
        if args.int:
            result = round(result)
    elif args.int:
        result = generator.integers(math.ceil(args.min), math.floor(args.max)+1)
    else:
        result = generator.uniform(args.min, args.max)
    print(result)

if __name__ == '__main__':
    main()
