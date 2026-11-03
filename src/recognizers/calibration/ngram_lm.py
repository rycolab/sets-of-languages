import collections
import enum
import itertools
import math
from collections.abc import Hashable, Iterable, Sequence
from typing import cast

class SpecialSymbol(enum.Enum):
    BOS = enum.auto()
    EOS = enum.auto()

def get_ngrams[T](s: Iterable[T], n: int) -> Iterable[tuple[T | SpecialSymbol, ...]]:
    queue: collections.deque[T | SpecialSymbol] = collections.deque(itertools.repeat(SpecialSymbol.BOS, n-1), maxlen=n)
    for a in cast(Iterable[T | SpecialSymbol], itertools.chain(s, [SpecialSymbol.EOS])):
        queue.append(a)
        yield tuple(queue)

class NGramLanguageModel[T: Hashable]:

    _n: int
    _counts: collections.Counter[tuple[T | SpecialSymbol, ...]]
    _totals: collections.Counter[tuple[T | SpecialSymbol, ...]]
    _vocabulary_size: int

    def __init__(self, n: int, vocabulary_size: int, examples: Iterable[Iterable[T]]) -> None:
        super().__init__()
        self._n = n
        # Add 1 for EOS.
        self._vocabulary_size = vocabulary_size + 1
        self._counts = collections.Counter()
        self._totals = collections.Counter()
        vocabulary: set[T | SpecialSymbol] = set()
        for example in examples:
            for ngram in get_ngrams(example, n):
                self._counts[ngram] += 1
                self._totals[ngram[:-1]] += 1
                vocabulary.add(ngram[-1])
        if len(vocabulary) > self._vocabulary_size:
            raise ValueError(
                f'examples contain more symbol types than vocabulary_size '
                f'({vocabulary_size})'
            )

    def log_probability(self, s: Iterable[T], k: float) -> float:
        return sum(
            self.conditional_log_probability(x[:-1], x[-1], k)
            for x in get_ngrams(s, self._n)
        )

    def conditional_log_probability(self,
        context: tuple[T | SpecialSymbol, ...],
        symbol: T | SpecialSymbol,
        k: float
    ) -> float:
        return _safe_log(
            (self._counts[(*context, symbol)] + k) /
            (self._totals[context] + k * self._vocabulary_size)
        )

def _safe_log(x: float) -> float:
    if x == 0:
        return -math.inf
    else:
        return math.log(x)

class NGramClassifier[T]:

    def __init__(self, n: int, examples: Iterable[tuple[Sequence[T], bool]]):
        super().__init__()
        positive_examples: list[Iterable[T]] = []
        negative_examples: list[Iterable[T]] = []
        vocabulary: set[T] = set()
        for example, label in examples:
            (positive_examples if label else negative_examples).append(example)
            vocabulary.update(example)
        self.positive_lm = NGramLanguageModel(n, len(vocabulary), positive_examples)
        self.negative_lm = NGramLanguageModel(n, len(vocabulary), negative_examples)

    def classify(self, s: Iterable[T], k: float = 1) -> bool:
        return self.positive_lm.log_probability(s, k) >= self.negative_lm.log_probability(s, k)
