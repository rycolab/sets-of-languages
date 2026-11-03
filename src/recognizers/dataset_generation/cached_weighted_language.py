import dataclasses
import math
import random
from collections.abc import Callable

from recognizers.theory.symbol import Symbol
from recognizers.dataset_generation.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse,
    String,
    ValidNextSymbolList,
    EmptyLanguageError
)

@dataclasses.dataclass
class CacheEntry:
    label: bool
    edit_distance: int | None = None

class CachedWeightedLanguage(WeightedLanguage):

    _cache: dict[String, CacheEntry]
    _symbol_to_str: Callable[[Symbol], str]

    def __init__(self, alphabet: list[str] | None):
        super().__init__()
        self._cache = {}
        if alphabet is not None:
            self._symbol_to_str = lambda s: alphabet[s]
        else:
            self._symbol_to_str = str

    def symbol_to_str(self, symbol: Symbol) -> str:
        return self._symbol_to_str(symbol)

    def with_length_range(self,
        length_range: tuple[int, int]
    ) -> LengthRestrictedWeightedLanguage:
        return LengthRestrictedCachedWeightedLanguage(self, length_range)

    def sample(self,
        length: int,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, float | None, ValidNextSymbolList | None]:
        s, log_probability, next_symbols = self.uncached_sample(
            length,
            generator,
            include_log_probability,
            include_next_symbols
        )
        cache_entry = self._cache.get(s)
        if cache_entry is None:
            cache_entry = self._cache[s] = CacheEntry(label=True)
        return s, log_probability, next_symbols

    def is_negative(self,
        s: String,
        include_edit_distance: bool
    ) -> tuple[bool, int | None]:
        # First check whether the length of s is even valid. This can avoid
        # expensive checks on certain negative strings. Don't even create a
        # cache entry.
        if not include_edit_distance and not self.is_valid_length(len(s)):
            return True, None
        cache_entry = self._cache.get(s)
        if cache_entry is None:
            if include_edit_distance:
                # If we need edit distance, then we didn't check whether the
                # length is valid, and we should check that before checking the
                # string.
                label = self.is_valid_length(len(s)) and self.uncached_label(s)
            else:
                label = self.uncached_label(s)
            cache_entry = self._cache[s] = CacheEntry(label=label)
        if include_edit_distance and cache_entry.edit_distance is None:
            if cache_entry.label:
                cache_entry.edit_distance = 0
            else:
                cache_entry.edit_distance = self.uncached_edit_distance(s)
        return not cache_entry.label, cache_entry.edit_distance

    def uncached_sample(self,
        length: int,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, float | None, ValidNextSymbolList | None]:
        raise NotImplementedError

    def uncached_label(self, s: String) -> bool:
        raise NotImplementedError

    def uncached_edit_distance(self, s: String) -> int:
        raise NotImplementedError

    def max_length(self) -> int:
        raise NotImplementedError

    def valid_lengths(self, length_range: tuple[int, int]) -> list[int]:
        raise NotImplementedError

    def is_valid_length(self, length: int) -> bool:
        raise NotImplementedError

    def total_length_log_probability(self, length: int) -> float:
        raise NotImplementedError

    def supports_log_probability(self) -> bool:
        raise NotImplementedError

    def supports_next_symbols(self) -> bool:
        raise NotImplementedError

    def supports_edit_distance(self) -> bool:
        raise NotImplementedError

class LengthRestrictedCachedWeightedLanguage(LengthRestrictedWeightedLanguage):

    _parent: CachedWeightedLanguage
    _min_length: int
    _max_length: int
    _valid_lengths: list[int]
    _log_num_lengths: float

    def __init__(self, parent: CachedWeightedLanguage, length_range: tuple[int, int]):
        super().__init__()
        self._parent = parent
        self._min_length, self._max_length = length_range
        if self._max_length > parent.max_length():
            raise ValueError(
                f'this language is prepared for sampling strings up to length '
                f'{parent.max_length()}, but {self._max_length} is required'
            )
        # Figure out which lengths are possible within this length range.
        self._valid_lengths = parent.valid_lengths(length_range)
        if not self._valid_lengths:
            raise EmptyLanguageError(
                f'no lengths are valid within the length range {length_range}'
            )
        # Precompute the log probability of selecting a length.
        self._log_num_lengths = math.log(len(self._valid_lengths))

    def supports_log_probability(self) -> bool:
        return self._parent.supports_log_probability()

    def supports_next_symbols(self) -> bool:
        return self._parent.supports_next_symbols()

    def supports_edit_distance(self) -> bool:
        return self._parent.supports_edit_distance()

    def sample(self,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, Parse]:
        length = self.sample_length(generator)
        s, log_probability, next_symbols = self._parent.sample(
            length,
            generator,
            include_log_probability,
            include_next_symbols
        )
        if log_probability is not None:
            # Renormalize the probability according to the length selected.
            log_probability = (
                log_probability
                - self._log_num_lengths
                - self._parent.total_length_log_probability(length)
            )
        return s, Parse(log_probability, next_symbols)

    def sample_length(self, generator):
        return generator.choice(self._valid_lengths)

    def is_negative(self,
        s: String,
        include_edit_distance: bool
    ) -> tuple[bool, int | None]:
        if not (self._min_length <= len(s) <= self._max_length):
            raise ValueError
        return self._parent.is_negative(s, include_edit_distance)
