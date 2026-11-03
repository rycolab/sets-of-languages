import random

import torch

from recognizers.dataset_generation.weighted_language import (
    String,
    ValidNextSymbolList
)
from recognizers.dataset_generation.cached_weighted_language import (
    CachedWeightedLanguage,
    LengthRestrictedCachedWeightedLanguage
)
from .weight_pushing import WeightedDeterministicFiniteAutomatonSampler

class DeterministicFiniteAutomatonLanguage(CachedWeightedLanguage):

    _sampler: WeightedDeterministicFiniteAutomatonSampler
    _dtype: torch.dtype
    _device: torch.device

    def __init__(self,
        sampler: WeightedDeterministicFiniteAutomatonSampler,
        alphabet: list[str] | None,
        dtype: torch.dtype,
        device: torch.device
    ) -> None:
        super().__init__(alphabet)
        self._sampler = sampler
        self._dtype = dtype
        self._device = device

    def alphabet_size(self) -> int:
        return self._sampler.alphabet_size()

    def uncached_sample(self,
        length: int,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, float | None, ValidNextSymbolList | None]:
        return self._sampler.sample(
            length,
            generator,
            include_log_probability,
            include_next_symbols
        )

    def uncached_label(self, s: String) -> bool:
        return self._sampler.accepts(s)

    def max_length(self) -> int:
        return self._sampler.max_length()

    def valid_lengths(self, length_range: tuple[int, int]) -> list[int]:
        return self._sampler.valid_lengths(length_range)

    def is_valid_length(self, length: int) -> bool:
        return self._sampler.is_valid_length(length)

    def total_length_log_probability(self, length: int) -> float:
        return self._sampler.total_length_log_weight(length)

    def supports_log_probability(self) -> bool:
        return True

    def supports_next_symbols(self) -> bool:
        return True

    def supports_edit_distance(self) -> bool:
        return False
