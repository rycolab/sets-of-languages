import math
from collections.abc import Callable

import torch

def count_to_bins(
    count: int,
    log_weight: float,
    weight_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    r"""Convert a count and a log weight to a vector in the log binning
    semiring.
    """
    result = torch.full((weight_size,), -math.inf, dtype=dtype, device=device)
    if count < weight_size:
        result[count] = log_weight
    return result
