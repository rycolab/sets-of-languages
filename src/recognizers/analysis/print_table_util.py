import dataclasses
import sys
from collections.abc import Callable
from typing import Any

import humanfriendly

@dataclasses.dataclass
class Row:
    label: str
    value: Any
    runs: int
    target_runs: int
    messages: list[str]
    bold_min: bool = False
    bold_max: bool = False

class Format:

    def __call__(self, x: Any, bold: bool) -> str:
        raise NotImplementedError

    def get_comparable_value(self, x: Any) -> Any:
        return x

@dataclasses.dataclass
class Column:
    heading: str
    specifier: str
    key: str
    format: Format
    bold_min: bool = False
    bold_max: bool = False

def wrap_bold(x: str, bold: bool) -> str:
    if bold:
        return f'\\bfseries {x}'
    else:
        return x

class format_text(Format):

    def __call__(self, x, bold):
        return wrap_bold(str(x), bold)

@dataclasses.dataclass
class format_float(Format):

    places: int = 2

    def __call__(self, x: float | None, bold: bool) -> str:
        if x is not None:
            if isinstance(x, float):
                return wrap_bold(f'{x:0.{self.places}f}', bold)
            else:
                raise TypeError
        else:
            return ''

class format_int(Format):

    def __call__(self, x: int | None, bold: bool) -> str:
        if x is not None:
            if isinstance(x, int):
                return wrap_bold(humanfriendly.format_number(x), bold)
            else:
                raise TypeError
        else:
            return ''

@dataclasses.dataclass
class format_mean_and_variance(Format):

    places: tuple[int, int]

    def __call__(self, x: tuple[float, float] | None, bold: bool) -> str:
        if x is not None:
            if len(x) == 2 and all(isinstance(xi, float) for xi in x):
                mean, std = x
                mean_str = f'{mean:0.{self.places[0]}f}'
                std_str = f'{std:.{self.places[1]}f}'.lstrip('0')
                macro = '\\meanAndVarBold' if bold else '\\meanAndVar'
                return f'{macro}{{{mean_str}}}{{{std_str}}}'
            else:
                raise TypeError
        else:
            return '&'

    def get_comparable_value(self, x: Any) -> Any:
        m, v = x
        return m

class Cache:

    _cache: dict[str, Any]
    _callbacks: dict[str, Callable[[Cache], Any]]

    def __init__(self, callbacks: dict[str, Callable[[Cache], Any]] | None = None) -> None:
        super().__init__()
        self._cache = {}
        self._callbacks = {}
        if callbacks is not None:
            self.set_callbacks(callbacks)

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def set_callback(self, key: str, func: Callable[[Cache], Any]) -> None:
        self._callbacks[key] = func

    def set_callbacks(self, callbacks: dict[str, Callable[[Cache], Any]]) -> None:
        self._callbacks.update(callbacks)

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        elif key in self._callbacks:
            result = self[key] = self._callbacks[key](self)
            return result
        else:
            raise KeyError(f'unable to get item in cache for key {key!r}')

    def clear(self) -> None:
        self._cache = {}

def num_digits_before(x: float, digits_after: int) -> int:
    return len(f'{abs(x):.{digits_after}f}'.split('.', 1)[0])

def print_table(
    *,
    header: str | None = None,
    rows: list[Row],
    columns: list[Column],
    callbacks: dict[str, Callable[[Cache], Any]]
) -> None:

    # If any runs are missing, show the number of runs per row in an extra
    # column.
    show_runs = any(row.runs != row.target_runs for row in rows)
    if show_runs:
        columns.append(Column('Runs', 'c', 'runs', format_int()))

    # Instantiate a separate cache for each row.
    caches = []
    for row in rows:
        cache = Cache(callbacks)
        cache['label'] = row.label
        cache['value'] = row.value
        cache['runs'] = row.runs
        caches.append(cache)

    # Do a pass over the data to figure out min and max values to bold.
    min_values = {}
    max_values = {}
    for c in columns:
        if c.bold_min or c.bold_max:
            all_values = (cache[c.key] for cache in caches)
            values = [c.format.get_comparable_value(x) for x in all_values if x is not None]
            if values:
                if c.bold_min:
                    min_values[c.key] = min(values)
                if c.bold_max:
                    max_values[c.key] = max(values)

    # Do a pass over the data to figure out how to format S columns.
    # For S columns, we need to update .specifier to include the correct number
    # of digits.
    # For S columns with variance, we need to update .specifier and .heading to
    # include two columns.
    for c in columns:
        if c.specifier == 'S':
            # Figure out the number of digits after the decimal.
            match c.format:
                case format_float(places=digits_after):
                    pass
                case format_mean_and_variance(places=(digits_after, _)):
                    pass
                case _:
                    raise ValueError
            # Figure out the max number of digits before the decimal.
            all_values = (cache[c.key] for cache in caches)
            values = [c.format.get_comparable_value(x) for x in all_values if x is not None]
            positive_digits_before = max((num_digits_before(x, digits_after) for x in values if x >= 0), default=1)
            negative_digits_before = max((num_digits_before(x, digits_after) for x in values if x < 0), default=1)
            has_negative = any(x < 0 for x in values)
            # If there's a negative sign, coalesce its space with one of the
            # digits of the positive numbers.
            if has_negative:
                positive_digits_before -= 1
            digits_before = max(positive_digits_before, negative_digits_before, 1)
            negative = '-' if has_negative else ''
            c.specifier = f'S[table-format={negative}{digits_before}.{digits_after}]'
            if isinstance(c.format, format_mean_and_variance):
                c.specifier += '@{}l'
                c.heading = f'\\multicolumn{{2}}{{c}}{{{c.heading}}}'
            else:
                c.heading = f'{{{c.heading}}}'

    # Print the table heading.
    column_spec = ''.join(c.specifier for c in columns)
    print(f'\\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}')
    print('\\toprule')
    if header:
        print(header)
    print(' & '.join(c.heading for c in columns) + ' \\\\')
    print('\\midrule')

    # Print the rows.
    for row, cache in zip(rows, caches, strict=True):
        cells = []
        row_min_value = None
        row_max_value = None
        if row.bold_min or row.bold_max:
            values = []
            for c in columns:
                if c.key != 'label':
                    optional_value = cache[c.key]
                    if optional_value is not None:
                        values.append(c.format.get_comparable_value(optional_value))
            if values:
                if row.bold_min:
                    row_min_value = min(values)
                if row.bold_max:
                    row_max_value = max(values)
        for c in columns:
            value = cache[c.key]
            comparable_value = c.format.get_comparable_value(value) if value is not None else None
            bold = (
                (c.bold_min and c.key in min_values and comparable_value == min_values[c.key]) or
                (c.bold_max and c.key in max_values and comparable_value == max_values[c.key]) or
                (row_min_value is not None and comparable_value == row_min_value) or
                (row_max_value is not None and comparable_value == row_max_value)
            )
            cell = c.format(value, bold)
            cells.append(cell)
        print(' & '.join(cells) + ' \\\\')

    # Print the table footer.
    print('\\bottomrule')
    print('\\end{tabular}')

    # After printing the table, print information about any missing data or
    # other messages.
    if show_runs:
        print(f'% info: results are not complete')
    else:
        print(f'% info: all results are complete')
    for row in rows:
        for message in row.messages:
            print(f'% {message}', file=sys.stderr)
