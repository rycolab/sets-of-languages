import dataclasses
import pathlib
from collections.abc import Sequence

@dataclasses.dataclass
class Bubble:
    id: str

    def width_variable(self) -> str:
        return f'\\{self.id}Width'

    def height_variable(self) -> str:
        return f'\\{self.id}Height'

    def bottom_variable(self) -> str:
        return f'\\{self.id}Bottom'

    def border_radius_variable(self) -> str:
        return f'\\{self.id}BorderRadius'

    def to_latex(self, num_columns: int) -> str:
        global_variables = variable_defs_to_latex([
            (
                '\\baseWidth',
                r'2*\classPadding + '
                f'{num_columns}' r'*2*\languageRadius + '
                f'{num_columns-1}' r'*\languageGutter'
            )
        ])
        variables, drawing = self._to_latex(num_columns, '0')
        return f'{global_variables}{variables}{drawing}'

    def _to_latex(self, num_columns: int, bottom: str) -> tuple[str, str]:
        raise NotImplementedError

@dataclasses.dataclass
class LanguageClass(Bubble):
    label: str
    languages: Sequence[bool] = dataclasses.field(default_factory=list)
    subclasses: Sequence[Bubble] = dataclasses.field(default_factory=list)
    num_levels: int = 1

    def _to_latex(self, num_columns: int, bottom: str) -> tuple[str, str]:
        num_full_rows, num_leftover_dots = divmod(len(self.languages), num_columns)
        num_rows = num_full_rows + int(num_leftover_dots > 0)
        if self.subclasses:
            width = f'{self.num_levels}' r'*2*\classPadding + ' f'{self.subclasses[0].width_variable()}'
            border_radius = f'{self.subclasses[0].border_radius_variable()}' r' + ' f'{self.num_levels}' r'*\classPadding'
        else:
            width = r'\baseWidth'
            border_radius = r'\borderRadius'
        height_terms = [r'\labelHeight']
        num_sub_items = int(num_rows > 0) + len(self.subclasses)
        if num_sub_items > 0:
            height_terms.append(r'\labelBottomSpace + \classPadding')
        if num_sub_items > 1:
            height_terms.append(f'{num_sub_items-1}' r'*\verticalGutter')
        if num_rows > 0:
            height_terms.append(f'{num_rows}' r'*2*\languageRadius')
        if num_rows > 1:
            height_terms.append(f'{num_rows-1}' r'*\languageGutter')
        for c in self.subclasses:
            height_terms.append(c.height_variable())
        height = ' + '.join(height_terms)
        bottom_variable = self.bottom_variable()
        width_variable = self.width_variable()
        height_variable = self.height_variable()
        border_radius_variable = self.border_radius_variable()
        language_bottom_variable = self.language_bottom_variable()
        variables_before = variable_defs_to_latex([(bottom_variable, bottom)])
        subclass_bottom = f'{bottom_variable}' r' + \classPadding'
        subclass_latex = []
        for c in reversed(self.subclasses):
            subclass_latex.append(c._to_latex(num_columns, subclass_bottom))
            subclass_bottom = f'{c.bottom_variable()} + {c.height_variable()}' r' + \verticalGutter'
        subclass_latex.reverse()
        variables_after_defs = [
            (width_variable, width),
            (height_variable, height),
            (border_radius_variable, border_radius),
        ]
        if num_rows > 0:
            variables_after_defs.append((language_bottom_variable, subclass_bottom))
        variables = ''.join([
            variable_defs_to_latex([(bottom_variable, bottom)]),
            *(x for x, _ in reversed(subclass_latex)),
            variable_defs_to_latex(variables_after_defs)
        ])
        class_drawing = (
            r'\draw[rounded corners=' f'{border_radius_variable}' r' cm, thick, fill=' f'{self.id}' r'] '
            f'(-0.5*{width_variable}, {bottom_variable}) rectangle (0.5*{width_variable}, {bottom_variable}+{height_variable});\n'
            r'\node[anchor=north] at (0, ' f'{bottom_variable}+{height_variable}' r'+\labelTopOffset) {\classLabel{'
            f'{self.label}'
            r'}};' '\n'
        )
        language_dots = []
        for i, language in enumerate(self.languages):
            row, column = divmod(i, num_columns)
            num_columns_in_row = num_columns if row < num_full_rows else num_leftover_dots
            x = (
                f'-{num_columns_in_row}' r'*\languageRadius'
                r' - 0.5*' f'{num_columns_in_row-1}' r'*\languageGutter'
                r' + ' f'{column}' r'*2*\languageRadius'
                r' + ' f'{column}' r'*\languageGutter'
                r' + \languageRadius'
            )
            y = (
                f'{language_bottom_variable}'
                r' + ' f'{row}' r'*2*\languageRadius'
                r' + ' f'{row}' r'*\languageGutter'
                r' + \languageRadius'
            )
            language_dots.append(
                r'\draw[thick' f'{(r', fill=black' if language else '')}' r'] (' f'{x}' r', ' f'{y}' r') circle (\languageRadius);' '\n'
            )
        drawing = ''.join([class_drawing, *language_dots, *(x for _, x in subclass_latex)])
        return variables, drawing

    def language_bottom_variable(self) -> str:
        return f'\\{self.id}LanguageBottom'

@dataclasses.dataclass
class HorizontalBubbles(Bubble):
    bubbles: Sequence[tuple[str, str]]

    def _to_latex(self, num_columns: int, bottom: str) -> tuple[str, str]:
        bottom_variable = self.bottom_variable()
        width_variable = self.width_variable()
        height_variable = self.height_variable()
        border_radius_variable = self.border_radius_variable()
        sub_width_variable = f'\\{self.bubbles[0][1]}Width'
        n = len(self.bubbles)
        variables = [
            (bottom_variable, bottom),
            (width_variable, r'\baseWidth'),
            (height_variable, r'\labelHeight'),
            (border_radius_variable, r'\borderRadius'),
            (sub_width_variable, f'\\baseWidth/{n}-\\classPadding+\\classPadding/{n}')
        ]
        drawings = []
        for i, (label, bubble_id) in enumerate(self.bubbles):
            left_variable = f'\\{bubble_id}Left'
            variables.append((
                left_variable,
                f'-0.5*\\baseWidth + {i}*{sub_width_variable} + {i}*\\classPadding'
            ))
            drawings.append(
                r'\draw[rounded corners=' f'{border_radius_variable}' r' cm, thick, fill=' f'{bubble_id}' r'] '
                f'({left_variable}, {bottom_variable}) rectangle ({left_variable}+{sub_width_variable}, {bottom_variable}+{height_variable});\n'
                r'\node[anchor=north] at ' f'({left_variable}+{sub_width_variable}/2, {bottom_variable}+{height_variable}' r'+\labelTopOffset) {\classLabel{'
                f'{label}'
                r'}};' '\n'
            )
        return variable_defs_to_latex(variables), ''.join(drawings)

    def left_variable(self) -> str:
        return f'\\'

def variable_defs_to_latex(pairs: list[tuple[str, str]]) -> str:
    return ''.join(
        r'\pgfmathsetmacro{' f'{k}' r'}{' f'{v}' '}\n'
        for k, v in pairs
    )

def print_language_class(
    language_class: LanguageClass,
    num_columns: int,
    file_name: pathlib.Path
) -> None:
    print(f'writing {file_name}')
    file_name.parent.mkdir(parents=True, exist_ok=True)
    with file_name.open('w') as fout:
        print(language_class.to_latex(num_columns), end='', file=fout)
