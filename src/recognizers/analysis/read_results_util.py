import json
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

from rau.tools.torch.saver import read_logs as read_model_logs

def read_test_accuracy(model_dir: pathlib.Path, print_error: bool = False) -> float | None:
    json_file_name = model_dir / 'eval' / 'test.json'
    try:
        with json_file_name.open() as fin:
            test_accuracy = json.load(fin)['scores']['recognition_accuracy']
    except FileNotFoundError:
        if print_error:
            print(f'missing test accuracy: {model_dir}', file=sys.stderr)
        return None
    return test_accuracy

def read_best_test_accuracy(trials_dir: pathlib.Path, num_trials: int) -> float | None:
    values = []
    for trial_no in range(1, num_trials+1):
        model_dir = trials_dir / str(trial_no)
        test_accuracy = read_test_accuracy(model_dir, print_error=True)
        if test_accuracy is not None:
            values.append(test_accuracy)
    if values:
        return max(values)
    else:
        return None

def read_language_txt_float(language_dir: pathlib.Path, name: str) -> float | None:
    file_name = language_dir / f'{name}.txt'
    try:
        result_str = file_name.read_text()
    except FileNotFoundError:
        print(f'missing: {file_name}', file=sys.stderr)
        return None
    else:
        return float(result_str.strip())

def read_logs(
    model_dir: pathlib.Path,
    single_event_names: Iterable[str]
) -> dict[str, dict[str, Any]] | None:
    single_events_left = set(single_event_names)
    result = {}
    try:
        with read_model_logs(model_dir) as logs:
            for event in logs:
                if not single_events_left:
                    break
                if event.type in single_events_left:
                    result[event.type] = event.data
                    single_events_left.remove(event.type)
        return result
    except FileNotFoundError:
        print(f'missing log file: {model_dir}')
        return None

def read_training_loop(model_dir: pathlib.Path) -> dict[str, Any] | None:
    json_file = model_dir / 'training-loop.json'
    try:
        with json_file.open() as fin:
            return json.load(fin)
    except FileNotFoundError:
        print(f'missing: {json_file}')
        return None
