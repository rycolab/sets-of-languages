import json
from collections.abc import Iterable
from io import Writer
from typing import Any

def write_json_line(data: Any, fout: Writer[str]) -> None:
    json.dump(data, fout, separators=(',', ':'))
    print(file=fout)

def load_jsonl_file(fin: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in fin:
        yield json.loads(line)
