"""Filesystem and JSON helpers used by the experiment entrypoint."""
import json
from pathlib import Path


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_file(file_path):
    with open(file_path, encoding='utf-8-sig') as stream:
        return json.load(stream)
