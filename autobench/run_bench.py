#!/usr/bin/env python

from os.path import abspath, basename, dirname, join
from pathlib import Path
from subprocess import run

import json


CURRENT_DIR = dirname(abspath(__file__))


cases_file = join(CURRENT_DIR, 'cases/cases.json')
with open(cases_file, 'r') as f:
    cases = json.load(f)

for i, case in enumerate(cases):
    for path in Path(case['bgo_path']).glob('*/bench'):
        print(f"Running case {i} ({path}):")
        run([path] + [str(arg) for arg in case['args']])
        print()
