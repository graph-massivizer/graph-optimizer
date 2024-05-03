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
        args = [str(arg) for arg in case['args']]
        print(f"Running case {i}\n"
              f"    path: {path}\n"
              f"    args: {', '.join(args)}")
        proc = run([path] + args)
        if proc.returncode != 0: 
            print("An error occurred!")
        print()
