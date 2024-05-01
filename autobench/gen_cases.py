#!/usr/bin/env python

"""
Generates a cases.json file with arguments to be used during benchmarking.
"""

import os
import sys

from pathlib import Path
import json
import random

from scipy.io import mmwrite

from config import DATA_DIR, BGOS_DIR


def RAND_VERT_VECTOR(size_verts):
    size = random.randrange(1, size_verts)
    vector = list(range(size_verts))
    random.shuffle(vector)
    vector = vector[:size]
    vector.sort()
    return vector


def gen_case(bgo, data):
    args, refs = [], {}
    for arg in bgo['args']:
        if arg['kind'] == 'GRAPH':
            args.append(data['path'])
            refs[arg['id']] = {
                'SIZE_VERTS': data['size_verts'],
                'SIZE_EDGES': data['size_edges'],
            }
        
        elif arg['kind'] == 'VECTOR':
            src, val = arg['value'].split('.')
            if val == 'RAND_VERT_VECTOR':
                vector = RAND_VERT_VECTOR(refs[src]['SIZE_VERTS'])
                args.append(vector)
        
            elif val == 'EMPTY_VERT_VECTOR':
                args.append(refs[src]['SIZE_VERTS'])

    return args

def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cases_dir = os.path.join(current_dir, 'cases')
    input_dir = os.path.join(cases_dir, 'input')
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)

    bgo_configs, data_configs = [], []
    for bgo_config_file in Path(os.path.abspath(os.path.join(current_dir, BGOS_DIR))).glob('*/config.json'):
        bgo_name = os.path.dirname(bgo_config_file)
        with open(bgo_config_file, 'r') as f:
            bgo_configs.append({**json.load(f), **{'path': bgo_name}})
    with open(os.path.abspath(os.path.join(current_dir, os.path.join(DATA_DIR, 'data.json'))), 'r') as f:
        data_configs = json.load(f)

    cases = []
    for bgo_config in bgo_configs:
        for data_config in data_configs[:1]:
            for _ in range(args.num):
                cases.append({
                    'bgo_path': bgo_config['path'],
                    'args': gen_case(bgo_config, data_config)
                    })

    for ci, case in enumerate(cases):
        for ai, arg in enumerate(case['args']):
            if type(arg) is list:
                path = os.path.join(input_dir, f'{ci}_{ai}.mtx')
                mmwrite(path, [arg])
                case['args'][ai] = path
    with open(os.path.join(cases_dir, 'cases.json'), 'w') as f:
        json.dump(cases, f, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=10)

    args = parser.parse_args()
    main(args)
