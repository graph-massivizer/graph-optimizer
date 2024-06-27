import os
import sys
import glob

from os.path import abspath, basename, dirname, exists, join, relpath
from pathlib import Path
from subprocess import run, PIPE

import json
import random
import time
from datetime import datetime

from csv import DictReader, DictWriter
from io import StringIO

from scipy.io import mminfo, mmwrite

from util.config import *


def RAND_VERT_VECTOR(size_verts):
    size = random.randrange(1, size_verts)
    vector = list(range(size_verts))
    random.shuffle(vector)
    vector = vector[:size]
    vector.sort()
    return vector


def generate_case(bgo_config, data):
    args, refs = [], {}

    if 'G' in data:
        rows, cols, entries, _, _, _ = mminfo(data['G'])
        assert(rows == cols)
        refs['G'] = {
            'SIZE_VERTS': rows,
            'SIZE_EDGES': entries,
            'PATH': data['G'],
        }

    for arg in bgo_config['in_args']:
        if arg['id'] == 'G':
            assert(arg['kind'] == 'GRAPH')
            args.append(data['G'])

        elif arg['kind'] == 'GRAPH':
            rows, cols, entries, _, _, _ = mminfo(data[arg['id']])
            assert(rows == cols)
            args.append(data[arg['id']])
            refs[arg['id']] = {
                'SIZE_VERTS': rows,
                'SIZE_EDGES':entries,
                'PATH': data[arg['id']],
            }

        elif arg['kind'] == 'VECTOR':
            src, val = arg['value'].split('.')
            if arg['id'] in data:
                rows, cols, _, _, _, _ = mminfo(data[arg['id']])
                assert(rows == 1)
                args.append(data[arg['id']])
                refs[arg['id']] = { 'SIZE': cols }
            elif val == 'RAND_VERT_VECTOR':
                vector = RAND_VERT_VECTOR(refs[src]['SIZE_VERTS'])
                args.append(vector)
                refs[arg['id']] = { 'SIZE': len(vector) }
            else:
                exit(-1)


        elif arg['kind'] == 'VERTEX':  #INT
            src, val = arg['value'].split('.')
            if arg['id'] in data:
                args.append(data[arg['id']])
            elif val == 'RAND_VERT':
                args.append(RAND_VERT_VECTOR(refs[src]['SIZE_VERTS'])[0])
            else:
                exit(-1)

    for arg in bgo_config['out_args']:
        if arg['id'] in data:
            args.append(data[arg['id']])
        elif arg['kind'] in ['GRAPH', 'VECTOR']:
            args.append('/dev/null')
        else:
            args.append('NONE')

    stats = {}
    if 'stats' in bgo_config:
        for stat in bgo_config['stats']:
            src, val = stat.split('.')
            stats[stat] = refs[src][val]

    return args, stats


def parse_arg_list(arg):
    """Parse arguments in a `key=value1,value2,value3` format."""
    key, value = arg.split('=')
    filepaths = []
    for fp in value.split(','):
        if '*' in fp or '?' in fp:
            filepaths.extend(glob.glob(fp))
        else:
            filepaths.append(fp)

    if len(filepaths) == 0:
        raise argparse.ArgumentTypeError(f"No (valid) filenames given (in --data) for key '{key}'!")
    return { key: filepaths }


if __name__ == '__main__':
    import argparse

    timestamp = int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds())

    parser = argparse.ArgumentParser()
    parser.add_argument('bgos', nargs='+', type=abspath)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--data', nargs='+', type=parse_arg_list)
    out_arg = parser.add_argument('--output', type=abspath, default=None)

    args = parser.parse_args()

    if args.output and len(args.bgos) > 1:
        parser.error("Cannot output to specific file when benchmarking multiple BGO's!")

    random.seed(args.seed)

    # Reshape the input data.
    data = { k: v for d in args.data for k, v in d.items() }
    data_len = max([len(v) for v in data.values()])
    for k, v in data.items():
        if len(v) == 1:
            data[k] = v * data_len
        elif len(v) != data_len:
            parser.error(f"Values of key '{k}' must be of length 1 or {data_len}.")

    for bgo_path in args.bgos:
        run('make bench'.split(' '), cwd=bgo_path)

        if args.output:
            results_file = args.output
        else:
            results_file = join(bgo_path, f'../results_{timestamp}.csv')

        with open(join(bgo_path, '../config.json'), 'r') as f:
            bgo_config = json.load(f)

        for i in range(data_len):
            cur_data = { k: v[i] for k, v in data.items() }
            for _ in range(args.num):
                bgo_args, bgo_stats = generate_case(bgo_config, cur_data)
                for j, arg in enumerate(bgo_args):
                    if type(arg) is list:
                        list_path = join(TEMP_DIR, f'arg{j}.mtx')
                        mmwrite(list_path, [arg])
                        bgo_args[j] = list_path

                bgo_args = [str(arg) for arg in bgo_args] + [str(args.runs)]
                print(f"Running case\n"
                    f"    path: {bgo_path}\n"
                    f"    args: {', '.join(bgo_args)}")
                proc = run([join(bgo_path, 'bench')] + bgo_args, stdout=PIPE)
                if proc.returncode != 0:
                    print("An error occurred!")
                print()

                reader = DictReader(StringIO(proc.stdout.decode('utf-8')))
                for row in reader:
                    result = {
                        'bgo_path': relpath(bgo_path, BASE_DIR),
                        'case': i,
                        **row,
                        **bgo_stats,
                    }

                    with open(results_file, 'a') as f:
                        writer = DictWriter(f, result.keys())
                        if os.path.getsize(results_file) == 0:
                            writer.writeheader()
                        writer.writerow(result)
