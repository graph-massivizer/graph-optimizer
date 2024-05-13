#!/usr/bin/env python

import sys

from os.path import abspath, basename, dirname, exists, join
from pathlib import Path
from subprocess import run

import json
import random

from scipy.io import mmwrite

from config import *


def RAND_VERT_VECTOR(size_verts):
    size = random.randrange(1, size_verts)
    vector = list(range(size_verts))
    random.shuffle(vector)
    vector = vector[:size]
    vector.sort()
    return vector


def generate_code(bgo):
    context = {
        'method': bgo['name'],
        'header': bgo['header'],
    }
    for phase in ['decl', 'init', 'free', 'name']:
        context[f'{phase}s'] = []
        for i, arg in enumerate(bgo['args'], start=1):
            if phase not in TRANSLATIONS[arg]:
                continue
            context[f'{phase}s'].append(TRANSLATIONS[arg][phase].format(i=i))
    with open(join(bgo['path'], 'bench.cpp'), 'w') as f:
        f.write(TEMPLATE.render(context))
    
    run('make bench'.split(' '), cwd=bgo['path'])


def generate_case(abs_bgo, data):
    args, refs = [], {}
    for arg in abs_bgo['args']:
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


if __name__ == '__main__':
    import argparse

    if not exists(CONFIG_FILE):
        print(f"Config file does not exist! ({CONFIG_FILE})", file=sys.stderr)
        exit(-1)
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    data = list(config['data'].keys())
    bgos = sum([list(config['bgos'][abs_bgo].keys()) for abs_bgo in config['bgos']], [])
    bgos = {
        imp_bgo: {
            'name': imp_bgo,
            'path': config['bgos'][abs_bgo]['implementations'][imp_bgo]['path'],
            'abs_bgo': abs_bgo,
            'args': config['bgos'][abs_bgo]['implementations'][imp_bgo]['args'],
            'header': config['bgos'][abs_bgo]['implementations'][imp_bgo]['header'],
        } for abs_bgo in config['bgos'] for imp_bgo in config['bgos'][abs_bgo]['implementations']
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data', nargs='+', choices=data, default=data)
    parser.add_argument('--bgos', nargs='+', choices=bgos, default=bgos)
    parser.add_argument('--num', type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)

    for bgo in args.bgos:
        generate_code(bgos[bgo])

        abs_bgo = config['bgos'][bgos[bgo]['abs_bgo']]

        print(bgos[bgo])

        for data in args.data:
            data = {**config['data'][data], 'path': data}
            for _ in range(args.num):
                args = generate_case(abs_bgo, data)
                for i, arg in enumerate(args):
                    if type(arg) is list:
                        path = join(TEMP_DIR, f'arg{i}.mtx')
                        mmwrite(path, [arg])
                        args[i] = path
                print(args)
                
                args = [str(arg) for arg in args]
                path = join(bgos[bgo]['path'], 'bench')
                print(f"Running case\n"
                    f"    path: {path}\n"
                    f"    args: {', '.join(args)}")
                proc = run([path] + args)
                if proc.returncode != 0:
                    print("An error occurred!")
                print()
