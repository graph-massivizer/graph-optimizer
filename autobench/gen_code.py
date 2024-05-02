#!/usr/bin/env python

"""
Generates C++ code and executable for each BGO implementation.
"""

from os.path import abspath, basename, dirname, join
from pathlib import Path
from subprocess import run

import json

from jinja2 import Environment, FileSystemLoader


CURRENT_DIR = dirname(abspath(__file__))
TEMPLATE = Environment(loader=FileSystemLoader(CURRENT_DIR)).get_template('main_template.cpp')


TRANSLATIONS = {
    'GrB_Matrix': {
        'decl': 'GrB_Matrix arg_{i};',
        'init': 'read_graph_GB(&arg_{i}, argv[{i}]);',
        'name': 'arg_{i}',
    },
    'LAGraph_Graph': {
        'decl': 'LAGraph_Graph arg_{i};',
        'init': 'read_graph_LA(&arg_{i}, argv[{i}]);',
        'name': 'arg_{i}',
    },
    'CArray<GrB_Index>': {
        'decl': 'CArray<GrB_Index> arg_{i} = CArray<GrB_Index>(atoi(argv[{i}]));',
        'name': 'arg_{i}',
    },
    'GrB_Vector*': {
        'decl': 'GrB_Vector arg_{i};',
        'name': '&arg_{i}',
    }
}


def generate_code(bgo_path):
    parent_config_file = abspath(join(bgo_path, '../config.json'))
    config_file = join(bgo_path, 'config.json')
    header_file = join(bgo_path, f'{basename(bgo_path)}.hpp')
    object_file = join(bgo_path, f'{basename(bgo_path)}.o')
    source_file = join(bgo_path, 'bench.cpp')
    binary_file = join(bgo_path, 'bench')

    with open(parent_config_file, 'r') as f:
        parent_config = json.load(f)
    with open(config_file, 'r') as f:
        config = json.load(f)

    context = {
        'method': config['name'],
        'header': header_file,
    }
    for phase in ['decl', 'init', 'free', 'name']:
        context[f'{phase}s'] = []
        for i, arg in enumerate(config['args'], start=1):
            if phase not in TRANSLATIONS[arg]:
                continue
            context[f'{phase}s'].append(TRANSLATIONS[arg][phase].format(i=i))
    
    with open(source_file, 'w') as f:
        f.write(TEMPLATE.render(context))
    
    run('make', cwd=bgo_path)


if __name__ == '__main__':
    bgos = [dirname(p) for p in Path(abspath(join(CURRENT_DIR, '../bgo'))).glob("*/*/config.json")]
    for bgo in bgos:
        generate_code(bgo)
