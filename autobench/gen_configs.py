#!/usr/bin/env python

"""
Generates a data.json file and .json file for each BGO implementation.
"""


import os
import sys

from pathlib import Path
import json

from scipy.io import mminfo
from clang.cindex import Index

from config import INCLUDES, DATA_DIR, BGOS_DIR


def genereate_data_config(data_dir):
    data = []
    for mtx_file in Path(data_dir).glob('**/*.mtx'):
        with open(mtx_file, 'r') as f:
            rows, cols, entries, form, field, symmetry = mminfo(f)
            if rows != cols:
                print(f"{mtx_file} skipped, number of rows and columns does not match!", file=sys.stderr)
                continue
            
            data.append({
                'id': str(mtx_file).split(f'{str(data_dir)}/')[1].split('.mtx')[0],
                'path': str(mtx_file),
                'size_verts': rows,
                'size_edges': entries,
            })
    
    with open(os.path.join(data_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent=4)


def read_header(header_file):
    signatures = []
    
    def traverse_ast(node):
        if str(node.location.file) == str(header_file) and node.kind.name == 'FUNCTION_DECL':
            signatures.append({
                'name': node.spelling,
                'return': node.result_type.spelling.replace(' ', ''),
                'args': [arg.type.spelling.replace(' ', '') for arg in node.get_arguments()]
            })
        
        for child in node.get_children():
            traverse_ast(child)

    index = Index.create()
    root = index.parse(header_file, args=[f'-I{os.path.abspath(os.path.expanduser(include))}' for include in INCLUDES]).cursor
    traverse_ast(root)
    return signatures


def generate_bgos_config(bgos_dir):
    for header_file in Path(bgos_dir).glob('*/*/*.hpp'):
        if os.path.basename(header_file).split('.')[0] != os.path.basename(os.path.dirname(header_file)):
            continue

        signatures = read_header(header_file)
        if len(signatures) != 1:
            print(f"{header_file} skipped, number of function definitions is not equal to one!")
            continue
        
        config_file = os.path.join(os.path.dirname(header_file), 'config.json')
        with open(config_file, 'w') as f:
            json.dump(signatures[0], f, indent=4)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.abspath(os.path.join(current_dir, DATA_DIR))
    genereate_data_config(data_dir)
    
    bgos_dir = os.path.abspath(os.path.join(current_dir, BGOS_DIR))
    generate_bgos_config(bgos_dir)


if __name__ == '__main__':
    main()
