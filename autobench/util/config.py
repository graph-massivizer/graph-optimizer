from os.path import abspath, dirname, expanduser, join
import re
from jinja2 import Environment, FileSystemLoader

BASE_DIR = abspath(join(dirname(__file__), '../../'))
DATA_DIR = join(BASE_DIR, 'data')
BGOS_DIR = join(BASE_DIR, 'bgo')
TEMP_DIR = join(BASE_DIR, '/tmp')

CONFIG_FILE = join(BASE_DIR, 'config.json')

INCLUDES = [
    expanduser('~/.local/include'),
    expanduser('~/graph-optimizer/include'),
]

TEMPLATE = Environment(loader=FileSystemLoader(BASE_DIR)).get_template('autobench/misc/template.cpp')

RESULT_PATTERN = re.compile(r'Runtime: (\d+) ns\nStatus: (\d+)')

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
    'CMatrix<int>': {
        'decl': 'CMatrix<int> arg_{i};',
        'init': 'read_graph_CMatrix(&arg_{i}, argv[{i}]);',
        'name': 'arg_{i}',
    },
    'CArray<int>': {
        'decl': 'CArray<int> arg_{i};',
        'init': 'read_vector_CArray(&arg_{i}, argv[{i}]);',
        'name': 'arg_{i}',
    },
    'CArray<int>*': {
        'decl': 'CArray<int> arg_{i} = CArray<int>(atoi(argv[{i}]));',
        'name': '&arg_{i}',
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
