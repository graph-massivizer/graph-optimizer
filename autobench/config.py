from os.path import abspath, dirname, expanduser, join

from jinja2 import Environment, FileSystemLoader

BASE_DIR = abspath(join(dirname(__file__), '../'))
DATA_DIR = join(BASE_DIR, 'data')
BGOS_DIR = join(BASE_DIR, 'bgo')
TEMP_DIR = join(BASE_DIR, 'autobench/cases')

CONFIG_FILE = join(BASE_DIR, 'config.json')

INCLUDES = [
    expanduser('~/.local/include'),
    expanduser('~/graph-optimizer/include'),
]

TEMPLATE = Environment(loader=FileSystemLoader(BASE_DIR)).get_template('autobench/main_template.cpp')

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
