import os

from clang.cindex import Index
from jinja2 import Environment, FileSystemLoader


INCLUDES = ['/home/knobel/.local/include']
TEMPLATE = 'main_template.cpp'

DECL_CODE = {
    'GrB_Matrix': 'GrB_Matrix arg_{i};',
    'GrB_Matrix*': 'GrB_Matrix arg_{i}',
}

INIT_CODE = {
    'GrB_Matrix': 'read_graph_GB(&arg_{i}, argv[{i}]);',
    'GrB_Matrix*': 'read_graph_GB(&arg_{i}, argv[{i}]);',
}

ARG_CODE = {
    'GrB_Matrix': 'arg_{i}',
    'GrB_Matrix*': '&arg_{i}',
}

FREE_CODE = {
    'GrB_Matrix': '',
    'GrB_Matrix*': '',
}


def read_header(filename):
    signatures = []
    
    def traverse_ast(node):
        if str(node.location.file) == filename and node.kind.name == 'FUNCTION_DECL':
            signatures.append({
                'name': node.spelling,
                'return': node.result_type.spelling.replace(' ', ''),
                'args': [arg.type.spelling.replace(' ', '') for arg in node.get_arguments()]
            })
        
        for child in node.get_children():
            traverse_ast(child)

    index = Index.create()
    root = index.parse(filename, args=[f'-I{include}' for include in INCLUDES]).cursor
    traverse_ast(root)
    return signatures


def generate_code(header, signature):
    env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
    template = env.get_template(TEMPLATE)

    args = signature['args']
    context = {
        'header': header,
        'decls': [DECL_CODE[arg].format(i=i+1) for i, arg in enumerate(args)],
        'inits': [INIT_CODE[arg].format(i=i+1) for i, arg in enumerate(args)],
        'args': [ARG_CODE[arg].format(i=i+1) for i, arg in enumerate(args)],
        'frees': [FREE_CODE[arg].format(i=i+1) for i, arg in enumerate(args)],
        'method': signature['name'],
    }

    return template.render(context)
