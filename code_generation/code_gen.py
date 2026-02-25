import argparse
import os
import json
from copy import copy

from itertools import zip_longest
from clang.cindex import Index
# for docker container:
from clang.cindex import Config
Config.set_library_file("/usr/lib/llvm-15/lib/libclang.so.1")

from autobench.util.config import TRANSLATIONS
from jinja2 import Environment, FileSystemLoader

Environment().globals.update(zip=zip)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def parse_bgo_header(headerfile, includes=''):
    """Read the first (!) function definition of a header file and return the signature."""
    def traverse_ast(node):
        if str(node.location.file) == str(headerfile) and node.kind.name == 'FUNCTION_DECL':
            return {
                'header': headerfile,
                'method': node.spelling,
                'return': node.result_type.spelling.replace(' ', ''),
                'args': [arg.type.spelling.replace(' ', '') for arg in node.get_arguments()],
            }
        for child in node.get_children():
            result = traverse_ast(child)
            if result:
                return result
        return None

    index = Index.create()
    root = index.parse(headerfile, args=includes.split(' ')).cursor
    return traverse_ast(root)

def generate_code(dag, user_input, host="H01", code_dir="code_generation/generated_code/", include_timing=True, print_output=False):
    includes = f"-I{BASE_DIR}/include -I{BASE_DIR}/include/gap"
    GPU = False
    phases = ['decl', 'init', 'save', 'free', 'name']
    context = {-1: {f"{phase}s": [] for phase in phases}} # Initiate context with -1, which refers to the external parameters.
    bgo_calls = [] # Used to save the funcion calls to the BGOs with correct arguments
    bgo_paths = [] # List of BGO paths, used for Makefile template
    for i, bgo in enumerate(dag):
        directory = f'{BASE_DIR}/bgo/{bgo["name"]}'

        implementation = next((p["implementation"] for p in bgo["performances"] if p['host'] == host), None)
        # If the implementation is GPU/... set the GPU flag to true to select the right template later on
        implementation_device, implementation_name = implementation.split("/")
        if implementation_device == "GPU":
            GPU = True
        # Load the header file
        bgo_path = f"{directory}/{implementation}"
        bgo_paths.append(bgo_path)
        header_file = f'{bgo_path}/{implementation_name}.hpp'
        if not os.path.isfile(header_file):
            print(f'{header_file} not found')
            exit(1)
        header = parse_bgo_header(header_file, includes)
        
        # Initiate context of current bgo with values from header file
        context[i] = copy(header)
        context[i].update({f"{phase}s": [] for phase in phases})

        # Load the config file for current bgo
        config_filename = f'{directory}/config.json'
        if not os.path.isfile(config_filename):
            print(f'{config_filename} not found')
            exit(1)
        
        # Read the config file and convert to JSON
        config_file = open(config_filename)
        config = json.load(config_file)
        config_file.close() 
        
        # Loop over list of all parameters from the header file, and match with config args and bgo inputs
        inputs = []
        outputs = []
        for (bgo_arg, config_arg), header_arg in zip(list(zip(bgo["inputs"], config["in_args"])) + [(None, arg) for arg in config['out_args']], header['args']):
            for phase in phases:
                if phase == "name":
                    if bgo_arg:
                        inputs.append(TRANSLATIONS[header_arg][phase].format(i=bgo_arg["source_id"]))
                    else:
                        outputs.append(TRANSLATIONS[header_arg][phase].format(i=config_arg["id"]))
                if phase not in TRANSLATIONS[header_arg]:
                    continue
                if bgo_arg and bgo_arg["source"] != -1:
                    continue

                bgo_id = i if not bgo_arg else bgo_arg["source"]
                translation = TRANSLATIONS[header_arg][phase].format(i=config_arg["id"] if not bgo_arg else bgo_arg["source_id"])
                phase_list = context[bgo_id][f"{phase}s"]
                phase_list.append(translation) if translation not in phase_list else phase_list

        in_args = ','.join(inputs)
        out_args = ','.join(outputs)

        bgo_calls.append(f"{header['method']}({in_args},{out_args});")
            
    # Include user defined arguments.
    for key, values in context[-1].items():
        # This should never pass, but is a backup to prevent crashes
        if key not in [f"{phase}s" for phase in phases]:
            continue

        new_values = []
        for x in values: 
            new_value = x
            for arg_name, arg_value in user_input.items():
                new_value = new_value.replace(f"argv[{arg_name}]", f'"{str(arg_value)}"')
            new_values.append(new_value)
        context[-1].update({key: new_values})
    
    headers = [v["header"] for _, v in context.items() if "header" in v]
    decls = [decl for _, v in context.items() if "decls" in v for decl in v["decls"]]
    inits = context[-1]['inits']

    code_template_values = {"headers": headers,
                            "decls": decls,
                            "inits": inits,
                            "bgo_calls": bgo_calls,
                            "final_outputs": outputs,
                            "include_timing": include_timing,
                            "print_output": print_output}
    makefile_template_values = {"includes": includes,
                                "lib_dir": f"{BASE_DIR}/lib",
                                "bgo_paths": bgo_paths,
                                "bgo_libs": {f"{n.upper()}_LIB": f"{p}/lib{p.split('/')[-1]}.a" for n, p in zip([bgo["name"] for bgo in dag], bgo_paths)}}
    
    extension = "cpp"# if GPU else "cpp"
    code_template_path = f"{BASE_DIR}/code_generation/template.{extension}"
    # TODO add cuda option
    makefile_template_path = f"{BASE_DIR}/code_generation/Makefile_template"
    code_template = Environment(loader=FileSystemLoader(os.path.dirname(code_template_path))).get_template(os.path.basename(code_template_path))
    makefile_template = Environment(loader=FileSystemLoader(os.path.dirname(makefile_template_path))).get_template(os.path.basename(makefile_template_path))
    generated_code = code_template.render(code_template_values)
    generated_makefile = makefile_template.render(makefile_template_values, zip=zip)
    
    os.makedirs(os.path.dirname(code_dir), exist_ok=True)
    with open(f"{code_dir}/main.{extension}", 'w') as f:
        f.write(generated_code)
    with open(f"{code_dir}/Makefile", 'w') as f:
        f.write(generated_makefile)
    
    return

if __name__=="__main__":
    dag = [{'id': 0,
  'name': 'pr',
  'dependencies': [],
  'inputs': [{'source': -1, 'source_id': 'G', 'target_id': 'G'}],
  'performances': [{'host': 'H01',
    'implementation': 'GPU/vertex_pull',
    'runtime': 58986703.3,
    'energy': 1363341.7}]},
 {'id': 1,
  'name': 'find_max',
  'dependencies': [0],
  'inputs': [{'source': 0, 'source_id': 'PR', 'target_id': 'values'}],
  'performances': [{'host': 'H01',
    'implementation': 'GPU/find_max_gpu',
    'runtime': 2.8534564647104044,
    'energy': 10.02278935619039}]},
 {'id': 2,
  'name': 'bfs',
  'dependencies': [0],
  'inputs': [{'source': -1, 'source_id': 'G', 'target_id': 'G'},
   {'source': -1, 'source_id': 'source', 'target_id': 'source'}],
  'performances': [{'host': 'H01',
    'implementation': 'GPU/vertex_push',
    'runtime': 6456753.1,
    'energy': 119537.59999999999}]},
 {'id': 3,
  'name': 'find_path',
  'dependencies': [1, 2],
  'inputs': [{'source': 2, 'source_id': 'parents', 'target_id': 'parents'},
   {'source': 1, 'source_id': 'result', 'target_id': 'end'},
   {'source': -1, 'source_id': 'source', 'target_id': 'start'}],
  'performances': [{'host': 'H01',
    'implementation': 'CPU/fp_ca',
    'runtime': 0.00013304792451113682,
    'energy': 4.656677357889789e-12}]}]
    user_input = {"G": "data/test_matrix_fully_connected.mtx", "source": 1}
    generate_code(dag, user_input)