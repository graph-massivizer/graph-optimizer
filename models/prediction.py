#! /usr/bin/python
import sys
import subprocess

import json
import importlib
import pkgutil
import os

import models
from models.evaluate_model import evaluate
from datetime import datetime
import models.utils as utils
from copy import deepcopy

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def import_models(base_pkg):
    base_path = os.path.dirname(base_pkg.__file__)
    modules = {}
    
    for finder, name, ispkg in pkgutil.walk_packages(base_pkg.__path__, base_pkg.__name__ + "."):
        if name.endswith(".energy_model") or name.endswith(".performance_model"):
            modules[name] = importlib.import_module(name)
    return modules

def analytical_prediction(hardware, bgo_dag, graph=None):
    return prediction(hardware, bgo_dag, graph)

def run_benchmark(bgo, graph_file):
    result = [item for item in subprocess.check_output(['python3', './autobench/run_bench.py', bgo, '--data', f'G={graph_file}']).decode('utf-8').rstrip().split('\n') if item != ''][-2:]
    values = dict(zip(*map(lambda x: x.split(','), result)))
    # if there is an error, return None
    if 'runtime_ns' not in values:
        return None, None

    return float(values['runtime_ns']), float(values['energy_joules'])

def sampling_based_prediction(annotated_dag, graph_properties, dampening_factor=1):
    """Do sampling based prediction for all implementations of an annotated dag
    which have no analytical models"""
    graph_sample = graph_properties["graph_sample"] if "graph_sample" in graph_properties else None
    sampling_rate = graph_properties["sampling_rate"] if "sampling_rate" in graph_properties else 0.1
    if not graph_sample:
        # Sample graph by calling c++ sampling code, and set name to sampled name
        graph_sample = f"{graph_properties['name'][:-4]}_sampled.mtx"
        subprocess.check_output([f"{BASE_DIR}/sampling/main",  graph_properties["name"], str(sampling_rate), str(graph_sample)])

    prediction_targets = ['runtime', 'energy']
    devices = ['CPU', 'GPU']
    # Where there are no analytical model predictions, use sampling
    for i, bgo in enumerate(annotated_dag):
        min_runtime = float('inf')
        for j, perf_entry in enumerate(bgo["performances"]):
            for device in devices:
                for implementation in perf_entry['runtime'][device]:
                    impl_path = f"./bgo/{bgo['name']}/{device}/{implementation}"
                    print(f"Estimating runtime for {device}/{implementation}...", end=' ')
                    if perf_entry['runtime'][device][implementation] == utils.ERROR_404:
                        for target, value in zip(prediction_targets, run_benchmark(impl_path, graph_sample)):
                            if value:
                                annotated_dag[i]["performances"][j][target][device][implementation] = value / sampling_rate * dampening_factor
                    runtime = annotated_dag[i]["performances"][j]['runtime'][device][implementation]
                    energy = annotated_dag[i]["performances"][j]['energy'][device][implementation]
                        
                    print(f"runtime: {runtime/1000000000}s, energy: {energy/1000000}MJ", end=' ')
                    if runtime < min_runtime:
                        print(f'-> new best implementation for {bgo["name"]}')
                        min_runtime = runtime
                    else:
                        print(f'-> discarded.')
    
    return annotated_dag

def select_worst_case(workflow, annotated_dag):
    performance_breakdown = deepcopy(workflow)
    host = annotated_dag[0]["performances"][0]["host"]

    # Select best implementations per bgo based on metric
    e2e_runtime = 0
    e2e_energy = 0
    devices = ['CPU', 'GPU']
    for i, bgo in enumerate(annotated_dag):
        # These tuples are (BGO_name, runtime, energy)
        max_runtime = (None, 0, 0)
        max_energy = (None, 0, 0)
        for j, perf_entry in enumerate(bgo["performances"]):
            for device in devices:
                for implementation in perf_entry['runtime'][device]:
                    name = f"{device}/{implementation}"
                    cur_runtime = annotated_dag[i]["performances"][j]['runtime'][device][implementation]
                    cur_energy = annotated_dag[i]["performances"][j]['energy'][device][implementation]
                    cur_tuple = (name, cur_runtime, cur_energy)
                    if type(cur_runtime) == float and type(cur_energy) == float:
                        max_runtime = max(cur_tuple, max_runtime, key=lambda x: x[1])
                        max_energy = max(cur_tuple, max_energy, key=lambda x: x[2])

        selected_noptimum = max_runtime

        e2e_runtime += selected_noptimum[1]
        e2e_energy += selected_noptimum[2]

        performance_breakdown[i]["performances"] = [{"host": host, "implementation": selected_noptimum[0], "runtime": selected_noptimum[1], "energy": selected_noptimum[2]}]

    return e2e_runtime, e2e_energy, performance_breakdown
    




def prediction(hardware, bgo_dag, graph=None):
    all_models = import_models(models)
    graph_characteristics = None
    graph_benchmarks = None

    if graph:
        if isinstance(graph, dict):
            graph_characteristics = graph
        else:
            graph_benchmarks = graph
            
    # Loop over bgo dag and do energy and performance predictions for all hardware configurations.
    for i, bgo in enumerate(bgo_dag):
        bgo_dag[i]['performances'] = []

        if graph_benchmarks is not None:
            for host in hardware['hosts']:
                performance, energy = utils.get_benchmark_result(graph_benchmarks, bgo['name'], host['name'])
                bgo_dag[i]['performances'].append({'host': host['name'], 'runtime': performance, 'energy': energy})
        else:
            # Each of these dictionaries stores as keys the implementation name of the specific bgo implementation, and as values the model functions.
            CPU_perf_models = {}
            CPU_energy_models = {}
            GPU_perf_models = {}
            GPU_energy_models = {}
            for name, model in all_models.items():
                model_parts = name.split('.')
                model_bgo = model_parts[1]
                model_device = model_parts[2]
                model_implementation = model_parts[3]
                model_type = model_parts[-1]
                
                if model_bgo != bgo["name"]:
                    continue

                if model_device == "CPU" and model_type == "performance_model":
                    CPU_perf_models[model_implementation] = model
                elif model_device == "CPU" and model_type == "energy_model":
                    CPU_energy_models[model_implementation] = model
                elif model_device == "GPU" and model_type == "performance_model":
                    GPU_perf_models[model_implementation] = model
                elif model_device == "GPU" and model_type == "energy_model":
                    GPU_energy_models[model_implementation] = model

            # Error handling
            if not CPU_perf_models and not GPU_perf_models:
                utils.exit_with_error(f'No performance model found for BGO {bgo["name"]}')
            if not CPU_energy_models and not GPU_energy_models:
                utils.exit_with_error(f'No energy model found for BGO {bgo["name"]}')

            # Predict performance and energy, and annotate bgo_dag with the prediction values for each host.
            for host in hardware['hosts']:
                performance = {'CPU': {implementation: model.predict(host) for implementation, model in CPU_perf_models.items()},
                               'GPU': {implementation: model.predict(host) for implementation, model in GPU_perf_models.items()}}
                energy      = {'CPU': {implementation: model.predict(host) for implementation, model in CPU_energy_models.items()},
                               'GPU': {implementation: model.predict(host) for implementation, model in GPU_energy_models.items()}}
                if graph_characteristics is not None:
                    performance = {'CPU': {implementation: evaluate(model.predict(host), graph_characteristics) for implementation, model in CPU_perf_models.items()},
                                   'GPU': {implementation: evaluate(model.predict(host), graph_characteristics) for implementation, model in GPU_perf_models.items()}}
                    energy      = {'CPU': {implementation: evaluate(model.predict(host), graph_characteristics) for implementation, model in CPU_energy_models.items()},
                                   'GPU': {implementation: evaluate(model.predict(host), graph_characteristics) for implementation, model in GPU_energy_models.items()}}

                bgo_dag[i]['performances'].append({'host': host['name'], 'runtime': performance, 'energy': energy})
    return bgo_dag


def convert_to_greenifier_input(prediction):
    greenifier_output = {"tasks": prediction}

    for bgo in greenifier_output['tasks']:
        bgo['runTimes'] = {perf['host']: int(perf['runtime']) for perf in bgo['performances']}
        bgo['energyConsumption'] = {perf['host']: int(perf['energy']) for perf in bgo['performances']}
        bgo['submissionTime'] = datetime.today().strftime('%Y-%m-%d')
        bgo['cpuCount'] = 1
        bgo['cpuUsage'] = 10000
        bgo['memCapacity'] = 1000000

        if 'inputs' in bgo:
            del bgo['inputs']
        del bgo['performances']

    return greenifier_output


if __name__ == '__main__':
    arguments = sys.argv[1:]

    # Error handling.
    if len(arguments) < 2:
        utils.exit_with_error('Usage: ./prediction.py <hardware_information_json> <bgo_dag> [graph_characteristics or graph_name]')

    # Read the hardware information from the first argument.
    hardware = utils.try_cast_json_dict(utils.check_file(arguments[0]), 'Invalid JSON format for hardware information')
    bgo_dag = utils.try_cast_json_list(utils.check_file(arguments[1]), 'Invalid JSON format for BGO DAG')

    # Evaluate the model if the last argument is 'evaluate_model'.
    graph_data = None
    if len(arguments) == 3:
        graph_data = utils.try_cast_graph_name_or_characteristics(arguments[2], 'Invalid graph characteristics or name format')

    bgo_dag = prediction(hardware, bgo_dag, graph_data)
    print(json.dumps(bgo_dag, indent=4))
