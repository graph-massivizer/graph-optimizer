#!/usr/bin/env python3
from bfs import performance_model as bfs_performance_model
from bfs import energy_model as bfs_energy_model
from bc import performance_model as bc_performance_model
from bc import energy_model as bc_energy_model
from find_max import performance_model as find_max_performance_model
from find_max import energy_model as find_max_energy_model
from find_path import performance_model as find_path_performance_model
from find_path import energy_model as find_path_energy_model
from evaluate_model import evaluate
import utils
import sys
import json


if __name__ == '__main__':
    arguments = sys.argv[1:]

    # Error handling.
    if len(arguments) < 2:
        utils.exit_with_error('Usage: ./prediction.py <hardware_information_json> <bgo_dag> [graph_characteristics]')

    # Read the hardware information from the first argument.
    hardware = utils.try_cast_json_dict(utils.check_file(arguments[0]), 'Invalid JSON format for hardware information')
    bgo_dag = utils.try_cast_json_list(utils.check_file(arguments[1]), 'Invalid JSON format for BGO DAG')
    graph_characteristics = None

    # Evaluate the model if the last argument is 'evaluate_model'.
    if len(arguments) == 3:
        graph_characteristics = utils.try_cast_json_dict(utils.check_file(arguments[2]), 'Invalid JSON format for graph characteristics')

    # Loop over bgo dag and do energy and performance predictions for all hardware configurations.
    for i, bgo in enumerate(bgo_dag):
        performance_model = globals().get(f'{bgo["name"]}_performance_model')
        energy_model = globals().get(f'{bgo["name"]}_energy_model')

        # Error handling
        if performance_model is None:
            utils.exit_with_error(f'No performance model found for BGO {bgo["name"]}')
        if energy_model is None:
            utils.exit_with_error(f'No energy model found for BGO {bgo["name"]}')

        bgo_dag[i]['performances'] = []

        # Predict performance and energy, and annotate bgo_dag with the prediction values for each host.
        for host in hardware['hosts']:
            performance = performance_model.predict(host)
            energy = energy_model.predict(host)
            if graph_characteristics is not None:
                performance = evaluate(performance, graph_characteristics)
                energy = evaluate(energy, graph_characteristics)

            bgo_dag[i]['performances'].append({'host': host['name'], 'runtime': performance, 'energy': energy})

    print(json.dumps(bgo_dag, indent=4))