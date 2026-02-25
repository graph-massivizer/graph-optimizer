import subprocess
import os
from models.prediction import analytical_prediction, sampling_based_prediction
import models.utils as utils
from copy import deepcopy

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# result = [{'id': 0, 'name': 'pr', 'dependencies': [], 'inputs': [{'source': -1, 'source_id': 'G', 'target_id': 'G'}], 'performances': [{'host': 'H01', 'runtime': {'CPU': {'pr_gap': 88209233.69999999, 'pr_gb': 23438874033.399998, 'pr_openmp': 'No analytical model available...', 'pr_sequential': 79267393545461.4}, 'GPU': {'edgelist': 413722555.4, 'rev_edgelist': 220847930.1, 'rev_struct_edgelist': 224477628.89999998, 'struct_edgelist': 201375548.5, 'vertex_pull': 300451137.7, 'vertex_pull_nodiv': 208546916.89999998, 'vertex_pull_warp': 311398999.09999996, 'vertex_pull_warp_nodiv': 280842281.3, 'vertex_push': 236767864.89999998, 'vertex_push_warp': 251228324.89999998}}, 'energy': {'CPU': {'pr_gap': 2585863.6999999997, 'pr_gb': 592626932.0999999, 'pr_openmp': 'No analytical model available...', 'pr_sequential': 2774358.774091149}, 'GPU': {'edgelist': 7854356.999999999, 'rev_edgelist': 3903224.4999999995, 'rev_struct_edgelist': 4002830.3, 'struct_edgelist': 3329910.5, 'vertex_pull': 5594092.699999999, 'vertex_pull_nodiv': 3759011.1999999997, 'vertex_pull_warp': 5960059.0, 'vertex_pull_warp_nodiv': 5311413.8, 'vertex_push': 4159616.3, 'vertex_push_warp': 5017537.0}}}]}, {'id': 1, 'name': 'find_max', 'dependencies': [0], 'inputs': [{'source': 0, 'source_id': 'PR', 'target_id': 'values'}], 'performances': [{'host': 'H01', 'runtime': {'CPU': {'find_max_ca': 2.8534564647104044, 'find_max_gb': 9831640.7, 'find_max_omp_custom_reduction': 'No analytical model available...', 'find_max_omp_local_reduction': 'No analytical model available...', 'find_max_threads': 'No analytical model available...'}, 'GPU': {'find_max_gpu': 'No analytical model available...'}}, 'energy': {'CPU': {'find_max_ca': 10.02278935619039, 'find_max_gb': 143525.9, 'find_max_omp_custom_reduction': 'No analytical model available...', 'find_max_omp_local_reduction': 'No analytical model available...', 'find_max_threads': 'No analytical model available...'}, 'GPU': {'find_max_gpu': 'No analytical model available...'}}}]}, {'id': 2, 'name': 'bfs', 'dependencies': [0], 'inputs': [{'source': -1, 'source_id': 'G', 'target_id': 'G'}, {'source': -1, 'source_id': 'source', 'target_id': 'source'}], 'performances': [{'host': 'H01', 'runtime': {'CPU': {'bfs_gap': 16576277.2, 'bfs_lagr': 165110808.79999998, 'bfs_naive': 1337710.0004947442}, 'GPU': {'bfs_gpu_naive': 'No analytical model available...', 'edgelist': 507859861.59999996, 'rev_edgelist': 184782590.29999998, 'rev_struct_edgelist': 188951316.39999998, 'struct_edgelist': 196996107.7, 'vertex_pull': 183452335.5, 'vertex_pull_warp': 191874569.6, 'vertex_push': 181686400.0, 'vertex_push_warp': 192149732.6}}, 'energy': {'CPU': {'bfs_gap': 194525.8, 'bfs_lagr': 3641992.9, 'bfs_naive': 0.04681985001731605}, 'GPU': {'bfs_gpu_naive': 'No analytical model available...', 'edgelist': 10687773.6, 'rev_edgelist': 3987547.9, 'rev_struct_edgelist': 3055738.6999999997, 'struct_edgelist': 3281902.4, 'vertex_pull': 3044968.5, 'vertex_pull_warp': 3140835.5999999996, 'vertex_push': 3505584.5999999996, 'vertex_push_warp': 3140831.4}}}]}, {'id': 3, 'name': 'find_path', 'dependencies': [1, 2], 'inputs': [{'source': 2, 'source_id': 'parents', 'target_id': 'parents'}, {'source': -1, 'source_id': 'source', 'target_id': 'start'}, {'source': 1, 'source_id': 'result', 'target_id': 'end'}], 'performances': [{'host': 'H01', 'runtime': {'CPU': {'fp_ca': 0.00013304792451113682, 'fp_gb': 0.00013304792451113682}, 'GPU': {}}, 'energy': {'CPU': {'fp_ca': 4.656677357889789e-12, 'fp_gb': 4.656677357889789e-12}, 'GPU': {}}}]}]

# TODO: different Hosts
def graph_optimizer_e2e(workflow, hardware, graph_properties, heuristic=0, dampening_factor=1):
    """
    Full end-to-end pipeline of the graph optimizer tool.
    
    Args:
        workflow: DAG in JSON format of the workflow of BGOs.
        hardware: JSON description of hardware platform, including microbenchmarks.
        graph_properties: Graph properties in JSON format.
        heuristic: performance/energy importance -> 0=full performance, 1=full energy.
        
    Returns:
        A tuple with as first 2 elements the e2e runtime and energy respecively,
        and as third item a JSON string containing end-to-end runtime and energy predictions,
        as well as a breakdown of the selected optimal BGOs and their runtimes.
    """
    # Get analytical models
    performance_breakdown = deepcopy(workflow)
    predictions = analytical_prediction(hardware, workflow, graph_properties)
    predictions = sampling_based_prediction(predictions, graph_properties, dampening_factor)

    host = predictions[0]["performances"][0]["host"]

    # Select best implementations per bgo based on metric
    e2e_runtime = 0
    e2e_energy = 0
    devices = ['CPU', 'GPU']
    for i, bgo in enumerate(predictions):
        # These tuples are (BGO_name, runtime, energy)
        min_runtime = (None, float('inf'), float('inf'))
        min_energy = (None, float('inf'), float('inf'))
        for j, perf_entry in enumerate(bgo["performances"]):
            for device in devices:
                for implementation in perf_entry['runtime'][device]:
                    name = f"{device}/{implementation}"
                    cur_runtime = predictions[i]["performances"][j]['runtime'][device][implementation]
                    cur_energy = predictions[i]["performances"][j]['energy'][device][implementation]
                    cur_tuple = (name, cur_runtime, cur_energy)
                    if type(cur_runtime) == float and type(cur_energy) == float:
                        min_runtime = min(cur_tuple, min_runtime, key=lambda x: x[1])
                        min_energy = min(cur_tuple, min_energy, key=lambda x: x[2])

        selected_optimum = min_runtime if heuristic == 0 else min_energy

        e2e_runtime += selected_optimum[1]
        e2e_energy += selected_optimum[2]

        performance_breakdown[i]["performances"] = [{"host": host, "implementation": selected_optimum[0], "runtime": selected_optimum[1], "energy": selected_optimum[2]}]


    print("DONE! Optimal BGO implementations have been calculated.")

    return e2e_runtime, e2e_energy, performance_breakdown, predictions