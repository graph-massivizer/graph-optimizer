#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(r"C:\Users\BartD\Documents\PhD\shared_VM\graph-optimizer"))
import models.utils as utils

def evaluate(model, arguments):
    # sort arguments based on key length
    sorted_arguments = sorted(arguments.items(), key=lambda item: len(item[0]), reverse=True)

    for key, value in sorted_arguments:
        model = model.replace(key, str(value))

    return eval(model)

if __name__=='__main__':
    if len(sys.argv) != 3:
        utils.exit_with_error('Usage: ./evaluate_model.py <calibrated_analytical_model> <model_arguments_json>')

    model = utils.check_file(sys.argv[1])
    arguments = utils.try_cast_json_dict(utils.check_file(sys.argv[2]), 'Invalid JSON format for model arguments')

    print(evaluate(model, arguments))
