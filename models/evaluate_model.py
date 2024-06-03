#!/usr/bin/env python3
import sys
import json
import utils

def evaluate(model, arguments):
    for key, value in arguments.items():
        model = model.replace(key, str(value))

    return eval(model)

if __name__=='__main__':
    if len(sys.argv) != 3:
        utils.exit_with_error('Usage: ./evaluate_model.py <calibrated_analytical_model> <model_arguments_json>')

    model = utils.check_file(sys.argv[1])
    arguments = utils.try_cast_json_dict(utils.check_file(sys.argv[2]), 'Invalid JSON format for model arguments')

    print(evaluate(model, arguments))
