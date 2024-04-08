#!/usr/bin/env python3
import sys
import json


def exit_with_error(message):
    print("Error:", message)
    exit(1)


def check_file(string):
    try:
        with open(string, 'r') as f:
            return f.read()
    except:
        return string


def try_cast_json(string, error):
    try:
        return dict(json.loads(string))
    except:
        exit_with_error(error)


if __name__=='__main__':
    if len(sys.argv) != 3:
        exit_with_error('Usage: ./evaluate_model.py <calibrated_analytical_model> <model_arguments_json>')

    model = check_file(sys.argv[1])
    arguments = try_cast_json(check_file(sys.argv[2]), 'Invalid JSON format for model arguments')

    for key, value in arguments.items():
        model = model.replace(key, str(value))

    print(eval(model))
