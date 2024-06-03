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


def try_cast_json_dict(string, error):
    try:
        return dict(json.loads(string))
    except:
        exit_with_error(error)


def try_cast_json_list(string, error):
    try:
        return list(json.loads(string))
    except:
        exit_with_error(error)


def try_cast_float(string, error):
    try:
        return float(string)
    except ValueError:
        exit_with_error(error)


def try_cast_float_list(string, error):
    try:
        l = string.strip('][').split(',')

        if len(l) == 0:
            exit_with_error(error)

        return list(map(lambda x: float(x.strip()), l))
    except:
        exit_with_error(error)
