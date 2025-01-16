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


def avg_mem_access_time(miss_rates, mem_access_times):
    time = 0
    final_component = 1
    for i in range(len(miss_rates)):
        summand = 1
        for j in range(i):
            summand *= miss_rates[j]
        summand *= (1-miss_rates[i])*mem_access_times[i]
        time += summand
        final_component *= miss_rates[i]

    time += final_component*mem_access_times[-1]

    return time

def T_malloc(n):
    return 0