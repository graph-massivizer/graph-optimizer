#!/usr/bin/env python3
import sys


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


def symbolic_model(T_q_front, T_q_pop, T_q_push, T_add, cache_linesizes, mem_access_times, int_size, float_size):
    miss_rates = [1 / (linesize/int_size) for linesize in cache_linesizes]
    G_miss_rates = [1 / (linesize/float_size) for linesize in cache_linesizes]
    T_mem_write = avg_mem_access_time(miss_rates, mem_access_times)
    T_mem_read = T_mem_write
    T_G_mem_read = avg_mem_access_time(G_miss_rates, mem_access_times)
    T_init = f"2*{T_mem_write}"
    T_while_loop = f"{T_q_front + T_q_pop}+n*{T_add + T_G_mem_read + T_mem_read}"
    T_visit_node = 2*mem_access_times[-1] + mem_access_times[0] + T_add + T_q_push
    T_bfs = f"n*{T_init}+n*({T_while_loop})+n*({T_visit_node})"

    return T_bfs


def exit_with_error(message):
    print("Error:", message)
    exit(1)


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


if __name__=="__main__":
    symbolical_model_parameters = 8
    arguments = sys.argv[1:]

    # Check if the correct number of arguments were given.
    if len(arguments) != symbolical_model_parameters:
        exit_with_error(f"Wrong number of arguments. Script needs {symbolical_model_parameters}, but {len(arguments)} were given.")

    # Read the argumends from commandline, and check if they are of the right type.
    T_q_front = try_cast_float(arguments[0], "Expected a number for parameter 'T_q_front' on position 1")
    T_q_pop = try_cast_float(arguments[1], "Expected a number for parameter 'T_q_pop' on position 2")
    T_q_push = try_cast_float(arguments[2], "Expected a number for parameter 'T_q_push' on position 3")
    T_add = try_cast_float(arguments[3], "Expected a number for parameter 'T_add' on position 4")
    cache_linesizes = try_cast_float_list(arguments[4], "Expected a list of numbers for parameter 'cache_linesizes' on position 5")
    mem_access_times = try_cast_float_list(arguments[5], "Expected a list of numbers for parameter 'mem_access_times' on position 6")
    int_size = try_cast_float(arguments[6], "Expected a number for parameter 'int_size' on position 7")
    float_size = try_cast_float(arguments[7], "Expected a number for parameter 'float_size' on position 8")

    # Check if length of mem_access_times is 1 more than cache_linesizes
    if len(mem_access_times) != len(cache_linesizes) + 1:
        exit_with_error("Expected the length of 'mem_access_times' to be 1 more than the length of 'cache_linesizes'")

    # Calculate the symbolic model
    T_bfs = symbolic_model(T_q_front, T_q_pop, T_q_push, T_add, cache_linesizes, mem_access_times, int_size, float_size)
    print(T_bfs)
