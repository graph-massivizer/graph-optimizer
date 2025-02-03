#include <iostream>
#include <chrono>

#include "GraphBLAS.h"
#include "LAGraph.h"

#include "utils.hpp"
#include "gap/builder.h"
#include "command_line.h"

#include "{{ header }}"


int main(int argc, char **argv) {
    char msg[LAGRAPH_MSG_LEN];
    LAGraph_Init(msg);

    if (argc < {{ argc + 1 }} || argc > {{ argc + 2 }}) {
        std::cerr << "Invalid number of arguments!" << std::endl;
        return -1;
    }

    int runs = 1;
    if (argc == {{ argc + 2 }}) {
        runs = atoi(argv[{{ argc + 1 }}]);
    }

    std::cout << "run,status,runtime_ns" << std::endl;

    {% for decl in decls %}
    {{ decl }}
    {% endfor %}

    {% for init in inits %}
    {{ init }}
    {% endfor %}

    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::steady_clock::now();

        int status = {{ method }} (
            {% for name in names %}
            {{ name }} {{ ", " if not loop.last }}
            {% endfor %}
        );

        auto end = std::chrono::steady_clock::now();
        auto runtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << run << "," << status << "," << runtime_ns << std::endl;

        if (run == 0) {
            {% for save in saves %}
            {{ save }}
            {% endfor %}
        }

        {% for free in frees %}
        {{ free }}
        {% endfor %}

        if (status != 0) { return status; }
    }

    return 0;
}
