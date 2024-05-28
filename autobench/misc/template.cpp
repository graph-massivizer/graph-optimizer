#include <iostream>
#include <chrono>

#include "GraphBLAS.h"
#include "LAGraph.h"

#include "utils.hpp"

#include "{{ header }}"


int main(int argc, char **argv) {
    char msg[LAGRAPH_MSG_LEN];
    int status;

    {% for decl in decls %}
    {{ decl }}
    {% endfor %}

    if (argc != {{ argc + 1 }}) {
        return -1;
    }

    LAGraph_Init(msg);

    {% for init in inits %}
    {{ init }}
    {% endfor %}

    auto start = std::chrono::steady_clock::now();

    status = {{ method }} (
        {% for name in names %}
        {{ name }} {{ ", " if not loop.last }}
        {% endfor %}
    );

    auto end = std::chrono::steady_clock::now();
    auto runtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Runtime: " << runtime_ns << " ns" << std::endl;
    std::cout << "Status: " << status << std::endl;

    {% for free in frees %}
    {{ free }}
    {% endfor %}

    return status;
}
