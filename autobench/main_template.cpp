#include <iostream>
#include <chrono>

#include "GraphBLAS.h"
#include "LAGraph.h"

#include "utils.hpp"

#include "{{ header }}"


int main(int argc, char **argv) {
    char msg[LAGRAPH_MSG_LEN];

    {% for decl in decls %}
    {{ decl }}
    {% endfor %}

    if (argc != {{ names|length + 1 }}) {
        return -1;
    }

    LAGraph_Init(msg);

    {% for init in inits %}
    {{ init }}
    {% endfor %}

    auto start = std::chrono::steady_clock::now();

    {{ method }} (
        {% for name in names %}
        {{ name }} {{ ", " if not loop.last }}
        {% endfor %}
    );

    auto end = std::chrono::steady_clock::now();
    auto runtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << runtime_ns << std::endl;

    {% for free in frees %}
    {{ free }}
    {% endfor %}

    return 0;
}
