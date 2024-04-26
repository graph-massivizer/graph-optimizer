#include <iostream>
#include <chrono>

#include "GraphBLAS.h"
#include "LAGraph.h"

#include "utils.hpp"

#include "{{ header }}"


char msg[LAGRAPH_MSG_LEN];

int main(int argc, char **argv) {
    {% for decl in decls %}
    {{ decl }}
    {% endfor %}

    if (argc != {{ args|length + 1 }}) {
        return -1;
    }

    LAGraph_Init(msg);

    {% for init in inits %}
    {{ init }}
    {% endfor %}

    auto start = std::chrono::steady_clock::now();

    {{ method }} (
        {% for arg in args %}
        {{ arg }} {{ ", " if not loop.last }}
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
