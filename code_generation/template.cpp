#include "GraphBLAS.h"

#include "utils.hpp"
#include "gap/builder.h"
#include "gap/command_line.h"

#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <chrono>
#include <cppJoules.h>

{% for header in headers %}
#include "{{ header }}"
{% endfor %}

{% if include_timing %}
// Capture and parse tracker.print_energy()
long long capture_total_energy(EnergyTracker tracker) {
    // Save original stdout
    fflush(stdout);
    int stdout_fd = dup(STDOUT_FILENO);

    // Create pipe
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return -1;
    }

    // Redirect stdout to pipe
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    // Call print_energy() (output goes into pipe)
    tracker.print_energy();

    // Restore stdout
    fflush(stdout);
    dup2(stdout_fd, STDOUT_FILENO);
    close(stdout_fd);

    // Read from pipe
    char buffer[4096];
    ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
    close(pipefd[0]);
    if (n <= 0) return -1;
    buffer[n] = '\0';

    // --- Parse ---
    std::istringstream iss(buffer);
    std::string label;
    double time_val = 0.0;
    long long energy_sum = 0;

    while (iss >> label) {
        if (label == "Time") {
            iss >> time_val; // skip runtime
        } else {
            long long val;
            if (iss >> val) {
                energy_sum += val;
            }
        }
    }
    return energy_sum;
}
{% endif %}

int main() {
    {% if include_timing %}
    EnergyTracker tracker;
    {% endif %}

    {% for decl in decls %}
    {{ decl }}
    {% endfor %}

    {% for init in inits %}
    {{ init }}
    {% endfor %}

    {% if include_timing %}
    auto start = std::chrono::steady_clock::now();
    tracker.start();
    {% endif %}

    {% for bgo_call in bgo_calls %}
    {{ bgo_call }}
    {% endfor %}

    {% if include_timing %}
    tracker.stop();
    auto end = std::chrono::steady_clock::now();
    tracker.calculate_energy();
    auto runtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto total_energy = capture_total_energy(tracker);
    {% endif %}
    
    {% if print_output %}
    {% for output in final_outputs %}
    pretty_print({{ output }});
    {% endfor %}
    {% endif %}
    
    {% if include_timing %}
    printf("Running BGOs took %.6f s, and %.6f MJ\n", (float)runtime_ns/1000000000, (float)total_energy/1000000);
    {% endif %}
}