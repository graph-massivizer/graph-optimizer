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


#include "/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/pr/GPU/vertex_pull/vertex_pull.hpp"

#include "/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_max/GPU/find_max_gpu/find_max_gpu.hpp"

#include "/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/bfs/GPU/vertex_push/vertex_push.hpp"

#include "/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_path/CPU/fp_ca/fp_ca.hpp"



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


int main() {
    
    EnergyTracker tracker;
    

    
    Reader<int32_t> r = Reader<int32_t>("data/test_matrix_fully_connected.mtx"); bool needs_weights = false; pvector<SGEdge> el = r.ReadFile(needs_weights); BuilderBase<int32_t> b = BuilderBase<int32_t>(); CSRGraph<int32_t> arg_G;
    
    int arg_source = (int) atoi("1");
    
    CArray<float> arg_PR;
    
    int arg_result;
    
    CArray<int32_t> arg_levels;
    
    CArray<int> arg_parents;
    
    std::vector<int> arg_path;
    

    
    arg_G = b.MakeGraphFromEL(el);
    

    
    auto start = std::chrono::steady_clock::now();
    tracker.start();
    

    
    pr_vertex_pull(arg_G,&arg_PR);
    
    find_max_gpu(arg_PR,&arg_result);
    
    bfs_vertex_push(arg_G,arg_source,&arg_levels,&arg_parents);
    
    find_path(arg_parents,arg_result,arg_source,arg_path);
    

    
    tracker.stop();
    auto end = std::chrono::steady_clock::now();
    tracker.calculate_energy();
    auto runtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto total_energy = capture_total_energy(tracker);
    
    
    
    pretty_print(arg_path);
    
    
    
    printf("Running BGOs took %lds, and %lldMJ\n", runtime_ns/1000000000, total_energy/1000000);
    
}