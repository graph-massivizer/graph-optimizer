#include <vector>
#include <queue>
#include "bfs.hpp"

void bfs(int *level, int *parent, std::vector<float> *G, int src, int num_nodes) {
    // Initialize level and parent arrays
    for (int i = 0; i < num_nodes; ++i) {
        level[i] = -1;  // -1 represents an unreachable node
        parent[i] = -1; // -1 represents no parent
    }

    // Create a queue for BFS
    std::queue<int> queue;

    // Mark the source node as visited and enqueue it
    level[src] = 0;
    queue.push(src);

    while (!queue.empty()) {
        // Dequeue a vertex from queue and print it
        int u = queue.front();
        queue.pop();

        // Get all adjacent vertices of the dequeued vertex u
        // If an adjacent vertex has not been visited, then mark it visited and enqueue it
        for (int v = 0; v < num_nodes; ++v) {
            if (G[u][v] > 0 && level[v] == -1) {
                parent[v] = u;
                level[v] = level[u] + 1;
                queue.push(v);
            }
        }
    }
}