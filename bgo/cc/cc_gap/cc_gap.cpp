// Copyright (c) 2018, The Hebrew University of Jerusalem (HUJI, A. Barak)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "cc_gap.hpp"


/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Authors: Michael Sutton, Scott Beamer

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Afforest subgraph sampling algorithm [1],
which restructures and extends the Shiloach-Vishkin algorithm [2].

[1] Michael Sutton, Tal Ben-Nun, and Amnon Barak. "Optimizing Parallel 
    Graph Connectivity Computation via Subgraph Sampling" Symposium on 
    Parallel and Distributed Processing, IPDPS 2018.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.
*/


using namespace std;

// Place nodes u and v in same component of lower component ID
void Link(NodeID u, NodeID v, pvector<NodeID>& comp) {
  NodeID p1 = comp[u];
  NodeID p2 = comp[v];
  while (p1 != p2) {
    NodeID high = p1 > p2 ? p1 : p2;
    NodeID low = p1 + (p2 - high);
    NodeID p_high = comp[high];
    // Was already 'low' or succeeded in writing 'low'
    if ((p_high == low) ||
        (p_high == high && compare_and_swap(comp[high], high, low)))
      break;
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}


// Reduce depth of tree for each component to 1 by crawling up parents
void Compress(const Graph &g, pvector<NodeID>& comp) {
  #pragma omp parallel for schedule(dynamic, 16384)
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    while (comp[n] != comp[comp[n]]) {
      comp[n] = comp[comp[n]];
    }
  }
}


NodeID SampleFrequentElement(const pvector<NodeID>& comp, int64_t num_samples = 1024) {
  std::unordered_map<NodeID, int> sample_counts(32);
  using kvp_type = std::unordered_map<NodeID, int>::value_type;
  // Sample elements from 'comp'
  std::mt19937 gen;
  std::uniform_int_distribution<NodeID> distribution(0, comp.size() - 1);
  for (NodeID i = 0; i < num_samples; i++) {
    NodeID n = distribution(gen);
    sample_counts[comp[n]]++;
  }
  // Find most frequent element in samples (estimate of most frequent overall)
  auto most_frequent = std::max_element(
    sample_counts.begin(), sample_counts.end(),
    [](const kvp_type& a, const kvp_type& b) { return a.second < b.second; });
  float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
  return most_frequent->first;
}


pvector<NodeID> Afforest(const Graph &g, int32_t neighbor_rounds = 2) {
  pvector<NodeID> comp(g.num_nodes());

  // Initialize each node to a single-node self-pointing tree
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    comp[n] = n;

  // Process a sparse sampled subgraph first for approximating components.
  // Sample by processing a fixed number of neighbors for each node (see paper)
  for (int r = 0; r < neighbor_rounds; ++r) {
  #pragma omp parallel for schedule(dynamic,16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u, r)) {
        // Link at most one time if neighbor available at offset r
        Link(u, v, comp);
        break;
      }
    }
    Compress(g, comp);
  }

  // Sample 'comp' to find the most frequent element -- due to prior
  // compression, this value represents the largest intermediate component
  NodeID c = SampleFrequentElement(comp);

  // Final 'link' phase over remaining edges (excluding the largest component)
  if (!g.directed()) {
    #pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      // Skip processing nodes in the largest component
      if (comp[u] == c)
        continue;
      // Skip over part of neighborhood (determined by neighbor_rounds)
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
        Link(u, v, comp);
      }
    }
  } else {
    #pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      if (comp[u] == c)
        continue;
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
        Link(u, v, comp);
      }
      // To support directed graphs, process reverse graph completely
      for (NodeID v : g.in_neigh(u)) {
        Link(u, v, comp);
      }
    }
  }
  // Finally, 'compress' for final convergence
  Compress(g, comp);
  return comp;
}

int cc_gap(const CCGraph &G, CArray<int> *components) {
    pvector<NodeID> result = Afforest(G);
    components->init(result.size());
    for (unsigned int n=0; n < result.size(); n++) {
        components->data[n] = result[n];
    }
    cout << endl;
    return 0;
}

