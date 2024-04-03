
#include <iostream>
#include <limits.h>
#include <vector>
#include <string.h>
#include <queue>
#include "../include/BinaryHeap.hpp"
using namespace std;

struct vprop {
	int dist;
	int num_paths;
};

vprop *bwc_dijkstra(vector<int> *G, int source, int N) {
	bool visited[N];
	vprop *props = (vprop*)(malloc(sizeof(vprop) * N));
	BinaryHeap heap(N);
	for (int i = 0; i < N; i++) {
		props[i].dist = INT_MAX;
		props[i].num_paths = 0;
		visited[i] = false;
	}
	props[source].dist = 0;
	props[source].num_paths = 1;
	visited[source] = true;
	heap.decrease_key(source, 0);

	while (!heap.is_empty()) {
		int i = heap.extract_min();
		visited[i] = true;
		for (int j = 0; j < N; j++) {
			if (!visited[j] && G[i][j] != 0) {
				if (props[j].dist > props[i].dist + 1) {
					props[j].dist = props[i].dist + 1;
					props[j].num_paths = props[i].num_paths;
					heap.decrease_key(j, props[j].dist);
				} else if (props[j].dist == props[i].dist + 1) {
					props[j].num_paths += props[i].num_paths;
				}
			}
		}
	}

	return props;
}