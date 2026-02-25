#ifndef BC_GAP_HPP
#define BC_GAP_HPP

#include <functional>
#include <iostream>
#include <vector>

#include "gap/bitmap.h"
#include "gap/builder.h"
#include "gap/command_line.h"
#include "gap/graph.h"
#include "gap/platform_atomics.h"
#include "gap/pvector.h"
#include "gap/sliding_queue.h"
#include "gap/timer.h"
#include "datastructures.hpp"


int bc_gap(CSR &G, CArray<int> sources, CArray<int> *centrality);

#endif