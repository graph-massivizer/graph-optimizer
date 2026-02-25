#ifndef CC_GAP_HPP
#define CC_GAP_HPP

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "datastructures.hpp"

int cc_gap(CSR &G, CArray<int> *components);

#endif