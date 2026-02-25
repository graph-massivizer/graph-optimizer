#ifndef PR_GAP_HPP
#define PR_GAP_HPP

#include <algorithm>
#include <iostream>
#include <vector>

#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "datastructures.hpp"


int pr_gap(CSR &G, CArray<float> *PR);

#endif