# Graph Optimizer Autobench

Autobench can be used to automatically generate benchmarking code and gather performance results from BGO implementations.

The autobench infrastructure expects that BGO's (in the `bgo` directory) are implemented in a two layer structure. At the highest level, an abstract description of the BGO's is formulated in the `config.json` files. Each subdirectory then contains an actual implementation of the BGO.

## Usage Instructions

### Add a new abstract BGO
First, create a new directory in the `bgo` directory. Preferably named after the BGO. Then, create a new file named `config.json` in the new directory. The contents of this file must describe the arguments of the BGO on an abstract level. See the example below:
```JSON
{
    "in_args": [
        {
            "id": "G",
            "kind": "GRAPH"
        },
        {
            "id": "sources",
            "kind": "VECTOR",
            "value": "G.RAND_VERT_VECTOR"
        }
    ],
    "out_args": [
        {
            "id": "centrality",
            "kind": "VECTOR"
        }
    ]
}
```
The arguments must be supplied in-order. Output arguments always come after the input arguments. For a list of accepted argument kinds and values, see the appendix below.

### Add a new implemented BGO
Based on the abstract description of a BGO, you can add multiple implementations. Each implementation is contained a seperate subdirectory. This subdirectory must contain a `.cpp` and `.hpp` file, as well as a Makefile. The `.hpp` file must contain one function definition that alligns with the abstract BGO description. See the example below:
```C++
int bc_lagr(LAGraph_Graph G, CArray<GrB_Index> sources, GrB_Vector *centrality);
```
Check the appendix below for the accepted argument types.

A generic makefile can be found at `autobench/misc/Makefile_generic`. You can create a symbolic link to this file using the following command:
```sh
ln -s ../../../autobench/misc/Makefile_generic Makefile
```

### Configure & run the benchmark
To run the benchmark, a populated `config.json` file is required in the root directory. You can create it by running `autobench/configure`.

You can run a benchmark using `autobench/run`. You may also use one of the options like:
- `--num N` : to specify the number of randomly generated inputs per graph/BGO pair.
- `--data PATH` : to specify one or multiple graphs.
- `--bgos NAME` : to specify one or multiple BGO implementations.

Example:
```sh
autobench/run --num 1 --data data/RO_edges.mtx --bgos bc_lagr
```

### Appendix: Example slurm job
In case you are running the benchmark on a cluster with SLURM installed you may want to create a SLURM job.

Example `autobench.job` file:
```sh
#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1

. env/bin/activate

autobench/run --num 1 --data data/RO_edges.mtx
```
You can run the job using `sbatch autobench.job`.


### Appendix: Known argument types

- `GRAPH`
    - Implementations:
        - `CMatrix<int>`
        - `GrB_Matrix`
        - `LAGraph_Graph`
    - Statistics:
        - `GRAPH.SIZE_VERTS`
        - `GRAPH.SIZE_EDGES`
- `VECTOR`
    - Implementations:
        - `CArray<int>`
        - `CArray<GrB_Index>`
        - `GrB_Vector`
    - Values
        - `GRAPH.RAND_VERT_VECTOR`
    - Statistics:
        - `VECTOR.SIZE`
