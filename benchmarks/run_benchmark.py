#
# HOW TO USE THIS SCRIPT:
#
# Run the script with three arguments.
# The first argument is the path to the benchmark executable.
# This executable should take three argument, the filename of the input graph, the directory of the output file, and an output csv file name.
# The second argument is the path to the directory containing the input graphs.
# The final argument is the path to the output directory to which the results will be written.
#

import os
import sys
import subprocess

print("test")
print(sys.argv[2])

for filename in os.listdir(sys.argv[2]):
    print(filename)
    if filename.endswith(".mtx"):
        print("=====Running benchmark with input file: " + filename + "=====")
        output = subprocess.check_output([sys.argv[1], sys.argv[2] + filename, sys.argv[3] + filename[:-4]])
