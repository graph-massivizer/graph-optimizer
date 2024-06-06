# Script to calibrate and test the model for the conversion from GrB Matrix
# to full-size C Matrix.
#
# Model: a * |V|^2 + b * |E| + c

import sys
import random

import pandas as pd
import numpy as np

df = pd.read_csv(sys.argv[1])

# Split the dataframe up into a train- and test-set.
filenames = df['G_in.PATH'].unique()
random.shuffle(filenames)
train_files, test_files =  filenames[len(filenames)//4:], filenames[:len(filenames)//4],
train, test = df[df['G_in.PATH'].isin(train_files)], df[df['G_in.PATH'].isin(test_files)]


# Convert the training data into a matrix A with |V|^2, |E|, 1 and output vector B with runtimes.
train_data = train[['G_in.SIZE_VERTS', 'G_in.SIZE_EDGES', 'runtime_ns']].to_numpy()
A_train = np.column_stack((train_data[:, 0]**2, train_data[:, 1], np.ones(len(train_data))))  # Known values
B_train = train_data[:, 2]  # Measured runtime

# Obtain the coeficients (a,b,c) of the model using the least squares method.
coef, _, _, _ = np.linalg.lstsq(A_train, B_train, rcond=None)


# Also convert the testing data.
test_data = test[['G_in.SIZE_VERTS', 'G_in.SIZE_EDGES', 'runtime_ns']].to_numpy()
A_test = np.column_stack((test_data[:, 0]**2, test_data[:, 1], np.ones(len(test_data))))
B_test = test_data[:, 2]

# Predict the runtimes for the test data.
predicted = np.dot(A_test, coef)

# Compute error metrics.
error = np.abs(B_test - predicted) / B_test * 100
print(f"Mean    Error: {np.mean(error):8.2f}%")
print(f"Median  Error: {np.median(error):8.2f}%")
print(f"Minimum Error: {np.min(error):8.2f}%")
print(f"Maximum Error: {np.max(error):8.2f}%")
