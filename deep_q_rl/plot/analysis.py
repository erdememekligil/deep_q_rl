
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import sys

# Modify this to do some smoothing...
kernel = np.array([1.] * 4)
kernel = kernel / np.sum(kernel)
root_folder = sys.argv[1]

analysis_map = {}
analysis = np.loadtxt(open("analysis.txt", "rb"), skiprows=1, delimiter="\t", dtype=str)


for line in analysis:
    print(line)
