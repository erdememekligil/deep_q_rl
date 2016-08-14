#! /usr/bin/env python
"""Plots data corresponding to Figure 2 in

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

Usage:

plot_results.py RESULTS_CSV_FILE
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import sys

# Modify this to do some smoothing...
kernel = np.array([1.] * 1)
kernel = kernel / np.sum(kernel)
csv_file = sys.argv[1]
spl = os.path.split(csv_file)
folder_name = os.path.split(spl[0])[1]

results = np.loadtxt(open(csv_file, "rb"), delimiter=",", skiprows=1)
# plt.subplot(1, 2, 1)

if folder_name.startswith("maze"):
    plt.ylabel('Total score per episode')
    plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), '-')
else:
    plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-')
    plt.ylabel('Average score per episode')
plt.xlabel('Training Epochs')
# plt.ylabel('Total s3core per episode')
# plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), '-')
#plt.ylim([0, 250])
# plt.subplot(1, 2, 2)
# plt.plot(results[:, 0], results[:, 4], '-')
# plt.xlabel('Training Epochs')
# plt.ylabel('Average action value')
#plt.ylim([0, 4])

plt.subplot(1, 2, 1)
plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-')
plt.ylabel('Average score per episode')
plt.subplot(1, 2, 2)
plt.ylabel('Total score per episode')
plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), '-')
# plt.subplot(1, 3, 3)
# plt.ylabel('Average action value')
# plt.plot(results[:, 0], results[:, 4], '-')
plt.savefig("{}.jpg".format(os.path.join(spl[0], folder_name)))
plt.show()
