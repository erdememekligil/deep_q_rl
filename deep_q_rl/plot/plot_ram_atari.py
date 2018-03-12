
import os

import numpy as np
import matplotlib.pyplot as plt
import sys

kernel = np.array([1.] * 9)
kernel = kernel / np.sum(kernel)
results = np.loadtxt(open(r"D:\dev\projects\ale-ram-learner\results\pong_1.csv", "rb"), delimiter="\t", skiprows=1)

plt.figure()
plt.plot(results[:,0], np.convolve(results[:, 2], kernel, mode='same'))


kernel = np.array([1.] * 19)
kernel = kernel / np.sum(kernel)
results = np.loadtxt(open(r"D:\dev\projects\ale-ram-learner\results\box_1.csv", "rb"), delimiter="\t", skiprows=1)

plt.figure()
plt.plot(results[:,0], np.convolve(results[:, 2], kernel, mode='same'))


kernel = np.array([1.] * 25)
kernel = kernel / np.sum(kernel)
results = np.loadtxt(open(r"D:\dev\projects\ale-ram-learner\results\breakout_1.csv", "rb"), delimiter="\t", skiprows=1)

plt.figure()
plt.plot(results[:,0], np.convolve(results[:, 2] + 40, kernel, mode='same'))