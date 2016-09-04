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
root_folder = sys.argv[1]


def draw_plot(folder_name, dir):

    try:
        results = np.loadtxt(open("{}/{}".format(dir, "results.csv"), "rb"), delimiter=",", skiprows=1)
    except:
        print("Error with {}".format(folder_name))
        return

    if len(results) == 0:
        print("Zero len {}".format(folder_name))
        return
    plt.subplot(1, 2, 1)
    plt.xlabel('Training Epochs')
    plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-')
    plt.ylabel('Average score per episode')
    plt.subplot(1, 2, 2)
    plt.xlabel('Training Epochs')
    plt.ylabel('Total score per episode')
    plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), '-')
    try:
        plt.savefig("{}.jpg".format(dir))
        plt.savefig("{}.jpg".format(os.path.join(dir, folder_name)))
    except:
        print("Error saving {}".format(folder_name))
    # plt.show()
    plt.clf()
    print(folder_name)




for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            draw_plot(d, os.path.join(subdir, d))
        except:
            print("Error plotting {}".format(d))
