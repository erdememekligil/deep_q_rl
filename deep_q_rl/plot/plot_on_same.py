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

maze_name_mapping = {"maze_empty" : 0,
                    "maze_one_wall" : 1,
                    "maze_two_wall" : 2}

color_map = {1 : "r",
             2 : "g",
             3 : "b",
             4 : "c",
             5 : "m"}

type_legend_labels = {1 : "8x8",
                 2 : "12x12",
                 3 : "16x12",
                 4 : "21x21"}

size_legend_labels = {1 : "no wall",
                 2 : "one wall",
                 3 : "two walls"}


file_size_list = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", dtype=str)

def get_maze_name_val(full_name):
    for m in maze_name_mapping:
        if m in full_name:
            return maze_name_mapping[m]
    return -1

def draw_plot(folder_name, dir, t, color, legend_label):
    try:
        results = np.loadtxt(open("{}/{}".format(dir, "results.csv"), "rb"), delimiter=",", skiprows=1)
    except:
        print("Error with {}".format(folder_name))
        return

    if len(results) == 0:
        print("Zero len {}".format(folder_name))
        return
    plt.xlabel('Training Epochs')
    plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), color_map[color] + '-', label = legend_label)
    if any(results[:, 3] > 99):
        horizontal_line_pos = np.argmax(results[:, 3] > 99) + 1
        plt.axvline(horizontal_line_pos, c="black", ls="dashed")
    plt.ylabel('Average score per episode')

# file_size_mapping = {}
# for a,b in file_size_list:
#     file_size_mapping[b] = a


prev_size_prefix = ""
i = 1
for size_prefix, folder_name, enemy_count in file_size_list:
    draw_plot(folder_name, os.path.join(root_folder, folder_name), None, i, "{} enemy".format(enemy_count))
    i += 1

plt.title("")
# plt.show()
plt.legend(loc='lower right')
fig = plt.gcf()
fig.set_size_inches(18, 18)
fig.savefig("{}.jpg".format(os.path.join(root_folder, "maze_enemy_plot_empty")))
plt.show()
# plt.clf()
#
# sorted_by_1 = file_size_list[np.argsort(file_size_list[:, 1])]
#
# prev_name_prefix = ""
# sub_plot_ind = 1
# i = 1
# for size_prefix, folder_name in sorted_by_1:
#     if prev_name_prefix != get_maze_name_val(folder_name):
#         try:
#             plt.subplot(3, 1, sub_plot_ind)
#             plt.gca().set_xlim([0, 250])
#             sub_plot_ind += 1
#             i = 1
#         except:
#             print("Error saving ".format(folder_name))
#     prev_name_prefix = get_maze_name_val(folder_name)
#     draw_plot(folder_name, os.path.join(root_folder, folder_name), "Wall count: {}".format(prev_name_prefix), i, type_legend_labels[i])
#     i += 1
#
# # plt.show()
#
# plt.legend(loc='lower right')
# fig = plt.gcf()
# fig.set_size_inches(18, 18)
# fig.savefig("{}.jpg".format(os.path.join(root_folder, "maze_enemy_plot_by_type")))