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

maze_name_mapping = {"maze_empty" : 0,
                    "maze_one_wall" : 1,
                    "maze_two_wall" : 2}

color_map = {1 : "r",
             2 : "g",
             3 : "b",
             4 : "c",
             5 : "m",
             6 : "p"}

dash_map = {1 : "-",
             2 : "-",
             3 : "-",
             4 : "-",
             5 : "-",
             6 : "-"}

type_legend_labels = {1 : "8x8",
                 2 : "12x12",
                 3 : "16x12",
                 4 : "21x21"}

size_color_map = {"8" : 1,
                  "08" : 1,
                 "12" : 2,
                 "16" : 3,
                 "21" : 4}

enemy_color_map = {"0" : 6,
                  "1" : 1,
                 "2" : 2,
                 "3" : 3,
                 "4" : 4}

size_legend_labels = {1 : "no wall",
                 2 : "one wall",
                 3 : "two walls"}

# Modify this to do some smoothing...
kernel = np.array([1.] * 1)
kernel = kernel / np.sum(kernel)
#args root folder, txt path
root_folder = sys.argv[1]
pacman_txt = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", dtype=str)

def get_maze_name_val(full_name):
    for m in maze_name_mapping:
        if m in full_name:
            return maze_name_mapping[m]
    return -1

legend_use_map = []

def draw_plot(folder_name, dir, t, color, legend_label):
    try:
        results = np.loadtxt(open("{}/{}".format(dir, "results.csv"), "rb"), delimiter=",", skiprows=1)
    except:
        print("Error with {}".format(folder_name))
        return

    if len(results) == 0:
        print("Zero len {}".format(folder_name))
        return
    # plt.xlabel('Training Epochs')
    print folder_name, color, legend_label
    if legend_label not in legend_use_map:
        plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), color_map[color] + dash_map[color], label = legend_label)
        legend_use_map.append(legend_label)
    else:
        plt.plot(results[:, 0], np.convolve(results[:, 2], kernel, mode='same'), color_map[color] + dash_map[color], label = "")
    # if any(results[:, 3] > 390):
    #     horizontal_line_pos = np.argmax(results[:, 3] > 390) + 1
    #     plt.axvline(horizontal_line_pos, c=color_map[color], ls=dash_map[color])
    # plt.ylabel('Average score per episode')


prev_size_prefix = ""
i = 1
for enemy_count, folder_name, explanation,_ in pacman_txt:
    if enemy_count == '2':
        continue
    draw_plot(folder_name, os.path.join(root_folder, folder_name), None, i, "{}".format(explanation))
    i += 1

plt.title("")
# plt.show()
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(9, 12)
fig.savefig("{}.jpg".format(os.path.join(root_folder, "pacman_0enemy")), bbox_inches='tight')
fig.savefig("{}.pdf".format(os.path.join(root_folder, "pacman_0enemy")), format="pdf", bbox_inches='tight')
plt.show()
# plt.clf()