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
             2 : "--",
             3 : "-.",
             4 : ":",
             5 : "--",
             6 : "-."}

type_legend_labels = {1 : "8x8",
                 2 : "12x12",
                 3 : "16x12",
                 4 : "21x21"}

size_color_map = {"8" : 1,
                  "08" : 1,
                 "12" : 2,
                 "16" : 3,
                 "21" : 4}

enemy_color_map = {"0" : 1,
                  "1" : 2,
                 "2" : 3,
                 "3" : 4,
                 "4" : 5}

size_legend_labels = {1 : "no wall",
                 2 : "one wall",
                 3 : "two walls"}

# Modify this to do some smoothing...
kernel = np.array([1.] * 1)
kernel = kernel / np.sum(kernel)
#args root folder, 08path, 12path, 16path, output path
root_folder = sys.argv[1]
split_path = os.path.split(sys.argv[2])
file_size_list_08 = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", dtype=str)
file_size_list_12 = np.loadtxt(open(sys.argv[3], "rb"), delimiter=",", dtype=str)
file_size_list_16 = np.loadtxt(open(sys.argv[4], "rb"), delimiter=",", dtype=str)
output_name = sys.argv[5]

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
        plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), color_map[color] + dash_map[color], label = legend_label)
        legend_use_map.append(legend_label)
    else:
        plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), color_map[color] + dash_map[color], label = "")
    if any(results[:, 3] > 99):
        horizontal_line_pos = np.argmax(results[:, 3] > 99) + 1
        plt.axvline(horizontal_line_pos, c=color_map[color], ls=dash_map[color])
    # plt.ylabel('Average score per episode')

def draw_sub_plot(file_size_list):
    i = 1
    try:
        for size_prefix, folder_name, note in file_size_list:
            try:
                maze_wall_val = get_maze_name_val(folder_name)
                draw_plot(folder_name, os.path.join(root_folder, folder_name), None, maze_wall_val+1, "{}".format(size_legend_labels[maze_wall_val+1]))
                print("{},{},{}".format(size_prefix, folder_name, note))
                i += 1
            except:
                i += 1
    except ValueError:
        for size_prefix, folder_name, note, note2 in file_size_list:
            try:
                draw_plot(folder_name, os.path.join(root_folder, folder_name), None, enemy_color_map[note], "{}".format(note))
                i += 1
            except:
                i += 1

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(6, 6))
plt.gca().set_ylim([0, 100])
plt.gca().set_xlim([0, 200])

fig.add_subplot(3,1,1)
plt.gca().set_ylim([0, 100])
# plt.gca().set_xlim([0, 200])
legend_use_map = []
draw_sub_plot(file_size_list_08)
plt.title("8x8 maze")

fig.add_subplot(3,1,2)
# plt.gca().set_xlim([0, 200])
legend_use_map = []
draw_sub_plot(file_size_list_12)
plt.title("12x12 maze")

fig.add_subplot(3,1,3)
plt.gca().set_ylim([0, 100])
plt.gca().set_xlim([0, 200])
legend_use_map = []
draw_sub_plot(file_size_list_16)
plt.title("16x16 maze")

ax[2].set_xlabel('Training epochs')
fig.text(0.04, 0.5, 'Average score per episode', va='center', rotation='vertical')

# plt.subplot(3, 1, 1)
# plt.gca().set_xlim([0, 200])
# draw_sub_plot(file_size_list_08)
# plt.subplot(3, 1, 2)
# plt.gca().set_xlim([0, 200])
# draw_sub_plot(file_size_list_12)
# plt.subplot(3, 1, 3)
# plt.gca().set_xlim([0, 200])
# draw_sub_plot(file_size_list_16)

# plt.title("")
# # plt.show()
plt.legend(loc='lower right')
# fig = plt.gcf()
fig.set_size_inches(9, 12)
fig.savefig("{}.jpg".format(output_name), bbox_inches='tight')
fig.savefig("{}.pdf".format(output_name), format="pdf", bbox_inches='tight')
plt.show()