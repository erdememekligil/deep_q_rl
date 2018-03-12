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

# RANDOM_SEED = ['1']
# RANDOM_SEED = ['2']
# RANDOM_SEED = ['3']
# RANDOM_SEED = ['4']
RANDOM_SEED = ['5']
# RANDOM_SEED = ['1', '2', '3', '4', '5']

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
             5 : "m",
             6 : "p"}

dash_map = {1 : "-",
             2 : "--",
             3 : "-.",
             4 : ":",
             5 : "-x",
             6 : "-o"}

type_legend_labels = {'08' : "8x8",
                 '12' : "12x12",
                 '16' : "16x12",
                 '21' : "21x21"}

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

#D:\dev\projects\deep_q_rl\deep_q_rl\results D:\dev\projects\deep_q_rl\deep_q_rl\plot\plot_empty_5_run.txt maze_plot_empty

split_path = os.path.split(sys.argv[2])
file_size_list = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", dtype=str)
output_name = sys.argv[3]

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
    plt.xlabel('Test Epochs')
    if legend_label not in legend_use_map:
        plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), color_map[color] + dash_map[color], label = legend_label)
        legend_use_map.append(legend_label)
    else:
        plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), color_map[color] + dash_map[color], label = "")
    # if any(results[:, 3] > 99):
    #     horizontal_line_pos = np.argmax(results[:, 3] > 99) + 1
    #     plt.axvline(horizontal_line_pos, c=color_map[color], ls="dashed")
    plt.ylabel('Average score per episode')


prev_size_prefix = ""
i = 1
try:
    for size_prefix, folder_name, note in file_size_list:
        try:

            if not note in RANDOM_SEED:
                continue
            draw_plot(folder_name, os.path.join(root_folder, folder_name), None, size_color_map[size_prefix], "{}".format(type_legend_labels[size_prefix]))
            print("{},{},{}".format(size_prefix, folder_name, note))
            i += 1
        except:
            i += 1
except ValueError:
    for size_prefix, folder_name, note, note2 in file_size_list:
        try:

            if not note in RANDOM_SEED:
                continue
            draw_plot(folder_name, os.path.join(root_folder, folder_name), None, enemy_color_map[note], "{}".format(note))
            i += 1
        except:
            i += 1

plt.title("")
# plt.show()
plt.legend(loc='lower right')
fig = plt.gcf()
fig.set_size_inches(7, 4)
fig.savefig("graphics/{}_{}.jpg".format(output_name, str.join('-', RANDOM_SEED)), bbox_inches='tight')
fig.savefig("graphics/{}_{}.pdf".format(output_name, str.join('-', RANDOM_SEED)), format="pdf", bbox_inches='tight')
# fig.savefig("{}.jpg".format(os.path.join(root_folder, split_path[split_path.__len__() - 1])), bbox_inches='tight')
# fig.savefig("{}.pdf".format(os.path.join(root_folder, split_path[split_path.__len__() - 1])), format="pdf", bbox_inches='tight')
# print("Saved {}.pdf".format(os.path.join(root_folder, split_path[split_path.__len__() - 1])))
# plt.show()