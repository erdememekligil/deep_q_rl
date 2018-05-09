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
# RANDOM_SEED = ['5']
RANDOM_SEED = ['1', '2', '3', '4', '5']

maze_name_mapping = {"maze_empty" : 0,
                    "maze_one_wall" : 1,
                    "maze_two_wall" : 2}

color_map = {1 : "r",
             2 : "g",
             3 : "b",
             4 : "c",
             5 : "m",
             6 : "k",
             7 : "y"}

dash_map = {1 : "-",
             2 : "--",
             3 : "-.",
             4 : ":",
             5 : "-x",
             6 : "-+"}

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
                 "4" : 4,
                 "5" : 5}

size_legend_labels = {1 : "no wall",
                 2 : "one wall",
                 3 : "two walls"}

explanation_map = {"empty": "a",
                   "intermed": "b",
                   "empty+wall" : "c",
    "intermed+wall": "d",
    "empty+wall_hard": "e",
    "intermed+wall_hard": "f"
}

explanation_colors = {"empty": 1,
                   "intermed": 2,
                   "empty+wall" : 3,
    "intermed+wall": 4,
    "empty+wall_hard": 5,
    "intermed+wall_hard": 6
}

# Modify this to do some smoothing...
kernel = np.array([1.] * 1)
kernel = kernel / np.sum(kernel)
#args root folder, txt path
#D:\dev\projects\deep_q_rl\results D:\dev\projects\deep_q_rl\deep_q_rl\plot\pacma_resultsn.txt
root_folder = sys.argv[1]
pacman_txt = np.loadtxt(open(sys.argv[2], "rb"), delimiter=",", dtype=str)

def get_maze_name_val(full_name):
    for m in maze_name_mapping:
        if m in full_name:
            return maze_name_mapping[m]
    return -1

legend_use_map = []

def draw_plot(folder_name, dir, enemy_count, color, legend_label):
    column_index = 3
    if enemy_count == 0:
        column_index = 2
        plt.ylabel('Total score per test epoch')
    else:
        plt.ylabel('Average score per episode')

    try:
        results = np.loadtxt(open("{}/{}".format(dir, "results.csv"), "rb"), delimiter=",", skiprows=1)
    except:
        print("Error with {}".format(folder_name))
        return

    if len(results) == 0:
        print("Zero len {}".format(folder_name))
        return
    plt.xlabel('Test Epochs')
    print folder_name, color, legend_label
    if legend_label not in legend_use_map:
        if legend_label in explanation_map:
            color = explanation_colors[legend_label]
            legend_label = explanation_map[legend_label]
        plt.plot(results[:, 0], np.convolve(results[:, column_index], kernel, mode='same'), color_map[color] + dash_map[color], label = legend_label)
        legend_use_map.append(legend_label)
    else:
        plt.plot(results[:, 0], np.convolve(results[:, column_index], kernel, mode='same'), color_map[color] + dash_map[color], label = "")
    # if any(results[:, 3] > 390):
    #     horizontal_line_pos = np.argmax(results[:, 3] > 390) + 1
    #     plt.axvline(horizontal_line_pos, c=color_map[color], ls=dash_map[color])
    # plt.ylabel('Average score per episode')


i = 1
for enemy_count, folder_name, explanation, note in pacman_txt:
    if not note in RANDOM_SEED:
        continue
    if enemy_count == '0':
        draw_plot(folder_name, os.path.join(root_folder, folder_name), int(enemy_count), i, "{}".format(explanation))
        i += 1

plt.title("")
# plt.show()
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(((8, 4.5)))
fig.savefig("{}_{}.jpg".format(os.path.join(root_folder, "pacman_0enemy"), str.join('-', RANDOM_SEED)), bbox_inches='tight')
fig.savefig("{}_{}.pdf".format(os.path.join(root_folder, "pacman_0enemy"), str.join('-', RANDOM_SEED)), format="pdf", bbox_inches='tight')
# plt.show()
plt.clf()


legend_use_map = []
i = 1
for enemy_count, folder_name, explanation, note in pacman_txt:
    if not note in RANDOM_SEED:
        continue
    if enemy_count == '2':
        draw_plot(folder_name, os.path.join(root_folder, folder_name), int(enemy_count), i, "{}".format(explanation))
        i += 1

plt.title("")
# plt.show()
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(((8, 4.5)))
fig.savefig("{}_{}.jpg".format(os.path.join(root_folder, "pacman_2enemy"), str.join('-', RANDOM_SEED)), bbox_inches='tight')
fig.savefig("{}_{}.pdf".format(os.path.join(root_folder, "pacman_2enemy"), str.join('-', RANDOM_SEED)), format="pdf", bbox_inches='tight')
# plt.show()
plt.clf()

exit()

legend_use_map = []
i = 1
for enemy_count, folder_name, explanation,_ in pacman_txt:
    if enemy_count == '0' and ('intermed+wall_hard' in explanation or 'empty+wall_hard' in explanation):
        draw_plot(folder_name, os.path.join(root_folder, folder_name), int(enemy_count), i, "{}".format(explanation))
        i += 1

plt.title("")
# plt.show()
# plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(((7, 8)))
fig.savefig("{}.jpg".format(os.path.join(root_folder, "pacman_hard_0enemy")), bbox_inches='tight')
fig.savefig("{}.pdf".format(os.path.join(root_folder, "pacman_hard_0enemy")), format="pdf", bbox_inches='tight')
# plt.show()
plt.clf()

legend_use_map = []
i = 1
for enemy_count, folder_name, explanation,_ in pacman_txt:
    if enemy_count == '2' and ('intermed+wall_hard' in explanation or 'empty+wall_hard' in explanation):
        draw_plot(folder_name, os.path.join(root_folder, folder_name), int(enemy_count), i, "{}".format(explanation))
        i += 1

plt.title("")
# plt.show()
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(((7, 8)))
fig.savefig("{}.jpg".format(os.path.join(root_folder, "pacman_hard_2enemy")), bbox_inches='tight')
fig.savefig("{}.pdf".format(os.path.join(root_folder, "pacman_hard_2enemy")), format="pdf", bbox_inches='tight')
# plt.show()
plt.clf()

legend_use_map = []
i = 1
for enemy_count, folder_name, explanation,_ in pacman_txt:
    if enemy_count == '0' and ('intermed+wall_hard' not in explanation and 'empty+wall_hard' not in explanation):
        draw_plot(folder_name, os.path.join(root_folder, folder_name), int(enemy_count), i, "{}".format(explanation))
        i += 1

plt.title("")
# plt.show()
# plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(((7, 8)))
fig.savefig("{}.jpg".format(os.path.join(root_folder, "pacman_rest_0enemy")), bbox_inches='tight')
fig.savefig("{}.pdf".format(os.path.join(root_folder, "pacman_rest_0enemy")), format="pdf", bbox_inches='tight')
# plt.show()
plt.clf()

legend_use_map = []
i = 1
for enemy_count, folder_name, explanation,_ in pacman_txt:
    if enemy_count == '2' and ('intermed+wall_hard' not in explanation and 'empty+wall_hard' not in explanation):
        draw_plot(folder_name, os.path.join(root_folder, folder_name), int(enemy_count), i, "{}".format(explanation))
        i += 1

plt.title("")
# plt.show()
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(((7, 8)))
fig.savefig("{}.jpg".format(os.path.join(root_folder, "pacman_rest_2enemy")), bbox_inches='tight')
fig.savefig("{}.pdf".format(os.path.join(root_folder, "pacman_rest_2enemy")), format="pdf", bbox_inches='tight')
# plt.show()
plt.clf()