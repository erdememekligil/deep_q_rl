#! /usr/bin/env python
"""Plots data corresponding to Figure 2 in

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

Usage:

plot_results.py RESULTS_CSV_FILE
"""
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import sys

# Modify this to do some smoothing...
kernel = np.array([1.] * 4)
kernel = kernel / np.sum(kernel)
root_folder = sys.argv[1]

original_results_map = {}
learning_speeds_map = {}
original_results = np.loadtxt(open("atari_original_results.txt", "rb"), delimiter="\t", skiprows=1, dtype=str)
cluster1 = np.loadtxt(open("cluster1.txt", "rb"), delimiter="\t", dtype=str)
analysis = np.loadtxt(open("analysis.txt", "rb"), skiprows=1, delimiter="\t", dtype=str)

for _, dname, learning_speed, human_score, paper_score, _ in analysis:
    if learning_speed not in learning_speeds_map:
        learning_speeds_map[learning_speed] = []
    learning_speeds_map[learning_speed].append(dname)
    original_results_map[dname] = float(paper_score.replace(",","."))

def draw_plot(folder_name, dir):

    try:
        results = np.loadtxt(open("{}/{}".format(dir, "results.csv"), "rb"), delimiter=",", skiprows=1)
    except:
        print("Error with {}".format(folder_name))
        return

    if len(results) == 0:
        print("Zero len {}".format(folder_name))
        return
    # plt.xlabel('Training Epochs')
    plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-')
    gca1 = plt.gca()
    # gca1.set_ylim(bottom=0)
    gca1.axes.get_xaxis().set_visible(False)
    gca1.axes.get_yaxis().set_visible(False)
    # plt.ylabel('Average score per episode')
    # try:
    #     plt.savefig("{}.jpg".format(dir))
    #     plt.savefig("{}.jpg".format(os.path.join(dir, folder_name)))
    # except:
    #     print("Error saving {}".format(folder_name))
    print(folder_name)
    return results.shape[0]

count = 0
for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        count += 1

row_count = math.ceil(math.sqrt(count))

i = 1
for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d)-21]
            # if game_name not in cluster1:
            #     continue
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(row_count, row_count, i)
            plt.title(game_name)
            x_limit = draw_plot(d, os.path.join(subdir, d))

            if game_name in original_results_map:
                original_game_score = original_results_map[game_name]
                plt.plot((0, x_limit), (original_game_score, original_game_score), 'w-')
            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(22, 22)
plt.gcf().subplots_adjust()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "all_atari")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "all_atari")),  bbox_inches='tight')
# plt.show()

plt.clf()
i = 1
for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d)-21]
            if game_name not in cluster1:
                continue
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(row_count, row_count, i)
            plt.title(game_name)
            x_limit = draw_plot(d, os.path.join(subdir, d))

            if game_name in original_results_map:
                original_game_score = original_results_map[game_name]
                plt.plot((0, x_limit), (original_game_score, original_game_score), 'w-')
            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(22, 22)
plt.gcf().subplots_adjust()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "all_atari_cluster1")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "all_atari_cluster1")),  bbox_inches='tight')


for l in range(1, 5):
    plt.clf()
    i = 1
    for subdir, dirs, files in os.walk(root_folder):
        for d in dirs:
            try:
                game_name = d[0:len(d)-21]
                if game_name not in learning_speeds_map[str(l)]:
                    continue
                plt.subplot(row_count, row_count, i)
                plt.title(game_name)
                x_limit = draw_plot(d, os.path.join(subdir, d))

                if game_name in original_results_map:
                    original_game_score = original_results_map[game_name]
                    plt.plot((0, x_limit), (original_game_score, original_game_score), 'w-')
                i += 1
            except:
                print("Error plotting {}".format(d))

    plt.gcf().set_size_inches(22, 22)
    plt.gcf().subplots_adjust()
    plt.savefig("{}.pdf".format(os.path.join(root_folder, "all_atari_learning_speed" + str(l))),  bbox_inches='tight')
    plt.savefig("{}.jpg".format(os.path.join(root_folder, "all_atari_learning_speed" + str(l))),  bbox_inches='tight')
