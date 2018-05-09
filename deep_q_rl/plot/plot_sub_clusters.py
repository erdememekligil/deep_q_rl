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

c1_dtw = ['assault',
'centipede',
'chopper_command',
'Enduro',
'Fishing_Derby',
'frostbite',
'gravitar',
'krull',
'kung_fu_master',
'private_eye',
'seaquest',
'star_gunner',
'time_pilot',
'Tutankham',
'up_n_down',
'wizard_of_wor']
# c1_dtw = ['freeway', 'venture', 'video_pinball', 'atlantis', 'centipede',
#       'frostbite', 'kung_fu_master', 'chopper_command', 'seaquest', 'wizard_of_wor', 'private_eye', 'assault']

c2_dtw =['alien',
'amidar',
'asterix',
'bank_heist',
'battle_zone',
'Bowling',
'boxing',
'breakout',
'crazy_climber',
'demon_attack',
'double_dunk',
'gopher',
'hero',
'ice_hockey',
'kangaroo',
'ms_pacman',
'name_this_game',
'pong',
'Qbert',
'road_runner',
'robotank',
'space_invaders',
'zaxxon']
# c2_dtw = ['alien', 'ms_pacman', 'bank_heist', 'pong', 'space_invaders', 'boxing', 'hero', 'asterix', 'battle_zone', 'kangaroo',
#      'Qbert', 'breakout']


c3_dtw =['atlantis',
'freeway',
'venture',
'video_pinball']

c1 = ['freeway', 'venture', 'video_pinball', 'atlantis', 'assault',
      'frostbite', 'Tutankham', 'chopper_command', 'seaquest', 'centipede', 'kung_fu_master', 'gravitar', 'star_gunner']
c2= ['road_runner', 'bank_heist', 'space_invaders', 'ms_pacman', 'alien', 'boxing', 'pong']
c3 =['hero', 'amidar', 'kangaroo', 'Qbert', 'gopher']
c4 = ['battle_zone', 'name_this_game', 'zaxxon', 'demon_attack', 'private_eye', 'krull', 'Bowling', 'robotank', 'breakout',
      'double_dunk', 'Fishing_Derby', 'crazy_climber', 'Enduro', 'ice_hockey']

name_map = {'assault': 'Assault',
'centipede': 'Centipede',
'chopper_command': 'Chopper Command',
'Enduro': 'Enduro',
'Fishing_Derby': 'Fishing Derby',
'frostbite': 'Frostbite',
'gravitar': 'Gravitar',
'krull': 'Krull',
'kung_fu_master': 'Kung-Fu Master',
'private_eye': 'Private Eye',
'seaquest': 'Seaquest',
'star_gunner': 'Star Gunner',
'time_pilot': 'Time Pilot',
'Tutankham': 'Tutankham',
'up_n_down': 'Up and Down',
'wizard_of_wor': 'Wizard of Wor',
'alien': 'Alien',
'amidar': 'Amidar',
'asterix': 'Asterix',
'bank_heist': 'Bank Heist',
'battle_zone': 'Battle Zone',
'Bowling': 'Bowling',
'boxing': 'Boxing',
'breakout': 'Breakout',
'crazy_climber': 'Crazy Climber',
'demon_attack': 'Demon Attack',
'double_dunk': 'Double Dunk',
'gopher': 'Gopher',
'hero': 'Hero',
'ice_hockey': 'Ice Hockey',
'kangaroo': 'Kangaroo',
'ms_pacman': 'Ms. Pacman',
'name_this_game': 'Name This Game',
'pong': 'Pong',
'Qbert': 'Q*bert',
'road_runner': 'Road Runner',
'robotank': 'Robotank',
'space_invaders': 'Space Invaders',
'zaxxon': 'Zaxxon',
'riverraid': 'River Raid',
'atlantis': 'Atlantis',
'freeway': 'Freeway',
'venture': 'Venture',
'video_pinball': 'Video Pinball',
'tennis': 'Tennis'}


original_results_map = {}
learning_speeds_map = {}
analysis = np.loadtxt(open("analysis.txt", "rb"), skiprows=1, delimiter="\t", dtype=str)

for _, dname, learning_speed, human_score, paper_score, _ in analysis:
    if learning_speed not in learning_speeds_map:
        learning_speeds_map[learning_speed] = []
    learning_speeds_map[learning_speed].append(dname)
    original_results_map[dname] = float(paper_score.replace(",","."))
original_results_map = {}

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
    data = results[:, 3]
    data = (data - data.min())/ (data.max() - data.min())
    plt.plot(results[:, 0], np.convolve(data, kernel, mode='same'), '-', linewidth=2.0)
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
            game_name = d[0:len(d)-21]
            if game_name not in c1:
                continue
            else:
                count += 1

row_count = math.ceil(math.sqrt(count))

i = 1
for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d)-21]
            if game_name not in c1_dtw:
                continue
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(row_count, row_count, i)
            plt.title(name_map[game_name])
            x_limit = draw_plot(d, os.path.join(subdir, d))

            if game_name in original_results_map:
                original_game_score = original_results_map[game_name]
                plt.plot((0, x_limit), (original_game_score, original_game_score), 'w-')
            i += 1
        except:
            print("Error plotting {}".format(d))


#i = i + 2

plt.gcf().set_size_inches(16, 16)
plt.gcf().subplots_adjust()
#plt.show()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "dtw_atari_cluster1")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "dtw_atari_cluster1")),  bbox_inches='tight')

plt.clf()
i = 1

for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d) - 21]
            if game_name not in c2_dtw:
                continue
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(6, 4, i)
            plt.title(name_map[game_name])
            x_limit = draw_plot(d, os.path.join(subdir, d))

            if game_name in original_results_map:
                original_game_score = original_results_map[game_name]
                plt.plot((0, x_limit), (original_game_score, original_game_score), 'w-')
            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(16, 20)
plt.gcf().subplots_adjust()
#plt.show()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "dtw_atari_cluster2")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "dtw_atari_cluster2")),  bbox_inches='tight')


plt.clf()
i = 1

for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d) - 21]
            if game_name not in c3_dtw:
                continue
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(2, 2, i)
            plt.title(name_map[game_name])
            x_limit = draw_plot(d, os.path.join(subdir, d))

            if game_name in original_results_map:
                original_game_score = original_results_map[game_name]
                plt.plot((0, x_limit), (original_game_score, original_game_score), 'w-')
            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(10, 10)
plt.gcf().subplots_adjust()
#plt.show()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "dtw_atari_cluster3")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "dtw_atari_cluster3")),  bbox_inches='tight')


atari_path = r'D:\GoogleDrive\MSc\tez\atari'
import matplotlib.image as mpimg


plt.clf()
i = 1

for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d) - 21]
            if game_name not in c1_dtw:
                continue

            full_name = ''
            for atari_image_name in os.listdir(atari_path):
                if game_name.lower() in atari_image_name.lower():
                    full_name = atari_image_name
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(row_count, row_count, i)
            plt.title(name_map[game_name])
            plt.imshow(mpimg.imread(atari_path +'/' + full_name))
            gca1 = plt.gca()
            gca1.axes.get_xaxis().set_visible(False)
            gca1.axes.get_yaxis().set_visible(False)

            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(16, 16)
plt.gcf().subplots_adjust()
#plt.show()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "dtw_atari_cluster1_img")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "dtw_atari_cluster1_img")),  bbox_inches='tight')

plt.clf()
i = 1

for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d) - 21]
            if game_name not in c2_dtw:
                continue

            full_name = ''
            for atari_image_name in os.listdir(atari_path):
                if game_name.lower() in atari_image_name.lower():
                    full_name = atari_image_name
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(6, 4, i)
            plt.title(name_map[game_name])
            plt.imshow(mpimg.imread(atari_path +'/' + full_name))
            gca1 = plt.gca()
            gca1.axes.get_xaxis().set_visible(False)
            gca1.axes.get_yaxis().set_visible(False)

            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(16, 20)
plt.gcf().subplots_adjust()
#plt.show()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "dtw_atari_cluster2_img")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "dtw_atari_cluster2_img")),  bbox_inches='tight')

plt.clf()
i = 1

for subdir, dirs, files in os.walk(root_folder):
    for d in dirs:
        try:
            game_name = d[0:len(d) - 21]
            if game_name not in c3_dtw:
                continue

            full_name = ''
            for atari_image_name in os.listdir(atari_path):
                if game_name.lower() in atari_image_name.lower():
                    full_name = atari_image_name
            # if game_name not in learning_speeds_map['1']:
            #     continue
            plt.subplot(2, 2, i)
            plt.title(name_map[game_name])
            plt.imshow(mpimg.imread(atari_path +'/' + full_name))
            gca1 = plt.gca()
            gca1.axes.get_xaxis().set_visible(False)
            gca1.axes.get_yaxis().set_visible(False)

            i += 1
        except:
            print("Error plotting {}".format(d))

plt.gcf().set_size_inches(10, 10)
plt.gcf().subplots_adjust()
#plt.show()
plt.savefig("{}.pdf".format(os.path.join(root_folder, "dtw_atari_cluster3_img")),  bbox_inches='tight')
plt.savefig("{}.jpg".format(os.path.join(root_folder, "dtw_atari_cluster3_img")),  bbox_inches='tight')