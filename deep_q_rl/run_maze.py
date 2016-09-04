#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015

"""
import sys
from ale_agent import NeuralAgent

from q_network import DeepQLearner
import launcher


class Parameters:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    steps_per_epoch = 250000
    epochs = 2000
    steps_per_test = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    base_rom_path = "../roms/"
    # rom = "maze_complex"
    # rom = "maze_empty"
    rom = "maze_one_wall"
    all_actions = False
    frame_skip = 1
    repeat_action_probability = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    update_rule = 'deepmind_rmsprop'
    batch_accumulator = 'sum'
    learning_rate = .00025
    discount = .99
    rms_decay = .95 # (Rho)
    rms_epsilon = .01
    momentum = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    clip_delta = 1.0
    epsilon_start = 1.0
    epsilon_min = .1
    epsilon_decay = 120 * steps_per_epoch
    phi_length = 4
    update_frequency = 4
    replay_memory_size = 1000000
    batch_size = 32
    network_type = "nature_dnn"
    freeze_interval = 10000
    input_scale = 255.
    replay_start_size = 50000
    resize_method = 'scale_nearest'
    resized_width = 84
    resized_height = 84
    death_ends_episode = 'true'
    max_start_nullops = 30
    deterministic = 'true'
    cudnn_deterministic = 'false'
    display_screen = False

    agent_type = NeuralAgent
    qlearner_type = DeepQLearner

    # maze_type = 'maze_complex_01_pre'
    # maze_type = 'maze_empty'
    maze_type = 'maze_one_wall'
    # maze_size = (16, 16)
    # maze_target = (14, 14)
    maze_init = (1, 1)
    #
    # maze_size = (8, 8)
    # maze_target = (6, 6)

    # maze_size = (12, 12)
    # maze_target = (10, 10)

    # maze_size = (10, 10)
    # maze_target = (8, 8)
    # #
    maze_size = (16, 16)
    maze_target = (14, 14)

    # maze_size = (18, 18)
    # maze_target = (16, 16)
    #
    # maze_size = (21, 21)
    # maze_target = (19, 19)
    random_maze_agent = True
    random_maze_target = True

    maze_max_action_count = (maze_size[0]+maze_size[1])*10
    maze_enemy_count = 2
    maze_gate_reward_size = 0
    maze_force_opposite_sides = True

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Parameters, __doc__)
