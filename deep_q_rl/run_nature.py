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
    # rom = 'Qbert.bin'
    rom = 'Bowling.bin'
    # rom = 'atlantis.bin'
    # rom = 'Fishing_Derby.bin'
    # rom = 'space_invaders.bin'
    # rom = 'star_gunner.bin'
    # rom = 'name_this_game.bin'
    # rom = 'video_pinball.bin'
    # rom = 'Tutankham.bin'
    # rom = 'ms_pacman.bin'
    # rom = 'riverraid.bin'
    # rom = 'gopher.bin'
    # rom = 'crazy_climber.bin'
    # rom = 'ice_hockey.bin'
    # rom = 'robotank.bin'
    # rom = 'pong.bin'
    # rom = 'asterix.bin'
    # rom = 'beam_rider.bin'
    # rom = 'Asteroids.bin'
    # rom = 'frostbite.bin'
    # rom = 'jamesbond.bin'
    # rom = 'private_eye.bin'
    # rom = 'assault.bin'
    # rom = 'kung_fu_master.bin'
    # rom = 'krull.bin'
    # rom = 'tennis.bin'
    # rom = 'alien.bin'
    # rom = 'road_runner.bin'
    # rom = 'kangaroo.bin'
    # rom = 'up_n_down.bin'
    # rom = 'zaxxon.bin'
    # rom = 'wizard_of_wor.bin'
    # rom = 'venture.bin'
    # rom = 'time_pilot.bin'
    # rom = 'freeway.bin'
    # rom = 'double_dunk.bin'
    # rom = 'demon_attack.bin'
    # rom = 'hero.bin'
    # rom = 'battle_zone.bin'
    # rom = 'centipede.bin'
    # rom = 'bank_heist.bin'
    # rom = 'chopper_command.bin'
    # rom = 'gravitar.bin'
    # rom = 'amidar.bin'
    # rom = 'montezuma_revenge.bin'
    # rom = 'Enduro.bin'
    # rom = 'boxing.bin'
    all_actions = False
    frame_skip = 4
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
    epsilon_decay = 250000 * 4
    phi_length = 4
    update_frequency = 4
    replay_memory_size = 1000000
    batch_size = 32
    network_type = "nature_dnn"
    freeze_interval = 10000
    input_scale = 255.
    replay_start_size = 50000
    resize_method = 'scale'
    resized_width = 84
    resized_height = 84
    death_ends_episode = 'true'
    max_start_nullops = 30
    deterministic = 'true'
    cudnn_deterministic = 'false'
    display_screen = False
    random_seed = 2

    agent_type = NeuralAgent
    qlearner_type = DeepQLearner

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Parameters, __doc__)
