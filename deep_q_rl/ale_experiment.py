"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging
import numpy as np
import cv2

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
import time
from deep_q_rl.visual_interface import VisualInterface

CROP_OFFSET = 8


class ALEExperiment(object):
    def __init__(self, ale, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng, display_screen, all_actions):
        self.ale = ale
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        if all_actions:
            self.min_action_set = ale.getLegalActionSet()
        else:
            self.min_action_set = ale.getMinimalActionSet()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = ale.getScreenDims()

        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length,
                                       self.height, self.width),
                                      dtype=np.uint8)

        self.terminal_lol = False  # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng
        self.display_screen = display_screen

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """

        self.agent.initialize(self.min_action_set)
        if self.display_screen:
            self.vis = VisualInterface(self.agent.network, self.agent.test_data_set)

        for epoch in range(1, self.num_epochs + 1):
            self.agent.start_epoch(epoch)
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing(epoch)
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False  # Make sure each epoch starts with a reset.
        steps_left = num_steps
        total_steps = num_steps
        episode = 0
        episode_start_time = time.time()
        epoch_avg_reward = 0.0
        t = episode_start_time

        while steps_left > 0:
            prefix = "Testing" if testing else "Training"
            # logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " + str(steps_left))
            _, num_steps, episode_reward = self.run_episode(steps_left, testing)

            epoch_avg_reward = (epoch_avg_reward * episode + episode_reward) / (episode + 1)
            episode += 1

            episode_time = time.time() - t
            t = time.time()
            total_time = t - episode_start_time

            steps_left -= num_steps

            logging.info("{} episode {} of epoch {} completed with reward {} in {:.1f} sec. Total time: {:.1f}. Epoch avg reward: {:.2f}. Steps: {}/{}. Steps/sec: {:.2f}".format(
                    prefix, episode, epoch, episode_reward, episode_time, total_time, epoch_avg_reward, total_steps - steps_left, total_steps, num_steps / episode_time))

    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        if not self.terminal_lol or self.ale.game_over():
            self.ale.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self._act(0)  # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self._act(0)
        self._act(0)

    def _act(self, action):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        reward = self.ale.act(action)
        index = self.buffer_count % self.buffer_length

        self.ale.getScreenGrayscale(self.screen_buffer[index, ...])

        self.buffer_count += 1
        return reward

    def _step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self._act(action)

        return reward

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()

        start_lives = self.ale.lives()

        action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        episode_reward = 0
        terminal = False

        while True:
            reward = self._step(self.min_action_set[action])
            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.ale.lives() < start_lives)
            terminal = self.ale.game_over() or self.terminal_lol
            episode_reward += reward
            num_steps += 1

            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self.get_observation())
            if testing and self.display_screen:
                self.vis.draw(num_steps, self.ale.getScreenRGB())


        return terminal, num_steps, episode_reward

    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
            crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')
