
from maze_interface import MazeInterface
import numpy as np
from numpy.random import random_integers as rand
import matplotlib.pyplot as pyplot
import operator
import maze_actions
import cPickle
import os
import logging

WHITE = 255
PLAYER = 170
TARGET = 85
BLACK = 0

MAZE_PICKLE_FOLDER = 'maze'

class MazeGenerator(MazeInterface):

    def __init__(self, maze_type="maze_empty", maze_size=(21, 21), maze_init=(1, 1), maze_target=(20, 20)):
        super(MazeGenerator, self).__init__()
        self.maze_type = maze_type
        self.height = maze_size[0]
        self.width = maze_size[1]
        self.initial_pos = maze_init
        self.agent_pos = self.initial_pos
        self.target_pos = maze_target

        file_name = maze_type + '.maze'
        file_name = os.path.join(MAZE_PICKLE_FOLDER, file_name)
        if maze_type.endswith("_pre"):
            # load pre defined maze
            handle = open(file_name, 'rb')
            try:
                self.maze = cPickle.load(handle)
            except EOFError:
                logging.error("Read error " + file_name)
            handle.close()
        else:
            # generate and save
            self.maze = self.generate_maze()
            logging.info("Maze generated.")
            with open(file_name, 'wb') as export_file:
                cPickle.dump(self.maze, export_file, -1)
                logging.info("Maze dumped " + file_name)

    def getScreenDims(self):
        return self.width, self.height

    def getScreenGrayscale(self, screen_data=None):
        if screen_data is None:
            screen_data = np.empty((self.height, self.width), dtype=np.uint8)
        screen_data[:] = self.maze
        screen_data[self.agent_pos[1]][self.agent_pos[0]] = PLAYER
        screen_data[self.target_pos[1]][self.target_pos[0]] = TARGET
        return screen_data

    def act(self, action_index):
        action = maze_actions.get_action(action_index).value
        next_pos = tuple(map(operator.add, self.agent_pos, action))
        if self.maze[next_pos[1]][next_pos[0]] != BLACK:
            self.agent_pos = next_pos
        if self.agent_pos == self.target_pos:
            return 100
        else:
            return 0

    def reset_game(self):
        self.agent_pos = self.initial_pos

    def game_over(self):
        if self.agent_pos == self.target_pos:
            return True
        else:
            return False

    def lives(self):
        return 1

    def getMinimalActionSet(self):
        return maze_actions.get_minimal_action_set()

    def getLegalActionSet(self):
        return maze_actions.get_minimal_action_set()

    def getScreenRGB(self):
        gray = self.getScreenGrayscale()
        vis = np.empty((gray.shape[0], gray.shape[1], 3), np.uint8)
        vis[:,:,0] = gray
        vis[:,:,1] = gray
        vis[:,:,2] = gray
        return vis

    def generate_maze(self, complexity=.75, density=.75):

        if self.maze_type == "maze_empty":
            Z = np.ones((self.height, self.width), np.uint8) * WHITE
            Z[0, 0:self.width] = BLACK
            Z[self.height-1, 0:self.width] = BLACK
            Z[0:self.height, 0] = BLACK
            Z[0:self.height, self.width-1] = BLACK

            return Z
        else:
            # Only odd shapes
            shape = ((self.height // 2) * 2 + 1, (self.width // 2) * 2 + 1)
            # Adjust complexity and density relative to maze size
            complexity = int(complexity * (5 * (shape[0] + shape[1])))
            density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
            # Build actual maze
            Z = np.ones(shape, dtype=np.uint8) * WHITE
            # Fill borders
            Z[0, :] = Z[-1, :] = BLACK
            Z[:, 0] = Z[:, -1] = BLACK
            # Make aisles
            for i in range(density):
                x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
                Z[y, x] = BLACK
                for j in range(complexity):
                    neighbours = []
                    if x > 1:
                        neighbours.append((y, x - 2))
                    if x < shape[1] - 2:
                        neighbours.append((y, x + 2))
                    if y > 1:
                        neighbours.append((y - 2, x))
                    if y < shape[0] - 2:
                        neighbours.append((y + 2, x))
                    if len(neighbours):
                        y_, x_ = neighbours[rand(0, len(neighbours) - 1)]
                        if Z[y_, x_] == WHITE:
                            Z[y_, x_] = BLACK
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = BLACK
                            x, y = x_, y_
            return Z


if __name__ == "__main__":
    m = MazeGenerator(50, 50)
    i = 0
    while(not m.game_over() and i < 500):
        action = rand(0,4)
        prev = m.agent_pos
        r = m.act(action)
        next_state = m.agent_pos
        print("{} -- {} --> {} {}".format(prev, maze_actions.get_action(action).name, next_state, r))
        i += 1
    # m.reset_game()
    pyplot.figure(figsize=(10, 5))
    # pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()