
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
ENEMY = 127
TARGET = 85
BLACK = 0
WALL = WHITE
EMPTY_SPACE = BLACK

MAZE_PICKLE_FOLDER = 'maze'


class MazeGenerator(MazeInterface):
    
    def __init__(self, maze_type="maze_empty", maze_size=(21, 21), maze_init=(1, 1), maze_target=(19, 19),
                 random_maze_agent=False, random_maze_target=False, max_action_count=400, enemy_count=0):
        super(MazeGenerator, self).__init__()
        self.maze_type = maze_type
        self.height = maze_size[0]
        self.width = maze_size[1]
        self.initial_pos = maze_init
        self.agent_pos = self.initial_pos
        self.target_pos = maze_target
        self.random_maze_target = random_maze_target
        self.random_maze_agent = random_maze_agent
        self.action_count = 0
        self.max_action_count = max_action_count
        self.enemy_count = enemy_count
        self.enemies = []
        self.gates = []

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
            try:
                with open(file_name, 'wb') as export_file:
                    cPickle.dump(self.maze, export_file, -1)
                    logging.info("Maze dumped " + file_name)
            except Exception as e:
                logging.error("Maze dump err {}".format(e))

    def getScreenDims(self):
        return self.width, self.height

    def getScreenGrayscale(self, screen_data=None):
        if screen_data is None:
            screen_data = np.empty((self.height, self.width), dtype=np.uint8)
        screen_data[:] = self.maze
        screen_data[self.agent_pos[1]][self.agent_pos[0]] = PLAYER
        screen_data[self.target_pos[1]][self.target_pos[0]] = TARGET
        for enemy in self.enemies:
            screen_data[enemy[1]][enemy[0]] = ENEMY
        return screen_data

    def act(self, action_index):
        action = maze_actions.get_action(action_index).value
        next_pos = tuple(map(operator.add, self.agent_pos, action))
        if self.maze[next_pos[1]][next_pos[0]] != WALL:
            self.agent_pos = next_pos

        self.action_count += 1
        if self.agent_pos == self.target_pos:
            return 100
        else:
            return 0

    def reset_game(self):
        self.maze = self.generate_maze()

        if self.random_maze_agent:
            self.agent_pos = self.generate_random_position_without_wall()
        else:
            self.agent_pos = self.initial_pos

        if self.random_maze_target:
            self.target_pos = self.generate_random_position_without_wall()
            while self.target_pos == self.agent_pos:
                self.target_pos = self.generate_random_position_without_wall()

        self.enemies = []
        for i in range(0, self.enemy_count):
            enemy = self.generate_random_position_without_wall()
            while enemy == self.agent_pos or enemy == self.target_pos or enemy in self.enemies or self.collides_with_gate(enemy):
                enemy = self.generate_random_position_without_wall()
            self.enemies.append(enemy)

        self.action_count = 0

    def collides_with_gate(self, pos):
        collides = pos in self.gates
        for g in self.gates:
            temp = (g[0]-1, g[1])
            collides = collides or temp == pos
            temp = (g[0]+1, g[1])
            collides = collides or temp == pos
            temp = (g[0], g[1]-1)
            collides = collides or temp == pos
            temp = (g[0], g[1]+1)
            collides = collides or temp == pos
        return collides

    def generate_random_position_without_wall(self):
        random_pos = self.generate_random_position()
        while self.maze[random_pos[1]][random_pos[0]] != EMPTY_SPACE:
            random_pos = self.generate_random_position()
        return random_pos

    def generate_random_position(self):
        x = np.random.randint(1, self.width - 1) # -1 for the wall
        y = np.random.randint(1, self.height - 1) # -1 for the watch
        return x, y

    def game_over(self):
        if self.agent_pos == self.target_pos:
            return True
        elif self.action_count >= self.max_action_count:
            return True
        elif self.agent_pos in self.enemies:
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

    def generate_maze(self):
        if self.maze_type == "maze_empty":
            return self.generate_maze_empty()
        elif self.maze_type == "maze_complex":
            return self.generate_maze_complex()
        elif self.maze_type == "maze_one_wall":
            return self.generate_maze_one_wall()
        elif self.maze_type == "maze_two_wall":
            return self.generate_maze_two_wall()

    def generate_maze_empty(self):
        Z = np.ones((self.height, self.width), np.uint8) * EMPTY_SPACE
        Z[0, 0:self.width] = WALL
        Z[self.height-1, 0:self.width] = WALL
        Z[0:self.height, 0] = WALL
        Z[0:self.height, self.width-1] = WALL
        self.gates = []
        return Z

    def generate_maze_one_wall(self):
        Z = self.generate_maze_empty()

        r = np.random.randint(0, 2)  # 0 or 1
        y = Z.shape[0] / 2
        x = Z.shape[1] / 2
        if r == 1:
            Z[0:self.height, y] = WALL
            r = np.random.randint(1, self.height-1)
            Z[r, y] = EMPTY_SPACE
        else:
            Z[x, 0:self.width] = WALL
            r = np.random.randint(1, self.width-1)
            Z[x, r] = EMPTY_SPACE
        return Z

    def generate_maze_two_wall(self):
        Z = self.generate_maze_empty()

        r = np.random.randint(0, 2)  # 0 or 1
        y = Z.shape[0] / 2
        x = Z.shape[1] / 2
        Z[x, 0:self.width] = WALL
        Z[0:self.height, y] = WALL
        if r == 1:
            r = np.random.randint(1, x-1)
            Z[r, y] = EMPTY_SPACE
            self.gates.append((y, r))
            r = np.random.randint(x+1, self.height-1)
            Z[r, y] = EMPTY_SPACE
            self.gates.append((y, r))

            r = np.random.randint(1, self.width-1)
            while r == y:
                r = np.random.randint(1, self.width-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))
        else:
            r = np.random.randint(1, y-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))
            r = np.random.randint(y+1, self.width-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))

            r = np.random.randint(1, self.height-1)
            while r == x:
                r = np.random.randint(1, self.height-1)
            Z[r, y] = EMPTY_SPACE
            self.gates.append((y, r))
        return Z

    def generate_maze_complex(self, complexity=.75, density=.75):
        # Only odd shapes
        shape = ((self.height // 2) * 2 + 1, (self.width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        Z = np.ones(shape, dtype=np.uint8) * EMPTY_SPACE
        # Fill borders
        Z[0, :] = Z[-1, :] = WALL
        Z[:, 0] = Z[:, -1] = WALL
        # Make aisles
        for i in range(density):
            x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
            Z[y, x] = WALL
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
                    if Z[y_, x_] == EMPTY_SPACE:
                        Z[y_, x_] = WALL
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = WALL
                        x, y = x_, y_
        return Z

if __name__ == "__main__":
    m = MazeGenerator("maze_two_wall", maze_size=(12, 12), maze_target=(10, 10), random_maze_agent=True, random_maze_target=True, enemy_count=10)
    m.reset_game()
    i = 0
    while(not m.game_over() and i < 500):
        action = rand(0,3)
        prev = m.agent_pos
        r = m.act(action)
        next_state = m.agent_pos
        print("{} -- {} --> {} {}".format(prev, maze_actions.get_action(action).name, next_state, r))
        i += 1
    pyplot.figure(figsize=(10, 5))
    pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()
