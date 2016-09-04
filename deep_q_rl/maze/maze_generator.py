
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
GATE = 212
PLAYER = 170
ENEMY = 127
TARGET = 85
BLACK = 0
WALL = WHITE
EMPTY_SPACE = BLACK

MAZE_PICKLE_FOLDER = 'maze'

GATE_REWARD = 10

class MazeGenerator(MazeInterface):
    
    def __init__(self, maze_type="maze_empty", maze_size=(21, 21), maze_init=(1, 1), maze_target=(19, 19),
                 random_maze_agent=False, random_maze_target=False, max_action_count=400, enemy_count=0,
                 maze_gate_reward_size=0, maze_force_opposite_sides=False, rng=np.random.RandomState()):
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
        self.gate_reward_size = maze_gate_reward_size
        self.enemies = []
        self.gates = []
        self.gate_rewards = []
        self.areas = []
        self.reward_given = 0
        self.force_opposite_sides = maze_force_opposite_sides
        self.rng = rng

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

        for gate in self.gate_rewards:
            if self.gate_reward_size == 0:
                break
            elif self.gate_reward_size == 1:
                screen_data[gate[1]][gate[0]] = GATE
            elif self.gate_reward_size == 2:
                if all(screen_data[gate[1]-1: gate[1]+2, gate[0]] == EMPTY_SPACE):
                    screen_data[gate[1]-1: gate[1]+2, gate[0]] = GATE
                elif all(screen_data[gate[1], gate[0]-1: gate[0]+2] == EMPTY_SPACE):
                    screen_data[gate[1], gate[0]-1: gate[0]+2] = GATE
            elif self.gate_reward_size == 3:
                empty_spaces = screen_data[gate[1]-1: gate[1]+2, gate[0]-1: gate[0]+2] == EMPTY_SPACE
                for i in range(0, 3):
                    for j in range(0, 3):
                        if empty_spaces[i, j]:
                            screen_data[gate[1]-1+i, gate[0]-1+j] = GATE
            else:
                break
        return screen_data

    def act(self, action_index):
        action = maze_actions.get_action(action_index).value
        next_pos = tuple(map(operator.add, self.agent_pos, action))
        if self.maze[next_pos[1]][next_pos[0]] != WALL:
            self.agent_pos = next_pos

        self.action_count += 1
        if self.agent_pos == self.target_pos:  # end episode
            rwrd = 100 - self.reward_given
            self.reward_given = 100
            return rwrd
        elif self.check_gate_reward():
            self.reward_given += GATE_REWARD
            return GATE_REWARD
        else:
            return 0

    # Returns true if agent hit the gate reward, and removes the gate reward.
    def check_gate_reward(self):
        if self.gate_reward_size == 0:
            return False
        elif self.gate_reward_size == 1 and self.agent_pos in self.gate_rewards:
            self.gate_rewards.remove(self.agent_pos)
            return True
        elif self.gate_reward_size == 2:
            collides, i = self.collides_with_gate(self.agent_pos, self.gate_rewards)
            if collides:
                self.gate_rewards.pop(i)
                return True
            else:
                return False
        elif self.gate_reward_size == 3: # 3x3 check
            collides, i = self.collides_with_gate(self.agent_pos, self.gate_rewards, True)
            if collides:
                self.gate_rewards.pop(i)
                return True
            else:
                return False
        else:
            #Not supported.
            return False

    def reset_game(self):
        self.maze = self.generate_maze()

        # generate player
        if self.random_maze_agent:
            self.agent_pos = self.generate_random_position_without_wall()
        else:
            self.agent_pos = self.initial_pos

        # generate target
        if self.random_maze_target:
            self.target_pos = self.generate_random_position_without_wall()
            while self.target_pos == self.agent_pos or (self.force_opposite_sides and not self.check_opposite_sides()):
                self.target_pos = self.generate_random_position_without_wall()

        # generate enemies.
        self.enemies = []
        for i in range(0, self.enemy_count):
            enemy = self.generate_random_position_without_wall()
            while enemy == self.agent_pos or enemy == self.target_pos or enemy in self.enemies or self.collides_with_gate(enemy, self.gates):
                enemy = self.generate_random_position_without_wall()
            self.enemies.append(enemy)

        self.action_count = 0
        self.reward_given = 0

    # returns false if both agent and target are in same area.
    def check_opposite_sides(self):
        if len(self.areas) <= 1:
            return True
        for a in self.areas:
            if self.agent_pos in a and self.target_pos in a:
                return False
        return True

    # cross shape check
    def collides_with_gate(self, pos, gates, full_shape=False):
        for ind in range(0, len(gates)):
            g = gates[ind]
            collides = g == pos
            temp = (g[0]-1, g[1])
            collides = collides or temp == pos
            temp = (g[0]+1, g[1])
            collides = collides or temp == pos
            temp = (g[0], g[1]-1)
            collides = collides or temp == pos
            temp = (g[0], g[1]+1)
            collides = collides or temp == pos
            if full_shape:
                temp = (g[0]-1, g[1]-1)
                collides = collides or temp == pos
                temp = (g[0]-1, g[1]+1)
                collides = collides or temp == pos
                temp = (g[0]+1, g[1]-1)
                collides = collides or temp == pos
                temp = (g[0]+1, g[1]+1)
                collides = collides or temp == pos

            if collides:
                return True, ind
        return False, -1

    def generate_random_position_without_wall(self):
        random_pos = self.generate_random_position()
        while self.maze[random_pos[1]][random_pos[0]] != EMPTY_SPACE:
            random_pos = self.generate_random_position()
        return random_pos

    def generate_random_position(self):
        x = self.rng.randint(1, self.width - 1) # -1 for the wall
        y = self.rng.randint(1, self.height - 1) # -1 for the watch
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
        self.areas = []
        if self.maze_type == "maze_empty":
            maze = self.generate_maze_empty()
        elif self.maze_type == "maze_complex":
            maze = self.generate_maze_complex()
        elif self.maze_type == "maze_one_wall":
            maze = self.generate_maze_one_wall()
        elif self.maze_type == "maze_two_wall":
            maze = self.generate_maze_two_wall()

        self.gate_rewards = list(self.gates)

        return maze

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

        r = self.rng.randint(0, 2)  # 0 or 1
        y = Z.shape[0] / 2
        x = Z.shape[1] / 2
        if r == 1:
            Z[0:self.height, y] = WALL
            r = self.rng.randint(1, self.height-1)
            Z[r, y] = EMPTY_SPACE
            self.gates.append((y, r))
            self.append_rect_area((1, 1), (y, self.width-1))
            self.append_rect_area((y+1, 1), (self.width-1, self.height-1))
            # self.append_rect_area((1, 1), (self.width-1, y))
            # self.append_rect_area((1, y+1), (self.width-1, self.height-1))
        else:
            Z[x, 0:self.width] = WALL
            r = self.rng.randint(1, self.width-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))
            self.append_rect_area((1, 1), (self.height-1, x))
            self.append_rect_area((1, x+1), (self.width-1, self.height-1))
            # self.append_rect_area((1, 1), (x, self.height-1))
            # self.append_rect_area((x+1, 1), (self.width-1, self.height-1))
        return Z

    def append_rect_area(self, left_corner, right_corner):
        l = list()
        for i in range(left_corner[0], right_corner[0]):
            for j in range(left_corner[1], right_corner[1]):
                l.append((i, j))
        self.areas.append(l)

    def generate_maze_two_wall(self):
        Z = self.generate_maze_empty()

        r = self.rng.randint(0, 2)  # 0 or 1
        y = Z.shape[0] / 2
        x = Z.shape[1] / 2
        Z[x, 0:self.width] = WALL
        Z[0:self.height, y] = WALL
        if r == 1:
            r = self.rng.randint(1, x-1)
            Z[r, y] = EMPTY_SPACE
            self.gates.append((y, r))
            r = self.rng.randint(x+1, self.height-1)
            Z[r, y] = EMPTY_SPACE
            self.gates.append((y, r))

            r = self.rng.randint(1, self.width-1)
            while r == y:
                r = self.rng.randint(1, self.width-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))
        else:
            r = self.rng.randint(1, y-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))
            r = self.rng.randint(y+1, self.width-1)
            Z[x, r] = EMPTY_SPACE
            self.gates.append((r, x))

            r = self.rng.randint(1, self.height-1)
            while r == x:
                r = self.rng.randint(1, self.height-1)
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
    m = MazeGenerator("maze_one_wall", maze_size=(12, 12), maze_target=(10, 10), random_maze_agent=True, random_maze_target=True, enemy_count=0, maze_force_opposite_sides=False, maze_gate_reward_size=2, rng=np.random.RandomState())
    m.reset_game()
    i = 0
    while(not m.game_over() and i < 0):
        action = rand(0,3)
        prev = m.agent_pos
        r = m.act(action)
        next_state = m.agent_pos
        print("{} -- {} --> {} {}".format(prev, maze_actions.get_action(action).name, next_state, r))
        i += 1
    m.reset_game()
    print m.agent_pos
    print m.target_pos
    print m.check_opposite_sides()
    print m.areas
    pyplot.figure(figsize=(10, 5))
    pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()
