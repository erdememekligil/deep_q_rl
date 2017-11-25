## My Stuff
from deep_q_rl.maze.maze_interface import MazeInterface
from pacman import *
from keyboardAgents import KeyboardAgent
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util, layout
## End my stuff

import numpy as np
from numpy.random import random_integers as rand
import matplotlib.pyplot as pyplot
import operator
import cPickle
import os
import logging

BLACK = 0
ENEMY = 42
FOOD = 84
SCARED_ENEMY = 126
PACMAN = 168
POWERUP = 210
WHITE = 255

WALL = WHITE
EMPTY_SPACE = BLACK

color_map = {
    0: WALL,
    1: PACMAN,
    2: ENEMY,
    3: SCARED_ENEMY,
    4: FOOD,
    5: POWERUP
}


class PacmanGenerator(MazeInterface):
    def __init__(self, l="originalClassic", n=60, x=50, timeout=30):
        super(PacmanGenerator, self).__init__()
        # self.maze_type = l
        # self.maze_layout = layout.getLayout(l)
        # self.height = self.maze_layout.height
        # self.width = self.maze_layout.width

        ## Set Game rule set
        # self.rules = ClassicGameRules(timeout)
        ## Set Keyboard Agent
        # agentOpts = {}
        # agentOpts['numTraining'] = n
        # pacmanType = loadAgent("KeyboardAgent", False)
        # self.pacmanAgent = pacmanType(**agentOpts)        ## -p parametresi
        # cmd = "-p ApproximateQAgent -a extractor=SimpleExtractor,epsilon=0.1,alpha=0.1,gamma=0.95 -x 5 -n 100 -l originalMSClassic --frameTime 0.0001"
        # cmd = "-p KeyboardAgent --textGraphics --frameTime 0"
        # cmd = "-p KeyboardAgent --quietTextGraphics --frameTime 0"
        # cmd = "-p KeyboardAgent --quietTextGraphics --frameTime 0 --layout smallClassic"
        cmd = "-p KeyboardAgent --quietTextGraphics --frameTime 0 --layout smallClassic --numghosts 1"

        self.args = readCommand( cmd.split(" ") ) # Get game components based on input
        rules1, layout1, pacman, ghosts, gameDisplay, beQuiet, catchExceptions = self.get_game_config(**self.args)
        self.rules = rules1
        self.maze_layout = layout1
        self.height = self.maze_layout.height
        self.width = self.maze_layout.width
        self.game = self.rules.newGame( layout1, pacman, ghosts, gameDisplay, beQuiet, catchExceptions)
        self.minimalActionSet = []
        for direction in Actions.getAllPossibleActions():
            self.minimalActionSet.append(Actions.directionToIndex(direction))
        # import textDisplay
        # gameDisplay = textDisplay.NullGraphics()
        # self.rules.quiet = True
        # runGames( **args )

        ## Aet Ghost Agents
        # ghostType = loadAgent("RandomGhost", True)
        # self.ghosts = [ghostType(gi + 1) for gi in xrange(1,4)]
        # import graphicsDisplay
        # self.gameDisplay = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1)
        # self.beQuiet = True
        # self.gamesPlayed = []

    def getScreenDims(self):
        # return self.height, self.width#burasi karismis
        return self.width, self.height#todo fix
        # return (self.width + 1) * 3, (self.height + 1) * 3  ## bu pixelmi?

    def getScreenGrayscale(self, screen_data=None):
        curr_state = self.game.getSelfStateMatrices()
        # return curr_state[:,:,1]
        ## 84*84 olacak sekilde 27*27 lik bir alani (28*28)x 3 seklinde yapiyorum
        # H = (self.height + 1)
        # W = (self.width + 1)
        # if screen_data is None:
        #     screen_data = np.empty((H * 3, W * 3), dtype=np.uint8)

        #transpoed
        # screen_data = np.zeros(curr_state.shape[0:2], dtype=np.uint8)
        # for color_ind in range(0, curr_state.shape[2]):
        #     color = color_map[color_ind]
        #     for i in range(0, screen_data.shape[0]):
        #         for j in range(0, screen_data.shape[1]):
        #             if curr_state[i, j, color_ind] == 1:
        #                 screen_data[i, j] = color

        #todo fixed
        width = curr_state.shape[0] #20
        height = curr_state.shape[1] #11
        screen_data = np.zeros((height, width), dtype=np.uint8)
        for color_ind in range(0, curr_state.shape[2]):
            color = color_map[color_ind]
            for i in range(0, height):
                for j in range(0, width):
                    if curr_state[j, i, color_ind] == 1:
                        screen_data[i, j] = color


        # for i in range(0, screen_data.shape[0]):
        #     for j in range(0, screen_data.shape[1]):
        #         value = BLACK
        #         if self.maze_layout.isWall(i, j):
        #             value = WALL
        #         else:
        #             if self.game.state.hasFood(i,j):
        #                 value = PELLET
        #             else:
        #                 if self.game.state.hasCapsule(i,j):
        #                     value = SUPER_PELLET
        #                 else:
        #                     value = EMPTY_SPACE
        #             for a in range(len(self.game.agents)):
        #                 agent = self.game.agents[a]
        #                 if a == 0:
        #                     value = PLAYER
        #                 else:
        #                     value = ENEMY
        #         X = i * 3
        #         Y = j * 3
        #         screen_data[X][Y] = value
        #         screen_data[X][Y+1] = value
        #         screen_data[X][Y+2] = value
        #
        #         screen_data[X+1][Y] = value
        #         screen_data[X+1][Y+1] = value
        #         screen_data[X+1][Y+2] = value
        #
        #         screen_data[X+2][Y] = value
        #         screen_data[X+2][Y+1] = value
        #         screen_data[X+2][Y+2] = value

        return screen_data

    def act(self, action_index):
        prev = self.game.state.getScore()
        self.game.runOneStep(Actions.indexToDirection(action_index))
        return self.game.state.getScore() - prev

        ## game.run fonksionunu icinde bir step olacak sekilde
        ##action = maze_actions.get_action(action_index).value
        ##next_pos = tuple(map(operator.add, self.agent_pos, action))
        ##if self.maze[next_pos[1]][next_pos[0]] != WALL:
        ##    self.agent_pos = next_pos

        ##self.action_count += 1
        ##if self.agent_pos == self.target_pos:  # end episode
        ##    rwrd = 100 - self.reward_given
        ##    self.reward_given = 100
        ##    return rwrd
        ##elif self.check_gate_reward():
        ##    self.reward_given += GATE_REWARD
        ##    return GATE_REWARD
        ##else:
        ##    return 0

    def reset_game(self):
        rules1, layout1, pacman, ghosts, gameDisplay, beQuiet, catchExceptions = self.get_game_config(**self.args)
        self.rules = rules1
        self.maze_layout = layout1
        self.height = self.maze_layout.height
        self.width = self.maze_layout.width
        self.game = self.rules.newGame( layout1, pacman, ghosts, gameDisplay, beQuiet, catchExceptions)
        self.minimalActionSet = []
        for direction in Actions.getAllPossibleActions():
            self.minimalActionSet.append(Actions.directionToIndex(direction))

        self.action_count = 0
        self.reward_given = 0

    def game_over(self):
        return self.game.is_game_over()

    def lives(self):
        return 1

    def getMinimalActionSet(self):
        return self.minimalActionSet

    def getLegalActionSet(self):
        #self.PacmanRules.getLegalActions() var amaaaaa
        ##return self.game.state.getLegalActions(0)
        return self.minimalActionSet

    def getScreenRGB(self):
        gray = self.getScreenGrayscale()
        vis = np.empty((gray.shape[0], gray.shape[1], 3), np.uint8)
        vis[:, :, 0] = gray
        vis[:, :, 1] = gray
        vis[:, :, 2] = gray
        return vis

    @staticmethod
    def get_game_config(outputFilePath, layout, pacman, ghosts, display, numGames, record, numTraining = 0, catchExceptions=False, timeout=30 ):

        rules = ClassicGameRules(timeout)
        beQuiet = False
        if beQuiet:
            # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            #gameDisplay = textDisplay.NullGraphics()
            gameDisplay = display
            rules.quiet = False
        return rules, layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions

if __name__ == "__main__":
    m = PacmanGenerator("originalClassic", 60, 50)
    # m.reset_game()
    # pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
    # pyplot.show()
    i = 0
    total_r = 0
    while (not m.game_over()):
        action = rand(0, 3)
        r = m.act(action)
        print action, r

        scr = m.getScreenRGB()
        pyplot.figure(figsize=(10, 5))
        pyplot.imshow(scr, cmap='Greys_r', interpolation='nearest')
        pyplot.xticks([]), pyplot.yticks([])
        # pyplot.show()
        # pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
        # pyplot.show()
        # print("{} -- {} --> {} {}".format(prev, Actions.getAllPossibleActions()[action], next_state, r))
        i += 1
        total_r += r

    print("total_r: ", str(total_r))
    print("reset")
    m.reset_game()
    while (not m.game_over()):
        action = rand(0, 3)
        r = m.act(action)
        print action, r

        m.getScreenRGB()

        # pyplot.imshow(m.getScreenRGB(), cmap='Greys_r', interpolation='nearest')
        # pyplot.show()
        # print("{} -- {} --> {} {}".format(prev, Actions.getAllPossibleActions()[action], next_state, r))
        i += 1

    scr = m.getScreenRGB()
    pyplot.figure(figsize=(10, 5))
    pyplot.imshow(scr, cmap='Greys_r', interpolation='nearest')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()
