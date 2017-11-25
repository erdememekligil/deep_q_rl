# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
import numpy
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.qValues = util.Counter()
    print "ALPHA", self.alpha
    print "DISCOUNT", self.discount
    print "EXPLORATION", self.epsilon

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    return self.qValues[(state, action)]


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    possibleStateQValues = util.Counter()
    for action in self.getLegalActions(state):
    	possibleStateQValues[action] = self.getQValue(state, action)
    
    return possibleStateQValues[possibleStateQValues.argMax()]

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    possibleStateQValues = util.Counter()
    possibleActions = self.getLegalActions(state)
    if len(possibleActions) == 0:
    	return None
    
    for action in possibleActions:
    	possibleStateQValues[action] = self.getQValue(state, action)
    
    if possibleStateQValues.totalCount() == 0:
    	return random.choice(possibleActions)
    else:
    	return possibleStateQValues.argMax()

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    if len(legalActions) > 0:
    	if util.flipCoin(self.epsilon):
    		action = random.choice(legalActions)
    	else:
			action = self.getPolicy(state)

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    print "State: ", state, " , Action: ", action, " , NextState: ", nextState, " , Reward: ", reward
    print "QVALUE", self.getQValue(state, action)
    print "VALUE", self.getValue(nextState)
    self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()
    self.isStrlInitialized = False
    self.visitAmount = [[[]]]
    self.catchAmount = [[[]]]

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    qValue = 0.0
    features = self.featExtractor.getFeatures(state, action)
    for key in features.keys():
    	qValue += (self.weights[key] * features[key])
    return qValue

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    features = self.featExtractor.getFeatures(state, action)




    "*** STRL CODE BEGIN ***"

    pacmanX,pacmanY = state.getPacmanPosition()

    if(not self.isStrlInitialized):
      self.visitAmount = [[[0.0001 for i in xrange(1)] for j in xrange(state.data.layout.height)] for k in xrange(state.data.layout.width)]
      self.catchAmount = [[[0 for i in xrange(1)] for j in xrange(state.data.layout.height)] for k in xrange(state.data.layout.width)]
      self.isStrlInitialized = True
    else:
      "* Visit iteration *"
      if len(self.visitAmount[pacmanX][pacmanY]) < state.getTimeTick():
        self.visitAmount[pacmanX][pacmanY].extend(numpy.full(state.getTimeTick() - len(self.visitAmount[pacmanX][pacmanY]) + 1, 0.0001))

      self.visitAmount[pacmanX][pacmanY][state.getTimeTick()-1] += 1

      "* Catch iteration *"
      if len(self.catchAmount[pacmanX][pacmanY]) < state.getTimeTick():
        self.catchAmount[pacmanX][pacmanY].extend(numpy.zeros(state.getTimeTick() - len(self.catchAmount[pacmanX][pacmanY]) + 1))

      for gPosX, gPosY in state.getGhostPositions():
       if state.isLose() or (gPosX == pacmanX and gPosY == pacmanY):
          self.catchAmount[pacmanX][pacmanY][state.getTimeTick()-1] += 1

    modifiedReward = 0.0
    try:
      modifiedReward = reward - reward * self.catchAmount[pacmanX][pacmanY][state.getTimeTick()-1] / self.visitAmount[pacmanX][pacmanY][state.getTimeTick()-1]
    except Exception, data:
      print ("x:" + str(pacmanX) + " y:" + str(pacmanY) + " Tick:" + str(state.getTimeTick()))
      raise data
    "*** STRL CODE END ***"

    possibleStateQValues = []
    for act in self.getLegalActions(state):
    	possibleStateQValues.append(self.getQValue(state, act))
    for key in features.keys():
    	self.weights[key] += self.alpha * (modifiedReward + self.discount * ((1-self.epsilon)*self.getValue(nextState)+(self.epsilon/len(possibleStateQValues))*(sum(possibleStateQValues))) - self.getQValue(state, action)) * features[key]

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
