This code used the Pacman framework provided by UC Berkeley. I've modified the feature extraction code in order to enable the pacman eat ghosts when it uses a power pellet.  

The learning algorithm used is On-policy Expected Sarsa

In this project, you will implement value iteration and Q-learning. You will test your agents first on Gridworld (from class), then apply them to a simulated robot controller (Crawler) and Pacman.

Some sample scenarios to try with are:

$ python gridworld.py -a q -k 100 
$ python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
$ python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

For STRL
$ python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l originalClassic


All the interesting code is in qlearningAgents.py and featureExtractors.py.

----------------------------------------------------------------------------------------------------
_____For Deep Learning Interface_____

pacman => ApproximateQAgent 

pacmanType = loadAgent("ApproximateQAgent", True)	## -p parametresi
agentOpts = parseAgentArgs("extractor=SimpleExtractor") ## -a parametresi
  if options.numTraining > 0:				## -x parametresi How many episodes are training (suppresses output)
    args['numTraining'] = options.numTraining
    if 'numTraining' not in agentOpts: agentOpts['numTraining'] = options.numTraining
pacman = pacmanType(**agentOpts) 		# Instantiate Pacman with agentArgs




args['pacman'] = pacman
args['layout'] = layout.getLayout("originalClassic")	##-l parametresi
args['numGames'] = 60					##-n parametresi the number of GAMES to play


pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor,epsilon=0.1,alpha=0.1,gamma=0.95 -x 50 -n 100 -l originalMSClassic --frameTime 0.0001 -o TestResults.txt