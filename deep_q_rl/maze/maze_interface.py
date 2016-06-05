from abc import ABCMeta, abstractmethod

class MazeInterface:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def getScreenDims(self):
        pass

    @abstractmethod
    def getLegalActionSet(self):
        pass

    @abstractmethod
    def getMinimalActionSet(self):
        pass

    @abstractmethod
    def act(self, action):
        pass

    @abstractmethod
    def getScreenRGB(self):
        pass

    @abstractmethod
    def getScreenGrayscale(self, screen_data=None):
        pass

    @abstractmethod
    def game_over(self):
        pass

    @abstractmethod
    def reset_game(self):
        pass

    @abstractmethod
    def lives(self):
        pass