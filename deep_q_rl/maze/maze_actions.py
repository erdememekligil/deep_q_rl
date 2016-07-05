
class MazeAction():

    def __init__(self, value, name, ale_mapping):
        self.value = value
        self.name = name
        self.ale_mapping = ale_mapping

    def __str__(self):
        return "{} {}".format(self.name, self.value)

NOOP = MazeAction((0, 0), "NOOP", 0)
ACTION_UP = MazeAction((0, -1), "ACTION_UP", 2)
ACTION_RIGHT = MazeAction((1, 0), "ACTION_RIGHT", 3)
ACTION_LEFT = MazeAction((-1, 0), "ACTION_LEFT", 4)
ACTION_DOWN = MazeAction((0, 1), "ACTION_DOWN", 5)

# ACTIONS = [NOOP, ACTION_UP, ACTION_RIGHT, ACTION_LEFT, ACTION_DOWN]
ACTIONS = [ACTION_UP, ACTION_RIGHT, ACTION_LEFT, ACTION_DOWN]

ACTION_ALE_MAPPINGS = {}
# ACTION_ALE_MAPPINGS[0] = NOOP
ACTION_ALE_MAPPINGS[2] = ACTION_UP
ACTION_ALE_MAPPINGS[3] = ACTION_RIGHT
ACTION_ALE_MAPPINGS[4] = ACTION_LEFT
ACTION_ALE_MAPPINGS[5] = ACTION_DOWN

def get_action(action_index):
    return ACTIONS[action_index]

def get_action_index(action):
    return ACTIONS.index(action)

def get_minimal_action_set():
    return range(0, len(ACTIONS))

if __name__ == "__main__":
    print(get_action(4))
    print(get_action_index(ACTION_LEFT))
    print(get_minimal_action_set())
