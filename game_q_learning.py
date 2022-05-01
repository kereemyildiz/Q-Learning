import numpy as np

from game import *


class CarGameQL(CarGame):
    def __init__(self, render=True, human_player=False):
        super(CarGameQL, self).__init__(render, human_player)

        """
        DEFINE YOUR OBSERVATION SPACE DIMENSIONS HERE FOR EACH MODE.
        JUST CHANGING THE "obs_space_dim" VARIABLE SHOULD BE ENOUGH
        
            Try making your returned state from get_state function
        a 1-D array if its not, it will make things simpler for you
        
            For the first Q-Learning part, you must use a more compact
        game state than raw game array
        """
        obs_space_dim = len(self.get_state())
        self.observation_space = spaces.Box(
            0, obs_space_dim, shape=(obs_space_dim,))

    def get_state(self):
        """
        Define your state representation here

        self.game_state gives you original [6][5] game grid array
        """
        state = np.array(self.game_state).ravel()
        #  fill here
        return str(state)

    def get_reward(self):
        """
        Define your reward calculations here
        """
        #  fill here

        blues = []
        red = None
        state = self.game_state
        state = np.asarray(state)

        for i in range(6):
            for j in range(5):
                if (state[i][j] == "b"):
                    blues.append((i, j))
                elif (state[i][j] == "r"):
                    red = j
                else:
                    pass

        for blue in blues:
            x = blue[0]
            y = blue[1]

        if (self.IsCrashed()):
            self.reward = -5
        elif(x == 4 and y == red):
            self.reward = -3
        else:
            self.reward = 2
        self.total_reward += self.reward
        return self.reward
