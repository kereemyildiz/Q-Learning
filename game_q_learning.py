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
        obs_space_dim = 30
        self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        
       

    def get_state(self):
        """
        Define your state representation here
        
        self.game_state gives you original [6][5] game grid array
        """
        state = None
        
        
        #  fill here
        
        
        
        return state
        

    def get_reward(self):
        """
        Define your reward calculations here
        """
        self.reward = None
        
        
        #  fill here


        self.total_reward += self.reward
        return self.reward



