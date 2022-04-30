from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk

from macros import * 


import time
import os.path
import random
from collections import defaultdict
from copy import deepcopy


import tkinter as tk
from tkinter import messagebox


import numpy as np


#  openai gym environment
import gym
from gym.utils import seeding
from gym import spaces



class CarGame(gym.Env):
    
    """
    #  game window variables
    window_width = None
    window_height = None
    
    
    #  image variables
    background_image = None
    """
    
    def __init__(self, render=True, human_player=True):   
        super(CarGame, self).__init__()
        
        
        self.GUI_created = False
        self.human_player = human_player
        
        
        #  gym/RL related variables
        self.np_random = 0
        
        self.done = False
        self.seed()
        self.reward = 0
        self.total_reward = 0
        self.current_time_step = 0
        self.observation_space = spaces.Box(0, 1, shape=(1,))  #  change in derived class later?
        self.action_space = spaces.Discrete(3)  #  R, L, P
        self.state_space = None
        
        self.Init()
        
        if (render):
            self.CreateGUI()
    
    
    
    """
        gym functions start
    """
    def seed(self, seed=None):
        print("setting seed")
        self.np_random, seed = seeding.np_random(seed)
        print("seed set to:", seed)
        return [seed]
    
    
    def render(self):
        if (self.GUI_created):
            img = self.DrawGameState(self.game_state, self.window_width, self.window_height)
            self.BlitImageToWindow(img)
            
            if not (self.human_player):
                self.UpdateWindowManually()
    
    
    #  resets game state
    def reset(self):
        #  create initial game state
        self.row_count = 6
        self.last_row_index = self.row_count-1
        self.column_count = 5
        self.last_column_index = self.column_count-1
        self.game_state = []
        for r in range(self.row_count):
            self.game_state.append([])
            for c in range(self.column_count):
                self.game_state[r].append("e")
        
        #  put red car in the middle of bottom row
        self.player_row = self.row_count-1
        self.player_column = self.column_count//2
        self.game_state[self.player_row][self.player_column] = "r" 
        
        
        #  create enemy cars
        self.enemy_spawn_method = ENEMY_SPAWN_METHOD_SAFE
        self.enemy_positions = []
        row = 0
        col = 0
        self.enemy_positions.append( [row, col] )
        
        for pos in self.enemy_positions:
            self.game_state[pos[0]][pos[1]] = "b"   
        
        
        #  gym/RL related variables
        self.done = False
        self.reward = 0
        #self.total_reward = 0
        self.current_time_step = 0
        
        return self.get_state()
    
    
    def get_state(self):
        return NotImplementedError
    
    
    def get_reward(self):
        return NotImplementedError
        
        
    def step(self, action):
        """
            0 - right
            1 - left
            2 - pass
        """
        
        action_dir_string = None
        
        #  check if action is a valid integer
        if (action == 0 or action == 1 or action == 2):
            #  valid integer, convert to string
            action_dir_string = convert_action_to_direction(action)
        
        #  check if action is a string
        elif (type(action) == str):
            action_upper = action.upper()
            if (action_upper == "L" or action_upper == "R" or action_upper == "P"):
                action_dir_string = action_upper
        
        #  it is something else
        else:
            raise ValueError("Action must be an integer from [0, 1, 2] or a string from [R, L, P]!") 
        
        #action_dir_string = convert_action_to_direction(action)
        game_result = self.PerformAction(action_dir_string)
        
        
        self.done = (game_result == GAME_RESULT_CRASH)
        return self.get_state(), self.get_reward(), self.done, {}
        
        
    
    
    """
        gym functions end
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def Init(self):
        self.window_width = 512
        self.window_height = 750
        
        self.reset()
        
        
        self.x = (self.window_width // (self.column_count+1))
        self.y = (self.window_height // (2*self.row_count))
        
        
        #  add an offset to x due to road border size
        self.x_offset = 10*self.window_width // 750
        self.topleft_x = self.x + self.x_offset
        self.topleft_y = self.y
        
        print("x:", self.x)
        print("y:", self.y)
        print("topleft position xy:({},{})".format(self.topleft_x, self.topleft_y))
        
        
        #  load images
        self.background_image = Image.open("tunnel_road_x3.png").resize((self.window_width, self.window_height))
        
        self.car_width = int(self.x * 2 * 0.8)
        self.car_height = int(self.y * 2 * 0.8)
        print("car width:{}, height:{}".format(self.car_width, self.car_height))
        self.red_car_image = Image.open("car_red.png").resize((self.car_width, self.car_height))
        self.blue_car_image = Image.open("car_blue_rotated.png").resize((self.car_width, self.car_height))
        self.explosion_image = Image.open("explosion.png").resize((self.car_width, self.car_height))
        
        
        
        
    """
        GUI functions start
    """
    def CreateGUI(self):
        #  create game window
        print("window width:{}, height:{}".format(self.window_width, self.window_height))
        
        self.window, self.window_panel = self.CreateGameWindow(self.window_width, self.window_height)
        
        #  create a PIL image from current board state
        img = self.DrawGameState(self.game_state, self.window_width, self.window_height)

        #  convert to tkinter window compatible image
        img_tk = ImageTk.PhotoImage(img)

        #  put game image into panel
        self.window_panel.configure(image=img_tk)
        self.window_panel.image = img_tk
        
        #  window event loop
        if (self.human_player):
            print("mainloop activated")
            self.window.mainloop()
        else:
            pass
        
        
        
    def CreateGameWindow(self, window_width=300, window_height=300):
        #  define event callback functions
        def window_key_event(event):
            print ("pressed", repr(event.char))
            ch = event.char
            if (ch == "r"):
                self.Init()
                img = self.DrawGameState(self.game_state, self.window_width, self.window_height)
                self.BlitImageToWindow(img)
                messagebox.showinfo("Restart", "Restart!")
            pass
        
        def window_left_key(event):
            print("left key pressed")
            game_result = self.PerformAction("L")
            img = self.DrawGameState(self.game_state, self.window_width, self.window_height)
            self.BlitImageToWindow(img)
            
            if (game_result == GAME_RESULT_CRASH):
                messagebox.showinfo("Game Finished", "Crash!")
                self.window.destroy()

        def window_right_key(event):
            print("right key pressed")
            game_result = self.PerformAction("R")
            img = self.DrawGameState(self.game_state, self.window_width, self.window_height)
            self.BlitImageToWindow(img)
            
            if (game_result == GAME_RESULT_CRASH):
                messagebox.showinfo("Game Finished", "Crash!")
                self.window.destroy()
        
        def window_space_key(event):
            print("space key pressed")
            game_result = self.PerformAction("P")
            img = self.DrawGameState(self.game_state, self.window_width, self.window_height)
            self.BlitImageToWindow(img)
            
            if (game_result == GAME_RESULT_CRASH):
                messagebox.showinfo("Game Finished", "Crash!")
                self.window.destroy()
            
            
        def window_left_click_event(event):
            print ("left clicked at", event.x, event.y)
            pass
            
            
        #  create app window
        window = tk.Tk()
        window.title("CarGame")
        window.geometry("{}x{}".format(window_width, window_height))
        #window.configure(background="grey")
        
        
        #  bind keyboard events
        window.bind("<Key>", window_key_event)
        
        #  bind left mouse click event
        window.bind("<Button-1>", window_left_click_event)
        
        #  bind left and right arrow keys
        window.bind('<Left>', window_left_key)
        window.bind('<Right>', window_right_key)
        
        #  bind space key
        window.bind('<space>', window_space_key)
        
        
        #  create a panel for game image
        window_panel = tk.Label(window)#tk.Label(window, image=img_tk)
        window_panel.pack(side="bottom", fill="both", expand="yes")
        
        
        self.GUI_created = True
        
        return window, window_panel
    
    
    
    def UpdateWindowManually(self):
        self.window.update()
    
    """
        GUI functions end
    """
    
    
    
    
    
    def DrawGameState(self, state, width=768, height=512):
        final_image = deepcopy(self.background_image)
        
        
        row_count = len(state)
        column_count = len(state[0])
        for r in range(row_count):
            for c in range(column_count):
                letter = state[r][c]
                center_pos_x = self.topleft_x + c*self.x - self.car_width//2
                center_pos_y = self.topleft_y + 2*r*self.y - self.car_height//2
                if (letter == "e" or letter == "."):
                    pass
                elif (letter == "r"):
                    final_image.paste(self.red_car_image, (center_pos_x, center_pos_y), self.red_car_image)
                elif (letter == "b"):
                    final_image.paste(self.blue_car_image, (center_pos_x, center_pos_y), self.blue_car_image)
                elif (letter == "x"):
                    final_image.paste(self.explosion_image, (center_pos_x, center_pos_y), self.explosion_image)
        
        
        return final_image
    
    
    
    def PerformAction(self, a):
        #  remove player from old position
        self.game_state[self.player_row][self.player_column] = "e"
        
        
        if (a == "R"):
            self.player_column = min(self.player_column + 1, self.last_column_index)
        elif (a == "L"):
            self.player_column = max(self.player_column - 1, 0)
        elif (a == "P"):
            pass
            
            
        #  put player to new position
        self.game_state[self.player_row][self.player_column] = "r"
        
        
        #  move all enemy cars 1 row down
        new_enemy_positions = []
        for i in range(len(self.enemy_positions)):
            enemy_pos = self.enemy_positions[i]
            enemy_row = enemy_pos[0]
            enemy_col = enemy_pos[1]
            
            #  remove enemy from old position
            self.game_state[enemy_row][enemy_col] = "e"
            
            new_enemy_row = enemy_row + 1
            new_enemy_col = enemy_col
            
            
            
            if (new_enemy_row > self.last_row_index):
                #  enemy removed from game
                pass
            else:
                #  add to new list
                new_enemy_positions.append([new_enemy_row, new_enemy_col])
                
                #  put enemy to new position
                self.game_state[new_enemy_row][new_enemy_col] = "b"
        
        
        if (self.enemy_spawn_method == ENEMY_SPAWN_METHOD_SAFE):
            #  check if timestep is an even number
            if (self.current_time_step%2 == 1):
                #  spawn a new enemy
                #TODO: use gym environment seed here
                possible_spawn_indices = [0, 2, 4]
                random_index = random.randint(0, 2)
                selected_column_index = possible_spawn_indices[random_index]
                
                new_enemy_positions.append([0, selected_column_index])
                self.game_state[0][selected_column_index] = "b"
                
        
        #  update enemy positions variable
        self.enemy_positions = new_enemy_positions
        
        
        #  check if an enemy and player is collided
        result = GAME_RESULT_CONTINUE
        for enemy_pos in self.enemy_positions:
            enemy_row = enemy_pos[0]
            enemy_col = enemy_pos[1]
            
            crash_enemy_from_left = (enemy_row == self.player_row) and (enemy_col == self.player_column+1)
            crash_enemy_from_middle = (enemy_row == self.player_row) and (enemy_col == self.player_column)
            crash_enemy_from_right = (enemy_row == self.player_row) and (enemy_col == self.player_column-1)
            
            if (crash_enemy_from_left or crash_enemy_from_middle or crash_enemy_from_right):
                self.game_state[enemy_row][enemy_col] = "x"
                self.game_state[self.player_row][self.player_column] = "x"
                
                #  mark game as finished
                result = GAME_RESULT_CRASH
        
        #  increase timestep
        self.current_time_step += 1
                
        
        return result
    
    
    def BlitImageToWindow(self, img):
        #  convert to tkinter window compatible image
        img_tk = ImageTk.PhotoImage(img)

        #  put game image into panel
        self.window_panel.configure(image=img_tk)
        self.window_panel.image = img_tk
    
    
    
    """
        find "r" in self.game_state and 
    returns a list of [player_row, player_column]
    """
    def FindPlayerPosition(self):
        row_count = len(self.game_state)
        column_count = len(self.game_state[0])
        
        #  find player position
        player_row = None
        player_col = None
        for r in range(row_count):
            for c in range(column_count):
                letter = self.game_state[r][c]
                if (letter == "r"):
                    player_row = r
                    player_col = c
                    return [player_row, player_col]
        
        return None
    
    
    """
        checks if there is an "x" in the self.game_state
    """
    def IsCrashed(self):
        row_count = len(self.game_state)
        column_count = len(self.game_state[0])
        
        #  find crash position
        for r in range(row_count):
            for c in range(column_count):
                letter = self.game_state[r][c]
                if (letter == "x"):
                    return True
        
        return False
    
    
        
        




if __name__ == "__main__":
    oyun = CarGame(render=True)
