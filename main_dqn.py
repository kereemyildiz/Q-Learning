from collections import namedtuple
import time
import os


import numpy as np


from game_dqn import CarGameDQN
from DQN import DQN



"""
    Parameters related to training process
"""

# Are we rendering or not, keep this false for faster training
RENDER = False

# Number of steps to train
NUM_STEPS = 50000

#  Print some information about training process every N step
PRINT_INFO_STEP = 100

# The step limit per episode, since we do not want infinite loops inside episodes
MAX_STEPS_PER_EPISODE = 200

# Frequency of target network update 
TARGET_NETWORK_UPDATE_STEP = 1000




# Here, we are creating a data structure to store our transitions
# This is just a convenient way to store
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])




# Frequency of testing our agent with greedy policy (without exploration)
TEST_EVERY_EPISODE = 200

# Number of episodes to run while testing the agent
#average of these episodes will be considered as test result
TEST_EPISODE_COUNT = 10

# Number of timesteps your agent must survive to be considered as successful
TEST_GOAL_STEP_COUNT = 180

"""
    End of training process parameters
"""




# Here, we are creating the environment with our predefined observation space
env = CarGameDQN(render=RENDER)


# Observation and action space
obs_space = env.observation_space
number_of_states = env.observation_space.shape[0]

action_space = env.action_space
number_of_actions = env.action_space.n
print("The observation space: {}".format(obs_space))
# Output: The observation space: Box(n,)
print("The action space: {}".format(action_space))
# Output: The action space: Discrete(m)



#  Hyperparameters of DQN agent
#Parameters related to DQN algorithm and its memory buffer, adjust the ones with value of "1"
HYPERPARAMETERS = {
    "number_of_states": number_of_states,
    "number_of_actions": number_of_actions,
    "number_of_steps": NUM_STEPS,
    "replay_buffer_capacity": 1,
    "learning_rate": 1,
    "batch_size": 1,
    "gamma": 1,
    "epsilon_annealing": "linear"
}











"""
    YOU ARE NOT ALLOWED TO MODIFY ANYTHING BELOW HERE
IN YOUR "SUBMITTED HOMEWORK FILE"!. You can change/add some
code to debug your training while working on the homework, but
while evaluating your homework, only the changes you did
in the parameters and in the other files (as mentioned in the pdf file)
will be considered
"""



def main():
    agent = DQN(HYPERPARAMETERS)

    episodes = 0
    steps = 0
    episode_rewards = []
    continue_training = True
    
    while continue_training and steps < NUM_STEPS:
    #while episodes < NUM_EPISODES:
        #break
        
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        while True:
            steps += 1
            # The select action method inside DQN will select based on policy or random, depending on the epsilon value
            action = agent.select_action(state)
            #print("selected action:", action)

            # Here, we will step the environment with the action
            # Next_state: the state after the action is taken
            # Reward: The reward agent will get. It is generally
            # 1 if the agent wins the game, -1 if the agent loses, 0 otherwise
            # You can add intermediate rewards other than win-lose conditions
            # Done: is the game finished
            # Info: Further info you can get from the environment, you can ignore this part
            next_state, reward, done, info = env.step(action)
            episode_steps += 1

            # Render each frame?
            if RENDER:
                env.render()

            transition = Transition(state, action, reward, next_state, done)
            agent.store_transition(transition)
            
            
            #  We want to update our DQN agent
            agent.update()
            
            if (steps % TARGET_NETWORK_UPDATE_STEP == 0):
                agent.update_target_network()
            

            episode_reward += reward
            #print("done:", done)
            #print("env current timestep:", env.current_time_step)
            
            
            #  We do not want the env to run indefinitely
            #When a done condition is met, we finish
            if done or episode_steps >= MAX_STEPS_PER_EPISODE:
                #print("episode:", episodes, ", episode reward:", episode_reward, ", steps:", steps, ", done:", done, ", reward:", reward)
                episodes += 1
                episode_rewards.append(episode_reward)
                
                
                #  Check if its time to test actual performance of our agent
                #by following greedy policy
                if (episodes % TEST_EVERY_EPISODE == 0):
                    #  time to test
                    print("Time to test, episode:", episodes)
                    test_episode_rewards = []
                    test_episode_step_counts = []
                    for t in range(TEST_EPISODE_COUNT):
                        state = env.reset()
                        test_step_count = 0
                        test_total_reward = 0.0
                        while (True):
                            action = agent.select_action_test(state)
                            next_state, reward, done, info = env.step(action)
                            test_total_reward += reward
                            test_step_count += 1
                            if (done or test_step_count >= 200):
                                test_episode_rewards.append(test_total_reward)
                                test_episode_step_counts.append(test_step_count)
                                break
                            
                            state = next_state
                        
                        
                    test_average_reward = np.mean(test_episode_rewards)
                    test_average_step_count = np.mean(test_episode_step_counts)
                    print("Test average reward:", test_average_reward, ", test average step count:", test_average_step_count)
                    if (test_average_step_count >= TEST_GOAL_STEP_COUNT):
                        print("Your agent learned the environment!")
                        agent.save(os.path.join(".", "araba_successful_weights"), "target_Q.pt")
                        continue_training = False
                        #exit(0)
                        break 
                
                break

            
            if steps % PRINT_INFO_STEP == 0:
                if episodes > 10:
                    print("Step: {}, Epsilon: {}, Mean Reward for last 10 episode: {}".format(steps, agent.e, np.average(episode_rewards[:-10])))

            state = next_state

    agent.save(os.path.join(".", "araba_weights_latest"), "target_Q.pt")

    # Delete the current game instance
    #env.quit()

    print(episode_rewards)


    #  Test our best agent (or current agent) with visualizations enabled
    test_with_render(agent)





def test_with_render(agent):
    # Create a new environment with render enabled
    render_env = CarGameDQN(render=True)
    
    
    success_network_path = os.path.join(".", "araba_successful_weights", "target_Q.pt")
    latest_network_path = os.path.join(".", "araba_weights_latest", "target_Q.pt")
    if (os.path.exists(success_network_path)):
        print("Loading successful network weights")
        agent.load(success_network_path)
    else:
        print("Loading latest network weights")
        agent.load(latest_network_path)
    
    #  reset environment and get initial state
    state = render_env.reset()
    print("Starting test with render:")
    test_step_count = 0
    while True:
        action = agent.select_action_test(state)
        next_state, reward, done, info = render_env.step(action)
        render_env.render()
        if done:
            #break
            state = render_env.reset()
            test_step_count = 0
        else:
            state = next_state
            test_step_count += 1
        
        print("Current test step:", test_step_count)
        
        time.sleep(0.1)


if __name__ == '__main__':
    main()


