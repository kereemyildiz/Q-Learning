from collections import namedtuple
import time
import operator
import numpy as np
from collections import defaultdict


from game_q_learning import CarGameQL
from macros import *


# Are we rendering or not
RENDER_TRAIN = False


"""
    Parameters related to training process and Q-Learning
"""

# Number of episodes to train
NUM_EPISODES = 1000

# epsilon parameter of e-greedy policy
EPSILON = 0.1

# learning rate parameter of q learning
LEARNING_RATE = 0.3

# discount rate parameter of q learning
DISCOUNT_RATE = 0.9

# The step limit per episode, since we do not want infinite loops inside episodes
MAX_STEPS_PER_EPISODE = 200

"""
    Parameters end
"""


# Here, we are creating the environment with our predefined observation space
env = CarGameQL(render=RENDER_TRAIN)

# Observation and action space
obs_space = env.observation_space
number_of_states = env.observation_space.shape[0]

action_space = env.action_space
number_of_actions = env.action_space.n
print("The observation space: {}".format(obs_space))
# Output: The observation space: Box(n,)
print("The action space: {}".format(action_space))
# Output: The action space: Discrete(m)


q_table = {}


def choose_action_greedy(state, q_table):

    action = max(q_table[state].items(), key=operator.itemgetter(1))[0]

    return action


def choose_action_e_greedy(state, q_table):
    action = None

    probs = np.ones(number_of_actions, dtype=float) * \
        EPSILON / number_of_actions

    try:
        best = max(q_table[state].items(), key=operator.itemgetter(1))[0]

    except KeyError:
        q_table[state] = {"R": 0, "L": 0, "P": 0}
        best = max(q_table[state].items(), key=operator.itemgetter(1))[0]

    best = convert_direction_to_action(best)

    probs[best] += (1.0 - EPSILON)

    action = np.random.choice(np.arange(len(probs)), p=probs)
    action = convert_action_to_direction(action)
    return action


def main():

    #  "Loop for each episode:"
    for e in range(NUM_EPISODES):
        #  "Initialize S"
        s0 = env.reset()

        #  "Loop for each step of episode:"
        episode_steps = 0
        while (episode_steps < MAX_STEPS_PER_EPISODE):
            #
            #  "Choose A from S using policy derived from Q (e.g., e-greedy)"
            #
            action = choose_action_e_greedy(s0, q_table)

            #  "Take action A, observe R, S'"
            s1, reward, done, info = env.step(action)

            #
            #  "Q(S,A) <-- Q(S,A) + alpha*[R + gamma* maxa(Q(S', a)) - Q(S, A)]"
            #
            try:
                best_next_action = max(
                    q_table[s1].items(), key=operator.itemgetter(1))[0]
            except KeyError:
                q_table[s1] = {"R": 0, "L": 0, "P": 0}
                best_next_action = max(
                    q_table[s1].items(), key=operator.itemgetter(1))[0]

            td_target = reward + DISCOUNT_RATE * q_table[s1][best_next_action]
            td_delta = td_target - q_table[s0][action]
            q_table[s0][action] += LEARNING_RATE * td_delta
            #  "S <-- S'"
            s0 = s1

            # until S is terminal
            if (done):
                break

        #  print number of episodes so far
        if (e % 100 == 0):
            print("episode {} completed".format(e))

    #  test our trained agent
    test_agent(q_table)


def test_agent(q_table):
    print("Initializing test environment:")
    test_env = CarGameQL(render=True, human_player=False)
    state = env.reset()
    steps = 0
    # while (steps < 200):
    while (True):
        action = choose_action_greedy(state, q_table)
        print("chosen action:", action)
        next_state, reward, done, info = test_env.step(
            convert_direction_to_action(action))
        print("state:", state, " , next_state:", next_state)
        test_env.render()
        if done:
            break
        else:
            state = next_state
        steps += 1
        print("test current step:", steps)

        time.sleep(0.3)

    print(env.total_reward)


if __name__ == '__main__':
    main()
