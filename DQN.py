import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from utility import linear_annealing, exponential_annealing



class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(PolicyNetwork, self).__init__()
        """
        :param num_states: Input size for the network
        :param num_actions: Output size for the network
        
        Define your neural network here
        This network will be used for both active and target networks
        Do not create over-sized networks
            *** You probably do not need more than 2 layers. 3 at most. 
            *** Your layers should probably only need 128 neurons at max. 
            *** Try to create a small network since more neurons means more processing power; increasing training time
            *** This is not a requirement, just a suggestion. You may create any network you want as long as it learns
        """
        self.layer1 = None


    def forward(self, x):
        """
        :param x: Input to the network
        :return: The action probabilities for each action

        This is the method that is called when you send the state to the network
        You send the input x (which is state) through the layers in order
        After each layer, do not forget to pass the output from an activation function (relu, tanh etc.)
        """
        return x


class DQN:
    memory_count = 0        # Amount of data pushed into mem
    update_count = 0        # Number of updates done

    def __init__(self, HYPERPARAMETERS):
        super(DQN, self).__init__()
        self.num_states = HYPERPARAMETERS["number_of_states"]
        self.num_actions = HYPERPARAMETERS["number_of_actions"]
        self.capacity = HYPERPARAMETERS["replay_buffer_capacity"]   # Memory capacity
        self.learning_rate = HYPERPARAMETERS["learning_rate"]       # Alpha in DQN formula
        self.batch_size = HYPERPARAMETERS["batch_size"]             # Number of batches to process at each update
        self.gamma = HYPERPARAMETERS["gamma"]                       # Discount factor

        self.target_net = PolicyNetwork(self.num_states, self.num_actions)
        self.act_net = PolicyNetwork(self.num_states, self.num_actions)
        self.memory = [None] * self.capacity
        self.losses = []

        # The epsilon value for e-greedy action selection
        # At the start, the agent will select a random action with %90 probability
        # That value will drop down as we take action, until %10 (it is always good to have some randomness/noise).
        # Linearly or exponentially (your call)
        self.e = 0.9
        if HYPERPARAMETERS["epsilon_annealing"] == 'linear':
            self.epsilon = linear_annealing(
                self.e,
                0.1,
                HYPERPARAMETERS["number_of_steps"]
            )
        else:
            self.epsilon = exponential_annealing(
                self.e,
                0.1,
                HYPERPARAMETERS["number_of_steps"]
            )

        # We will use Adam optimizer here
        self.optimizer = optim.Adam(self.act_net.parameters(),
                                    self.learning_rate)
        # Mean-squared error will be enough for this project
        self.loss_func = nn.MSELoss()

    def select_action(self, state):
        # To select an action, we need to feed it to Neural Net
        # NN only accepts tensors, so we need to convert the state
        
        # Here, the exploitation-exploration balance is handled
        # We get the next epsilon value based on the current step amount
        self.e = next(self.epsilon)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action = None
        raise NotImplementedError("You should write a function for action selection")
        return action
    
    
    def select_action_test(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        with torch.no_grad():
            action = None
            raise NotImplementedError("You should write a function for action selection")
            return action
        
    

    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def save(self, folder_path, file_name):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.target_net.state_dict(), os.path.join(folder_path, file_name))
        
        
        
    #def save(self, filename):
    #    import os
    #    os.makedirs(filename, exist_ok=True)
    #    torch.save(self.target_net.state_dict(), filename + "/target_Q.pt")
    
    def load(self, file_path):
        weights = torch.load(file_path)
        self.act_net.load_state_dict(weights)

    
    def update(self):
        #  You can change anything in this function as you want,
        #or write it from scratch. It is up to you.


        if self.memory_count >= self.capacity:
            # Read state, action, reward, next_state from mem
            state, action, reward, next_state = [], [], [], []
            for t in self.memory:
                state.append(t.state)
                action.append(t.action)
                reward.append(t.reward)
                next_state.append(t.next_state)

            state = torch.tensor(state).float()
            action = torch.LongTensor(action).view(-1, 1).long()
            reward = torch.tensor(reward).float()
            next_state = torch.tensor(next_state).float()
            # The view method reshapes the tensor without any copy
            # operation. It is super fast and efficient


            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            # Take a look at this reward calculation. We have a tensor
            # (1D vector) of rewards, we are calculating mean and std
            # of this tensor, and subtract mean from all elements.
            # Then divide all elements by (std + some small value) to
            # prevent division by zero. What do we get? A normalized
            # tensor of rewards. This is a normalization technique. If
            # the rewards are too divergent, it will affect the training
            # negatively, thus we normalize them for each batch.


            # Calculate target_Q values by using Bellman equation
            # Note that we do not want to calculate gradients for this


            #  Update...
            #Below is the missing part you should complete
            #Get a set of random indices to fetch them in memory
            #You may use a different sampling method instead of using the code below
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
                                      drop_last=False):
                # Get the current Q values
                # Notice we are using active network and
                # calculating gradients.


                # The optimization loop
                # Call zero_grad to clear previous grads
                # Then make back propagation
                # Then step
                

                raise NotImplementedError


        else:
            print("Memory Buffer is too small")


    def update_target_network(self):
        #  You can perform a direct update or update with TAU parameter, both are accepted
        raise NotImplementedError
        print("updated target network")
