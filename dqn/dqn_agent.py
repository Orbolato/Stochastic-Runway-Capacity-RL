import numpy as np
import random
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from collections import deque

###############################################################################
# Auxiliary functions
###############################################################################

def wrap_input(arr, device, dtype=torch.float, reshape=False):
    output = torch.from_numpy(np.array(arr)).type(dtype).to(device)
    if reshape:
        output = output.reshape(-1, 1)

    return output

###############################################################################
# Deep Q-Network
###############################################################################

class DQN(nn.Module):
    def __init__(self, layer_sizes):
        
        super(DQN, self).__init__()

        # Neural network
        modules = []
        for i in range(1,len(layer_sizes)-1):
            modules.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, obs):

        return self.net(obs)

###############################################################################
# Memory Replay
###############################################################################

class Memory(object):
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, n_samples):
        return zip(*random.sample(self.memory, n_samples))

    def __len__(self):
        return len(self.memory)

###############################################################################
# DQN Agent
###############################################################################

class DQNAgent:
    
    def __init__(self, layer_sizes, 
                 # in_scaler_min, in_scaler_max, # out_scaler_min, out_scaler_max, 
                 step_size = 0.5, discount_rate = 0.9,
                 epsilon = 0.1, final_epsilon = 0.1,
                 epsilon_decay = 10_000, learning_rate = 1e-4,
                 dqn_model = None, mem_buffer = [], max_mem_buffer = 1_000, 
                 batch_size = 32,  mean_G = 0):
        
        # Reinforcement Learning hyperparameters
        self.step_size = step_size # Step-size parameter
        self.discount_rate = discount_rate # Discount rate
        self.eps = epsilon
        self.initial_eps = epsilon
        self.final_eps = final_epsilon
        self.delta_eps = (epsilon - final_epsilon)/epsilon_decay

        # Deep Q-Network and Target Network
        if dqn_model != None:
            self.model = dqn_model
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DQN(layer_sizes).to(self.device)
            self.target = DQN(layer_sizes).to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        
        # Input data normalization
        # self.in_scaler = MinMaxScaler() # Input scaler
        # self.in_scaler.fit([in_scaler_min, in_scaler_max])
        
        # Output data normalization
        # self.out_scaler = MinMaxScaler()
        # self.out_scaler.fit([out_scaler_min, out_scaler_max])
        
        # Current cost and return
        self.mean_G = mean_G # Mean return
        
        # Memory buffer
        self.mem_buffer = Memory(max_mem_buffer)
        self.batch_size = batch_size
    
    ###########################################################################
    # Training methods
    ###########################################################################
    
    # Choose epsilon-greedy action
    def take_epsilon_greedy_action(self, state, set_of_actions):
        
        # Determining the inputs
        state_actions = [np.concatenate((state, action)) for action in set_of_actions]
        state_actions = np.array(state_actions)
        # state_actions = self.in_scaler.transform(np.array(state_actions))
              
        # Epsilon-greedy policy derived from Q
        if np.random.rand() < self.eps:
            i_ac = np.random.choice(range(len(set_of_actions)))
        else:
            state_actions = wrap_input(state_actions, self.device)#.unsqueeze(0)
            q_values = self.model(state_actions).detach().numpy().ravel()
            i_ac = np.argmax(q_values)
        action = set_of_actions[i_ac]
        
        # Return action
        return action
    
    # Store experience
    def store_exp(self, curr_state_action, curr_R, new_state, set_of_actions, terminal): # (curr_state_action, curr_R, new_state, set_of_actions, terminal)
        
        #We use a dictionnary to store them
        experience = (curr_state_action, {
            "current_state_action": curr_state_action,
            "reward": curr_R,
            "next_state": new_state,
            "set_of_actions": set_of_actions,
            "terminal": terminal
        })
    
        self.mem_buffer.store(experience)
    
    # Train DQN
    def train_model(self): # Train the supervised learning model
    
        # We shuffle the memory buffer and select a batch size of experiences
        curr_state_actions, batch_sample = self.mem_buffer.sample(min(self.batch_size, self.mem_buffer.__len__()))
        curr_q_values = []
        target_q_values = []
        
        # Train the model
        self.model.train() # Flag to indicate training mode
        
        # Current Q values
        # curr_state_actions = self.in_scaler.transform(np.array(curr_state_actions))
        curr_state_actions = wrap_input(curr_state_actions, device=self.device)
        curr_q_values = self.model(curr_state_actions)
                                        
        # We iterate over the selected experiences
        for experience in batch_sample:
            
            # We compute the Q-target using Bellman optimality equation
            q_target = experience['reward']
            
            if not experience['terminal']:
                
                # Determining the inputs
                state_actions = [np.concatenate((experience['next_state'], action)) 
                                 for action in experience['set_of_actions']]
                # state_actions = self.in_scaler.transform(np.array(state_actions))
                state_actions = wrap_input(state_actions, self.device)
                
                # Computing Q values for the next state
                q_values = self.model(state_actions).detach().numpy().ravel()
                
                # Calculating Q-target
                q_target += (self.discount_rate * np.max(q_values))
                
            # Storing Q target
            target_q_values.append(q_target)
                
        # New Q for the current state
        curr = curr_q_values.detach().numpy().ravel() # Current Q values
        new_q_values = curr + self.step_size*(target_q_values - curr) # New values
        new_q_values = wrap_input(new_q_values, device=self.device, reshape=True) # To torch tensor
        
        # Compute loss
        loss = self.loss_fn(curr_q_values, new_q_values) # Calculate losses
        self.optimizer.zero_grad() # Zero the previous gradients
        loss.backward() # Backward the losses

        # Backprop
        self.optimizer.step() # Do the update
    
    # Synchronize Target Network
    def target_sync(self):
        self.target.load_state_dict(self.model.state_dict())
    
    # Compute return average
    def compute_mean_return(self, G, weight, episode):

        dG = self.mean_G
        
        if weight == 1:
            self.mean_G = (self.mean_G*episode + G)/(episode + 1)
        
        elif weight < 1:
            self.mean_G = (self.mean_G*weight*(1 - weight**episode) / 
                                        (1 - weight) + G)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
        else:
            print('Invalid weight for the mean return.')

        dG = self.mean_G - dG
        
        return self.mean_G
    
    def update_epsilon(self, ep):
        self.eps = max(self.initial_eps - ep * self.delta_eps, self.final_eps)
        
    def get_epsilon(self):
        return round(self.eps, 2)
    
    def get_mem_len(self):
        return self.mem_buffer.__len__()
    
    ###########################################################################
    # Evaluation methods
    ###########################################################################
    
    def compute_single_action(self, state, set_of_actions):
        
        # Determining the inputs
        state_actions = [np.concatenate((state, action)) for action in set_of_actions]
        # state_actions = self.in_scaler.transform(np.array(state_actions))
        state_actions = wrap_input(state_actions, self.device)
        
        # Computing Q values
        q_values = self.model(state_actions).detach().numpy().ravel()
                
        # Greedy policy derived from Q
        action = set_of_actions[np.argmax(q_values)]
        
        # Return action
        return action
    
    def get_dqn_model(self):
        return self.model