import numpy as np
import random
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from typing import List, Tuple, Dict, Union, Optional, Any, Deque

###############################################################################
# Auxiliary functions
###############################################################################

def wrap_input(arr: np.ndarray, device: torch.device, dtype=torch.float, reshape: bool = False) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor with proper device placement.
    
    Args:
        arr: Input numpy array
        device: Target device (CPU or CUDA)
        dtype: Tensor data type
        reshape: Whether to reshape tensor to (N, 1)
        
    Returns:
        PyTorch tensor on specified device
    """
    output = torch.from_numpy(np.array(arr)).type(dtype).to(device)
    if reshape:
        output = output.reshape(-1, 1)
    return output

###############################################################################
# Deep Q-Network
###############################################################################

class DQN(nn.Module):
    def __init__(self, layer_sizes: List[int], activation=nn.LeakyReLU) -> None:
        """
        Deep Q-Network model implementation.
        
        Args:
            layer_sizes: List of neurons per layer (including input and output sizes)
            activation: Activation function class to use between layers
        """
        super(DQN, self).__init__()
        
        # Build neural network with proper layer initialization
        modules = []
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            
            # Xavier/Glorot initialization for better training stability
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            modules.append(linear)
            
            # Add activation for all but the last layer
            if i < len(layer_sizes) - 2:
                modules.append(activation())
                
        self.net = nn.Sequential(*modules)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.net(obs)

###############################################################################
# Memory Replay Buffer
###############################################################################

class Memory:
    def __init__(self, maxlen: int) -> None:
        """
        Experience replay buffer implementation.
        
        Args:
            maxlen: Maximum number of experiences to store
        """
        self.memory: Deque = deque(maxlen=maxlen)
        # Track priorities for potential future implementation of prioritized replay
        self.priorities = np.ones(maxlen)
        self.position = 0
        self.size = 0
        self.maxlen = maxlen

    def store(self, experience: Tuple[Any, Dict[str, Any]]) -> None:
        """
        Store a new experience in the buffer.
        
        Args:
            experience: Tuple of (state_action, experience_dict)
        """
        if self.size < self.maxlen:
            self.memory.append(experience)
            self.size += 1
        else:
            self.memory[self.position] = experience
            self.position = (self.position + 1) % self.maxlen

    def sample(self, n_samples: int) -> Tuple[List, List]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            Tuple of (states_actions, experience_dicts)
        """
        sample_size = min(n_samples, self.__len__())
        return zip(*random.sample(self.memory, sample_size))

    def __len__(self) -> int:
        """Return current size of memory buffer"""
        return len(self.memory)

###############################################################################
# DQN Agent
###############################################################################

class DQNAgent:
    def __init__(
        self,
        layer_sizes: List[int],
        step_size: float = 0.5,
        discount_rate: float = 0.99,
        epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        epsilon_decay: int = 10_000,
        learning_rate: float = 1e-4,
        target_update_freq: int = 1000,
        max_mem_buffer: int = 100_000,
        batch_size: int = 32,
        use_scaler: bool = False,
        in_scaler_min: Optional[List[float]] = None,
        in_scaler_max: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Deep Q-Network Agent implementation.
        
        Args:
            layer_sizes: List of neurons per layer (including input and output)
            step_size: Learning rate for Q-value update
            discount_rate: Discount factor for future rewards (gamma)
            epsilon: Initial exploration rate
            final_epsilon: Final exploration rate
            epsilon_decay: Steps to decay from initial to final epsilon
            learning_rate: Learning rate for optimizer
            target_update_freq: Frequency to update target network (in steps)
            max_mem_buffer: Maximum size of experience replay buffer
            batch_size: Batch size for training
            use_scaler: Whether to use input normalization
            in_scaler_min: Min values for input scaler
            in_scaler_max: Max values for input scaler
            device: Torch device (will autodetect if None)
        """
        # Set device (GPU if available, otherwise CPU)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Reinforcement Learning hyperparameters
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.eps = epsilon
        self.initial_eps = epsilon
        self.final_eps = final_epsilon
        self.delta_eps = (epsilon - final_epsilon) / epsilon_decay
        
        # Training parameters
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Initialize neural networks
        self.model = DQN(layer_sizes).to(self.device)
        self.target = DQN(layer_sizes).to(self.device)
        
        # Initialize target network with same weights
        self.target_sync()
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        
        # Input data normalization
        self.use_scaler = use_scaler
        if use_scaler and in_scaler_min is not None and in_scaler_max is not None:
            self.in_scaler = MinMaxScaler()
            self.in_scaler.fit([in_scaler_min, in_scaler_max])
        else:
            self.in_scaler = None
        
        # Performance tracking
        self.mean_G = 0  # Mean return
        self.training_losses = []
        self.episode_rewards = []
        
        # Memory buffer
        self.mem_buffer = Memory(max_mem_buffer)
    
    ###########################################################################
    # Training methods
    ###########################################################################
    
    def take_epsilon_greedy_action(self, state: np.ndarray, set_of_actions: List[np.ndarray]) -> np.ndarray:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            set_of_actions: List of possible actions
            
        Returns:
            Selected action
        """
        # Determine state-action pairs
        state_actions = [np.concatenate((state, action)) for action in set_of_actions]
        state_actions = np.array(state_actions)
        
        # Apply scaler if enabled
        if self.use_scaler and self.in_scaler is not None:
            state_actions = self.in_scaler.transform(state_actions)
        
        # Epsilon-greedy policy
        if np.random.rand() < self.eps:
            # Exploration: random action
            i_ac = np.random.choice(range(len(set_of_actions)))
        else:
            # Exploitation: greedy action
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                state_actions_tensor = wrap_input(state_actions, self.device)
                q_values = self.model(state_actions_tensor).cpu().numpy().ravel()
            i_ac = np.argmax(q_values)
            
        return set_of_actions[i_ac]
    
    def store_exp(self, 
                 curr_state_action: np.ndarray, 
                 curr_R: float, 
                 new_state: np.ndarray, 
                 set_of_actions: List[np.ndarray], 
                 terminal: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            curr_state_action: Current state-action pair
            curr_R: Current reward
            new_state: Next state
            set_of_actions: Available actions in next state
            terminal: Whether this is a terminal state
        """
        # Create experience dictionary
        experience = (curr_state_action, {
            "current_state_action": curr_state_action,
            "reward": curr_R,
            "next_state": new_state,
            "set_of_actions": set_of_actions,
            "terminal": terminal
        })
        
        # Store in replay buffer
        self.mem_buffer.store(experience)
    
    def train_model(self) -> float:
        """
        Train the DQN model using a batch from experience replay.
        
        Returns:
            Training loss value
        """
        # Check if we have enough samples
        if len(self.mem_buffer) < self.batch_size:
            return 0.0
            
        # Sample experiences from memory buffer
        curr_state_actions, batch_sample = self.mem_buffer.sample(self.batch_size)
        
        # Prepare current state-actions for network
        curr_state_actions_np = np.array(curr_state_actions)
        if self.use_scaler and self.in_scaler is not None:
            curr_state_actions_np = self.in_scaler.transform(curr_state_actions_np)
            
        # Set model to training mode
        self.model.train()
        
        # Get current Q values
        curr_state_actions_tensor = wrap_input(curr_state_actions_np, device=self.device)
        curr_q_values = self.model(curr_state_actions_tensor)
        
        # Calculate target Q values
        target_q_values = []
        for experience in batch_sample:
            # Compute Q-target using Bellman equation
            q_target = experience['reward']
            
            if not experience['terminal']:
                # Get next state-action values
                state_actions = [np.concatenate((experience['next_state'], action)) 
                               for action in experience['set_of_actions']]
                state_actions_np = np.array(state_actions)
                
                if self.use_scaler and self.in_scaler is not None:
                    state_actions_np = self.in_scaler.transform(state_actions_np)
                
                # Use target network for more stable learning
                with torch.no_grad():
                    state_actions_tensor = wrap_input(state_actions_np, self.device)
                    # Use target network for better stability
                    next_q_values = self.target(state_actions_tensor).cpu().numpy().ravel()
                    
                # Update target using max Q-value (DQN)
                q_target += self.discount_rate * np.max(next_q_values)
                
            target_q_values.append(q_target)
        
        # Convert current Q values to numpy
        curr_q_numpy = curr_q_values.detach().cpu().numpy().ravel()
        
        # Calculate new Q values with step size (learning rate)
        new_q_values = curr_q_numpy + self.step_size * (np.array(target_q_values) - curr_q_numpy)
        
        # Convert back to tensor
        target_q_tensor = wrap_input(new_q_values, device=self.device, reshape=True)
        
        # Calculate loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.loss_fn(curr_q_values, target_q_tensor)
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_sync()
        
        # Track training loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def target_sync(self) -> None:
        """Synchronize target network with policy network"""
        self.target.load_state_dict(self.model.state_dict())
    
    def compute_mean_return(self, G: float, weight: float, episode: int) -> float:
        """
        Update and return the running average of episode returns.
        
        Args:
            G: Episode return
            weight: Weight factor for moving average
            episode: Current episode number
            
        Returns:
            Updated mean return
        """
        prev_mean_G = self.mean_G
        
        if weight == 1:
            # Simple average
            self.mean_G = (self.mean_G * episode + G) / (episode + 1)
        elif 0 < weight < 1:
            # Weighted average
            self.mean_G = (self.mean_G * weight * (1 - weight**episode) / 
                          (1 - weight) + G) * ((1 - weight) / 
                                              (1 - weight**(episode + 1)))
        else:
            raise ValueError('Weight must be between 0 and 1')
        
        # Track episode reward
        self.episode_rewards.append(G)
        
        return self.mean_G
    
    def update_epsilon(self, step: int) -> None:
        """
        Update exploration rate using epsilon decay.
        
        Args:
            step: Current training step
        """
        self.eps = max(self.initial_eps - step * self.delta_eps, self.final_eps)
    
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return round(self.eps, 3)
    
    def get_mem_len(self) -> int:
        """Get current memory buffer size"""
        return len(self.mem_buffer)
    
    ###########################################################################
    # Evaluation methods
    ###########################################################################
    
    def compute_single_action(self, state: np.ndarray, set_of_actions: List[np.ndarray]) -> np.ndarray:
        """
        Compute best action for given state (no exploration).
        
        Args:
            state: Current state observation
            set_of_actions: List of possible actions
            
        Returns:
            Best action according to current policy
        """
        # Create state-action pairs
        state_actions = [np.concatenate((state, action)) for action in set_of_actions]
        state_actions_np = np.array(state_actions)
        
        # Apply scaling if enabled
        if self.use_scaler and self.in_scaler is not None:
            state_actions_np = self.in_scaler.transform(state_actions_np)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Compute Q values without gradient tracking
        with torch.no_grad():
            state_actions_tensor = wrap_input(state_actions_np, self.device)
            q_values = self.model(state_actions_tensor).cpu().numpy().ravel()
        
        # Select action with highest Q-value
        best_action_idx = np.argmax(q_values)
        
        return set_of_actions[best_action_idx]
    
    def save_model(self, path: str) -> None:
        """
        Save model weights to file.
        
        Args:
            path: File path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mean_G': self.mean_G,
            'epsilon': self.eps,
        }, path)
        
    def load_model(self, path: str) -> None:
        """
        Load model weights from file.
        
        Args:
            path: File path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.mean_G = checkpoint['mean_G']
        self.eps = checkpoint['epsilon']
        
    def get_dqn_model(self) -> nn.Module:
        """Return the underlying DQN model"""
        return self.model
