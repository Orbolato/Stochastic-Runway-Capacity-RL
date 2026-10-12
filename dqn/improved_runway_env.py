import numpy as np
from gymnasium import spaces
import gymnasium as gym
from typing import Optional, Dict, List, Tuple, Any, Union

class RunwayEnv(gym.Env):
    """
    Runway Configuration Management Environment for Air Traffic Control
    
    This environment simulates the decision-making process for runway configuration
    management at an airport, considering arrival and departure queues, weather conditions,
    and wind states.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self, 
        observation_low: List[float], 
        observation_high: List[float], 
        data: Any,
        training_days: List[int], 
        u: np.ndarray,
        u_max: int,
        v: np.ndarray,
        v_max: int,
        possible_actions: np.ndarray,
        valid_bool: np.ndarray,
        wind_states: List[List[int]], 
        alpha: float = 2.0, 
        delta: float = 0.3,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the runway environment.
        
        Parameters:
        -----------
        observation_low : List[float]
            Lower bounds for the observation space
        observation_high : List[float]
            Upper bounds for the observation space
        data : DataFrame
            Traffic data with METAR information
        training_days : List[int]
            Available days for training
        u : np.ndarray
            Allowed arrivals for each weather condition (shape: [wind_states, weather_conditions, runway_configs, max_arrivals])
        v : np.ndarray
            Maximum departures for each weather condition (shape: [wind_states, weather_conditions, runway_configs, max_arrivals])
        possible_actions : np.ndarray
            Set of all possible actions (shape: [n_actions, 3]) where each action is [runway_config, arrivals, departures]
        valid_bool : np.ndarray
            Boolean mask for valid actions given wind state, weather condition, arrival queue, departure queue
        wind_states : List[List[int]]
            Available runway configurations for each wind state
        alpha : float, optional
            Arrival priority factor, by default 2.0
        delta : float, optional
            Proportional cost of switching runway configuration, by default 0.3
        render_mode : str, optional
            Mode for rendering the environment ("human" or "rgb_array"), by default None
        """
        # Validate inputs
        assert len(observation_low) == len(observation_high), "Observation bounds must have same dimensions"
        
        # Observation space
        self.observation_space = spaces.Box(
            np.array(observation_low, dtype=np.float32),
            np.array(observation_high, dtype=np.float32)
        )
        
        # Action space is defined implicitly through the set of possible actions
        # We'll use a discrete space for clarity, though actual action selection uses masking
        self.action_space = spaces.Discrete(len(possible_actions))
        
        # Store parameters
        self.alpha = alpha
        self.delta = delta
        self.data = data
        self.training_days = training_days
        self.u = u
        self.u_max = u_max
        self.v = v
        self.v_max = v_max
        self.possible_actions = possible_actions
        self.valid_bool = valid_bool
        self.wind_states = wind_states
        
        # Initialize render mode
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Initialize state variables
        self.time_step = 0
        self.day = None
        self.day_data = None
        self.obs = None
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self.obs
    
    def _get_info(self) -> Dict[str, int]:
        """Get information about the current state."""
        return {
            'day': self.day,
            'runway_configuration': int(self.obs[0]),
            'arrival_queue': int(self.obs[1]),
            'departure_queue': int(self.obs[2]),
            'time_step': self.time_step
        }
    
    def compute_reward(
        self, 
        RC_prev: int, 
        RC_t: int, 
        a_prev: int, 
        d_prev: int, 
        a_t: int, 
        d_t: int
    ) -> float:
        """
        Compute the reward for a state transition.
        
        Parameters:
        -----------
        RC_prev : int
            Previous runway configuration
        RC_t : int
            Current runway configuration
        a_prev : int
            Previous arrival queue
        d_prev : int
            Previous departure queue
        a_t : int
            Current arrival queue
        d_t : int
            Current departure queue
            
        Returns:
        --------
        float
            The reward value
        """
        # Determine if runway configuration changed
        config_switch = int(RC_t != RC_prev)
        
        # Calculate queue penalties
        queue_penalty = -(self.alpha * a_t**2 + d_t**2)
        
        # Calculate switch penalty if applicable
        switch_penalty = -config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2)
        
        return queue_penalty + switch_penalty
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None, 
        day: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed, by default None
        options : Dict, optional
            Additional options for reset, by default None
        day : int, optional
            Specific day to use, by default None (random selection)
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            Initial observation and info dictionary
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset time step
        self.time_step = 0
        
        # Select day to start
        if day is None:
            self.day = self.np_random.choice(self.training_days)
        else:
            self.day = day
        
        # Get data for the selected day
        self.day_data = self.data.loc[self.data['day'] == self.day].reset_index(drop=True)
        
        if len(self.day_data) == 0:
            raise ValueError(f"No data available for day {self.day}")
        
        # Initialize observation
        self.obs = np.array([
            0,  # Runway configuration (0 = None, 1 = 10R_10L, 2 = 28L_28R)
            self.day_data.loc[self.time_step, 'arrival_demand_40'],  # Arrival queue
            self.day_data.loc[self.time_step, 'departure_demand'],  # Departure queue
            self.day_data.loc[self.time_step, 'weather_condition'],  # Weather condition
            self.day_data.loc[self.time_step, 'wind_state'],  # Wind state
        ])
        
        # Return observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action_index: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action_index : Union[int, np.ndarray]
            Index of the action to take or the action array itself
            
        Returns:
        --------
        Tuple[np.ndarray, float, bool, bool, Dict]
            New observation, reward, terminated flag, truncated flag, and info dictionary
        """
        # Convert action index to action if needed
        if isinstance(action_index, (int, np.integer)):
            action = self.possible_actions[action_index]
        else:
            action = action_index
        
        # Get current state
        RC_i = int(self.obs[0])  # Previous runway configuration
        a_i = int(self.obs[1])   # Arrival queue
        d_i = int(self.obs[2])   # Departure queue
        
        # Apply action
        RC_f = action[0]  # New runway configuration
        a_f = a_i - action[1]  # Updated arrival queue
        d_f = d_i - action[2]  # Updated departure queue
            
        # Validate queue values
        if a_f < 0 or d_f < 0:
            raise ValueError('Invalid queue value detected.')
        
        # Compute reward
        reward = self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f)
        
        # Check if episode is done
        terminated = self.time_step == len(self.day_data) - 1
        
        if not terminated:
            # Advance to next time step
            self.time_step += 1
            
            # Calculate new queues
            a_i = a_f + self.day_data.loc[self.time_step, 'arrival_demand_40']
            d_i = d_f + self.day_data.loc[self.time_step, 'departure_demand']
            
            # Update observation
            self.obs = np.array([
                RC_f,  # New runway configuration
                a_i,   # Updated arrival queue
                d_i,   # Updated departure queue
                self.day_data.loc[self.time_step, 'weather_condition'],  # Weather condition
                self.day_data.loc[self.time_step, 'wind_state'],  # Wind state
            ])
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get the set of valid actions for the current state.
        
        Returns:
        --------
        np.ndarray
            Array of valid actions
        """
        # Get current state
        a_i = int(self.obs[1])  # Arrival queue
        d_i = int(self.obs[2])  # Departure queue
        wc = int(self.obs[3])   # Weather condition
        ws = int(self.obs[4])   # Wind state
        
        # Limit queues by maximum possible action values
        a_i = min(a_i, self.u_max)
        d_i = min(d_i, self.v_max)
        
        # Validate queue values
        if a_i < 0 or d_i < 0:
            raise ValueError('Invalid queue value detected.')
            
        # # Creating set of valid actions for action masking
        # valid_actions = [] # Valid actions list
        # for rc in self.wind_states[ws]: # For each runway configuration available for the given wind state
        #     u_values = self.u[wc, rc]  # Precompute u values
        #     v_values = self.v[wc, rc]  # Precompute v values
        #     arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding obvious suboptimal actions)
        #     arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
        #     for arr in range(arr_min, arr_lim): # For each possible arrival count
        #         if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
        #             dep = v_values[arr] # Departure service rates are given by the departure capacity
        #             valid_actions.append(np.array([rc, arr, dep], dtype=np.int64)) # Append to the list of valid actions
        #     dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
        #     valid_actions.append(np.array([rc, arr_lim, dep], dtype=np.int64)) # Append to the list of valid actions
        # valid_actions = np.array(valid_actions, dtype=np.int64) # Convert the list to numpy array
        
        # # Action mask
        # return valid_actions
        
        # Get valid actions using the mask
        return self.possible_actions[self.valid_bool[ws, wc, a_i, d_i]]
    
    def base_action(self) -> np.ndarray:
        """
        Get the baseline action based on historical data.
        
        Returns:
            The baseline action as [RC, arrivals, departures]
        """
        # Extract current queues
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        
        # Get baseline values from historical data
        rc = self.day_data.loc[self.time_step, 'base_configuration']
        arrivals = self.day_data.loc[self.time_step, 'base_arrivals_40']
        departures = self.day_data.loc[self.time_step, 'base_departures']
        
        # Limit by available queues
        if arrivals > arrivals_queue:
            # Can't serve more than what's in queue
            overflow = arrivals - arrivals_queue
            arrivals = arrivals_queue
            
            # Apply overflow to next time step if not at end
            if self.time_step < len(self.day_data) - 1:
                self.day_data.loc[self.time_step + 1, 'base_arrivals_40'] += overflow
        
        if departures > departures_queue:
            # Can't serve more than what's in queue
            overflow = departures - departures_queue
            departures = departures_queue
            
            # Apply overflow to next time step if not at end
            if self.time_step < len(self.day_data) - 1:
                self.day_data.loc[self.time_step + 1, 'base_departures'] += overflow
        
        return np.array([rc, arrivals, departures], dtype=np.int64)
    
    def naive_action(self) -> np.ndarray:
        """
        Get the naive (greedy) action for the current state.
        
        Returns:
        --------
        np.ndarray
            Greedy action [runway_config, arrivals, departures]
        """
        # Get current state
        RC_i = int(self.obs[0])  # Runway configuration
        a_i = int(self.obs[1])   # Arrival queue
        d_i = int(self.obs[2])   # Departure queue
        
        # Validate queue values
        if a_i < 0 or d_i < 0:
            raise ValueError('Invalid queue value detected.')
        
        # Get valid actions
        valid_actions = self.get_valid_actions()
        
        if len(valid_actions) == 0:
            # Fallback if no valid actions
            return np.array([RC_i, 0, 0], dtype=np.int64)
        
        # Calculate rewards for all valid actions
        rewards = []
        for action in valid_actions:
            RC_f = action[0]
            a_f = a_i - action[1]
            d_f = d_i - action[2]
            rewards.append(self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f))
        
        # Return the action with the highest reward
        return valid_actions[np.argmax(rewards)]
    
    def _render_frame(self):
        """Render the current state of the environment."""
        if self.render_mode == "human" and self.window is None:
            try:
                import pygame
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((800, 600))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont(None, 30)
            except ImportError:
                self.render_mode = None
                return
        
        if self.window is not None:
            self.window.fill((255, 255, 255))
            
            # Draw runway configuration
            RC = int(self.obs[0])
            rc_text = ["None", "10R_10L", "28L_28R"][RC]
            rc_surf = self.font.render(f"RC: {rc_text}", True, (0, 0, 0))
            self.window.blit(rc_surf, (50, 50))
            
            # Draw queues
            a_queue = int(self.obs[1])
            d_queue = int(self.obs[2])
            a_surf = self.font.render(f"Arrival Queue: {a_queue}", True, (0, 0, 0))
            d_surf = self.font.render(f"Departure Queue: {d_queue}", True, (0, 0, 0))
            self.window.blit(a_surf, (50, 100))
            self.window.blit(d_surf, (50, 150))
            
            # Draw weather and wind state
            wc = int(self.obs[3])
            ws = int(self.obs[4])
            wc_text = ["VMC", "IMC", "LIMC"][wc]
            ws_text = ["Both available", "10R_10L available", "28L_28R available", "Both unavailable"][ws]
            wc_surf = self.font.render(f"Weather: {wc_text}", True, (0, 0, 0))
            ws_surf = self.font.render(f"Wind: {ws_text}", True, (0, 0, 0))
            self.window.blit(wc_surf, (50, 200))
            self.window.blit(ws_surf, (50, 250))
            
            # Draw day and time step
            time_surf = self.font.render(f"Day: {self.day}, Time: {self.time_step}", True, (0, 0, 0))
            self.window.blit(time_surf, (50, 300))
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
