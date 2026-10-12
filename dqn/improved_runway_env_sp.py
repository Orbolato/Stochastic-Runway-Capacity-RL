import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class RunwayEnv(gym.Env):
    """
    Runway Environment for reinforcement learning-based optimization of runway operations.
    
    This environment simulates a runway configuration optimization problem where an agent needs
    to decide on runway configurations and service rates based on arrival/departure queues,
    weather conditions, and wind states.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, alpha, delta, taf_data, std_taf_data, data,
                 congested_days, u, u_max, v, v_max, 
                 possible_actions, valid_bool, wind_states, 
                 observation_low, observation_high,
                 arr_mu, arr_sigma, dep_mu, dep_sigma,
                 arr_mean, arr_std, dep_mean, dep_std,
                 p_wc_trans, p_ws_trans,
                 p_wc_error, p_ws_error,
                 chosen_day=None, render_mode=None):
        """
        Initialize the runway environment.
        
        Parameters:
        -----------
        alpha : float
            Weight for arrival queue in reward function
        delta : float
            Weight for configuration switch cost in reward function
        taf_data : pandas.DataFrame
            Training data with forecast information
        std_taf_data : pandas.DataFrame
            Standardized training data
        data : pandas.DataFrame
            Test data
        congested_days : list
            List of congested days to select from
        u : numpy.ndarray
            Arrival service rates for different states
        v : numpy.ndarray
            Departure service rates for different states
        possible_actions : list
            List of all possible actions
        wind_states : list
            Available runway configurations for each wind state
        observation_low : list
            Lower bounds for observation space
        observation_high : list
            Upper bounds for observation space
        arr_mu, arr_sigma, dep_mu, dep_sigma : float
            Parameters for schedule error simulation
        arr_mean, arr_std, dep_mean, dep_std : numpy.ndarray
            Schedule error parameters per demand level
        p_wc_trans, p_ws_trans : numpy.ndarray
            Transition probabilities for weather condition and wind state
        p_wc_error, p_ws_error : numpy.ndarray
            Error probabilities for weather condition and wind state
        chosen_day : str, optional
            Specific day to simulate, or random if None
        render_mode : str, optional
            Rendering mode, one of None, "human", or "rgb_array"
        """
        self._init_rendering(render_mode = render_mode)
        self.window_size = 512  # Size of PyGame window
        
        # Store input parameters
        self._store_parameters(
            alpha, delta, taf_data, std_taf_data, data, congested_days,
            u, u_max, v, v_max, possible_actions, valid_bool, wind_states, 
            arr_mu, arr_sigma, dep_mu, dep_sigma, arr_mean, arr_std, dep_mean, dep_std,
            p_wc_trans, p_ws_trans, p_wc_error, p_ws_error
        )
        
        # Select day if not provided
        self.day = self._select_day(chosen_day)
        
        # Define observation and action spaces
        self._define_spaces(observation_low, observation_high)
        
        # Initialize rendering variables
        self._init_rendering(render_mode)
    
    def _store_parameters(self, alpha, delta, taf_data, std_taf_data, data,
                          congested_days, u, u_max, v, v_max,
                          possible_actions, valid_bool, wind_states,
                          arr_mu, arr_sigma, dep_mu, dep_sigma,
                          arr_mean, arr_std, dep_mean, dep_std,
                          p_wc_trans, p_ws_trans, p_wc_error, p_ws_error):
        """Store all input parameters as instance variables."""
        # Scheduling parameters
        self.arr_mu = arr_mu
        self.arr_sigma = arr_sigma
        self.dep_mu = dep_mu
        self.dep_sigma = dep_sigma
        self.arr_mean = arr_mean
        self.arr_std = arr_std
        self.dep_mean = dep_mean
        self.dep_std = dep_std
        
        # Transition probabilities
        self.p_wc_trans = p_wc_trans
        self.p_wc_error = p_wc_error
        self.p_ws_trans = p_ws_trans
        self.p_ws_error = p_ws_error
        
        # Data
        self.training_data = taf_data
        self.std_training_data = std_taf_data
        self.test_data = data
        self.days = congested_days
        
        # Service rates and action definitions
        self.u = u  # Arrival service rates
        self.u_max = u_max
        self.v = v  # Departure service rates
        self.v_max = v_max
        self.possible_actions = possible_actions
        self.valid_bool = valid_bool
        self.wind_states = wind_states
        
        # Reward function parameters
        self.alpha = alpha
        self.delta = delta
    
    def _select_day(self, chosen_day):
        """Select a day for simulation, either randomly or as specified."""
        if chosen_day is None:
            chosen_day = np.random.choice(self.days)
        print(f'Period {chosen_day} chosen.\n')
        return chosen_day
    
    def _define_spaces(self, observation_low, observation_high):
        """Define observation and action spaces."""
        # Observation space
        observation_low = np.array(observation_low, dtype=np.float32)
        observation_high = np.array(observation_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high)

        # Action space
        self.action_space = spaces.Discrete(len(self.possible_actions))
    
    def _init_rendering(self, render_mode):
        """Initialize rendering-related variables."""
        # Validate render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
        # Rendering attributes
        self.window = None
        self.clock = None
        self.screen_shot_count = 0
        self.screen_shot_button = pygame.Rect(10, 10, 120, 30)  # (x, y, width, height)
        
        # Colors for rendering
        self.colors = {
            "background": (240, 240, 240),
            "text": (20, 20, 20),
            "arrival_queue": (70, 130, 180),   # Steel blue
            "departure_queue": (220, 120, 50), # Orange
            "runway_10R_10L": (50, 200, 50),   # Green
            "runway_28L_28R": (200, 50, 50),   # Red
            "unavailable": (120, 120, 120),    # Gray
            "button": (100, 100, 240),         # Blue
            "button_hover": (130, 130, 255),   # Light blue
            "button_text": (255, 255, 255)     # White
        }

    def _get_obs(self):
        """
        Get the current observation.
        
        Returns:
        --------
        numpy.ndarray
            Current observation
        """
        return self.obs
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
        --------
        dict
            Dictionary with additional information
        """
        return {
            'day': self.day,                # Day being analyzed
            'runway_configuration': self.obs[0],  # Runway configuration in use
            'arrival_queue': self.obs[1],        # Arrival queue
            'departure_queue': self.obs[2]       # Departure queue
        }
    
    def compute_reward(self, RC_prev, RC_t, a_prev, d_prev, a_t, d_t):
        """
        Compute reward based on queue sizes and configuration switches.
        
        Parameters:
        -----------
        RC_prev : int
            Previous runway configuration
        RC_t : int
            Current runway configuration
        a_prev : int
            Previous arrival queue size
        d_prev : int
            Previous departure queue size
        a_t : int
            Current arrival queue size
        d_t : int
            Current departure queue size
            
        Returns:
        --------
        float
            Reward value
        """
        
        # Determine if runway configuration changed
        config_switch = int(RC_t != RC_prev)
        
        # Calculate queue penalties
        queue_penalty = -(self.alpha * a_t**2 + d_t**2)
        
        # Calculate switch penalty if applicable
        switch_penalty = -config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2)
        
        return queue_penalty + switch_penalty

    def _generate_initial_state(self, training=True):
        """
        Generate the initial state for an episode.
        
        Parameters:
        -----------
        training : bool
            Whether the environment is in training mode
            
        Returns:
        --------
        numpy.ndarray
            Initial state
        """
        if training:
            # Get training data for the selected day
            self.day_data = self.training_data.loc[self.training_data['day'] == self.day].reset_index(drop=True)
            
            # Generate schedule errors
            sched_arr = self.day_data.loc[self.time_step, 'arrival_demand_40'].astype(int)
            sched_dep = self.day_data.loc[self.time_step, 'departure_demand'].astype(int)
            arr_err = round(np.random.normal(loc=self.arr_mean[sched_arr], scale=self.arr_std[sched_arr]))
            dep_err = round(np.random.normal(loc=self.dep_mean[sched_dep], scale=self.dep_std[sched_dep]))
            
            # Calculate initial queues
            a_i = max(0, self.day_data.loc[self.time_step, 'arrival_demand_40'] + arr_err)
            d_i = max(0, self.day_data.loc[self.time_step, 'departure_demand'] + dep_err)
            
            # Sample initial weather condition and wind state
            wc = np.random.choice([0, 1, 2], p=self.p_wc_error[self.day_data.loc[self.time_step, 'weather_condition']])
            ws = np.random.choice([0, 1, 2, 3], p=self.p_ws_error[self.day_data.loc[self.time_step, 'wind_state']])
            self.prev_wc = wc
            self.prev_ws = ws
        else:
            # Get test data for the selected day
            self.day_data = self.test_data.loc[self.test_data['day'] == self.day].reset_index(drop=True)
            
            # Use actual values from test data
            a_i = self.day_data.loc[self.time_step, 'arrival_demand_40']
            d_i = self.day_data.loc[self.time_step, 'departure_demand']
            wc = self.day_data.loc[self.time_step, 'weather_condition']
            ws = self.day_data.loc[self.time_step, 'wind_state']
        
        # Return initial state vector
        return np.array([
            0,    # Runway configuration (0 = None, 1 = 10R_10L, 2 = 28L_28R)
            a_i,  # Arrival queue (integer)
            d_i,  # Departure queue (integer)
            wc,   # Weather condition (0 = VMC, 1 = IMC, 2 = LIMC)
            ws    # Wind state (0 = Both available, 1 = 10R_10L available, 2 = 28L_28R available, 3 = Both unavailable)
        ])
    
    def reset(self, seed=None, options=None, training=True):
        """
        Reset the environment to its initial state.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional options
        training : bool, optional
            Whether the environment is in training mode
            
        Returns:
        --------
        tuple
            Initial observation and info
        """
        # Seed the random number generator
        super().reset(seed=seed)
        
        # Initialize time step
        self.time_step = 0
        
        # Generate initial state
        self.obs = self._generate_initial_state(training)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _generate_next_state(self, RC_f, a_f, d_f, training=True):
        """
        Generate the next state given the current action results.
        
        Parameters:
        -----------
        RC_f : int
            Runway configuration after action
        a_f : int
            Arrival queue after action
        d_f : int
            Departure queue after action
        training : bool
            Whether the environment is in training mode
            
        Returns:
        --------
        numpy.ndarray
            Next state
        """
        # New queues calculation depends on training mode
        if training:
            # Generate schedule errors for demand
            sched_arr = self.day_data.loc[self.time_step, 'arrival_demand_40'].astype(int)
            sched_dep = self.day_data.loc[self.time_step, 'departure_demand'].astype(int)
            arr_err = round(np.random.normal(loc=self.arr_mean[sched_arr], scale=self.arr_std[sched_arr]))
            dep_err = round(np.random.normal(loc=self.dep_mean[sched_dep], scale=self.dep_std[sched_dep]))
            
            # New queues with errors
            a_i = a_f + max(0, self.day_data.loc[self.time_step, 'arrival_demand_40'] + arr_err)
            d_i = d_f + max(0, self.day_data.loc[self.time_step, 'departure_demand'] + dep_err)
            
            # Sample next weather condition and wind state with transitions
            p_wc = self.p_wc_error[self.day_data.loc[self.time_step, 'weather_condition']] * self.p_wc_trans[self.prev_wc] + 1e-12
            p_wc = p_wc / np.sum(p_wc)
            p_ws = self.p_ws_error[self.day_data.loc[self.time_step, 'wind_state']] * self.p_ws_trans[self.prev_ws] + 1e-12
            p_ws = p_ws / np.sum(p_ws)
            wc = np.random.choice([0, 1, 2], p=p_wc)
            ws = np.random.choice([0, 1, 2, 3], p=p_ws)
            self.prev_wc = wc
            self.prev_ws = ws
        else:
            # New queues from deterministic data
            a_i = a_f + self.day_data.loc[self.time_step, 'arrival_demand_40']
            d_i = d_f + self.day_data.loc[self.time_step, 'departure_demand']
            
            # Get weather condition and wind state from data
            wc = self.day_data.loc[self.time_step, 'weather_condition']
            ws = self.day_data.loc[self.time_step, 'wind_state']
        
        # Return next state vector
        return np.array([
            RC_f,  # Runway configuration
            a_i,   # Arrival queue
            d_i,   # Departure queue
            wc,    # Weather condition
            ws     # Wind state
        ])
    
    def step(self, action, training=True):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action_index : int
            Index of the action to take
        training : bool, optional
            Whether the environment is in training mode
            
        Returns:
        --------
        tuple
            Observation, reward, terminated, truncated, info
        """
        
        # Extract state and action components
        RC_i = int(self.obs[0])  # Previous runway configuration
        a_i = int(self.obs[1])   # Current arrival queue
        d_i = int(self.obs[2])   # Current departure queue
        RC_f = action[0]         # New runway configuration
        a_f = a_i - action[1]    # Arrival queue after action
        d_f = d_i - action[2]    # Departure queue after action
        
        # Calculate reward
        reward = self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f)
        
        # Check if episode is done
        terminated = self.time_step == len(self.day_data) - 1
        
        if not terminated:
            # Advance time step
            self.time_step += 1
            
            # Generate next state
            self.obs = self._generate_next_state(RC_f, a_f, d_f, training)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def set_of_actions(self) -> np.ndarray:
        """
        Get the set of valid actions for the current state.
        
        Returns:
            Array of valid actions
        """
        # Extract current state
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        
        # Limit queues by maximum possible action values
        arrivals_queue = min(arrivals_queue, self.u_max)
        departures_queue = min(departures_queue, self.v_max)
        
        # Validate queue values
        if arrivals_queue < 0 or departures_queue < 0:
            raise ValueError('Invalid queue value detected.')
            
        # Return valid actions using precomputed boolean mask
        return self.possible_actions[self.valid_bool[wind_state, 
                                                     weather_condition,
                                                     arrivals_queue, 
                                                     departures_queue]]
    
    def action_masks(self) -> np.ndarray:
        """
        Get the action mask for the current state.
        
        Returns:
            Boolean array indicating which actions are valid
        """
        # Extract current state
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        
        # Limit queues by maximum possible action values
        arrivals_queue = min(arrivals_queue, self.u_max)
        departures_queue = min(departures_queue, self.v_max)
        
        # Validate queue values
        if arrivals_queue < 0 or departures_queue < 0:
            raise ValueError('Invalid queue value detected.')
        
        # Return boolean mask for valid actions
        return self.valid_bool[wind_state, weather_condition, arrivals_queue, departures_queue]

    def naive_action(self):
        """
        Generate a naive greedy action based on immediate rewards.
        
        Returns:
        --------
        int
            Action index for the greedy action
        """
        
        # Extract state components
        RC_i = int(self.obs[0])
        a_i = int(self.obs[1])
        d_i = int(self.obs[2])
        if a_i < 0 or d_i < 0:
            print('Invalid queue value detected. Check the code.')
        
        # Get valid actions
        valid_actions = self.set_of_actions()
        
        # Calculate rewards for each action
        rewards = []
        for action in valid_actions:
            RC_f = action[0]
            a_f = a_i - action[1]
            d_f = d_i - action[2]
            rewards.append(self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f))
        
        # Select greedy action
        greedy_action = valid_actions[np.argmax(rewards)]

        # Return greedy action
        return greedy_action
    
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
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        return self._render_frame()
    
    def _render_frame(self):
        """
        Render a single frame of the environment.
        
        Returns:
        --------
        numpy.ndarray or None
            RGB array if render_mode is "rgb_array", otherwise None
        """
        import pygame
        
        # Initialize Pygame if not already done
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Runway Configuration Optimization")
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create the surface to draw on
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors["background"])
        
        # Extract current state
        rc = int(self.obs[0])
        arrivals = int(self.obs[1])
        departures = int(self.obs[2])
        weather = int(self.obs[3])
        wind = int(self.obs[4])
        
        # Render time step and day information
        self._render_text(canvas, f"Day: {self.day}", 20, 50)
        self._render_text(canvas, f"Time Step: {self.time_step}", 20, 80)
        
        # Render weather and wind information
        weather_labels = ["VMC", "IMC", "LIMC"]
        wind_labels = ["All Available", "10R_10L Available", "28L_28R Available", "None Available"]
        self._render_text(canvas, f"Weather: {weather_labels[weather]}", 20, 110)
        self._render_text(canvas, f"Wind: {wind_labels[wind]}", 20, 140)
        
        # Render queue information
        self._render_text(canvas, f"Arrival Queue: {arrivals}", 20, 180, self.colors["arrival_queue"])
        self._render_text(canvas, f"Departure Queue: {departures}", 20, 210, self.colors["departure_queue"])
        
        # Render runway configuration
        rc_labels = ["None", "10R_10L", "28L_28R"]
        self._render_text(canvas, f"Runway Configuration: {rc_labels[rc]}", 20, 250)
        
        # Draw queue bars
        self._draw_queue_bars(canvas, arrivals, departures)
        
        # Draw runway diagram
        self._draw_runway_diagram(canvas, rc, wind)
        
        # Draw screenshot button
        self._draw_button(canvas)
        
        # Copy canvas to window and handle events if render_mode is "human"
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.screen_shot_button.collidepoint(mouse_pos):
                        self._take_screenshot()
            
            # Maintain frame rate
            self.clock.tick(self.metadata["render_fps"])
        
        # Return canvas as RGB array if render_mode is "rgb_array"
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(canvas).transpose((1, 0, 2))
    
    def _render_text(self, canvas, text, x, y, color=None):
        """
        Render text on the canvas.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        text : str
            Text to render
        x, y : int
            Position to render text
        color : tuple, optional
            RGB color tuple, defaults to text color
        """
        import pygame
        
        if not hasattr(self, 'font'):
            self.font = pygame.font.SysFont(None, 24)
        
        if color is None:
            color = self.colors["text"]
        
        text_surface = self.font.render(text, True, color)
        canvas.blit(text_surface, (x, y))
    
    def _draw_queue_bars(self, canvas, arrivals, departures):
        """
        Draw bars representing queue sizes.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        arrivals : int
            Size of arrival queue
        departures : int
            Size of departure queue
        """
        import pygame
        
        # Constants for bar graph
        bar_width = 40
        max_height = 150
        max_queue = 50  # Scale bar height
        
        # Calculate heights (with max cap)
        arr_height = min(arrivals / max_queue * max_height, max_height)
        dep_height = min(departures / max_queue * max_height, max_height)
        
        # Draw arrival queue bar
        arr_bar = pygame.Rect(
            300, 
            320 - arr_height,
            bar_width, 
            arr_height
        )
        pygame.draw.rect(canvas, self.colors["arrival_queue"], arr_bar)
        pygame.draw.rect(canvas, self.colors["text"], arr_bar, 2)  # outline
        
        # Draw departure queue bar
        dep_bar = pygame.Rect(
            360, 
            320 - dep_height,
            bar_width, 
            dep_height
        )
        pygame.draw.rect(canvas, self.colors["departure_queue"], dep_bar)
        pygame.draw.rect(canvas, self.colors["text"], dep_bar, 2)  # outline
        
        # Draw labels
        self._render_text(canvas, "Arrivals", 290, 330)
        self._render_text(canvas, "Departures", 340, 330)
        
        # Draw horizontal line
        pygame.draw.line(canvas, self.colors["text"], (280, 320), (420, 320), 2)
    
    def _draw_runway_diagram(self, canvas, rc, wind):
        """
        Draw runway diagram showing current configuration.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        rc : int
            Runway configuration
        wind : int
            Wind state
        """
        import pygame
        
        # Constants for runway diagram
        center_x = self.window_size // 2
        center_y = 420
        runway_length = 200
        runway_width = 20
        
        # Draw N/S/E/W indicators
        self._render_text(canvas, "N", center_x - 5, center_y - 110)
        self._render_text(canvas, "S", center_x - 5, center_y + 100)
        self._render_text(canvas, "E", center_x + 100, center_y - 5)
        self._render_text(canvas, "W", center_x - 110, center_y - 5)
        
        # Function to draw a runway with given angle
        def draw_runway(angle, label, color):
            # Calculate endpoints based on angle
            rad_angle = math.radians(angle)
            x1 = center_x - (runway_length / 2) * math.cos(rad_angle)
            y1 = center_y - (runway_length / 2) * math.sin(rad_angle)
            x2 = center_x + (runway_length / 2) * math.cos(rad_angle)
            y2 = center_y + (runway_length / 2) * math.sin(rad_angle)
            
            # Draw runway line
            pygame.draw.line(canvas, color, (x1, y1), (x2, y2), runway_width)
            
            # Draw runway label
            label_x = center_x + (runway_length / 2 + 15) * math.cos(rad_angle)
            label_y = center_y + (runway_length / 2 + 15) * math.sin(rad_angle)
            self._render_text(canvas, label, label_x - 20, label_y - 10, color)
        
        # Draw runways based on configuration and wind state
        import math
        
        # Determine colors based on wind state and configuration
        color_10R_10L = self.colors["unavailable"]
        color_28L_28R = self.colors["unavailable"]
        
        if wind == 0:  # Both available
            color_10R_10L = self.colors["runway_10R_10L"]
            color_28L_28R = self.colors["runway_28L_28R"]
        elif wind == 1:  # Only 10R_10L available
            color_10R_10L = self.colors["runway_10R_10L"]
        elif wind == 2:  # Only 28L_28R available
            color_28L_28R = self.colors["runway_28L_28R"]
            
        # Highlight the active configuration
        if rc == 1:  # 10R_10L active
            color_10R_10L = self.colors["runway_10R_10L"]
            # Draw a highlight around the active runway
            pygame.draw.circle(canvas, color_10R_10L, (center_x, center_y), runway_length // 2 + 10, 3)
        elif rc == 2:  # 28L_28R active
            color_28L_28R = self.colors["runway_28L_28R"]
            # Draw a highlight around the active runway
            pygame.draw.circle(canvas, color_28L_28R, (center_x, center_y), runway_length // 2 + 10, 3)
            
        # Draw the runways
        draw_runway(100, "10R/10L", color_10R_10L)  # ~100 degrees for 10R/10L
        draw_runway(280, "28L/28R", color_28L_28R)  # ~280 degrees for 28L/28R
    
    def _draw_button(self, canvas):
        """
        Draw screenshot button on the canvas.
        
        Parameters:
        -----------
        canvas : pygame.Surface
            Surface to draw on
        """
        import pygame
        
        # Check if mouse is over button
        mouse_pos = pygame.mouse.get_pos() if self.render_mode == "human" else (0, 0)
        button_color = self.colors["button_hover"] if self.screen_shot_button.collidepoint(mouse_pos) else self.colors["button"]
        
        # Draw button
        pygame.draw.rect(canvas, button_color, self.screen_shot_button)
        pygame.draw.rect(canvas, self.colors["text"], self.screen_shot_button, 2)  # Border
        
        # Draw button text
        if not hasattr(self, 'font'):
            self.font = pygame.font.SysFont(None, 24)
        
        text_surface = self.font.render("Screenshot", True, self.colors["button_text"])
        text_rect = text_surface.get_rect(center=self.screen_shot_button.center)
        canvas.blit(text_surface, text_rect)
    
    def _take_screenshot(self):
        """Save current frame as a screenshot."""
        import pygame
        import os
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        
        # Generate filename
        filename = f"screenshots/runway_env_{self.day}_step_{self.time_step}_{self.screen_shot_count}.png"
        self.screen_shot_count += 1
        
        # Save screenshot
        pygame.image.save(self.window, filename)
        print(f"Screenshot saved as {filename}")
