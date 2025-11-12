import numpy as np
import gymnasium as gym
from gymnasium import spaces

# https://gymnasium.farama.org/environments/classic_control/mountain_car/
# https://gymnasium.farama.org/api/env/

class mountain_car_discrete_v1(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space: [position, velocity] # 
        low = np.array([-1.2, -0.07], dtype = np.float32)
        high = np.array([0.6, 0.07], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = left, 1 = nothing, 2 = right # 
        self.action_space = spaces.Discrete(3)

        # constants #
        self.force = 0.001
        self.gravity = 0.0025
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        if options is not None:
            self.force = options.get("force", 0.001)
            self.gravity = options.get("gravity", 0.0025)
            self.max_speed = options.get("max_speed", 0.07)
                    
        position = self.np_random.uniform(low = -0.6, high = -0.4)       # random position in [-0.6, -0.4]
        velocity = 0.0
        self.state = np.array([position, velocity], dtype = np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = self.state

        # physics # 
        velocity = velocity + (action - 1) * self.force - np.cos(3 * position) * self.gravity
        velocity = np.clip(velocity, self.min_speed, self.max_speed)
        position = position + velocity
        position = np.clip(position, self.min_position, self.max_position)

        # collision at left wall # 
        if position <= self.min_position and velocity < 0:
            velocity = 0.0

        self.steps += 1
        self.state = np.array([position, velocity], dtype = np.float32)

        # reward # 
        reward = -1

        termination = position >= self.goal_position        
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}




class mountain_car_discrete_v2(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space: [position, velocity] #
        low = np.array([-1.2, -0.07], dtype = np.float32)
        high = np.array([0.6, 0.07], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = left, 1 = nothing, 2 = right #
        self.action_space = spaces.Discrete(3)

        # constants #
        self.force = 0.001
        self.gravity = 0.0025
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        if options is not None:
            self.force = options.get("force", 0.001)
            self.gravity = options.get("gravity", 0.0025)
            self.max_speed = options.get("max_speed", 0.07)
                    
        position = self.np_random.uniform(low = -0.6, high = -0.4)        # random position in [-0.6, -0.4], velocity = 0
        velocity = 0.0
        self.state = np.array([position, velocity], dtype = np.float32)
        self.steps = 0
        self.furthest_position = position
        self.highest_velocity = velocity
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = self.state

        # physics #
        velocity = velocity + (action - 1) * self.force - np.cos(3 * position) * self.gravity
        velocity = np.clip(velocity, self.min_speed, self.max_speed)
        position = position + velocity
        position = np.clip(position, self.min_position, self.max_position)

        # collision at left wall #
        if position <= self.min_position and velocity < 0:
            velocity = 0.0

        self.steps += 1
        self.state = np.array([position, velocity], dtype = np.float32)

        # reward #
        reward = -1
        if position > self.furthest_position:
            self.furthest_position = position
            reward += 0.1
        if abs(velocity) > self.highest_velocity:
            self.highest_velocity = abs(velocity)
            reward += 0.1
            
        termination = position >= self.goal_position        
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}




class steeper_hill(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space: [position, velocity] # 
        low = np.array([-1.2, -0.07], dtype = np.float32)
        high = np.array([0.6, 0.07], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = left, 1 = nothing, 2 = right # 
        self.action_space = spaces.Discrete(3)

        # constants #
        self.force = 0.001
        self.gravity = 0.0025
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.state = None
        self.steps = 0
        self.max_steps = 300

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)  
        position = self.np_random.uniform(low = -0.6, high = -0.4)       # random position in [-0.6, -0.4]
        velocity = 0.0
        self.state = np.array([position, velocity], dtype = np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = self.state

        # physics # 
        velocity = velocity + (action - 1) * self.force - 1.5 * np.cos(3 * position) * self.gravity
        velocity = np.clip(velocity, self.min_speed, self.max_speed)
        position = position + velocity
        position = np.clip(position, self.min_position, self.max_position)

        # collision at left wall # 
        if position <= self.min_position and velocity < 0:
            velocity = 0.0

        self.steps += 1
        self.state = np.array([position, velocity], dtype = np.float32)

        # reward # 
        reward = -1

        termination = position >= self.goal_position        
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}




class mirrored_terrain(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space: [position, velocity] # 
        low = np.array([0.4, -0.07], dtype = np.float32)
        high = np.array([2.2, 0.07], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = left, 1 = nothing, 2 = right # 
        self.action_space = spaces.Discrete(3)

        # constants #
        self.force = 0.001
        self.gravity = 0.0025
        self.min_position = 0.4
        self.max_position = 2.2
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        position = self.np_random.uniform(low = 1.4, high = 1.6)
        velocity = 0.0
        self.state = np.array([position, velocity], dtype = np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = self.state

        # physics # 
        velocity = velocity + (action - 1) * self.force - np.cos(3 * position) * self.gravity
        velocity = np.clip(velocity, self.min_speed, self.max_speed)
        position = position + velocity
        position = np.clip(position, self.min_position, self.max_position)

        # collision at right wall # 
        if position >= self.max_position and velocity > 0:
            velocity = 0.0

        self.steps += 1
        self.state = np.array([position, velocity], dtype = np.float32)

        # reward # 
        reward = -1

        termination = position <= self.goal_position        
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}




class extended_track(gym.Env):
    def __init__(self):
        super().__init__()

        self.move = 2 * np.pi/3
        # observation space: [position, velocity] # 
        low = np.array([-1.2, -0.07], dtype = np.float32)
        high = np.array([0.6 + self.move, 0.07], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = left, 1 = nothing, 2 = right # 
        self.action_space = spaces.Discrete(3)

        # constants #
        self.force = 0.001
        self.gravity = 0.0025
        self.min_position = -1.2 - self.move
        self.max_position = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.state = None
        self.steps = 0
        self.max_steps = 400

    def reset(self, seed = None, options = None):
        super().reset(seed = seed) 
        position = self.np_random.uniform(low = -0.6 - self.move, high = -0.4 - self.move)   
        velocity = 0.0
        self.state = np.array([position, velocity], dtype = np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = self.state

        # physics # 
        velocity = velocity + (action - 1) * self.force - np.cos(3 * position) * self.gravity
        velocity = np.clip(velocity, self.min_speed, self.max_speed)
        position = position + velocity
        position = np.clip(position, self.min_position, self.max_position)

        # collision at left wall # 
        if position <= self.min_position and velocity < 0:
            velocity = 0.0

        self.steps += 1
        self.state = np.array([position, velocity], dtype = np.float32)

        # reward # 
        reward = -1

        termination = position >= self.goal_position        
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}




        