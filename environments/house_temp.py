import numpy as np
import gymnasium as gym
from gymnasium import spaces

class house_temp_v1_0(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space: [inside temp, target temp, window, heater] #
        low = np.array( [ 0, 10, 0, 0], dtype = np.float32)
        high = np.array([30, 20, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = do nothing, 1 = window open/close, 2 = heater on/off #
        self.action_space = spaces.Discrete(3)

        # constants #
        self.inside_min = 0
        self.inside_max = 30
        self.probability = 0.5

        self.state = None
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        inside_temp = self.np_random.uniform(low = 0, high = 30)                      # random inside temp in [0, 30]
        target_temp = self.np_random.uniform(low = 10, high = 20)                     # random target temp in [0, 30]
        self.state = np.array([inside_temp, target_temp, 0, 0], dtype = np.float32)   # initial state
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, target_temp, window, heater = self.state

        if action == 1:           # window open/close
            window = 1 - window
        elif action == 2:         # heater on/off
            heater = 1 - heater

        # physics # 
        inside_temp = inside_temp + (window * -1.0) + (heater * 1.0)

        if np.random.rand() <= self.probability:           # random cooling/heating
            inside_temp += np.random.choice([-0.5, 0.5])
            
        inside_temp = np.clip(inside_temp, self.inside_min, self.inside_max)
        self.state = np.array([inside_temp, target_temp, window, heater], dtype = np.float32)
        
        # reward # 
        if target_temp - 1.0 < inside_temp < target_temp + 1.0:
            reward = 0.0
        else:
            reward = -1.0 * abs(target_temp - inside_temp)

        self.steps += 1
        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}




class house_temp_v2_0(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler] #
        low = np.array( [ 0,  2,  0,  0,  1, 0, 0], dtype = np.float32)
        high = np.array([30, 28, 30, 30, 30, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = do nothing, 1 = heater on/off, 2 = cooler on/off #
        self.action_space = spaces.Discrete(3)

        # constants #
        self.temp_min = 0
        self.temp_max = 30
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 576)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        change = self.np_random.choice([-1, 1]) * self.np_random.uniform(low = 4, high = 6)   # how the next target moves
        start = self.np_random.uniform(low = 4, high = 6)
        starting_time = self.np_random.choice(self.max_steps)

        if options is not None:
            self.temp_min = options.get("temp_min", 0)
            self.temp_max = options.get("temp_max", 30)
            target_temp = options.get("target_temp", self.np_random.uniform(low = 12.5, high = 17.5))
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3))
        else:
            target_temp = self.np_random.uniform(low = 12.5, high = 17.5)                                  # random target temp in [12.5, 17.5]
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])                # random inside temp from target_temp
            outside_temp_curve = (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3)   # change the outside temperature
            
        next_target_temp = target_temp + change
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 1]

        self.steps = 0
        switch_time = (30 - (self.steps % 30))
        state_space = [inside_temp, self.outside_temp[0], target_temp, next_target_temp, switch_time, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, target_temp, next_target_temp, time, heater, cooler = self.state

        if action == 1:           # heater on/off
            heater = 1 - heater
        elif action == 2:         # cooler on/off
            cooler = 1 - cooler

        # physics # 
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        if target_temp - 1.0 < inside_temp < target_temp + 1.0:
            reward = 0.5
        else:
            reward = -1.0 * abs(target_temp - inside_temp)

        if time <= 5:
            if next_target_temp - 2.0 < inside_temp < next_target_temp + 2.0:
                reward = 0 
            else: 
                reward = -1.0 * abs(next_target_temp - inside_temp)

        self.steps += 1
        switch_time = (30 - (self.steps % 30))

        if self.steps > 0 and self.steps % 30 == 0:
            target_temp = next_target_temp
            next_target_temp += np.random.choice([-1, 1]) * np.random.uniform(low = 4, high = 6)
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)
        
        state_space = [inside_temp, self.outside_temp[self.steps], target_temp, next_target_temp, switch_time, heater, cooler]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, target_temp, next_target_temp, switch_time, heater, cooler = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Target: {target_temp:.2f} | Next Target: {next_target_temp:.2f} | Switch Time: {switch_time} | Heater: {heater} | Cooler: {cooler}")




class house_temp_v2_1(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler] #
        low = np.array( [ 0,  2,  0,  0,  1, 0, 0], dtype = np.float32)
        high = np.array([30, 28, 30, 30, 30, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = do nothing, 1 = heater on/off, 2 = cooler on/off #
        self.action_space = spaces.Discrete(3)

        # constants #
        self.temp_min = 0
        self.temp_max = 30
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 576)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        change = self.np_random.choice([-1, 1]) * self.np_random.uniform(low = 4, high = 6)   # how the next target moves
        start = self.np_random.uniform(low = 4, high = 6)
        starting_time = self.np_random.choice(self.max_steps)

        if options is not None:
            self.temp_min = options.get("temp_min", 0)
            self.temp_max = options.get("temp_max", 30)
            target_temp = options.get("target_temp", self.np_random.uniform(low = 12.5, high = 17.5))
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3))
        else:
            target_temp = self.np_random.uniform(low = 12.5, high = 17.5)                                  # random target temp in [12.5, 17.5]
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])                # random inside temp from target_temp
            outside_temp_curve = (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3)   # change the outside temperature
            
        next_target_temp = target_temp + change
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 1]

        self.steps = 0
        switch_time = (30 - (self.steps % 30))
        state_space = [inside_temp, self.outside_temp[0], target_temp, next_target_temp, switch_time, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, target_temp, next_target_temp, time, heater, cooler = self.state

        if action == 1:           # heater on/off
            heater = 1 - heater
        elif action == 2:         # cooler on/off
            cooler = 1 - cooler

        # physics # 
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        if time == 1 and (next_target_temp - 1.0 < inside_temp < next_target_temp + 1.0):
            reward = 15.0
        elif target_temp - 1.0 < inside_temp < target_temp + 1.0:
            reward = 0.0
        else:
            reward = -1.0 * abs(target_temp - inside_temp)

        self.steps += 1
        switch_time = (30 - (self.steps % 30))

        if self.steps > 0 and self.steps % 30 == 0:
            target_temp = next_target_temp
            next_target_temp += np.random.choice([-1, 1]) * np.random.uniform(low = 4, high = 6)
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)
        
        state_space = [inside_temp, self.outside_temp[self.steps], target_temp, next_target_temp, switch_time, heater, cooler]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, target_temp, next_target_temp, switch_time, heater, cooler = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Target: {target_temp:.2f} | Next Target: {next_target_temp:.2f} | Switch Time: {switch_time} | Heater: {heater} | Cooler: {cooler}")




class house_temp_v3_0(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler, window] #
        low = np.array( [-5,  2, -5, -5,  1, 0, 0, 0], dtype = np.float32)
        high = np.array([35, 28, 35, 35, 30, 1, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = heater on/off, 1 = cooler on/off, 2 = window on/off #
        self.action_space = spaces.Discrete(8)
        self.action_dict = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 
                            4: (1, 1, 0), 5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 1, 1)}

        # constants #
        self.temp_min = -5
        self.temp_max = 35
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 576)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        change = self.np_random.choice([-1, 1]) * self.np_random.uniform(low = 4, high = 6)   # how the next target moves
        start = self.np_random.uniform(low = 3, high = 5)
        starting_time = self.np_random.choice(self.max_steps)

        if options is not None:
            self.temp_min = options.get("temp_min", -5)
            self.temp_max = options.get("temp_max", 35)
            target_temp = options.get("target_temp", self.np_random.uniform(low = 12.5, high = 17.5))
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3))
        else:
            target_temp = self.np_random.uniform(low = 12.5, high = 17.5)                                  # random target temp in [12.5, 17.5]
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])                # random inside temp from target_temp
            outside_temp_curve = (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3)   # change the outside temperature
            
        next_target_temp = target_temp + change
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 1]

        self.steps = 0
        switch_time = (30 - (self.steps % 30))
        state_space = [inside_temp, self.outside_temp[0], target_temp, next_target_temp, switch_time, 0, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, target_temp, next_target_temp, time, heater, cooler, window = self.state
        heater, cooler, window = self.action_dict[action]

        # physics # 
        if window == 1:
            self.alpha = -0.0175 * 3
        else:
            self.alpha = -0.0175
        
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        if target_temp - 1.0 < inside_temp < target_temp + 1.0:
            reward = 0.5 
        else:
            reward = -1.0 * abs(target_temp - inside_temp)

        if time <= 5:
            if next_target_temp - 2.0 < inside_temp < next_target_temp + 2.0:
                reward = 0.0
            else: 
                reward = -1.0 * abs(next_target_temp - inside_temp)

        if heater == 1 and cooler == 1:  
            reward -= 1.0
        if heater == 1 and window == 1 or window == 1 and cooler == 1:
            reward -= 0.25

        if outside_temp > inside_temp and (target_temp > inside_temp or next_target_temp > inside_temp):
            if window == 1:
                reward += 0.25
        if outside_temp < inside_temp and (target_temp < inside_temp or next_target_temp < inside_temp):
            if window == 1:
                reward += 0.25

        self.steps += 1
        switch_time = (30 - (self.steps % 30))

        if self.steps > 0 and self.steps % 30 == 0:
            target_temp = next_target_temp
            next_target_temp += np.random.choice([-1, 1]) * np.random.uniform(low = 4, high = 6)
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)
        
        state_space = [inside_temp, self.outside_temp[self.steps], target_temp, next_target_temp, switch_time, heater, cooler, window]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Target: {target_temp:.2f} | Next Target: {next_target_temp:.2f} | Switch Time: {switch_time} | Heater: {heater} | Cooler: {cooler} | Window: {window}")




class house_temp_v3_1(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler, window] #
        low = np.array( [-5,  2, -5, -5,  1, 0, 0, 0], dtype = np.float32)
        high = np.array([35, 28, 35, 35, 30, 1, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = heater on/off, 1 = cooler on/off, 2 = window on/off #
        self.action_space = spaces.Discrete(8)
        self.action_dict = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 
                            4: (1, 1, 0), 5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 1, 1)}

        # constants #
        self.temp_min = -5
        self.temp_max = 35
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 576)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        change = self.np_random.choice([-1, 1]) * self.np_random.uniform(low = 4, high = 6)   # how the next target moves
        start = self.np_random.uniform(low = 3, high = 5)
        starting_time = self.np_random.choice(self.max_steps)

        if options is not None:
            self.temp_min = options.get("temp_min", -5)
            self.temp_max = options.get("temp_max", 35)
            target_temp = options.get("target_temp", self.np_random.uniform(low = 12.5, high = 17.5))
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3))
        else:
            target_temp = self.np_random.uniform(low = 12.5, high = 17.5)                                  # random target temp in [12.5, 17.5]
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])                # random inside temp from target_temp
            outside_temp_curve = (10 * np.sin(self.x) + 15) + self.np_random.uniform(low = -3, high = 3)   # change the outside temperature
            
        next_target_temp = target_temp + change
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 1]

        self.steps = 0
        switch_time = (30 - (self.steps % 30))
        state_space = [inside_temp, self.outside_temp[0], target_temp, next_target_temp, switch_time, 0, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, target_temp, next_target_temp, time, heater, cooler, window = self.state
        heater, cooler, window = self.action_dict[action]

        # physics # 
        if window == 1:
            self.alpha = -0.0175 * 3
        else:
            self.alpha = -0.0175
        
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        energy_penalty = (heater * -0.5) + (cooler * -0.5) + (window * 0.5)
        if time == 1 and (next_target_temp - 1.0 < inside_temp < next_target_temp + 1.0):
            reward = 15.0 + energy_penalty
        elif target_temp - 1.0 < inside_temp < target_temp + 1.0:
            reward = 0.0 + energy_penalty
        else:
            reward = -1.0 * abs(target_temp - inside_temp) + energy_penalty

        self.steps += 1
        switch_time = (30 - (self.steps % 30))

        if self.steps > 0 and self.steps % 30 == 0:
            target_temp = next_target_temp
            next_target_temp += np.random.choice([-1, 1]) * np.random.uniform(low = 4, high = 6)
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)
        
        state_space = [inside_temp, self.outside_temp[self.steps], target_temp, next_target_temp, switch_time, heater, cooler, window]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Target: {target_temp:.2f} | Next Target: {next_target_temp:.2f} | Switch Time: {switch_time} | Heater: {heater} | Cooler: {cooler} | Window: {window}")




class house_temp_v4_0(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler, window] #
        low = np.array( [-11, -6, -6, -11, -11,   1, 0, 0, 0], dtype = np.float32)
        high = np.array([ 30, 25, 25,  30,  30, 288, 1, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = heater on/off, 1 = cooler on/off, 2 = window on/off #
        self.action_space = spaces.Discrete(8)
        self.action_dict = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 
                            4: (1, 1, 0), 5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 1, 1)}

        # constants #
        self.temp_min = -11
        self.temp_max = 30
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 600)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula
        self.probability = 0.5                            # chance to change temperature
        self.switch_value = 24                            # how often to switch temperature
        self.low_change = 4                               # lower bound for target switch
        self.high_change = 6                              # upper bound for target swtich
        
        self.seasonal_variation = {"winter": {"mean":  4.0, "fluctuation":  5.0, "noise": (-5, 0), "temp_min": -11, "temp_max": 14},
                                   "spring": {"mean":  8.0, "fluctuation": 10.0, "noise": (-3, 3), "temp_min":  -4, "temp_max": 20},
                                   "summer": {"mean": 15.0, "fluctuation":  5.0, "noise":  (0, 5), "temp_min":   5, "temp_max": 30},
                                   "autumn": {"mean": 10.0, "fluctuation": 10.0, "noise": (-3, 3), "temp_min":  -2, "temp_max": 22}}

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        starting_time = self.np_random.choice(self.max_steps)
        start = self.np_random.uniform(low = 2, high = 4)
        self.probability = 0.5
        season = self.np_random.choice(["winter", "spring", "summer", "autumn"])

        if options is not None:
            self.switch_value = options.get("switch_value", 24)
            self.low_change = options.get("low_change", 4)
            self.high_change = options.get("high_change", 6)
            seasonal_variation = options.get("seasonal_variation", self.seasonal_variation[season])
            mean, fluctuation, noise, self.temp_min, self.temp_max = seasonal_variation.values()
            noise = self.np_random.uniform(low = noise[0], high = noise[1])
            target_temp = options.get("target_temp", mean)
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (fluctuation * np.sin(self.x) + mean) + noise)
        else:
            self.switch_value = 24
            self.low_change = 4
            self.high_change = 6
            mean, fluctuation, noise, self.temp_min, self.temp_max = self.seasonal_variation[season].values()
            noise = self.np_random.uniform(low = noise[0], high = noise[1])
            target_temp = mean                                                                
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])   
            outside_temp_curve = (fluctuation * np.sin(self.x) + mean) + noise

        if np.random.rand() >= self.probability:
            next_target_temp = target_temp + self.np_random.uniform(low = self.low_change, high = self.high_change)
            self.probability += 0.1
        else:
            next_target_temp = target_temp - self.np_random.uniform(low = self.low_change, high = self.high_change)
            self.probability -= 0.1
            
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 2]
        self.steps = 0
        switch_time = (self.switch_value - (self.steps % self.switch_value))
        state_space = [inside_temp, self.outside_temp[0], self.outside_temp[1], target_temp, next_target_temp, switch_time, 0, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, time, heater, cooler, window = self.state
        heater, cooler, window = self.action_dict[action]

        # physics # 
        if window == 1:
            self.alpha = -0.0175 * 3
        else:
            self.alpha = -0.0175
        
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        if target_temp - 0.5 < inside_temp < target_temp + 0.5:
            reward = 0.5 
        else:
            reward = -1.0 * abs(target_temp - inside_temp)

        if time <= 8:
            if next_target_temp - 1.5 < inside_temp < next_target_temp + 1.5:
                reward = 0 
            else: 
                reward = -1.0 * abs(next_target_temp - inside_temp)

        if heater == 1 and cooler == 1:  
            reward -= 1.0
        if heater == 1 and window == 1 or window == 1 and cooler == 1:
            reward -= 0.25

        if outside_temp > inside_temp and (target_temp > inside_temp or next_target_temp > inside_temp):
            if window == 1:
                reward += 0.25
        if outside_temp < inside_temp and (target_temp < inside_temp or next_target_temp < inside_temp):
            if window == 1:
                reward += 0.25

        self.steps += 1
        switch_time = (self.switch_value - (self.steps % self.switch_value))

        if self.steps > 0 and self.steps % self.switch_value == 0:
            target_temp = next_target_temp
            if np.random.rand() >= self.probability:
                next_target_temp = target_temp + np.random.uniform(low = self.low_change, high = self.high_change)
                self.probability += 0.1
            else:
                next_target_temp = target_temp - np.random.uniform(low = self.low_change, high = self.high_change)
                self.probability -= 0.1
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)
        
        next_outside_temp = self.outside_temp[self.steps + 1]
        state_space = [inside_temp, self.outside_temp[self.steps], next_outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Outside+1: {next_outside_temp:.2f} | Target: {target_temp:.2f} | Target+1: {next_target_temp:.2f} | Time: {int(switch_time)} | Heater: {int(heater)} | Cooler: {int(cooler)} | Window: {int(window)}")




class house_temp_v4_1(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler, window] #
        low = np.array( [-11, -6, -6, -11, -11,   1, 0, 0, 0], dtype = np.float32)
        high = np.array([ 30, 25, 25,  30,  30, 288, 1, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = heater on/off, 1 = cooler on/off, 2 = window on/off #
        self.action_space = spaces.Discrete(8)
        self.action_dict = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 
                            4: (1, 1, 0), 5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 1, 1)}

        # constants #
        self.temp_min = -11
        self.temp_max = 30
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 600)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula
        self.probability = 0.5                            # chance to change temperature
        self.switch_value = 24                            # how often to switch temperature
        self.low_change = 4                               # lower bound for target switch
        self.high_change = 6                              # upper bound for target swtich
        
        self.seasonal_variation = {"winter": {"mean":  4.0, "fluctuation":  5.0, "noise": (-5, 0), "temp_min": -11, "temp_max": 14},
                                   "spring": {"mean":  8.0, "fluctuation": 10.0, "noise": (-3, 3), "temp_min":  -4, "temp_max": 20},
                                   "summer": {"mean": 15.0, "fluctuation":  5.0, "noise":  (0, 5), "temp_min":   5, "temp_max": 30},
                                   "autumn": {"mean": 10.0, "fluctuation": 10.0, "noise": (-3, 3), "temp_min":  -2, "temp_max": 22}}

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        starting_time = self.np_random.choice(self.max_steps)
        start = self.np_random.uniform(low = 2, high = 4)
        self.probability = 0.5
        season = self.np_random.choice(["winter", "spring", "summer", "autumn"])

        if options is not None:
            self.switch_value = options.get("switch_value", 24)
            self.low_change = options.get("low_change", 4)
            self.high_change = options.get("high_change", 6)
            seasonal_variation = options.get("seasonal_variation", self.seasonal_variation[season])
            mean, fluctuation, noise, self.temp_min, self.temp_max = seasonal_variation.values()
            noise = self.np_random.uniform(low = noise[0], high = noise[1])
            target_temp = options.get("target_temp", mean)
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (fluctuation * np.sin(self.x) + mean) + noise)
        else:
            self.switch_value = 24
            self.low_change = 4
            self.high_change = 6
            mean, fluctuation, noise, self.temp_min, self.temp_max = self.seasonal_variation[season].values()
            noise = self.np_random.uniform(low = noise[0], high = noise[1])
            target_temp = mean                                                                
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])   
            outside_temp_curve = (fluctuation * np.sin(self.x) + mean) + noise

        if np.random.rand() >= self.probability:
            next_target_temp = target_temp + self.np_random.uniform(low = self.low_change, high = self.high_change)
            self.probability += 0.1
        else:
            next_target_temp = target_temp - self.np_random.uniform(low = self.low_change, high = self.high_change)
            self.probability -= 0.1
            
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 2]
        self.steps = 0
        switch_time = (self.switch_value - (self.steps % self.switch_value))
        state_space = [inside_temp, self.outside_temp[0], self.outside_temp[1], target_temp, next_target_temp, switch_time, 0, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, time, heater, cooler, window = self.state
        heater, cooler, window = self.action_dict[action]

        # physics # 
        if window == 1:
            self.alpha = -0.0175 * 3
        else:
            self.alpha = -0.0175
        
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        energy_penalty = (heater * -0.5) + (cooler * -0.5) + (window * 0.5)
        if time == 1 and (next_target_temp - 1.0 < inside_temp < next_target_temp + 1.0):
            reward = 20.0 + energy_penalty
        elif target_temp - 0.5 < inside_temp < target_temp + 0.5:
            reward = 0.0 + energy_penalty
        else:
            reward = -1.0 * abs(target_temp - inside_temp) + energy_penalty

        self.steps += 1
        switch_time = (self.switch_value - (self.steps % self.switch_value))

        if self.steps > 0 and self.steps % self.switch_value == 0:
            target_temp = next_target_temp
            if np.random.rand() >= self.probability:
                next_target_temp = target_temp + np.random.uniform(low = self.low_change, high = self.high_change)
                self.probability += 0.1
            else:
                next_target_temp = target_temp - np.random.uniform(low = self.low_change, high = self.high_change)
                self.probability -= 0.1
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)

        next_outside_temp = self.outside_temp[self.steps + 1]
        state_space = [inside_temp, self.outside_temp[self.steps], next_outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Outside+1: {next_outside_temp:.2f} | Target: {target_temp:.2f} | Target+1: {next_target_temp:.2f} | Time: {int(switch_time)} | Heater: {int(heater)} | Cooler: {int(cooler)} | Window: {int(window)}")




class house_temp_v4_2(gym.Env):
    def __init__(self):
        super().__init__()

        # observation space # 
        # [inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, time_until_next_target, heater, cooler, window] #
        low = np.array( [-11, -6, -6, -11, -11,   1, 0, 0, 0], dtype = np.float32)
        high = np.array([ 30, 25, 25,  30,  30, 288, 1, 1, 1], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # action space: 0 = heater on/off, 1 = cooler on/off, 2 = window on/off #
        self.action_space = spaces.Discrete(8)
        self.action_dict = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 
                            4: (1, 1, 0), 5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 1, 1)}

        # constants #
        self.temp_min = -11
        self.temp_max = 30
        self.outside_temp = 0
        self.x = np.linspace(3.5, 3.5 + 4 * np.pi, 600)   # x values for the temperature curve
        self.alpha = -0.0175                              # cooling formula
        self.probability = 0.5                            # chance to change temperature
        self.switch_value = 24                            # how often to switch temperature
        self.low_change = 4                               # lower bound for target switch
        self.high_change = 6                              # upper bound for target swtich
        
        self.seasonal_variation = {"winter": {"mean":  4.0, "fluctuation":  5.0, "noise": (-5, 0), "temp_min": -11, "temp_max": 14},
                                   "spring": {"mean":  8.0, "fluctuation": 10.0, "noise": (-3, 3), "temp_min":  -4, "temp_max": 20},
                                   "summer": {"mean": 15.0, "fluctuation":  5.0, "noise":  (0, 5), "temp_min":   5, "temp_max": 30},
                                   "autumn": {"mean": 10.0, "fluctuation": 10.0, "noise": (-3, 3), "temp_min":  -2, "temp_max": 22}}

        self.state = None
        self.steps = 0
        self.max_steps = 288   # each step is 5 minutes

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        starting_time = self.np_random.choice(self.max_steps)
        start = self.np_random.uniform(low = 2, high = 4)
        self.probability = 0.5
        season = self.np_random.choice(["winter", "spring", "summer", "autumn"])

        if options is not None:
            self.switch_value = options.get("switch_value", 24)
            self.low_change = options.get("low_change", 4)
            self.high_change = options.get("high_change", 6)
            seasonal_variation = options.get("seasonal_variation", self.seasonal_variation[season])
            mean, fluctuation, noise, self.temp_min, self.temp_max = seasonal_variation.values()
            noise = self.np_random.uniform(low = noise[0], high = noise[1])
            target_temp = options.get("target_temp", mean)
            inside_temp = options.get("inside_temp", self.np_random.choice([target_temp - start, target_temp + start]))
            outside_temp_curve = options.get("outside_temp_curve", (fluctuation * np.sin(self.x) + mean) + noise)
        else:
            self.switch_value = 24
            self.low_change = 4
            self.high_change = 6
            mean, fluctuation, noise, self.temp_min, self.temp_max = self.seasonal_variation[season].values()
            noise = self.np_random.uniform(low = noise[0], high = noise[1])
            target_temp = mean                                                                
            inside_temp = self.np_random.choice([target_temp - start, target_temp + start])   
            outside_temp_curve = (fluctuation * np.sin(self.x) + mean) + noise

        if np.random.rand() >= self.probability:
            next_target_temp = target_temp + self.np_random.uniform(low = self.low_change, high = self.high_change)
            self.probability += 0.1
        else:
            next_target_temp = target_temp - self.np_random.uniform(low = self.low_change, high = self.high_change)
            self.probability -= 0.1
            
        self.outside_temp = outside_temp_curve[starting_time: starting_time + self.max_steps + 2]
        self.steps = 0
        switch_time = (self.switch_value - (self.steps % self.switch_value))
        state_space = [inside_temp, self.outside_temp[0], self.outside_temp[1], target_temp, next_target_temp, switch_time, 0, 0, 0]
        self.state = np.array(state_space, dtype = np.float32)   # initial state
        return self.state.copy(), {}

    def step(self, action):
        inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, time, heater, cooler, window = self.state
        heater, cooler, window = self.action_dict[action]

        # physics # 
        if window == 1:
            self.alpha = -0.0175 * 3
        else:
            self.alpha = -0.0175
        
        natural_cooling = self.alpha * (inside_temp - outside_temp)
        heater_effect = heater * (0.5)
        cooler_effect = cooler * (-0.5)
        inside_temp = inside_temp + natural_cooling + heater_effect + cooler_effect
        inside_temp = np.clip(inside_temp, self.temp_min, self.temp_max)

        # reward # 
        if target_temp - 0.5 < inside_temp < target_temp + 0.5:
            reward = 0.0 
        else:
            reward = -1.0 * abs(target_temp - inside_temp)

        self.steps += 1
        switch_time = (self.switch_value - (self.steps % self.switch_value))

        if self.steps > 0 and self.steps % self.switch_value == 0:
            target_temp = next_target_temp
            if np.random.rand() >= self.probability:
                next_target_temp = target_temp + np.random.uniform(low = self.low_change, high = self.high_change)
                self.probability += 0.1
            else:
                next_target_temp = target_temp - np.random.uniform(low = self.low_change, high = self.high_change)
                self.probability -= 0.1
            next_target_temp = np.clip(next_target_temp, self.temp_min, self.temp_max)
        
        next_outside_temp = self.outside_temp[self.steps + 1]
        state_space = [inside_temp, self.outside_temp[self.steps], next_outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window]
        self.state = np.array(state_space, dtype = np.float32)

        termination = False
        truncation = self.steps >= self.max_steps
        
        return self.state.copy(), reward, termination, truncation, {}

    def render(self):
        inside_temp, outside_temp, next_outside_temp, target_temp, next_target_temp, switch_time, heater, cooler, window = self.state
        print(f"Step: {self.steps} | Inside: {inside_temp:.2f} | Outside: {outside_temp:.2f} | Outside+1: {next_outside_temp:.2f} | Target: {target_temp:.2f} | Target+1: {next_target_temp:.2f} | Time: {int(switch_time)} | Heater: {int(heater)} | Cooler: {int(cooler)} | Window: {int(window)}")







