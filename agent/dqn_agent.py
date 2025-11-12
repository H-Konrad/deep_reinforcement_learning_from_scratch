import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
# https://spinningup.openai.com/en/latest/spinningup/keypapers.html

class dqn_agent_mse:
    def __init__(self, model, state_dim, action_dim, gamma, lr, epsilon, epsilon_min, decay_steps, buffer_size, batch_size, device):
        self.action_dim = action_dim                               # number of actions
        self.gamma = gamma                                         # how much the next state matters
        self.epsilon = epsilon                                     # starting exploration
        self.epsilon_min = epsilon_min                             # smallest exploration
        self.epsilon_decay = (epsilon - epsilon_min)/decay_steps   # how quickly exploration decreases
        self.buffer_size = buffer_size                             # size of the memory
        self.batch_size = batch_size                               # size of the batch

        self.states = np.zeros((buffer_size, state_dim), dtype = np.float32)        # memory for states
        self.actions = np.zeros((buffer_size, 1), dtype = np.int64)                 # memory for actions
        self.rewards = np.zeros(buffer_size, dtype = np.float32)                    # memory for rewards
        self.next_states = np.zeros((buffer_size, state_dim), dtype = np.float32)   # memory for next states
        self.done = np.zeros(buffer_size, dtype = np.float32)                       # memory for termination
        self.counter = 0                                                            # used to index the memory
        self.upper_value = 0                                                        # check if the memory has been filled

        self.device = device
        self.main_model = model(state_dim, action_dim).to(self.device)
        self.target_model = model(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.main_model.parameters(), lr = lr)

    def act(self, state): # provides the action that will be taken in the environment
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        with torch.no_grad():
            q_values = self.main_model(torch.tensor(state, dtype = torch.float32).to(self.device))
            return torch.argmax(q_values).item()

    def get_batch(self): # gets a batch of observations
        if self.upper_value == 0:
            indices = np.random.choice(self.counter, self.batch_size)       # only samples values when the array is not full yet
        else: 
            indices = np.random.choice(self.upper_value, self.batch_size)   # sampling from the full memory

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)          
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        done = torch.from_numpy(self.done[indices]).to(self.device)
        return states, actions, rewards, next_states, done

    def train_step(self):
        if self.upper_value == 0 and self.counter < self.batch_size:   # checks if memory holds batch size amount
            return None
        
        states, actions, rewards, next_states, done = self.get_batch()

        # dqn formula
        q_values = torch.gather(self.main_model(states), dim = 1, index = actions).squeeze()
        with torch.no_grad():
            next_q, _ = torch.max(self.target_model(next_states), dim = 1)
            target_q = rewards + self.gamma * next_q * (1 - done)
    
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_memory(self, state, action, reward, next_state, done): # updates memory
        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.next_states[self.counter] = next_state
        self.done[self.counter] = done
        
        self.counter += 1                         # cycle through array
        if self.counter == self.buffer_size:
            self.counter = 0                      # resets when array is full
            self.upper_value = self.buffer_size   # one cycle has been complete

    def update_target(self): # updates the target model weights
        self.target_model.load_state_dict(self.main_model.state_dict())

    def decay_epsilon(self): # lowers exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay




            

class dqn_agent_huber:
    def __init__(self, model, state_dim, action_dim, gamma, lr, epsilon, epsilon_min, decay_steps, buffer_size, batch_size, device):
        self.action_dim = action_dim                               # number of actions
        self.gamma = gamma                                         # how much the next state matters
        self.epsilon = epsilon                                     # starting exploration
        self.epsilon_min = epsilon_min                             # smallest exploration
        self.epsilon_decay = (epsilon - epsilon_min)/decay_steps   # how quickly exploration decreases
        self.buffer_size = buffer_size                             # size of the memory
        self.batch_size = batch_size                               # size of the batch

        self.states = np.zeros((buffer_size, state_dim), dtype = np.float32)        # memory for states
        self.actions = np.zeros((buffer_size, 1), dtype = np.int64)                 # memory for actions
        self.rewards = np.zeros(buffer_size, dtype = np.float32)                    # memory for rewards
        self.next_states = np.zeros((buffer_size, state_dim), dtype = np.float32)   # memory for next states
        self.done = np.zeros(buffer_size, dtype = np.float32)                       # memory for termination
        self.counter = 0                                                            # used to index the memory
        self.upper_value = 0                                                        # check if the memory has been filled

        self.device = device
        self.main_model = model(state_dim, action_dim).to(self.device)
        self.target_model = model(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.main_model.parameters(), lr = lr)

    def act(self, state): # provides the action that will be taken in the environment
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        with torch.no_grad():
            q_values = self.main_model(torch.tensor(state, dtype = torch.float32).to(self.device))
            return torch.argmax(q_values).item()

    def get_batch(self): # gets a batch of observations
        if self.upper_value == 0:
            indices = np.random.choice(self.counter, self.batch_size)       # only samples values when the array is not full yet
        else: 
            indices = np.random.choice(self.upper_value, self.batch_size)   # sampling from the full memory

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)          
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        done = torch.from_numpy(self.done[indices]).to(self.device)
        return states, actions, rewards, next_states, done

    def train_step(self):
        if self.upper_value == 0 and self.counter < self.batch_size:   # checks if memory holds batch size amount
            return None
        
        states, actions, rewards, next_states, done = self.get_batch()

        # dqn formula
        q_values = torch.gather(self.main_model(states), dim = 1, index = actions).squeeze()
        with torch.no_grad():
            next_q, _ = torch.max(self.target_model(next_states), dim = 1)
            target_q = rewards + self.gamma * next_q * (1 - done)
    
        loss = nn.HuberLoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_memory(self, state, action, reward, next_state, done): # updates memory
        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.next_states[self.counter] = next_state
        self.done[self.counter] = done
        
        self.counter += 1                         # cycle through array
        if self.counter == self.buffer_size:
            self.counter = 0                      # resets when array is full
            self.upper_value = self.buffer_size   # one cycle has been complete

    def update_target(self): # updates the target model weights
        self.target_model.load_state_dict(self.main_model.state_dict())

    def decay_epsilon(self): # lowers exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay