import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from useful import trees

# https://arxiv.org/abs/1511.05952

class pddqn_agent_mse:
    def __init__(self, model, state_dim, action_dim, gamma, alpha, beta, lr, epsilon, epsilon_min, decay_steps, buffer_size, batch_size, n_episodes, device):
        self.action_dim = action_dim                               # number of actions
        self.gamma = gamma                                         # how much the next state matters
        self.alpha = alpha                                         # level of prioritisation, alpha = 0 corresponds to the uniform case
        self.beta = beta                                           # how much prioritisation to apply, usually annealing during training
        self.beta_growth = (1.0 - beta)/n_episodes                 # linear growth for prioritisation
        self.epsilon = epsilon                                     # starting exploration
        self.epsilon_min = epsilon_min                             # smallest exploration
        self.epsilon_decay = (epsilon - epsilon_min)/decay_steps   # how quickly exploration decreases
        self.buffer_size = buffer_size                             # size of the memory
        self.batch_size = batch_size                               # size of the batch 
        self.priority = 1.0                                        # priority 1 at first since there is no td error 
        self.error = 1e-4                                          # small error to ensure no zero values
        
        self.states = np.zeros((buffer_size, state_dim), dtype = np.float32)        # memory for states
        self.actions = np.zeros((buffer_size, 1), dtype = np.int64)                 # memory for actions
        self.rewards = np.zeros(buffer_size, dtype = np.float32)                    # memory for rewards
        self.next_states = np.zeros((buffer_size, state_dim), dtype = np.float32)   # memory for next states
        self.done = np.zeros(buffer_size, dtype = np.float32)                       # memory for termination
        self.counter = 0                                                            # used to index the memory
        self.upper_value = 0                                                        # check if the memory has been filled
        self.sumtree = trees.sum_tree(buffer_size)                                  # memory for priorities

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

    def grab_batch_index(self): # grabs the batch index and priority based on priority
        priority_sum = self.sumtree.data_tree[0]        
        index = np.zeros(self.batch_size, dtype = np.int64)
        priority = np.zeros(self.batch_size, dtype = np.float32)
        
        for i in range(self.batch_size):
            random_value = np.random.uniform(0, priority_sum)
            index[i], priority[i] = self.sumtree.derive_data(random_value)
        return index, priority

    def get_batch(self, indices): # gets a batch of observations
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)          
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        done = torch.from_numpy(self.done[indices]).to(self.device)
        return states, actions, rewards, next_states, done

    def weights(self, priority_j): # grab the weights 
        priority_sum = self.sumtree.data_tree[0] 
        N = self.upper_value or (self.counter + 1)
        probability = priority_j.astype(np.float64)/priority_sum
        probability = np.clip(probability, min = self.error, max = None)
        weights = (N * probability) ** (-self.beta)
        normalised_weights = weights/weights.max()
        return torch.from_numpy(normalised_weights.astype(np.float32)).to(self.device)

    def train_step(self):
        if self.upper_value == 0 and self.counter < self.batch_size:   # checks if memory holds batch size amount
            return None

        indices, priority_j = self.grab_batch_index()
        states, actions, rewards, next_states, done = self.get_batch(indices)
        weights_j = self.weights(priority_j)

        # ddqn formula 
        q_values = torch.gather(self.main_model(states), dim = 1, index = actions).squeeze()
        with torch.no_grad():
            next_actions = torch.argmax(self.main_model(next_states), dim = 1).unsqueeze(1)
            q_target = torch.gather(self.target_model(next_states), dim = 1, index = next_actions).squeeze()
            target_q = rewards + self.gamma * q_target * (1 - done)

        td_error = (torch.abs(target_q - q_values).detach() + self.error) ** self.alpha   # calculates the new priority values
        td_error = torch.clamp(td_error, min = self.error)                                # checks for small values 
        if not torch.isfinite(td_error).all():                                            # checks for nan values
            td_error = torch.where(torch.isfinite(td_error), td_error, self.error)        # converts invalid values
            
        for i in range(self.batch_size):
            self.sumtree.add_data(indices[i], td_error[i].item())   # switch the priority for the batch
        
        tensor_loss = nn.MSELoss(reduce = "none")(q_values, target_q)   # work out loss for the entire batch
        loss = torch.mean(weights_j * tensor_loss)                      # weighting each loss by importance

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.increase_beta()

        return loss

    def update_memory(self, state, action, reward, next_state, done): # updates memory and priority
        self.priority = self.sumtree.largest_value()         # set priority as the largest priority in the tree
        self.sumtree.add_data(self.counter, self.priority)   # add priority
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

    def increase_beta(self): # increases prioritisation
        self.beta += self.beta_growth





class pddqn_agent_huber:
    def __init__(self, model, state_dim, action_dim, gamma, alpha, beta, lr, epsilon, epsilon_min, decay_steps, buffer_size, batch_size, n_episodes, device):
        self.action_dim = action_dim                               # number of actions
        self.gamma = gamma                                         # how much the next state matters
        self.alpha = alpha                                         # level of prioritisation, alpha = 0 corresponds to the uniform case
        self.beta = beta                                           # how much prioritisation to apply, usually annealing during training
        self.beta_growth = (1.0 - beta)/n_episodes                 # linear growth for prioritisation
        self.epsilon = epsilon                                     # starting exploration
        self.epsilon_min = epsilon_min                             # smallest exploration
        self.epsilon_decay = (epsilon - epsilon_min)/decay_steps   # how quickly exploration decreases
        self.buffer_size = buffer_size                             # size of the memory
        self.batch_size = batch_size                               # size of the batch 
        self.priority = 1.0                                        # priority 1 at first since there is no td error 
        self.error = 1e-4                                          # small error to ensure no zero values
        
        self.states = np.zeros((buffer_size, state_dim), dtype = np.float32)        # memory for states
        self.actions = np.zeros((buffer_size, 1), dtype = np.int64)                 # memory for actions
        self.rewards = np.zeros(buffer_size, dtype = np.float32)                    # memory for rewards
        self.next_states = np.zeros((buffer_size, state_dim), dtype = np.float32)   # memory for next states
        self.done = np.zeros(buffer_size, dtype = np.float32)                       # memory for termination
        self.counter = 0                                                            # used to index the memory
        self.upper_value = 0                                                        # check if the memory has been filled
        self.sumtree = trees.sum_tree(buffer_size)                                  # memory for priorities

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

    def grab_batch_index(self): # grabs the batch index and priority based on priority
        priority_sum = self.sumtree.data_tree[0]        
        index = np.zeros(self.batch_size, dtype = np.int64)
        priority = np.zeros(self.batch_size, dtype = np.float32)
        
        for i in range(self.batch_size):
            random_value = np.random.uniform(0, priority_sum)
            index[i], priority[i] = self.sumtree.derive_data(random_value)
        return index, priority

    def get_batch(self, indices): # gets a batch of observations
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)          
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        done = torch.from_numpy(self.done[indices]).to(self.device)
        return states, actions, rewards, next_states, done

    def weights(self, priority_j): # grab the weights 
        priority_sum = self.sumtree.data_tree[0] 
        N = self.upper_value or (self.counter + 1)
        probability = priority_j.astype(np.float64)/priority_sum
        probability = np.clip(probability, min = self.error, max = None)
        weights = (N * probability) ** (-self.beta)
        normalised_weights = weights/weights.max()
        return torch.from_numpy(normalised_weights.astype(np.float32)).to(self.device)

    def train_step(self):
        if self.upper_value == 0 and self.counter < self.batch_size:   # checks if memory holds batch size amount
            return None

        indices, priority_j = self.grab_batch_index()
        states, actions, rewards, next_states, done = self.get_batch(indices)
        weights_j = self.weights(priority_j)

        # ddqn formula 
        q_values = torch.gather(self.main_model(states), dim = 1, index = actions).squeeze()
        with torch.no_grad():
            next_actions = torch.argmax(self.main_model(next_states), dim = 1).unsqueeze(1)
            q_target = torch.gather(self.target_model(next_states), dim = 1, index = next_actions).squeeze()
            target_q = rewards + self.gamma * q_target * (1 - done)

        td_error = (torch.abs(target_q - q_values).detach() + self.error) ** self.alpha   # calculates the new priority values
        td_error = torch.clamp(td_error, min = self.error)                                # checks for small values 
        if not torch.isfinite(td_error).all():                                            # checks for nan values
            td_error = torch.where(torch.isfinite(td_error), td_error, self.error)        # converts invalid values
            
        for i in range(self.batch_size):
            self.sumtree.add_data(indices[i], td_error[i].item())   # switch the priority for the batch
        
        tensor_loss = nn.HuberLoss(reduce = "none")(q_values, target_q)   # work out loss for the entire batch
        loss = torch.mean(weights_j * tensor_loss)                        # weighting each loss by importance

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.increase_beta()

        return loss

    def update_memory(self, state, action, reward, next_state, done): # updates memory and priority
        self.priority = self.sumtree.largest_value()         # set priority as the largest priority in the tree
        self.sumtree.add_data(self.counter, self.priority)   # add priority
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

    def increase_beta(self): # increases prioritisation
        self.beta += self.beta_growth




