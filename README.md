# Deep Reinforcement Learning Project
This project applies reinforcement learning to the Mountain Car environment and a custom temperature regulation control environment. This includes the implementation of DQN, Double DQN, Duelling DQN, and Prioritised Experience Replay (PER), with the goal of analysing how different reward structures influence agent learning and generalisation behaviour.

For an in-depth explanation of reinforcement learning theory, methods, and results, see the project report: 
[View Full Report (PDF)](https://github.com/H-Konrad/project_reports/blob/main/reinforcement_learning_write_up.pdf)

---

## Repository Structure

### Agent
- `ddqn_agent.py` : Implementation of Double DQN 
- `dqn_agent.py` : Implementation of DQN
- `pddqn_agent.py` : Implementation of DQN with PER

### Environments
- `house_temp.py` : Versions of the custom reinforcement learning environment
- `mountain_car.py` : Versions of the Mountain Car environment

### Notebooks
- `house_temp_param.ipynb` : Hyperparameter testing for the custom environment models
- `house_temp_testing.ipynb` : Testing the custom environment agents
- `house_temp_training.ipynb` : Final training and environmental parameter testing
- `mountain_car_testing.ipynb` : Testing the Mountain Car environment
- `mountain_car_training.ipynb` : Hyperparameter testing and training for the Mountain Car environment
- `temp_breakdown.ipynb` : A breakdown of seasonal temperature

### Other Functions
- `trees.py` : A sum-tree function used in the PER implementation


