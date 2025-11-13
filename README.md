# Deep Reinforcement Learning Project
A reinforcement learning project developed for my MSc dissertation. Includes implementations of DQN and its variants, reproduction of the Mountain Car experiment, and a custom environment for temperature regulation.

---

## Overview 
This project 

For an in-depth explanation of methods and results, see the project report: 
[View Full Report (PDF)](https://github.com/H-Konrad/project_reports/blob/main/reinforcement_learning_write_up.pdf)

---

## Repository Structure

### Agent
- `ddqn_agent.py` : Implementation of Double DQN 
- `dqn_agent.py` : Implementation of DQN
- `pddqn_agent.py` : Implementation of DQN with prioritised experience replay (PER)

### Environments
- `house_temp.py` : Versions of the custom reinforcement learning envrionment
- `mountain_car.py` : Versions of the Mountain Car environment

### Notebooks
- `house_temp_param.ipynb` : Hyperparameter testing for the custom environment models
- `house_temp_testing.ipynb` : Testing the custom envrionment agents
- `house_temp_training.ipynb` : Final training and envrionmental paramater testing
- `mountain_car_testing.ipynb` : Testing the Mountain Car environment
- `mountain_car_training.ipynb` : Hyperparameter testing and training for the Mountain Car envrionment
- `temp_breakdown.ipynb` : A breakdown of seasonal temperature

### Other Functions
- `trees.py` : A sum-tree function used in the PER implementation


