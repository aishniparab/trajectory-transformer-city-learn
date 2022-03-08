import sys
sys.path.append('/Users/aishniparab/Documents/coursework/winter_2022_coursework/cs269_deep_gen_models/project/trajectory-transformer-city-learn')

import os
import numpy as np
import torch
import pdb

from trajectory.models.transformers import GPT

from citylearn.citylearn import CityLearn
from pathlib import Path
import pickle


# Load city learn environment
climate_zone = 5
data_path = Path("citylearn/data/Climate_Zone_"+str(climate_zone))
building_ids = ["Building_"+str(i) for i in [1,2,3,4,5,6,7,8,9]]
sim_period = (0, 8760*4-1)
params = {'data_path':data_path, 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':building_ids,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': sim_period, 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False }

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)

print("env: ", env, type(env))

with open(f'citylearn/data/Climate_Zone_5_RBC_concate_origin.pkl', "rb") as f:
    dataset = pickle.load(f)
    
N = dataset['rewards'].shape[0]

print(dataset.keys())
