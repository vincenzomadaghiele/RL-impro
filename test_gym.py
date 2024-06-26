import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C
import os
import improvisationEnv 


if __name__ == '__main__':

	synth_name = 'sin'
	model_dir = f"./00_synths/{synth_name}/gym_models/models"
	env = gym.make('improvisation-matching-v0', 
					#synth_name='granular', 
					#N_synth_parameters=4, 
					render_mode=None)

	model = A2C.load(f'{model_dir}/a2c_20000', env=env)
	# Run a test
	obs = env.reset()[0]
	max_n_episodes = 10
	N_EPISODES = 0
	terminated = False
	while N_EPISODES < max_n_episodes:
		action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
		obs, _, terminated, _, _ = env.step(action.tolist())
		if terminated:
			N_EPISODES += 1
			obs = env.reset()[0]

