import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import argparse
import sys
import pickle
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time
import improvisationEnv 

# to visualize: python3 -m tensorboard.main --logdir ./00_synths/sin/gym_models/logs

if __name__ == '__main__':


	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--ENVIRONMENT_SETTINGS', type=str, default='environment_settings.json',
						help='path to the json file containing environment settings')
	parser.add_argument('--AGENT_TYPE', type=str, default='PPO',
						help='stable baselines3 model type: PPO or A2C')
	parser.add_argument('--TIMESTEPS', type=int, default=50000,
						help='number of timesteps for a training session')
	parser.add_argument('--ITERATIONS', type=int, default=100,
						help='number of timesteps for a training session')
	parser.add_argument('--UBUNTU', type=bool, default=False,
						help='True if the script runs on Ubuntu')
	args = parser.parse_args(sys.argv[1:])

	ENVIRONMENT_SETTINGS_PATH = args.ENVIRONMENT_SETTINGS
	AGENT_TYPE = args.AGENT_TYPE
	TIMESTEPS = args.TIMESTEPS
	ITERATIONS = args.ITERATIONS
	UBUNTU = args.UBUNTU

	# LOAD ENV SETTINGS
	f = open(ENVIRONMENT_SETTINGS_PATH)
	environment_settings = json.load(f)
	f.close()

	synth_name = environment_settings['instrument_name']
	corpus_name = environment_settings['corpus_name']

	# initialize training environment
	N_synth_params = environment_settings['N_synth_params']    
	features_keep = environment_settings['features_keep']
	features_reward = environment_settings['features_reward']
	reward_noise = environment_settings['reward_noise']
	training_mode = environment_settings['training_mode']
	reward_type = environment_settings['reward_type']
	step_size = environment_settings['step_size']
	ip_send = environment_settings['ip_send']
	agent_port_send = environment_settings['port_send']
	target_port_send = environment_settings['port_send_optimal']
	verbose = True if environment_settings['verbose'] == 'True' else False
	live = True if environment_settings['live'] == 'True' else False
	max_episode_duration = environment_settings['max_episode_duration']
	seed = None if environment_settings['seed'] == 'None' else environment_settings['seed']
	if live:
		render_mode = 'human'
	else:
		render_mode = None

	model_dir = f"./00_synths/{synth_name}/gym_models/models/{AGENT_TYPE}-{int(time.time())}"
	log_dir = f"./00_synths/{synth_name}/gym_models/logs/{AGENT_TYPE}-{int(time.time())}"
	settings_dir = f"./00_synths/{synth_name}/gym_models/settings"
	os.makedirs(model_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(settings_dir, exist_ok=True)

	json_filename = f'{AGENT_TYPE}-{int(time.time())}-environment.json'
	with open(os.path.join(settings_dir,json_filename), 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

	env = gym.make('improvisation-matching-v0', 
					features_keep=features_keep,
					features_reward=features_reward,
					synth_name=synth_name, 
					N_synth_parameters=N_synth_params, 
					corpus_name=corpus_name,
					step_size=step_size,
					reward_noise=reward_noise, 
					training_mode=training_mode,
					ip_send=ip_send, agent_port_send=agent_port_send, target_port_send=target_port_send,
					max_episode_duration=max_episode_duration,
					render_mode=render_mode,
					seed=seed, 
					UBUNTU=UBUNTU)

	if AGENT_TYPE == 'A2C':
		model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
	else:
		model = PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

	for i in range(ITERATIONS):
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=AGENT_TYPE) # train
		model.save(f"{model_dir}/a2c_{TIMESTEPS*i}")


