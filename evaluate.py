import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import json
import sys
import pickle
from stable_baselines3 import A2C, PPO, DQN
import os
import improvisationEnv 
import constants

def evaluate(synth_name, model_name, model_iteration, 
			corpus_name=None, training_mode=None,
			N_EVAL_EPISODES=10, print_interval=100, UBUNTU=False):

	AGENT_TYPE = model_name.split('-')[1]

	## LOAD MODEL
	MODEL_DIR = f'./00_synths/{synth_name}/gym_models/models/{model_name}/{model_iteration}'

	# LOAD ENV SETTINGS
	ENVIRONMENT_SETTINGS_PATH = f'./00_synths/{synth_name}/gym_models/settings/{model_name}-environment.json'
	f = open(ENVIRONMENT_SETTINGS_PATH)
	environment_settings = json.load(f)
	f.close()

	synth_name = environment_settings['instrument_name']
	if not corpus_name:
		corpus_name = environment_settings['corpus_name']

	# initialize training environment
	N_synth_params = environment_settings['N_synth_params']
	features_keep = environment_settings['features_keep']
	features_reward = environment_settings['features_reward']
	reward_noise = environment_settings['reward_noise']
	if not training_mode:
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
					render_mode=None,
					seed=seed, 
					UBUNTU=UBUNTU)

	N_tot_features = len(constants.feature_names)
	print(f'Loading model {model_name}')
	print('-'*50)
	print(f'Synthesizer name: {synth_name}')
	print(f'Number synth parameters: {N_synth_params}')
	print(f'Update step: {step_size}')
	print(f'Features used as state: {features_keep}')
	print()


	if AGENT_TYPE == 'A2C':
		model = A2C.load(MODEL_DIR, env=env)
	elif AGENT_TYPE == 'DQN':
		model = DQN.load(MODEL_DIR, env=env)
	else:
		model = PPO.load(MODEL_DIR, env=env)

	# Run a test
	obs = env.reset()[0]
	n_episode = 0
	step = 0
	cumulative_reward = []
	reward_means = []
	reward_stds = []
	cumulative_RMSE = []
	RMSE_means = []
	RMSE_stds = []
	terminated = False
	while n_episode < N_EVAL_EPISODES:
		action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
		obs, _, terminated, _, info = env.step(action.tolist())
		cumulative_reward.append(info['reward'])
		cumulative_RMSE.append(info['RMSE'])
		step += 1
		if step % print_interval == 0:
			print(f'Episode step {step}: mean reward {np.array(cumulative_reward).mean():.3f}, mean RMSE {np.array(cumulative_RMSE).mean():.3f}')
			print(f'		std reward {np.array(cumulative_reward).std():.3f}, std RMSE {np.array(cumulative_RMSE).std()}:.3f')
		if terminated:
			print(f'Finished evaluation of episode {n_episode}: mean reward {np.array(cumulative_reward).mean():.3f}, mean RMSE {np.array(cumulative_RMSE).mean():.3f}')
			print(f'				std reward {np.array(cumulative_reward).std():.3f}, std RMSE {np.array(cumulative_RMSE).std():.3f}')
			print('-'* 50)
			n_episode += 1
			obs = env.reset()[0]
			step = 0
			print(f'Evaluating episode {n_episode}')
			reward_means.append(np.array(cumulative_reward).mean())
			RMSE_means.append(np.array(cumulative_RMSE).mean())
			reward_stds.append(np.array(cumulative_reward).std())
			RMSE_stds.append(np.array(cumulative_RMSE).std())

	return reward_means, RMSE_means, reward_stds, RMSE_stds


if __name__ == '__main__':

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--SYNTH_NAME', type=str, default='sin',
						help='name of the folder containing the synth to be used')
	parser.add_argument('--MODEL_NAME', type=str, default='DQN-1719412489',
						help='stable baselines3 model type: PPO, DQN or A2C with corresponding training timestamp')
	parser.add_argument('--MODEL_ITERATION', type=str, default="DQN_2000000",
						help='training iteration of the model to select eg. DQN_2000000')
	parser.add_argument('--UBUNTU', type=bool, default=False,
						help='True if the script runs on Ubuntu')
	args = parser.parse_args(sys.argv[1:])

	## DEFINE SCRIPT PARAMETERS
	synth_name = args.SYNTH_NAME
	model_name = args.MODEL_NAME
	model_iteration = args.MODEL_ITERATION
	UBUNTU = args.UBUNTU

	reward_means, RMSE_means, reward_stds, RMSE_stds = evaluate(synth_name, model_name, model_iteration,
																corpus_name='GuitarSet_test', training_mode='corpus', 
																UBUNTU=UBUNTU)
	
	print()
	print(f'Reward mean: {np.array(reward_means).mean():.3f}')
	print(f'Reward std: {np.array(reward_means).std():.3f}')
	print(f'RMSE mean: {np.array(RMSE_means).mean():.3f}')
	print(f'RMSE std: {np.array(RMSE_means).std():.3f}')


