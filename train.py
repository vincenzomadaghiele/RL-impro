import os
import time
import random
import json
import argparse
import sys
import pickle

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

import improvisationEnv
import evaluate

# to visualize: python3 -m tensorboard.main --logdir ./00_synths/{synth-name}/gym_models/logs

if __name__ == '__main__':


	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--ENVIRONMENT_SETTINGS', type=str, default='environment_settings.json',
						help='path to the json file containing environment settings')
	parser.add_argument('--AGENT_TYPE', type=str, default='PPO',
						help='stable baselines3 model type: PPO, DQN or A2C')
	parser.add_argument('--TIMESTEPS', type=int, default=50000,
						help='number of timesteps for a training session')
	parser.add_argument('--ITERATIONS', type=int, default=40,
						help='number of timesteps for a training session')
	parser.add_argument('--N_EVAL_EPISODES', type=int, default=5,
						help='number of timesteps for a training session')
	parser.add_argument('--UBUNTU', type=bool, default=False,
						help='True if the script runs on Ubuntu')
	args = parser.parse_args(sys.argv[1:])

	ENVIRONMENT_SETTINGS_PATH = args.ENVIRONMENT_SETTINGS
	AGENT_TYPE = args.AGENT_TYPE
	TIMESTEPS = args.TIMESTEPS
	ITERATIONS = args.ITERATIONS
	N_EVAL_EPISODES = args.N_EVAL_EPISODES
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

	agent_name = f'{int(time.time())}-{AGENT_TYPE}'
	model_dir = f"./00_synths/{synth_name}/gym_models/models/{agent_name}"
	log_dir = f"./00_synths/{synth_name}/gym_models/logs/{agent_name}"
	settings_dir = f"./00_synths/{synth_name}/gym_models/settings"
	evaluation_dir = f"./00_synths/{synth_name}/gym_models/evaluation"
	os.makedirs(model_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(settings_dir, exist_ok=True)
	os.makedirs(evaluation_dir, exist_ok=True)

	with open(os.path.join(settings_dir, f'{agent_name}-environment.json'), 'w', encoding='utf-8') as f:
		json.dump(environment_settings, f, ensure_ascii=False, indent=4)

	## INSTANTIATE TRAINING ENVIRONMENT
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


	## INSTANTIATE AGENT
	if AGENT_TYPE == 'A2C':
		model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
	elif AGENT_TYPE == 'DQN':
		model = DQN('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
	else:
		model = PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)


	## TRAIN
	eval_interval = 5 # evaluate every iterations
	best_model_rew = -np.inf
	for i in range(ITERATIONS):
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=AGENT_TYPE) # train
		model.save(f"{model_dir}/{AGENT_TYPE}_{TIMESTEPS*i}")
		if i % eval_interval:
			mean_rew, std_rew = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
			print(mean_rew, std_rew)
			if mean_rew > best_model_rew:
				best_model_rew = mean_rew
				model.save(f"{model_dir}/{AGENT_TYPE}_best")


	## EVALUATE BEST MODEL
	model_itaration_for_eval = f'{AGENT_TYPE}_best'
	model_evaluation = {}
	print('Evaluating best model')
	reward_means, RMSE_means, reward_stds, RMSE_stds = evaluate.evaluate(synth_name, agent_name, model_itaration_for_eval,
																		corpus_name='GuitarSet', training_mode='corpus')

	model_evaluation["train_rew_mean"] = np.array(reward_means).mean()
	model_evaluation["train_rew_std"] = np.array(reward_means).std()
	model_evaluation["train_RMSE_mean"] = np.array(RMSE_means).mean()
	model_evaluation["train_RMSE_std"] = np.array(RMSE_means).std()
	print()
	print(f'Training reward mean: {np.array(reward_means).mean():.3f}')
	print(f'Training reward std: {np.array(reward_means).std():.3f}')
	print(f'Training RMSE mean: {np.array(RMSE_means).mean():.3f}')
	print(f'Training RMSE std: {np.array(RMSE_means).std():.3f}')

	reward_means, RMSE_means, reward_stds, RMSE_stds = evaluate.evaluate(synth_name, agent_name, model_itaration_for_eval,
																		corpus_name='GuitarSet_test', training_mode='corpus')


	model_evaluation["test_rew_mean"] = np.array(reward_means).mean()
	model_evaluation["test_rew_std"] = np.array(reward_means).std()
	model_evaluation["test_RMSE_mean"] = np.array(RMSE_means).mean()
	model_evaluation["test_RMSE_std"] = np.array(RMSE_means).std()
	print()
	print(f'Test reward mean: {np.array(reward_means).mean():.3f}')
	print(f'Test reward std: {np.array(reward_means).std():.3f}')
	print(f'Test RMSE mean: {np.array(RMSE_means).mean():.3f}')
	print(f'Test RMSE std: {np.array(RMSE_means).std():.3f}')

	with open(os.path.join(evaluation_dir, f'{agent_name}-evaluation.json'), 'w', encoding='utf-8') as f:
		json.dump(model_evaluation, f, ensure_ascii=False, indent=4)

