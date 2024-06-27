import constants
import os
import pickle
import argparse
import sys
import json
import subprocess
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client

import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN

import improvisationEnv


# operate transformations: this has to be modified by hand
def filtering(features_vector):
	# spectral-0
	return features_vector


def process(arrayIn):

	# separate components
	current_synth_params = np.array(arrayIn[N_tot_features*2:])
	synth_state = np.array(arrayIn[:N_tot_features])
	target_state = np.array(arrayIn[N_tot_features:N_tot_features*2])
	lookup_synth_state = np.array(env.unwrapped.synth_agent.parameters2features(current_synth_params)).reshape(1,-1)
	#print(current_synth_params, synth_state, target_state)

	# filter array as in dataset
	filtered_synth_state = filtering(synth_state)
	filtered_target_state = filtering(target_state)

	# select features used in model
	index_feats_keep = [constants.feature_names.index(feat) for feat in env.get_wrapper_attr('features_keep')]
	synth_state_kept = filtered_synth_state[index_feats_keep]
	target_state_kept = filtered_target_state[index_feats_keep]

	# scaler
	scaled_synth_state = env.unwrapped.synth_agent.scaler.transform(synth_state_kept.reshape(1, -1))
	scaled_target_state = env.unwrapped.synth_agent.scaler.transform(target_state_kept.reshape(1, -1))
	observation = np.concatenate((scaled_synth_state, scaled_target_state, current_synth_params.reshape(1, -1)), axis=1).reshape(-1,)
	lookup_observation = np.concatenate((lookup_synth_state, scaled_target_state, current_synth_params.reshape(1, -1)), axis=1).reshape(-1,)
	#print(lookup_observation)

	action, _ = model.predict(observation=lookup_observation, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
	synth_param_actions = env.unwrapped.synth_agent.actions_dict[action.tolist()]
	# Update synthesis paramters
	for parameter, synth_param_action in enumerate(synth_param_actions):
		current_synth_params[parameter] += synth_param_action.value * step_size
	current_synth_params = np.clip(current_synth_params, 0, 1)
	print(f'action number: {action.tolist()}')	
	print(f'action: {synth_param_actions}')
	print(f'synthesis parameters: {current_synth_params}')

	return current_synth_params


def liveFeaturesIn_handler(address, *args):
    #print(f"{address}: {args}")

	# check which feature is received
	feature = address.split('/')[-1]
	if feature == 'loudness':
		loudness = np.array(args).tolist()
		#print(f"state: {loudness}")
		result = process(loudness)
		sendControlsToPD(result, client)


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")



def sendControlsToPD(resultArray, client):
    # send osc data to PD
    # resultArray is a list

    # convert to string and format message
	resultArray = [str(n) for n in resultArray]
	msg = ' '.join(resultArray)
	#print(msg)

	# send
	client.send_message("/agent-params", msg)




if __name__ == '__main__': 

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--SYNTH_NAME', type=str, default='sin',
						help='name of the folder containing the synth to be used')
	parser.add_argument('--MODEL_NAME', type=str, default='DQN-1719412489',
						help='stable baselines3 model type: PPO, DQN or A2C with corresponding training timestamp')
	parser.add_argument('--MODEL_ITERATION', type=str, default="DQN_2000000",
						help='training iteration of the model to select')
	parser.add_argument('--IP', type=str, default="127.0.0.1",
						help='IP address where to send and receive the data')
	parser.add_argument('--PORT_SEND', type=int, default=6667,
						help='Port to send data to PD')
	parser.add_argument('--PORT_RECEIVE', type=int, default=6666,
						help='Port to receive data from PD')
	parser.add_argument('--UBUNTU', type=bool, default=False,
						help='True if the script runs on Ubuntu')
	args = parser.parse_args(sys.argv[1:])

	## DEFINE SCRIPT PARAMETERS
	synth_name = args.SYNTH_NAME
	model_name = args.MODEL_NAME
	model_iteration = args.MODEL_ITERATION
	ip = args.IP # localhost
	port_snd = args.PORT_SEND # send port to PD
	port_rcv = args.PORT_RECEIVE # receive port from PD
	UBUNTU = args.UBUNTU
	AGENT_TYPE = model_name.split('-')[0]


	## OPEN PD LIVE INTERFACE
	if not UBUNTU:
		pd_executable = constants.macos_pd_executable
	else:
		pd_executable = constants.ubuntu_pd_executable
	command = pd_executable + f' ./00_synths/{synth_name}/live.pd'
	subprocess.Popen(command, shell=True)


	## LOAD MODEL
	MODEL_DIR = f'./00_synths/{synth_name}/gym_models/models/{model_name}/{model_iteration}'

	# LOAD ENV SETTINGS
	ENVIRONMENT_SETTINGS_PATH = f'./00_synths/{synth_name}/gym_models/settings/{model_name}-environment.json'
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


	if AGENT_TYPE == 'A2C':
		model = A2C.load(MODEL_DIR, env=env)
	elif AGENT_TYPE == 'DQN':
		model = DQN.load(MODEL_DIR, env=env)
	else:
		model = PPO.load(MODEL_DIR, env=env)

	
	N_synth_params, synth_step = environment_settings['N_synth_params'], environment_settings['step_size']
	N_tot_features = len(constants.feature_names)
	print(f'Loading model {model_name}')
	print('-'*50)
	print(f'Synthesizer name: {synth_name}')
	print(f'Number synth parameters: {N_synth_params}')
	print(f'Update step: {synth_step}')
	print(f'Features used as state: {environment_settings["features_keep"]}')
	print()

	## OSC SERVER
	# define dispatcher
	dispatcher = Dispatcher()
	dispatcher.map("/feats/*", liveFeaturesIn_handler)
	dispatcher.set_default_handler(default_handler)

	# define client
	client = udp_client.SimpleUDPClient(ip, port_snd)

	# define server
	server = BlockingOSCUDPServer((ip, port_rcv), dispatcher)
	server.serve_forever()  # Blocks forever


