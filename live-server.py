# this server gets mfcc from the live player as input
# processes them through the model
# and sends synth control parameters as output

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import numpy as np

import os
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import TimeStep
import pickle

import SynthEnv as synth_env



def sendControlsToPD(resultArray, client):
    # send osc data to PD
    # resultArray is a list

    # convert to string and format message
	resultArray = [str(n) for n in resultArray]
	msg = ' '.join(resultArray)
	#print(msg)

	# send
	client.send_message("/result", msg)


# operate transformations: this has to be modified by hand
def filtering(features_vector):
	# spectral-0
	if features_vector[16] >= 9500:
		features_vector[16] = 0
	return features_vector


def process(arrayIn):

	# separate components
	current_synth_params = np.array(arrayIn[N_tot_features*2:])
	synth_state = np.array(arrayIn[:N_tot_features])
	target_state = np.array(arrayIn[N_tot_features:N_tot_features*2])
	#print(current_synth_params, synth_state, target_state)

	# filter array as in dataset
	filtered_synth_state = filtering(synth_state)
	filtered_target_state = filtering(target_state)

	# select features used in model
	index_feats_keep = [feature_names.index(feat) for feat in env_dict['features_keep']]
	synth_state_kept = filtered_synth_state[index_feats_keep]
	target_state_kept = filtered_target_state[index_feats_keep]

	# scaler
	scaled_synth_state = scaler.transform(synth_state_kept.reshape(1, -1))
	scaled_target_state = scaler.transform(target_state_kept.reshape(1, -1))
	#print(np.concatenate((scaled_synth_state, scaled_target_state, current_synth_params.reshape(1, -1)), axis=1).shape)
	observation = np.concatenate((scaled_synth_state, scaled_target_state, current_synth_params.reshape(1, -1)), axis=1).reshape(1, 1, -1)

	# feed to the agent
	observed_timestep = TimeStep(tf.convert_to_tensor(np.array([1]), dtype='int32'), 
								tf.convert_to_tensor(np.array([1]), dtype='float32'), 
								tf.convert_to_tensor(np.array([1]), dtype='float32'), 
								tf.convert_to_tensor(observation, dtype='float32'))
	action_step = saved_policy.action(observed_timestep)
	# decode action with dictionary from env
	action_num = action_step.action.numpy().tolist()[0]

	synth_commands = actions_dict[action_num]
	print(f'action number: {action_num} --> A_t: {synth_commands}')

	# perform action
	send_synth_params = np.zeros(N_synth_params)
	for param in range(len(synth_commands)):
		# action = 1 is move up
		if synth_commands[param] == 1:
			send_synth_params[param] = (current_synth_params[param] + synth_step).clip(0,1)
		# action = 0 is move down
		elif synth_commands[param] == 0:
			send_synth_params[param] = (current_synth_params[param] - synth_step).clip(0,1)
		else:
			send_synth_params[param] = current_synth_params[param]
		# action = 2 is not move

	print(f'synthesis parameters: {send_synth_params}')

	return send_synth_params


def liveFeaturesIn_handler(address, *args):
    #print(f"{address}: {args}")

	# check which feature is received
	feature = address.split('/')[-1]
	if feature == 'loudness':
		loudness = np.array(args).tolist()
		print(f"state: {loudness}")
		result = process(loudness)
		sendControlsToPD(result, client)


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")


if __name__ == '__main__': 

	# DEFINE
	ip = "127.0.0.1" # localhost
	port_rcv = 6666 # receive port from PD
	port_snd = 6667 # send port to PD

	# load model
	instrument_name = '02_FM4'
	model_name = '2024-03-22 06.06-target_loudness-0_loudness-1_loudness-2_bufmfcc_feats-0_bufmfcc_feats-1_bufmfcc_feats-2_bufmfcc_feats-4_bufmfcc_feats-5_bufmfcc_feats-6_spectral-0_spectral-1_spectral-2_spectral-3_spectral-4_spectral-5_spectral-6-16_features'
	save_dir = f'00_lookup_table/{instrument_name}/models/{model_name}/'

	feature_names = ["loudness-0", 
					"loudness-1", 
					"loudness-2",
					"bufmfcc_feats-0",
					"bufmfcc_feats-1",
					"bufmfcc_feats-2",
					"bufmfcc_feats-4",
					"bufmfcc_feats-5",
					"bufmfcc_feats-6",
					"bufmfcc_feats-7",
					"bufmfcc_feats-8",
					"bufmfcc_feats-9",
					"bufmfcc_feats-10",
					"bufmfcc_feats-11",
					"bufmfcc_feats-12",
					"bufmfcc_feats-13",  
					"spectral-0", 
					"spectral-1", 
					"spectral-2", 
					"spectral-3", 
					"spectral-4", 
					"spectral-5", 
					"spectral-6"
					]

	# load utils
	utils_dir = os.path.join(save_dir, 'utils')
	scaler_filename = "MinMaxScaler.pkl"
	scaler_save_dir = os.path.join(utils_dir, scaler_filename)
	action_dict_filename = "ActionDict.pkl"
	action_dict_save_dir = os.path.join(utils_dir, action_dict_filename)
	env_dict_filename = "EnvironmentDict.pkl"
	env_dict_save_dir = os.path.join(utils_dir, env_dict_filename)
	
	with open(scaler_save_dir, 'rb') as f:
		scaler = pickle.load(f)

	with open(action_dict_save_dir, 'rb') as f:
		actions_dict = pickle.load(f)

	with open(env_dict_save_dir, 'rb') as f:
		env_dict = pickle.load(f)

	N_synth_params, N_features, synth_step = env_dict['N_synth_params'], env_dict['N_features'], env_dict['step_size']
	print(N_synth_params, N_features, synth_step)
	synth_step = 0.02

	N_tot_features = len(feature_names)
	len_arrayIn = N_tot_features * 2 + env_dict['N_synth_params']

	# load policy
	checkpoint_dir = os.path.join(save_dir, 'checkpoint')
	policy_dir = os.path.join(save_dir, 'policy')
	saved_policy = tf.saved_model.load(policy_dir)


	# OSC SERVER
	# define dispatcher
	dispatcher = Dispatcher()
	dispatcher.map("/feats/*", liveFeaturesIn_handler)
	dispatcher.set_default_handler(default_handler)

	# define client
	client = udp_client.SimpleUDPClient(ip, port_snd)

	# define server
	server = BlockingOSCUDPServer((ip, port_rcv), dispatcher)
	server.serve_forever()  # Blocks forever



