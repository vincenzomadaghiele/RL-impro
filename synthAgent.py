import itertools
import random
import subprocess
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from pythonosc import udp_client
from enum import Enum

import constants


class SynthParamAction(Enum):
	UP = 1
	NOT_MOVE = 0
	DOWN = -1


class DiscreteSynthAgent:

	def __init__(self, 
					synth_name='sin', 
					N_synth_parameters=2, 
					features=['all'],
					step_size=0.05,
					ip_send="127.0.0.1", 
					port_send=6667):

		## AGENT PROPERTIES
		self.synth_name = synth_name
		self.N_synth_parameters = N_synth_parameters
		features = constants.abbreviations2feats(features)
		self.features = [feat for feat in constants.feature_names if feat in features]
		self.step_size = step_size
		self.last_action = ''

		print('Initializing agent')
		print('Synthesizer name:')
		print(self.synth_name)
		print('State features:')
		print(self.features)

		## INITIALIZE ACTION DICTIONARY BASED ON NUMBER OF PARAMETERS
		# possible actions are -1, 0 or 1 for each parameter
		possible_actions = [p for p in itertools.product(SynthParamAction, repeat=self.N_synth_parameters)]
		self.N_possible_actions = len(possible_actions)
		self.actions_dict = dict(zip(range(self.N_possible_actions), possible_actions))
		print('Possible actions:')
		print(self.actions_dict)
		print()

		## LOAD LOOKUP TABLE FOR FEATURES
		self.param_names = [f'param-{num_param}' for num_param in range(self.N_synth_parameters)]
		features_keep = self.param_names + self.features
		lookup_table_path = f'./00_synths/{synth_name}/features/lookup_table.csv'
		self.lookup_table = pd.read_csv(lookup_table_path)[features_keep]
		# normalize lookup table
		self.scaler = StandardScaler()
		self.normalized_lookup_table = self.scaler.fit_transform(self.lookup_table[self.features].values)


		# define osc client to PD
		self.ip_send = ip_send # localhost
		self.port_send = port_send # send port to PD
		self.osc_client = udp_client.SimpleUDPClient(self.ip_send, self.port_send)

		# load live PD interface for rendering
		self.RENDER = False
		if self.RENDER:
			synth_path = f'./00_synths/{self.synth_name}/live.pd'
			UBUNTU = False
			## OPEN PD LIVE INTERFACE
			if not UBUNTU:
				pd_executable = constants.macos_pd_executable
			else:
				pd_executable = constants.ubuntu_pd_executable
			command = pd_executable + ' ' + synth_path
			subprocess.Popen(command, shell=True)

		self.reset()


	def reset(self, seed=None):
		# Initialize synth parameters
		random.seed(seed)
		self.synth_parameter_values = [random.uniform(0,1) for _ in range(self.N_synth_parameters)]
		self.synth_state_features = self.parameters2features(self.synth_parameter_values)


	def perform_action(self, action:int):
		synth_param_actions = self.actions_dict[action]
		self.last_action = synth_param_actions
		# Update synthesis paramters
		for parameter, synth_param_action in enumerate(synth_param_actions):
			self.synth_parameter_values[parameter] += synth_param_action.value * self.step_size
			self.synth_parameter_values[parameter] = np.clip(self.synth_parameter_values[parameter], 0, 1)
		# update synthesizer state as described by the sound features
		self.synth_state_features = self.parameters2features(self.synth_parameter_values)

		#print(f'A_t: {action} --> {synth_param_actions}')
		#print(f'p_synth: {self.synth_parameter_values} ')
		#print(f'f_synth: {self.synth_state_features} ')
		return self.synth_parameter_values, self.synth_state_features


	def render(self):
		synthparams = [str(n) for n in self.synth_parameter_values]
		msg = ' '.join(synthparams)
		self.osc_client.send_message("/agent-params", msg)


	def parameters2features(self, parameters):
		# returns the normalized feature vales corresponding to the synthesis parameters
		closest = abs(parameters - self.lookup_table[self.param_names].values)
		closest_idx = np.argmin(np.sum((closest), axis=1))
		return self.normalized_lookup_table[closest_idx, :]

	def features2optimalparamteres(self, normalized_features):
		# returns the normalized feature vales corresponding to the synthesis parameters
		closest = abs(normalized_features - self.normalized_lookup_table)
		closest_idx = np.argmin(np.sum((closest), axis=1))
		return self.lookup_table.values[closest_idx, :self.N_synth_parameters], self.normalized_lookup_table[closest_idx, :]



# For unit testing
if __name__=="__main__":

	synth_name='sin'
	N_synth_parameters=2

	synthAgent = DiscreteSynthAgent(synth_name, N_synth_parameters)
	synthAgent.render()

	while(True):
		rand_action = random.randint(0, 3**N_synth_parameters-1)
		synthAgent.perform_action(rand_action)
		synthAgent.render()
		#print(rand_action)


