## EQUIVALENT ENVIRONMENT CODED IN GYMNASIUM

import os
import subprocess
import numpy as np
import pandas as pd
from pythonosc import udp_client

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import constants
import synthAgent as sa


# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
	id='improvisation-matching-v0',                                # call it whatever you want
	entry_point='improvisationEnv:ImprovisationMatchingEnv', # module_name:class_name
)

class ImprovisationMatchingEnv(gym.Env):
	# metadata is a required attribute
	# render_modes in our environment is either None or 'human'.
	# render_fps is not used in our env, but we are require to declare a non-zero value.
	metadata = {"render_modes": ["human"], 'render_fps': 4}

	def __init__(self, 
				features_keep=['all'],
				features_reward=['all'],
				synth_name='sin', 
				N_synth_parameters=2, 
				corpus_name='GuitarSet',
				step_size=0.05,
				reward_noise=0.1, 
				training_mode='mixed_random',
				ip_send="127.0.0.1", agent_port_send=6667, target_port_send=6668,
				max_episode_duration=3000,
				render_mode=None):

		# training parameters
		self.render_mode = render_mode
		self.training_mode = training_mode
		self.max_episode_duration = max_episode_duration

		# features
		self.features_keep = constants.abbreviations2feats(features_keep)
		self.N_features = len(self.features_keep)

		# synthesizer
		self.synth_name = synth_name
		self.N_synth_parameters = N_synth_parameters

		# agent
		self.synth_agent = sa.DiscreteSynthAgent(synth_name=synth_name,
												N_synth_parameters=N_synth_parameters, 
												features=features_keep,
												step_size=step_size,
												ip_send=ip_send, 
												port_send=agent_port_send)


		# target corpus
		self.target_corpus_path = f'./01_corpus/{corpus_name}/features/all-files-in-corpus' # path to target corpus feature csv files folder
		#self.target_features = np.ones((1,self.N_features))
		self.target_songs = []
		self.target_songs_names = []
		for file_path in os.listdir(self.target_corpus_path):
			csv_path = os.path.join(self.target_corpus_path, file_path)
			if csv_path.endswith('.csv'):
				#print(file_path.split('.')[0])
				#print(pd.read_csv(csv_path)[self.features_keep].columns)
				corpus_song = pd.read_csv(csv_path)[self.features_keep].values
				self.target_songs.append(corpus_song)
				self.target_songs_names.append(file_path.split('.')[0])
				#self.target_features = np.concatenate([self.target_features, corpus_song], axis=0)
		#self.target_features = self.target_features[1:,:]

		# normalize all corpus files using the agent's scaler
		# this allows to "project" the corpus in the space of the agent
		# normalize corpus song by song
		self.normalized_target_songs = [self.synth_agent.scaler.transform(target_file) for target_file in self.target_songs]


		# action space
		N_possible_actions = len(sa.SynthParamAction)**self.N_synth_parameters
		self.action_space = spaces.Discrete(N_possible_actions)

		# observation space
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(self.N_features * 2 + self.N_synth_parameters,),
			dtype=np.float64
		)

		# reward properties
		self.features_reward = constants.abbreviations2feats(features_reward) 
		reward_weights = []
		for feat in self.features_keep:
			if feat in self.features_reward:
				reward_weights.append(1)
			else:
				reward_weights.append(0)
		reward_weights.append(reward_noise)
		# reward rules for each feature
		if len(reward_weights) != self.N_features + 1:
			raise ValueError('reward_weights must be of length N_features+1')
		self.reward_weights = np.array(reward_weights)
		self.max_reward = np.sum(self.reward_weights)


		# define osc client to PD
		self.ip_send = ip_send # localhost
		self.agent_osc_client = udp_client.SimpleUDPClient(self.ip_send, agent_port_send)
		self.target_osc_client = udp_client.SimpleUDPClient(self.ip_send, target_port_send)

		# load live PD interface for rendering
		if(self.render_mode=='human'):
			synth_path = f'./00_synths/{self.synth_name}/live.pd'
			UBUNTU = False
			## OPEN PD LIVE INTERFACE
			if not UBUNTU:
				pd_executable = constants.macos_pd_executable
			else:
				pd_executable = constants.ubuntu_pd_executable
			command = pd_executable + ' ' + synth_path
			subprocess.Popen(command, shell=True)



	def reset(self, seed=None, options=None):

		super().reset(seed=seed)
		self.synth_agent.reset(seed=seed)

		# Additional info to return.
		info = {}
		self.episode_step = 0


		# select the mode of the next episode
		if self.training_mode == 'corpus':
			self.episode_mode = 'corpus'
		elif self.training_mode == 'static_random':
			self.episode_mode = 'static_random'
		elif self.training_mode == 'dynamic_random':
			self.episode_mode = 'dynamic_random'
		elif self.training_mode == 'mixed_random':
			if self.np_random.random() > 0.5:
				self.episode_mode = 'static_random'
			else:
				self.episode_mode = 'dynamic_random'
		elif self.training_mode == 'mixed_random+corpus':
			if self.np_random.random() > 0.5:
				self.episode_mode = 'corpus'
			else:
				if self.np_random.random() > 0.5:
					self.episode_mode = 'static_random'
				else:
					self.episode_mode = 'dynamic_random'

		info['episode_mode'] = self.episode_mode

		# initialize the targets based on episode mode
		if self.episode_mode == 'corpus':
			# select a random song from the corpus
			target_song_index = self.np_random.integers(0, len(self.target_songs))
			self.normalized_target_song = self.normalized_target_songs[target_song_index]
			self.episode_duration = self.normalized_target_song.shape[1]
			# update target features
			self.target_features = self.normalized_target_song[self.episode_step]
			self.optimal_target_synth_parameters, _ = self.synth_agent.features2optimalparamteres(self.target_features)
			info['target_song'] = self.target_songs_names[target_song_index]
			info['target_synth_parameters'] = self.optimal_target_synth_parameters
			# compute optimal target synth parameters

		elif self.episode_mode in ['static_random', 'dynamic_random']:
			self.episode_duration = self.max_episode_duration
			self.optimal_target_synth_parameters = np.array([self.np_random.random() for _ in range(self.N_synth_parameters)])
			# update target features
			self.target_features = self.synth_agent.parameters2features(self.optimal_target_synth_parameters)
			info['target_synth_parameters'] = self.optimal_target_synth_parameters

		# Construct the observation state:
		# [normalized_agent_features, normalzied_target_features, synth_parameters]
		obs = np.concatenate((self.synth_agent.synth_state_features, self.target_features, self.synth_agent.synth_parameter_values))

		# Render environment
		if(self.render_mode=='human'):
			self.render()

		# Return observation and info
		return obs, info


	def step(self, action):
		# update synthesis paramteres
		self.synth_agent.perform_action(action)

		# update target features
		if self.episode_mode == 'corpus':
			self.target_features = self.normalized_target_song[self.episode_step]
			self.optimal_target_synth_parameters, _ = self.synth_agent.features2optimalparamteres(self.target_features)

		elif self.episode_mode == 'static_random':
			pass
		elif self.episode_mode == 'dynamic_random':
			change_probability = 0.005 # probability to change target (eg. 0.5 = change target every 2 steps on avg)    
			if self.np_random.random() < change_probability:
				self.optimal_target_synth_parameters = np.array([self.np_random.random() for _ in range(self.N_synth_parameters)])
				self.target_features = self.synth_agent.parameters2features(self.optimal_target_synth_parameters)

		# update observed state
		obs = np.concatenate((self.synth_agent.synth_state_features, self.target_features, self.synth_agent.synth_parameter_values))

		# calculate reward
		reward, RMSE = self.weightedSimilarityReward(self.target_features, self.synth_agent.synth_state_features)

		info = {}
		info['episode_step'] = self.episode_step
		info['action'] = action
		info['action_commands'] = self.synth_agent.actions_dict[action]
		info['synth_features'] = self.synth_agent.synth_state_features
		info['target_features'] = self.target_features
		info['synth_parameteres'] = self.synth_agent.synth_parameter_values
		info['target_optimal_parameters'] = self.optimal_target_synth_parameters
		info['reward'] = reward
		info['RMSE'] = RMSE
		print(info)

		# update episode step
		if self.episode_step >= self.episode_duration - 1:
			terminated = True
		else:
			self.episode_step += 1
			terminated = False

		if(self.render_mode=='human'):
			self.render()

		return obs, reward, terminated, False, info


	def render(self):
		synthparams = [str(n) for n in self.synth_agent.synth_parameter_values]
		msg = ' '.join(synthparams)
		self.agent_osc_client.send_message("/agent-params", msg)

		synthparams = [str(n) for n in self.optimal_target_synth_parameters]
		msg = ' '.join(synthparams)
		self.target_osc_client.send_message("/target-params", msg)


	def weightedSimilarityReward(self, target_state, agent_state, reward_type='cubic'):

		weighted_agent_state = self.reward_weights * np.append(agent_state, np.array(self.np_random.random()))
		weighted_target = self.reward_weights * np.append(target_state, np.array(self.np_random.random()))

		RMSE = np.sqrt(np.sum((weighted_target - weighted_agent_state)**2)/abs(self.max_reward))
		
		if reward_type == 'cubic':
			reward = 8 * (1 - RMSE - 0.5)**3
		elif reward_type == 'linear':
			reward = 1 - RMSE
		elif reward_type == 'cubic+prize':
			reward = 4 * (1 - RMSE - 0.5)**3
			if RMSE <= 0.1:
				reward += 10
		elif reward_type == 'prize':
			if RMSE <= 0.1:
				reward = 10
			else: 
				reward = -10

		return reward, RMSE




# For unit testing
if __name__=="__main__":

	env = gym.make('improvisation-matching-v0', 
					#synth_name='granular', 
					#N_synth_parameters=4, 
					render_mode='human')
	print(env.observation_space)

	# Use this to check our custom environment
	print("Check environment begin")
	check_env(env.unwrapped)
	print("Check environment end")

	# Reset environment
	obs = env.reset()[0]

	# Take some random actions
	while(True):
	    rand_action = env.action_space.sample()
	    obs, reward, terminated, _, _ = env.step(rand_action)

	    if(terminated):
	        obs = env.reset()[0]



