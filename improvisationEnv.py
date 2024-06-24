import os
import numpy as np
import pandas as pd

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
				features_keep='all',
				synth_name='sin', 
				corpus_name='GuitarSet',
				N_synth_parameters=2, 
				step_size=0.05,
				ip_send="127.0.0.1", port_send=6667,
				render_mode=None):

		self.render_mode = render_mode

		# features
		self.features_keep = constants.abbreviations2feats(['all'])
		self.N_features = len(self.features_keep)

		# synthesizer
		self.synth_name = synth_name
		self.N_synth_parameters = N_synth_parameters

		# intialize agent
		self.synth_agent = sa.DiscreteSynthAgent(synth_name=synth_name,
												N_synth_parameters=N_synth_parameters, 
												features=features_keep,
												step_size=step_size,
												ip_send="127.0.0.1", 
												port_send=6667)


		# target corpus
		self.target_corpus_path = f'./01_corpus/{corpus_name}/features/all-files-in-corpus' # path to target corpus feature csv files folder
		#self.target_features = np.ones((1,self.N_features))
		self.target_songs = []
		for file_path in os.listdir(self.target_corpus_path):
			csv_path = os.path.join(self.target_corpus_path, file_path)
			if csv_path.endswith('.csv'):
				#print(file_path.split('.')[0])
				#print(pd.read_csv(csv_path)[self.features_keep].columns)
				corpus_song = pd.read_csv(csv_path)[self.features_keep].values
				self.target_songs.append(corpus_song)
				self.target_features = np.concatenate([self.target_features, corpus_song], axis=0)
		#self.target_features = self.target_features[1:,:]

        # normalize all corpus files using the agent's scaler
        # this allows to "project" the corpus in the space of the agent
        # normalize corpus song by song
        self.normalized_target_songs = [self.synth_agent.scaler.transform(target_file) for target_file in self.target_songs]


		# action space
		N_possible_actions = 3**len(sa.SynthParamAction)
		self.action_space = spaces.Discrete(N_possible_actions)

		# observation space
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(self.N_features * 2 + self.N_synth_parameters,),
			dtype=np.float32
		)


	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.synth_agent.reset(seed=seed)


		# Construct the observation state:
		# [agent_features, target_features, synth_parameters]
		obs = np.concatenate((self.synth_agent.synth_state_features, self.synth_agent.synth_parameter_values))

		# Additional info to return. For debugging or whatever.
		info = {}

		# Render environment
		if(self.render_mode=='human'):
		    self.render()

		# Return observation and info
		return obs, info


	def step(self, action):
		self.synth_agent.perform_action(action)
		obs = np.concatenate((self.synth_agent.synth_state_features, self.synth_agent.synth_parameter_values))
		reward = 1
		terminated = False
		if reward == 2:
			terminated = True

		info = {}

		return obs, reward, terminated, False, info


	def render(self):
		self.synth_agent.render()






# For unit testing
if __name__=="__main__":

	env = gym.make('improvisation-matching-v0', render_mode='human')

	# Use this to check our custom environment
	# print("Check environment begin")
	# check_env(env.unwrapped)
	# print("Check environment end")

	'''
	# Reset environment
	obs = env.reset()[0]

	# Take some random actions
	while(True):
	    rand_action = env.action_space.sample()
	    obs, reward, terminated, _, _ = env.step(rand_action)

	    if(terminated):
	        obs = env.reset()[0]

	'''
