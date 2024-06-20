import numpy as np
import pandas as pd
import itertools
import subprocess
import random
import pickle
import os 
from sklearn.preprocessing import StandardScaler
from pythonosc import udp_client

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


import matplotlib.pyplot as plt

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.utils import common
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver

#import SynthEnv as synth_env
import time
import json
import datetime


from tf_agents.trajectories import trajectory

import constants
import synthEnvironment as synthenv

'''
## define CONSTANTS (move to global constants.py script)

# number of features for each type
global num_loudness_features
global num_mfcc_features
global num_chroma_features
global num_specshape_features
global num_sinefeaturefreqs_features
global num_sinefeaturemags_features

num_loudness_features = 3
num_mfcc_features = 13
num_chroma_features = 12
num_specshape_features = 7
num_sinefeaturefreqs_features = 10
num_sinefeaturemags_features = 10

# define feature names
global loudness_feature_names
global mfcc_feature_names
global chroma_feature_names
global specshape_feature_names
global sinefeaturefreqs_feature_names
global sinefeaturemags_feature_names
global feature_names

loudness_feature_names = ['loudness-dBFS', 'loudness-TP', 'loudness-dB']
mfcc_feature_names = [f'mfcc-{i}' for i in range(num_mfcc_features)]
chroma_feature_names = ['chroma-A', 'chroma-A#', 'chroma-B', 
	                        'chroma-C', 'chroma-C#', 'chroma-D', 
	                        'chroma-D#', 'chroma-E', 'chroma-F', 
	                        'chroma-F#', 'chroma-G', 'chroma-G#']
specshape_feature_names = ['specshape-centroid', 'specshape-spread', 
	                           'specshape-skewness', 'specshape-kurtosis', 
	                           'specshape-rolloff', 'specshape-flatness', 
	                           'specshape-crest']
sinefeaturefreqs_feature_names = [f'sinefeaturefreqs-{i}' for i in range(num_sinefeaturefreqs_features)]
sinefeaturemags_feature_names = [f'sinefeaturemags-{i}' for i in range(num_sinefeaturemags_features)]
feature_names = loudness_feature_names + mfcc_feature_names + chroma_feature_names + specshape_feature_names + sinefeaturefreqs_feature_names + sinefeaturemags_feature_names
'''

'''
## environment for RL training

class DiscreteSynthMatchingEnv(py_environment.PyEnvironment):
    
    def __init__(self, N_synth_params, 
                    features_keep,
                    features_reward,
                    synth_name, 
                    corpus_name,
                    reward_noise=0.1, 
                    training_mode='mixed_random',
                    reward_type='cubic+prize',
                    step_size=0.05, ip_send="127.0.0.1", port_send=6667, port_send_optimal=6668,
                    verbose = False, live = True, max_episode_duration=3000):
        
        print('Initialized synth environment')
        print('-'*40)
        print('Synthesizer used:')
        print(synth_name)
        print()
        print('Target corpus data:')
        print(corpus_name)
        print()
        print('State features:')
        print(features_keep)
        print()
        print('Reward features:')
        print(features_reward)
        print()

        # user-defined parameters
        self.N_synth_params = N_synth_params # number of parameters of the synthesizer
        
        # features to use for state 
        features = []
        for feat in features_keep:
            if feat == 'loudness':
                features += constants.loudness_feature_names
            elif feat == 'mfcc':
                features += constants.mfcc_feature_names
            elif feat == 'chroma':
                features += constants.chroma_feature_names
            elif feat == 'specshape':
                features += constants.specshape_feature_names
            elif feat == 'sinefeaturefreqs':
                features += constants.sinefeaturefreqs_feature_names
            elif feat == 'sinefeaturemags':
                features += constants.sinefeaturemags_feature_names
            elif feat == 'all':
                features += constants.feature_names
            else:
                features.append(feat)
        
        features = [feat for feat in constants.feature_names if feat in features]
        self.features_keep = features # features to use for training
        self.N_features = len(self.features_keep) # number of features used for training
        #print(self.features_keep)


        # features to use for state 
        features_rew = []
        for feat in features_reward:
            if feat == 'loudness':
                features_rew += constants.loudness_feature_names
            elif feat == 'mfcc':
                features_rew += constants.mfcc_feature_names
            elif feat == 'chroma':
                features_rew += constants.chroma_feature_names
            elif feat == 'specshape':
                features_rew += constants.specshape_feature_names
            elif feat == 'sinefeaturefreqs':
                features_rew += constants.sinefeaturefreqs_feature_names
            elif feat == 'sinefeaturemags':
                features_rew += constants.sinefeaturemags_feature_names
            elif feat == 'all':
                features += constants.feature_names
            else:
                features_rew.append(feat)
        
        # compute reward weights based on user settings
        features_rew = [feat for feat in constants.feature_names if feat in features_rew]
        reward_weights = []
        for feat in self.features_keep:
            if feat in features_rew:
                reward_weights.append(1)
            else:
                reward_weights.append(0)
        reward_weights.append(reward_noise)
        #print(reward_weights)


        self.lookup_table_path = f'./00_synths/{synth_name}/features/lookup_table.csv' # path to the csv lookup table for the synthesizer 
        self.target_corpus_path = f'./01_corpus/{corpus_name}/features/all-files-in-corpus' # path to target corpus feature csv files folder
        self.step_size = step_size # synth parameter update step size
        self.verbose = verbose # print updates
        self.live = live # send actions to synth through osc
        self.max_episode_duration = max_episode_duration # episode duration in random corpus mode
        
        self.training_mode = training_mode # dictates if training on corpus, static random samples or dynamic random samples
        self.reward_type = reward_type

        # reward rules for each feature
        if len(reward_weights) != self.N_features + 1:
            raise ValueError('reward_weights must be of length N_features+1')
        self.reward_weights = np.array(reward_weights)
        self.max_reward = np.sum(self.reward_weights)

        # possible actions are 0, 1 or 2 for each parameter
        possible_actions = [p for p in itertools.product([0, 1, 2], repeat=self.N_synth_params)]
        self.N_possible_actions = len(possible_actions)
        self.actions_dict = dict(zip(range(self.N_possible_actions), possible_actions))
        print('Possible actions:')
        print(self.actions_dict)
        print()

        # action space = paramters of the synth (can go up, down or stay)
        self._action_spec = array_spec.BoundedArraySpec(
                    shape=(), dtype=np.int32, minimum=0, maximum=self.N_possible_actions-1, name='action')
        # observation space = observed features
        self._observation_spec = array_spec.BoundedArraySpec(
                    shape=(1,self.N_features * 2 + self.N_synth_params), dtype=np.float32, name='observation')
        
        # initialize algorithm variables
        self._state = np.random.rand(self.N_features * 2 + self.N_synth_params)
        self._current_synth_params = self._state[-self.N_synth_params:]
        self._episode_ended = False
        self.iteration = 0
        self.episode_duration = self.max_episode_duration


        # LOAD TARGET CORPUS
        self.target_features = np.ones((1,self.N_features))
        self.target_files = []
        for file_path in os.listdir(self.target_corpus_path):
            csv_path = os.path.join(self.target_corpus_path, file_path)
            if csv_path.endswith('.csv'):
                #print(file_path.split('.')[0])
                #print(pd.read_csv(csv_path)[self.features_keep].columns)
                corpus_song = pd.read_csv(csv_path)[self.features_keep].values
                self.target_files.append(corpus_song)
                self.target_features = np.concatenate([self.target_features, corpus_song], axis=0) # KEEP!!
        self.target_features = self.target_features[1:,:]

        # LOAD SYNTH LOOKUP TABLE
        param_names = [f'param-{num_param}' for num_param in range(self.N_synth_params)]
        features_keep_timbre = param_names + self.features_keep
        lookup_table = pd.read_csv(self.lookup_table_path)[features_keep_timbre].values
        self.timbre_params = lookup_table[:, :self.N_synth_params]
        self.timbre_features = lookup_table[:, self.N_synth_params:]
        
        # normalize
        self.scaler = StandardScaler()
        #self.scaled_target_features = self.scaler.fit_transform(self.target_features)
        #self.scaled_timbre_features = self.scaler.transform(self.timbre_features)
        self.scaled_timbre_features = self.scaler.fit_transform(self.timbre_features)
        self.scaled_target_features = self.scaler.transform(self.target_features)
        # normalize file by file
        self.target_files_scaled = [self.scaler.transform(target_file) for target_file in self.target_files]

        # define osc client to PD
        self.ip_send = ip_send # localhost
        self.port_send = port_send # send port to PD
        self.osc_client = udp_client.SimpleUDPClient(self.ip_send, self.port_send)

        # sending optimal parameters to pd for monitoring training
        self.port_send_optimal = port_send_optimal # send port to PD
        self.osc_client_optimal = udp_client.SimpleUDPClient(self.ip_send, self.port_send_optimal)

    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        
        # initialize algorithm
        self._state = np.random.rand(self.N_features * 2 + self.N_synth_params)
        self._current_synth_params = self._state[-self.N_synth_params:]
        self._episode_ended = False
        self.iteration = 0 
        
        if self.training_mode == 'corpus_full_episode':
            # select a random song from corpus 
            self.target_index = np.random.randint(len(self.target_files))
            self.next_file = self.target_files_scaled[self.target_index]
            self.episode_duration = self.next_file.shape[0]
        
        elif self.training_mode == 'corpus_random_segment':
            # choose a random corpus split
            episode_split = np.random.randint(self.corpus_split)
            self.start_index = episode_split * self.max_episode_duration
            self.episode_duration = self.max_episode_duration

        elif self.training_mode in ['static_random', 'dynamic_random', 'mixed_random']:
            # random values act as corpus generation 
            self.new_targets = np.random.rand(1,self.N_synth_params)
            scaled_target = self._params2feats(self.new_targets)
            self.optimal_params, self.optimal_params_state, self.unscaled_optimal_params_state = self._feats2optimalparams(scaled_target)
            self.episode_duration = self.max_episode_duration
            self.current_target = scaled_target
            if self.training_mode == 'mixed_random':
                if np.random.rand() > 0.5:
                    self.episode_mode = 'static'
                else:
                    self.episode_mode = 'dynamic'
        
        elif self.training_mode == 'mixed_random+corpus_full_episode':
            if np.random.rand() > 0.5:
                self.episode_mode = 'corpus'
                # select a random song from corpus 
                self.target_index = np.random.randint(len(self.target_files))
                self.next_file = self.target_files_scaled[self.target_index]
                self.episode_duration = self.next_file.shape[0]
            else:
                # random values act as corpus generation 
                self.new_targets = np.random.rand(1,self.N_synth_params)
                scaled_target = self._params2feats(self.new_targets)
                self.optimal_params, self.optimal_params_state, self.unscaled_optimal_params_state = self._feats2optimalparams(scaled_target)
                self.episode_duration = self.max_episode_duration
                self.current_target = scaled_target
                if np.random.rand() > 0.5:
                    self.episode_mode = 'static'
                else:
                    self.episode_mode = 'dynamic'
        
        return ts.restart(np.array([self._state], dtype=np.float32))
    
    def _step(self, action):
    
        # check for ended episode
        if self._episode_ended:
            return self.reset()
        
        # get commands for each parameter from integer state
        synth_commands = self.actions_dict[action.tolist()]

        # perform action
        for param in range(len(synth_commands)):
            # action = 1 is move up
            if synth_commands[param] == 1:
                self._current_synth_params[param] = (self._current_synth_params[param] + self.step_size).clip(0,1)
            # action = 0 is move down
            elif synth_commands[param] == 0:
                self._current_synth_params[param] = (self._current_synth_params[param] - self.step_size).clip(0,1)
            # action = 2 is not move

        
        # get target for this step according to training mode
        if self.training_mode == 'corpus_full_episode':
            # one episode is one full song from the corpus
            current_target = self.target_files_scaled[self.target_index][self.iteration, :]
            self.current_target = current_target
        
        elif self.training_mode == 'corpus_random_segment':
            # one episode is a segment from the whole corpus starting from a random index
            corpus_index = self.start_index + self.iteration
            current_target = self.scaled_target_features[corpus_index,:]

        elif self.training_mode == 'static_random':
            # one episode is a random combination of parameters
            current_target = self.current_target

        elif self.training_mode == 'dynamic_random':
            # random parameter values change dynamically during an episode with change probablity
            change_probability = 0.005 # probability to change target (eg. 0.5 = change target every 2 steps on avg)    
            if np.random.rand() < change_probability:
                self.new_targets = np.random.rand(1,self.N_synth_params)
                scaled_target = self._params2feats(self.new_targets)
                self.optimal_params, self.optimal_params_state, self.unscaled_optimal_params_state = self._feats2optimalparams(scaled_target)
                self.current_target = scaled_target
        
        elif self.training_mode == 'mixed_random':
            # in some episode the random paramter values to match are static, in others are dynamic 
            if self.episode_mode == 'static':
                current_target = self.current_target
            else:
                change_probability = 0.005 # probability to change target (eg. 0.5 = change target every 2 steps on avg)    
                if np.random.rand() < change_probability:
                    self.new_targets = np.random.rand(1,self.N_synth_params)
                    scaled_target = self._params2feats(self.new_targets)
                    self.optimal_params, self.optimal_params_state, self.unscaled_optimal_params_state = self._feats2optimalparams(scaled_target)
                    self.current_target = scaled_target


        elif self.training_mode == 'mixed_random+corpus_full_episode':
            # in some episode the random paramter values to match are static, in others are dynamic 
            if self.episode_mode == 'corpus':
                current_target = self.target_files_scaled[self.target_index][self.iteration, :] 
                self.current_target = current_target
            elif self.episode_mode == 'static':
                current_target = self.current_target
            else:
                change_probability = 0.005 # probability to change target (eg. 0.5 = change target every 2 steps on avg)    
                if np.random.rand() < change_probability:
                    self.new_targets = np.random.rand(1,self.N_synth_params)
                    scaled_target = self._params2feats(self.new_targets)
                    self.optimal_params, self.optimal_params_state, self.unscaled_optimal_params_state = self._feats2optimalparams(scaled_target)
                    self.current_target = scaled_target


        # target features
        current_target = self.current_target
        #current_target = np.ones(self.N_features) * 0.1

        current_features = self._params2feats(self._current_synth_params)

        # weighted feature vectors for reward
        weighted_state = self.reward_weights * np.append(current_features, np.array(random.random()))
        weighted_target = self.reward_weights * np.append(current_target, np.array(random.random()))
        
        # get optimal synth parameters for current target
        optimal_params, optimal_params_state, unscaled_optimal_params_state = self._feats2optimalparams(current_target)
        
        # reward based on similarity to features
        reward, RMSE = self.computeReward(weighted_target, 
                                            weighted_state, 
                                            reward_type=self.reward_type)

        # reward based on similarity to optimal params
        #reward = self.computeReward(optimal_params, 
        #                            self._current_synth_params, 
        #                            reward_type='cubic+prize')

        # compute new state
        self._state = np.concatenate((current_features, current_target, self._current_synth_params))
        
        if self.verbose:
            print('Action: ', action)
            print('Corresponding commands: ', synth_commands)
            print(f'current synth params: {self._current_synth_params}')
            print(f'current optimal params: {optimal_params}')
            print(f'current synth state: {self.scaler.inverse_transform(weighted_state[:-1].reshape(1,-1))}')
            print(f'optimal params state: {unscaled_optimal_params_state}')
            print(f'current target state: {self.scaler.inverse_transform(weighted_target[:-1].reshape(1,-1))}')
            print(f'scaled synth state: {weighted_state[:-1]}')
            print(f'scaled optimal params state: {optimal_params_state}')
            print(f'scaled target state: {weighted_target[:-1]}')
            print(f'RMSE: {RMSE}')
            print(f'reward: {reward}')
            print()
        
        if self.live:
            self.sendToSynth(self._current_synth_params)
            resultArray = [str(n) for n in optimal_params]
            msg = ' '.join(resultArray)
            self.osc_client_optimal.send_message("/params", msg)


        # end when reward is optimal or max iterations have been reached
        #if self.iteration >= self.episode_duration - 1  or abs(self.optimal_reward - reward) <= 0.05:
        if self.iteration >= self.episode_duration - 1:
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=np.float32), reward)
        else:
            self.iteration += 1
            return ts.transition(
                np.array([self._state], dtype=np.float32), reward, discount=1.0)

    def _params2feats(self, params):
        
        # find closest parameters in the table
        closest = abs(params - self.timbre_params)
        closest_idx = np.argmin(np.sum((closest), axis=1))
        # find corresponding features
        feats = self.scaled_timbre_features[closest_idx, :]
        
        return feats
    
    def _feats2optimalparams(self, scaled_feats):
        
        # find closest parameters in the table
        closest = abs(scaled_feats - self.scaled_timbre_features)
        closest_idx = np.argmin(np.sum((closest), axis=1))
        # find corresponding features
        optimal_params = self.timbre_params[closest_idx, :]
        corresponding_state = self.scaled_timbre_features[closest_idx, :]
        uscaled_corresponding_state = self.scaler.inverse_transform(self.scaled_timbre_features[closest_idx, :].reshape(1, -1))
        
        #print(scaled_feats)
        #print(corresponding_state)
        #print(self.scaler.inverse_transform(scaled_feats.reshape(1, -1)))
        #print(uscaled_corresponding_state)
        
        return optimal_params, corresponding_state, uscaled_corresponding_state
    
    def computeReward(self, target_state, agent_state, reward_type='cubic'):
        
        RMSE = np.sqrt(np.sum((target_state - agent_state)**2)/abs(self.max_reward))
        
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
    
    def sendToSynth(self, synthparams):
        
        resultArray = [str(n) for n in synthparams]
        msg = ' '.join(resultArray)
        self.osc_client.send_message("/params", msg)
    
    def getParams(self):
        
        params_dict = {}
        params_dict['N_synth_params'] = self.N_synth_params
        params_dict['features_keep'] = self.features_keep
        params_dict['N_features'] = self.N_features
        params_dict['lookup_table_path'] = self.lookup_table_path
        params_dict['N_possible_actions'] = self.N_possible_actions
        params_dict['actions_dict'] = self.actions_dict
        params_dict['target_corpus_path'] = self.target_corpus_path
        params_dict['reward_weights'] = self.reward_weights
        params_dict['step_size'] = self.step_size
        params_dict['ip_send'] = self.ip_send
        params_dict['port_send'] = self.port_send
        
        return params_dict


def cube(x):
    if x >= 0:
        return x**(1/3)
    elif x < 0:
        return -(abs(x)**(1/3))



## TRAINING UTILS FUNCTIONS

def compute_avg_return(environment, policy, num_episodes=10):
    
    total_return = 0.0
    for _ in range(num_episodes):
    
        time_step = environment.reset()
        episode_return = 0.0
    
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, replay_buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
  
    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

'''

## EVAL UTILS FUNCTIONS 

def test_metrics(model_dir, test_steps):
    
    saved_environment_settings_path = os.path.join(model_dir, 'environment_settings.json')
    f = open(saved_environment_settings_path)
    environment_settings = json.load(f)
    f.close()
    
    saved_model_settings_path = os.path.join(model_dir, 'model_settings.json')
    f = open(saved_model_settings_path)
    model_settings = json.load(f)
    f.close()
    
    # load policy
    checkpoint_dir = os.path.join(model_dir, 'checkpoint')
    policy_dir = os.path.join(model_dir, 'policy')
    saved_policy = tf.saved_model.load(policy_dir)


    instrument_name = environment_settings['instrument_name']
    corpus_name = environment_settings['corpus_name']
    
    # initialize training environment
    N_synth_params = environment_settings['N_synth_params']
    features_keep = environment_settings['features_keep']
    lookup_table_path = environment_settings['lookup_table_path']
    target_corpus_path = environment_settings['target_corpus_path']
    reward_weights = environment_settings['reward_weights']
    training_mode = environment_settings['training_mode']
    reward_type = environment_settings['reward_type']
    step_size = environment_settings['step_size']
    ip_send = environment_settings['ip_send']
    port_send = environment_settings['port_send']
    port_send_optimal = environment_settings['port_send_optimal']
    verbose = True if environment_settings['verbose'] == 'True' else False
    live = True if environment_settings['live'] == 'True' else False
    max_episode_duration = environment_settings['max_episode_duration']

    # Train corpus 
    corpus_name = "corpus_guitar_train"
    target_corpus_path = "01_target_corpus/corpus_guitar_test/all_features"
    training_mode = "corpus_full_episode"
    environment = synthenv.DiscreteSynthMatchingEnv(N_synth_params, 
                                             features_keep,
                                             features_reward,
                                             synth_name, 
                                             corpus_name,
                                             reward_noise, 
                                             training_mode,
                                             reward_type,
                                             step_size, 
                                             ip_send, 
                                             port_send, 
                                             port_send_optimal,
                                             verbose, 
                                             live, 
                                             max_episode_duration)
    
    environment = tf_py_environment.TFPyEnvironment(py_environment)
    
    time_step = environment.reset()
    action_step = saved_policy.action(time_step)
    time_step = environment.step(action_step.action)
    time_step = environment.reset()
    
    cumulative_rewards = []
    cumulative_RMSEs = []
    for step in range(test_steps):
        cumulative_reward = 0
        time_step = environment.reset()
        step_count = 0
        while not time_step.is_last():
            action_step = saved_policy.action(time_step)
            #print(action_step)
            time_step = environment.step(action_step.action)
            #print(time_step)
            #print(time_step.reward)
            cumulative_reward += time_step.reward
            step_count += 1

        avg_reward = cumulative_reward[0].numpy() / step_count
        avg_RMSE = 0.5 - synthenv.cube(avg_reward/4)
        cumulative_rewards.append(avg_reward)
        cumulative_RMSEs.append(avg_RMSE)
        if step % 10 == 0:
            print(f'Episode {step}')
            print(f'Cumulative reward = {avg_reward}')
            print(f'Cumulative RMSE = {avg_RMSE}')

        
    print('Finished evaluation on training corpus:')
    print('-'*40)
    print(f'Reward mean: {np.array(cumulative_rewards).mean()}')
    print(f'Reward std: {np.array(cumulative_rewards).std()}')
    print(f'RMSE mean: {np.array(cumulative_RMSEs).mean()}')
    print(f'RMSE std: {np.array(cumulative_RMSEs).std()}')
    print()
    
    train_rew_mean = np.array(cumulative_rewards).mean()
    train_rew_std = np.array(cumulative_rewards).std()
    train_RMSE_mean = np.array(cumulative_RMSEs).mean()
    train_RMSE_std = np.array(cumulative_RMSEs).std()
    
    # Test corpus 
    corpus_name = "corpus_guitar_test"
    target_corpus_path = "01_target_corpus/corpus_guitar_test/all_features"
    training_mode = "corpus_full_episode"
    environment = synthenv.DiscreteSynthMatchingEnv(N_synth_params, 
                                         features_keep,
                                         features_reward,
                                         synth_name, 
                                         corpus_name,
                                         reward_noise, 
                                         training_mode,
                                         reward_type,
                                         step_size, 
                                         ip_send, 
                                         port_send, 
                                         port_send_optimal,
                                         verbose, 
                                         live, 
                                         max_episode_duration)
    
    environment = tf_py_environment.TFPyEnvironment(py_environment)
    
    time_step = environment.reset()    
    action_step = saved_policy.action(time_step)
    time_step = environment.step(action_step.action)
    time_step = environment.reset()
    
    cumulative_rewards = []
    cumulative_RMSEs = []
    for step in range(test_steps):
        cumulative_reward = 0
        time_step = environment.reset()
        step_count = 0
        while not time_step.is_last():
            action_step = saved_policy.action(time_step)
            #print(action_step)
            time_step = environment.step(action_step.action)
            #print(time_step)
            #print(time_step.reward)
            cumulative_reward += time_step.reward
            step_count += 1
        
        avg_reward = cumulative_reward[0].numpy() / step_count
        avg_RMSE = 0.5 - synthenv.cube(avg_reward/4)
        cumulative_rewards.append(avg_reward)
        cumulative_RMSEs.append(avg_RMSE)
        if step % 10 == 0:
            print(f'Episode {step}')
            print(f'Cumulative reward = {avg_reward}')
            print(f'Cumulative RMSE = {avg_RMSE}')

    
    print('Finished evaluation on test corpus:')
    print('-'*40)
    print(f'Reward mean: {np.array(cumulative_rewards).mean()}')
    print(f'Reward std: {np.array(cumulative_rewards).std()}')
    print(f'RMSE mean: {np.array(cumulative_RMSEs).mean()}')
    print(f'RMSE std: {np.array(cumulative_RMSEs).std()}')
    print()
    
    test_rew_mean = np.array(cumulative_rewards).mean()
    test_rew_std = np.array(cumulative_rewards).std()
    test_RMSE_mean = np.array(cumulative_RMSEs).mean()
    test_RMSE_std = np.array(cumulative_RMSEs).std()

    return train_rew_mean, train_rew_std, train_RMSE_mean, train_RMSE_std, test_rew_mean, test_rew_std, test_RMSE_mean, test_RMSE_std





if __name__ == '__main__':
    
    
    def train(environment_settings_path, model_settings_path, pd_executable):

        f = open(environment_settings_path)
        environment_settings = json.load(f)
        f.close()
    
        f = open(model_settings_path)
        model_settings = json.load(f)
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
        port_send = environment_settings['port_send']
        port_send_optimal = environment_settings['port_send_optimal']
        verbose = True if environment_settings['verbose'] == 'True' else False
        live = True if environment_settings['live'] == 'True' else False
        max_episode_duration = environment_settings['max_episode_duration']
    
        if live:
            pd_live_script_path = f'./00_synths/{synth_name}/live.pd'
            command = pd_executable + ' ' + pd_live_script_path
            subprocess.Popen(command, shell=True)
    
    
        # check that environment works
        environment = synthenv.DiscreteSynthMatchingEnv(N_synth_params, 
                                             features_keep,
                                             features_reward,
                                             synth_name, 
                                             corpus_name,
                                             reward_noise, 
                                             training_mode,
                                             reward_type,
                                             step_size, 
                                             ip_send, 
                                             port_send, 
                                             port_send_optimal,
                                             verbose, 
                                             live, 
                                             max_episode_duration)
        
        print('Validating environment...')
        utils.validate_py_environment(environment, episodes=5)
        
    
        #%% Define agent parameters
    
        num_iterations = model_settings['num_iterations']
        
        initial_collect_steps = model_settings['initial_collect_steps']
        collect_steps_per_iteration = model_settings['collect_steps_per_iteration']
        replay_buffer_capacity = model_settings['replay_buffer_capacity']
        
        fc_layer_params = tuple(model_settings['fc_layer_params'])
        
        batch_size = model_settings['batch_size']
        learning_rate = model_settings['learning_rate']
        gamma = model_settings['gamma']
        log_interval = model_settings['log_interval']
        
        num_atoms = model_settings['num_atoms']
        min_q_value = model_settings['min_q_value']
        max_q_value = model_settings['max_q_value']
        n_step_update = model_settings['n_step_update']
        
        num_eval_episodes = model_settings['num_eval_episodes']
        eval_interval = model_settings['eval_interval']
    
    
        #%% Train agent
        
        
        # Saving environment parameters
    
        # model_name
        ct = str(datetime.datetime.now().strftime("%Y-%m-%d %H.%M"))
        model_name = f'{ct}'
        print('Model name:')
        print(model_name)
        
        # saving environment parameters
        save_dir = f'./00_synths/{synth_name}/models/{model_name}/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, 'policy'))
    
        # save utils files
        utils_dir = os.path.join(save_dir, 'utils')
        if not os.path.isdir(utils_dir):
            os.mkdir(utils_dir) 
        scaler_filename = "MinMaxScaler.pkl"
        scaler_save_dir = os.path.join(utils_dir, scaler_filename)
        
        with open(scaler_save_dir, 'wb') as f:
            pickle.dump(environment.scaler, f)
        
        action_dict_filename = "ActionDict.pkl"
        action_dict_save_dir = os.path.join(utils_dir, action_dict_filename)
        with open(action_dict_save_dir, 'wb') as f:
            pickle.dump(environment.actions_dict, f)
        
        env_dict_filename = "EnvironmentDict.pkl"
        env_dict_save_dir = os.path.join(utils_dir, env_dict_filename)
        with open(env_dict_save_dir, 'wb') as f:
            pickle.dump(environment.getParams(), f)
    
        env_settings_name = "environment_settings.json"
        json_environment_settings = json.dumps(environment_settings, indent=4)
        env_dict_save_dir = os.path.join(save_dir, env_settings_name)
        with open(env_dict_save_dir, "w") as outfile:
            outfile.write(json_environment_settings)
            
        model_settings_name = "model_settings.json"
        json_model_settings = json.dumps(model_settings, indent=4)
        model_dict_save_dir = os.path.join(save_dir, model_settings_name)
        with open(model_dict_save_dir, "w") as outfile:
            outfile.write(json_model_settings)
        
    
        # Initialize training environments    
        train_py_env = synthenv.DiscreteSynthMatchingEnv(N_synth_params, 
                                             features_keep,
                                             features_reward,
                                             synth_name, 
                                             corpus_name,
                                             reward_noise, 
                                             training_mode,
                                             reward_type,
                                             step_size, 
                                             ip_send, 
                                             port_send, 
                                             port_send_optimal,
                                             verbose, 
                                             live, 
                                             max_episode_duration)
        
        eval_py_env = synthenv.DiscreteSynthMatchingEnv(N_synth_params, 
                                             features_keep,
                                             features_reward,
                                             synth_name, 
                                             corpus_name,
                                             reward_noise, 
                                             training_mode,
                                             reward_type,
                                             step_size, 
                                             ip_send, 
                                             port_send, 
                                             port_send_optimal,
                                             verbose, 
                                             live, 
                                             max_episode_duration)
        
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    
        # initialize DQN network
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
                                                train_env.observation_spec(),
                                                train_env.action_spec(),
                                                num_atoms=num_atoms,
                                                fc_layer_params=fc_layer_params)
        
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.compat.v1.train.get_or_create_global_step()
    
        train_step_counter = tf.Variable(0)
        
        agent = categorical_dqn_agent.CategoricalDqnAgent(
                                        train_env.time_step_spec(),
                                        train_env.action_spec(),
                                        categorical_q_network=categorical_q_net,
                                        optimizer=optimizer,
                                        min_q_value=min_q_value,
                                        max_q_value=max_q_value,
                                        n_step_update=n_step_update,
                                        td_errors_loss_fn=common.element_wise_squared_loss,
                                        gamma=gamma,
                                        train_step_counter=train_step_counter)
        agent.initialize()    
        
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec())
        synthenv.compute_avg_return(eval_env, random_policy, num_eval_episodes)
        
        # initialize replay buffer
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                                                    data_spec=agent.collect_data_spec,
                                                    batch_size=train_env.batch_size,
                                                    max_length=replay_buffer_capacity)
        
        # collect initial steps
        for _ in range(initial_collect_steps):
            synthenv.collect_step(train_env, random_policy, replay_buffer)
        
        # generate dataset from replay buffer
        dataset = replay_buffer.as_dataset(
                                num_parallel_calls=3, sample_batch_size=batch_size,
                                num_steps=n_step_update + 1).prefetch(3)
        iterator = iter(dataset)
        
        # define trainer function
        agent.train = common.function(agent.train)
        
        # Reset the train step
        agent.train_step_counter.assign(0)
        
        # evaluate the agent's policy once before training.
        avg_return = synthenv.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        returns = [avg_return]
        
        # save policy and checkpoints
        policy_dir = os.path.join(save_dir, 'policy')
        save_checkpoints = False
        if save_checkpoints:
            checkpoint_dir = os.path.join(save_dir, 'checkpoint')
            train_checkpointer = common.Checkpointer(
                ckpt_dir=checkpoint_dir,
                max_to_keep=1,
                agent=agent,
                policy=agent.policy,
                replay_buffer=replay_buffer,
                global_step=global_step
            )
            train_checkpointer.initialize_or_restore()
            global_step = tf.compat.v1.train.get_global_step()
        
        
        # Training
        highest_return = float('-inf')
        for _ in range(num_iterations):
    
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                synthenv.collect_step(train_env, agent.collect_policy, replay_buffer)
    
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience)
    
            step = agent.train_step_counter.numpy()
            
            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            
            if step % eval_interval == 0:
                avg_return = synthenv.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
                returns.append(avg_return)
                if avg_return >= highest_return:
                    # keep best model
                    highest_return = avg_return
                    # save policy
                    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
                    tf_policy_saver.save(policy_dir)
                    
                    if save_checkpoints:
                        # save to checkpoint
                        train_checkpointer.save(global_step)
        
        
        #tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        #tf_policy_saver.save(policy_dir)
        #print(model_name)
    
        # training stats
        steps = range(0, num_iterations + 1, eval_interval)
        plt.plot(steps, returns)
        plt.ylabel('Average Return')
        plt.xlabel('Step')
        plt.title(f'Training model {model_name}')
        plt.savefig(os.path.join(save_dir,model_name +'.png'))
        #plt.show()
    
    
        # load and evaluate saved policy
        saved_policy = tf.saved_model.load(policy_dir)
        avg_return = synthenv.compute_avg_return(eval_env, saved_policy, num_eval_episodes)
        print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))



    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


    ## PD EXECUTABLE
    global pd_executable
    UBUNTU = False
    if not UBUNTU:
		# find the pd executable in your computer, the following works on mac
        pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' # on mac
    else:
        pd_executable = '/usr/bin/pd' # on linux

    environment_settings_path = "environment_settings.json"
    model_settings_path = "model_settings.json"

    train(environment_settings_path, model_settings_path, pd_executable)
    

    #%% Test on a new environment

    test_on_new_env = False
    if test_on_new_env:

        corpus_name = "corpus_guitar_test"
        target_corpus_path = "01_target_corpus/corpus_guitar_test/all_features"
        py_environment = synthenv.DiscreteSynthMatchingEnv(N_synth_params, 
                                             features_keep,
                                             features_reward,
                                             synth_name, 
                                             corpus_name,
                                             reward_noise, 
                                             training_mode,
                                             reward_type,
                                             step_size, 
                                             ip_send, 
                                             port_send, 
                                             port_send_optimal,
                                             verbose, 
                                             live, 
                                             max_episode_duration)
        
        # save utils files
        utils_dir = os.path.join(save_dir, 'utils')
        scaler_filename = "MinMaxScaler.pkl"
        scaler_save_dir = os.path.join(utils_dir, scaler_filename)
        #joblib.dump(py_environment.scaler, scaler_filename)
        
        with open(scaler_save_dir, 'wb') as f:
            pickle.dump(py_environment.scaler, f)
        
        action_dict_filename = "ActionDict.pkl"
        action_dict_save_dir = os.path.join(utils_dir, action_dict_filename)
        with open(action_dict_save_dir, 'wb') as f:
            pickle.dump(py_environment.actions_dict, f)
        
        env_dict_filename = "EnvironmentDict.pkl"
        env_dict_save_dir = os.path.join(utils_dir, env_dict_filename)
        with open(env_dict_save_dir, 'wb') as f:
            pickle.dump(py_environment.getParams(), f)
        
        environment = tf_py_environment.TFPyEnvironment(py_environment)
        time_step = environment.reset()
        print(time_step)
        
        
        action_step = agent.policy.action(time_step)
        #print(action_step)
        time_step = environment.step(action_step.action)
        print(time_step)
        #print(time_step.reward)

        time_step = environment.reset()
        print(time_step)
        
        cumulative_reward = 0
        while not time_step.is_last():
            action_step = saved_policy.action(time_step)
            #print(action_step)
            time_step = environment.step(action_step.action)
            #print(time_step)
            #print(time_step.reward)
            cumulative_reward += time_step.reward
            time.sleep(0.01)
        
        print(f'Cumulative reward = {cumulative_reward}')


    #%% Test metrics

    test = True
    test_steps = 20
    if test:
        train_rew_mean, train_rew_std, train_RMSE_mean, train_RMSE_std, test_rew_mean, test_rew_std, test_RMSE_mean, test_RMSE_std = test_metrics(save_dir, test_steps)
        
        
        model_evaluation = {}
        model_evaluation['train_rew_mean'] = train_rew_mean
        model_evaluation['train_rew_std'] = train_rew_std
        model_evaluation['train_RMSE_mean'] = train_RMSE_mean
        model_evaluation['train_RMSE_std'] = train_RMSE_std
        model_evaluation['test_rew_mean'] = test_rew_mean
        model_evaluation['test_rew_std'] = test_rew_std
        model_evaluation['test_RMSE_mean'] = test_RMSE_mean
        model_evaluation['test_RMSE_std'] = test_RMSE_std
        
        model_evaluation_name = "model_evaluation.json"
        json_model_evaluation = json.dumps(model_evaluation, indent=4)
        model_dict_save_dir = os.path.join(save_dir, model_evaluation_name)
        with open(model_dict_save_dir, "w") as outfile:
            outfile.write(json_model_evaluation)


