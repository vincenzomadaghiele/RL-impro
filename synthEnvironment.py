import itertools
import random
import os 

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from pythonosc import udp_client
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

import constants


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
        self.synth_name = synth_name
        print(synth_name)
        print()
        print('Target corpus data:')
        self.corpus_name = corpus_name
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
        
        ## FEATURES TO USE TO DESCRIBE STATE
        features = constants.abbreviations2feats([features_keep])        
        features = [feat for feat in constants.feature_names if feat in features]
        self.features_keep = features # features to use for training
        self.N_features = len(self.features_keep) # number of features used for training
        #print(self.features_keep)

        ## FEATURES TO USE TO COMPUTE REWARD
        self.features_rew = constants.abbreviations2feats([features_reward]) 
        reward_weights = []
        for feat in self.features_keep:
            if feat in self.features_rew:
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
        params_dict['synth_name'] = self.synth_name
        params_dict['corpus_name'] = self.corpus_name
        params_dict['N_synth_params'] = self.N_synth_params
        params_dict['features_keep'] = self.features_keep
        params_dict['features_reward'] = self.features_rew
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



