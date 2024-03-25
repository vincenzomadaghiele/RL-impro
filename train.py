import numpy as np
import pandas as pd
import itertools
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

import SynthEnv as synth_env
import time
import json
import datetime



if __name__ == '__main__':
    
    
    f = open("environment_settings.json")
    environment_settings = json.load(f)
    f.close()

    f = open("model_settings.json")
    model_settings = json.load(f)
    f.close()

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


    # check that environment works
    environment = synth_env.SynthEnv(N_synth_params, 
                                     features_keep, 
                                     lookup_table_path, 
                                     target_corpus_path,
                                     reward_weights, 
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


    #%% variable parameter for multi-training

    reward_weightss = [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       ]
    
    variation_name = ['loudness', 'spectral centroid', 'loudness and spectral centroid']
    variation_name = ['all']
    returns_per_mode = []
    for reward_weights in reward_weightss:

        # Saving environment parameters

        # model_name
        ct = str(datetime.datetime.now().strftime("%Y-%m-%d %H.%M"))
        target_features = [features_keep[i] for i in range(len(features_keep)) if reward_weights[i] != 0]
        target_features = '_'.join(target_features)
        model_name = f'{ct}-target_{target_features}-{len(features_keep)}_features'
        print('Model name:')
        print(model_name)
        
        # saving environment parameters
        save_dir = f'00_lookup_table/{instrument_name}/models/{model_name}/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

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
        train_py_env = synth_env.SynthEnv(N_synth_params, 
                                             features_keep, 
                                             lookup_table_path, 
                                             target_corpus_path,
                                             reward_weights, 
                                             training_mode,
                                             reward_type,
                                             step_size, 
                                             ip_send, 
                                             port_send, 
                                             port_send_optimal,
                                             verbose, 
                                             live, 
                                             max_episode_duration)
        
        eval_py_env = synth_env.SynthEnv(N_synth_params, 
                                             features_keep, 
                                             lookup_table_path, 
                                             target_corpus_path,
                                             reward_weights, 
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
        synth_env.compute_avg_return(eval_env, random_policy, num_eval_episodes)
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity)
            
        for _ in range(initial_collect_steps):
            synth_env.collect_step(train_env, random_policy, replay_buffer)
        
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=batch_size,
            num_steps=n_step_update + 1).prefetch(3)
        
        iterator = iter(dataset)
        
        
        agent.train = common.function(agent.train)
        
        # Reset the train step
        agent.train_step_counter.assign(0)
        
        # Evaluate the agent's policy once before training.
        avg_return = synth_env.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        returns = [avg_return]
        
        # save policy and checkpoints
        policy_dir = os.path.join(save_dir, 'policy')
        save_checkpoints = True
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
                synth_env.collect_step(train_env, agent.collect_policy, replay_buffer)
    
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience)
    
            step = agent.train_step_counter.numpy()
            
            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            
            if step % eval_interval == 0:
                avg_return = synth_env.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
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
        plt.show()
    
    
        # load and evaluate saved policy
        
        saved_policy = tf.saved_model.load(policy_dir)
        avg_return = synth_env.compute_avg_return(eval_env, saved_policy, num_eval_episodes)
        print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))

    
        returns_per_mode.append(returns)


        #%% Test on a new environment
    
        test_on_new_env = False
        if test_on_new_env:
    
            corpus_name = "corpus_guitar_test"
            target_corpus_path = "01_target_corpus/corpus_guitar_test/all_features"
            py_environment = synth_env.SynthEnv(N_synth_params, 
                                                 features_keep, 
                                                 lookup_table_path, 
                                                 target_corpus_path,
                                                 reward_weights, 
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
            train_rew_mean, train_rew_std, train_RMSE_mean, train_RMSE_std, test_rew_mean, test_rew_std, test_RMSE_mean, test_RMSE_std = synth_env.test_metrics(save_dir, test_steps)
            
            
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

            
    
    #%% training stats
    
    variation_name = ['loudness', 'spectral centroid', 'loudness and spectral centroid']
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    avg_step = 8
    steps = range(0, num_iterations + 1, eval_interval)
    ma_returns_per_mode = [moving_average(ret, avg_step) for ret in returns_per_mode]
    ma_returns_per_mode = [pd.Series(ret).rolling(window=avg_step).mean() for ret in returns_per_mode]
    std_returns_per_mode = [pd.Series(ret).rolling(window=avg_step).std() / 7 for ret in returns_per_mode]    
    
    fig, ax = plt.subplots()
    for i in range(len(ma_returns_per_mode)):
        ax.plot(steps, ma_returns_per_mode[i], label=variation_name[i])
        ax.fill_between(steps, (ma_returns_per_mode[i]-std_returns_per_mode[i]), (ma_returns_per_mode[0]+std_returns_per_mode[0]), alpha=.2)
    plt.ylabel('Average Return')
    plt.xlabel('Training Step')
    title = 'Moving average Return over training episodes with different matching features'
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_dir, title +'.png'))
    plt.show()
    
    #steps = range(0, num_iterations + 1, eval_interval)
    for i in range(len(ma_returns_per_mode)):
        plt.plot(steps, returns_per_mode[i], label=variation_name[i])
    plt.ylabel('Average Return')
    plt.xlabel('Training Step')
    title = 'Average Return over training episodes with different matching features'
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_dir, title +'.png'))
    plt.show()
    
    
