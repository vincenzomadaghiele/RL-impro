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
    

    ## DEFINE AGENT PARAM
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


    ## TRAIN AGENT
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
    
    return save_dir


def evaluate(model_dir, test_episodes, pd_executable):
    
    saved_environment_settings_path = os.path.join(model_dir, 'environment_settings.json')
    f = open(saved_environment_settings_path)
    environment_settings = json.load(f)
    f.close()
    
    # load policy
    policy_dir = os.path.join(model_dir, 'policy')
    saved_policy = tf.saved_model.load(policy_dir)
    
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
    
    environment = tf_py_environment.TFPyEnvironment(environment)
    
    time_step = environment.reset()
    action_step = saved_policy.action(time_step)
    time_step = environment.step(action_step.action)
    time_step = environment.reset()
    
    cumulative_rewards = []
    cumulative_RMSEs = []
    for step in range(test_episodes):
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


