import sys
import argparse

import constants
import training_utils

import tensorflow as tf


if __name__ == '__main__':


    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ENVIRONMENT_SETTINGS', type=str, default='environment_settings.json',
                        help='path to the json file containing environment settings')
    parser.add_argument('--MODEL_SETTINGS', type=str, default='model_settings.json',
                        help='path to the json file containing model settings')
    parser.add_argument('--UBUNTU', type=bool, default=False,
                        help='True if the script runs on Ubuntu')
    args = parser.parse_args(sys.argv[1:])


    ## ARGUMENTS OF THIS PROGRAM ARE:
    # synth name: name of the folder where to compute the lookup table
    # number of parameters: number of parameters of the synth in the folder
    # number of subdivisions: granularity of the lookup table
    # ubuntu or mac os: flag that tells the script where to find the pd executable
    ENVIRONMENT_SETTINGS = args.ENVIRONMENT_SETTINGS
    MODEL_SETTINGS = args.MODEL_SETTINGS
    UBUNTU = args.UBUNTU

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    #tf.config.experimental.set_memory_growth(gpu, True)
    
    ## PD EXECUTABLE
    if not UBUNTU:
        pd_executable = constants.macos_pd_executable
    else:
        pd_executable = constants.ubuntu_pd_executable # on linux
    
    
    model_save_dir = training_utils.train(ENVIRONMENT_SETTINGS, MODEL_SETTINGS, pd_executable)
    print(f'Model name : {model_save_dir}')
    print()
    num_evaluation_episodes = 10
    #model_save_dir = './00_synths/sin/models/2024-06-21 13.08'
    training_utils.evaluate(model_save_dir, num_evaluation_episodes, pd_executable)
    