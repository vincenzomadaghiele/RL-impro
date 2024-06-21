import constants
import training_utils

import tensorflow as tf


if __name__ == '__main__':
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    #tf.config.experimental.set_memory_growth(gpu, True)
    
    environment_settings_path = "environment_settings.json"
    model_settings_path = "model_settings.json"
    UBUNTU = False

    ## PD EXECUTABLE
    if not UBUNTU:
        pd_executable = constants.macos_pd_executable
    else:
        pd_executable = constants.ubuntu_pd_executable # on linux
    
    
    model_save_dir = training_utils.train(environment_settings_path, model_settings_path, pd_executable)
    print(f'Model name : {model_save_dir}')
    print()
    num_evaluation_episodes = 10
    model_save_dir = './00_synths/sin/models/2024-06-21 13.08'
    training_utils.evaluate(model_save_dir, num_evaluation_episodes, pd_executable)
    