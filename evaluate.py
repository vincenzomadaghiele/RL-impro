import numpy as np
import os 
import tensorflow as tf
from tf_agents.environments import tf_py_environment
import json

from multi_train_params_in_state import StaticSynthMatchingEnv, test_metrics

if __name__ == '__main__':


    test_model_dirs = ['00_lookup_table/00_sinusoid/models/2024-03-22 23.07-target_loudness-2-2_features',
                       '00_lookup_table/00_sinusoid/models/2024-03-22 23.41-target_spectral-0-2_features',
                       '00_lookup_table/00_sinusoid/models/2024-03-23 00.16-target_loudness-2_spectral-0-2_features'
                       ]
    
    for model_dir in test_model_dirs:
    
        train_rew_mean, train_rew_std, train_RMSE_mean, train_RMSE_std, test_rew_mean, test_rew_std, test_RMSE_mean, test_RMSE_std = test_metrics(model_dir, 100)
