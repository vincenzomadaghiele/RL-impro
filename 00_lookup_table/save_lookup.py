import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == '__main__': 

    
    N_PARAMS = 3
    
    instrument_name = '01_FM3'
    feature_name = 'loudness_mfcc_spectral'
    save_dir = f'{instrument_name}'
    
    lookup_filename = f'{instrument_name}/features/{instrument_name}_{feature_name}_lookup_table.pkl'

    lookup_save_dir = os.path.join(save_dir, lookup_filename)
    
    with open(lookup_filename, 'rb') as f:
    	lookup_table = pickle.load(f)
        
    lookup_table = np.array(lookup_table)
    
    #min_index = lookup_table[:,N_PARAMS].min()
    #max_index = lookup_table[:,N_PARAMS].max()
    
    #lookup_table = [row for row in  lookup_table if row[N_PARAMS] != min_index]
    #lookup_table = [row for row in  lookup_table if row[N_PARAMS] != min_index+1]
    #lookup_table = np.array(lookup_table)

    param_names = [f'p{num_param}' for num_param in range(N_PARAMS)]
    feature_names = ['loudness-0', 'loudness-1', 'loudness-2',
                     'bufmfcc_feats-0', 'bufmfcc_feats-1', 'bufmfcc_feats-2',
                     'bufmfcc_feats-4', 'bufmfcc_feats-5', 'bufmfcc_feats-6',
                     'bufmfcc_feats-7', 'bufmfcc_feats-8', 'bufmfcc_feats-9',
                     'bufmfcc_feats-10', 'bufmfcc_feats-11', 'bufmfcc_feats-12',
                     'bufmfcc_feats-13', 
                     'spectral-0', 'spectral-1', 'spectral-2', 'spectral-3',
                     'spectral-4', 'spectral-5', 'spectral-6']
    
    features_keep = ["loudness-0", "loudness-1", "loudness-2", "bufmfcc_feats-0", "bufmfcc_feats-1", "bufmfcc_feats-2","bufmfcc_feats-4", "bufmfcc_feats-5", "bufmfcc_feats-6","bufmfcc_feats-7", "bufmfcc_feats-8", "bufmfcc_feats-9","bufmfcc_feats-10", "bufmfcc_feats-11", "bufmfcc_feats-12", "bufmfcc_feats-13",  "spectral-0", "spectral-1", "spectral-2", "spectral-3", "spectral-4", "spectral-5", "spectral-6"]
    
    all_feature_names = param_names + ['col_idx'] + feature_names

    lookup_table_df = pd.DataFrame(data=np.array(lookup_table), columns=all_feature_names)
    lookup_table_df = lookup_table_df.groupby(param_names, as_index=False).mean()
    lookup_table_df = lookup_table_df.drop(columns=['col_idx'])
    lookup_table_df = lookup_table_df.set_index(param_names)
    
    # when vol is 0 set to min
    #min_cols = lookup_table_df.min()
    #lookup_table_df[lookup_table_df['loudness-2'] == 0] = min_cols

    # outliers
    lookup_df_no_outliers = lookup_table_df[(np.abs(stats.zscore(lookup_table_df)) < 3).all(axis=1)]
    lookup_table_df.to_csv(f'{instrument_name}/features/{instrument_name}_{feature_name}_lookup_table.csv')
    lookup_df_no_outliers.to_csv(f'{instrument_name}/features/{instrument_name}_{feature_name}_lookup_table_noOutliers.csv')
    
    def filtering(features_vector):
        if features_vector.iloc[16] >= 9500:
            features_vector.iloc[16] = 0
        #if features_vector.iloc[17] >= 2500:
        #    features_vector.iloc[17] = 0
        #if features_vector.iloc[20] >= 7000:
        #    features_vector.iloc[20] = 0
        return features_vector
    
    lookup_table_df_filtered = lookup_table_df.copy()
    lookup_table_df_filtered = lookup_table_df_filtered.apply(lambda row: filtering(row), axis=1)
    lookup_table_df_filtered.to_csv(f'{instrument_name}/features/{instrument_name}_{feature_name}_lookup_table_filtered.csv')


