import os
import numpy as np
from scipy import stats
import pickle 
import pandas as pd
import re


global pd_executable
pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' 


def analyseCorpus(pd_script_path, corpus_path, out_path):
	# computes audio corpus analysis through a pd script

	dir_list = os.listdir(corpus_path)
	for filepath in dir_list:
		
		filename = filepath.split('.')[0]
		result_path = os.path.join(out_path, filename)
		if not os.path.exists(result_path):
			os.mkdir(result_path) 
		command = pd_executable + f' -send "; filename {filename}" -nogui ' + pd_script_path
		print(f'analysing file {filepath}')
		os.system(command)

# sorting functions
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def loadCorpus(path_to_corpus):
    
    # only keep directories
    sample_paths = os.listdir(path_to_corpus)
    sample_paths = [path for path in sample_paths if os.path.isdir(os.path.join(path_to_corpus, path))]
    
    # save features in a dictionary
    print('loading data:')
    print('-'*15)
    data_entries = dict.fromkeys(sample_paths, 0)
    for sample_path in sample_paths:
        print(sample_path)
        path = os.path.join(path_to_corpus, sample_path)
        mfccs = os.listdir(path)
        data_entry = []
        # open single mfcc files
        #print([feat.split('.')[0] for feat in mfccs].sort(key=natural_keys))
        mfccs.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        #mfccs = [feat.split('.')[0] for feat in mfccs].sort(key=natural_keys)

        for mfcc in mfccs:
            mfcc_path = os.path.join(path, mfcc)
            if mfcc_path.endswith('.txt'):
                f = open(mfcc_path, "r")
                print(mfcc_path)
                txt_data = f.read()
                f.close()
                txt_data = txt_data.split(' ')
                mfccs_list = [float(i.split(';')[0]) for i in txt_data]
                data_entry.append(mfccs_list)
        
        mfccs_array = np.array(data_entry)
        data_entries[sample_path] = mfccs_array
    return data_entries


if __name__ == '__main__':

    feature = 'loudness'
    pd_script = f'extract-{feature}_exe.pd'
    corpus_path ='corpus'
    analyseCorpus(pd_script, corpus_path, feature)
    #%%
    data = loadCorpus(feature)
    with open(f'corpus_{feature}.pkl', 'wb') as f:
        pickle.dump(data, f)

    feature = 'mfcc'
    pd_script = f'extract-{feature}_exe.pd'
    corpus_path ='corpus'
    analyseCorpus(pd_script, corpus_path, feature)
    #%%
    data = loadCorpus(feature)
    with open(f'corpus_{feature}.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    feature = 'spectral'
    pd_script = f'extract-{feature}_exe.pd'
    corpus_path ='corpus'
    analyseCorpus(pd_script, corpus_path, feature)
    #%%
    data = loadCorpus(feature)
    with open(f'corpus_{feature}.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    
    #%% load full corpus
        
    with open('corpus_loudness.pkl', 'rb') as f:
        corpus_loudness = pickle.load(f)

    with open('corpus_mfcc.pkl', 'rb') as f:
        corpus_mfcc = pickle.load(f)
        
    with open('corpus_spectral.pkl', 'rb') as f:
        corpus_spectral = pickle.load(f)

    dict_all_features = dict.fromkeys(corpus_loudness.keys(), 0)    
    for key in corpus_loudness.keys():
        dict_all_features[key] = np.concatenate((corpus_loudness[key], corpus_mfcc[key], corpus_spectral[key]))

    with open('corpus_all.pkl', 'wb') as f:
        pickle.dump(dict_all_features, f)
     
    
    feature_names = ['loudness-0', 'loudness-1', 'loudness-2',
                     'bufmfcc_feats-0', 'bufmfcc_feats-1', 'bufmfcc_feats-2', 'bufmfcc_feats-3',
                     'bufmfcc_feats-4', 'bufmfcc_feats-5', 'bufmfcc_feats-6',
                     'spectral-0', 'spectral-1', 'spectral-2', 'spectral-3',
                     'spectral-4', 'spectral-5', 'spectral-6',]

    def filtering(features_vector):
        if features_vector.iloc[16] >= 9500:
            features_vector.iloc[16] = 0
        #if features_vector.iloc[17] >= 2500:
        #    features_vector.iloc[17] = 0
        #if features_vector.iloc[20] >= 7000:
        #    features_vector.iloc[20] = 0
        return features_vector
    
    for key in corpus_loudness.keys():
        corpus_df = pd.DataFrame(data=dict_all_features[key].T, columns=feature_names)
        corpus_df_no_outliers = corpus_df[(np.abs(stats.zscore(corpus_df)) < 2.4).all(axis=1)]
        print(f'Saving to dataset: {key}')
        print(corpus_df.shape, corpus_df_no_outliers.shape)
        corpus_df.to_csv(f'all_features/{key}.csv')
        corpus_df_no_outliers.to_csv(f'all_features_noOutliers/{key}.csv')
        
        
        corpus_df_filtered = corpus_df.copy()
        corpus_df_filtered = corpus_df_filtered.apply(lambda row: filtering(row), axis=1)
        corpus_df_filtered.to_csv(f'all_features_filtered/{key}.csv')



