## THIS SCRIPT COMPUTES THE LOOKUP TABLE FOR THE SYNTHESIZER SAVED IN '{synth_name}/synth.pd'
# it is possible to compute the table on a more powerful computer 
# and visualize everything just with the csv

import os
import numpy as np
import pandas as pd
import itertools
import shutil
import sys
import argparse
from subprocess import Popen # process on multiple threads


## DEFINE FUNCTIONS TO COMPUTE LOOKUP TABLE

def readFeatureTxt(feature_path):
	feature = []
	with open(feature_path, 'r') as f: 
		for line in f: 
			data = line.split()
			for n in data:
				feature.append(n)
	feature = feature[1:-1]
	feature = np.array([float(feat) for feat in feature])
	f.close()
	return feature.mean(), feature


def getFeatureMean(synth_name, audio_filename, feature_name, num_feats):

    # get the feature txt files
    path = f"./{synth_name}/features/{feature_name}"
    dir_list = os.listdir(path)
    features_path = []
    for i in range(num_feats):
        features_path.append(f'./{synth_name}/features/{feature_name}/{audio_filename}-{feature_name}-{i}.txt')
    
    # get features and calculate average
    features = []
    for feature_path in features_path:
        feature, _ = readFeatureTxt(feature_path)
        features.append(feature)

    return features


def computeLookup(synth_name):

	## DEFINE FEATURE NAMES AND QUANTITY
	# number of features for each type
	num_loudness_features = 3
	num_mfcc_features = 13
	num_chroma_features = 12
	num_specshape_features = 7
	num_sinefeaturefreqs_features = 10
	num_sinefeaturemags_features = 10

	# define feature names
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


	## OPEN SYNTH DATA
	path = f"./{synth_name}/audio"
	dir_list = os.listdir(path) 
	all_files = ['.'.join(dir.split('.')[:-1]) for dir in dir_list if '.'.join(dir.split('.')[:-1])]
	print(f'Reading dataset with {len(all_files)} sound files')

	# define parameter names 
	synth_params = np.array([audio_filename.split('-') for audio_filename in all_files]).astype(np.float64)
	param_names = [f'param-{i}' for i in range(synth_params.shape[1])]
	column_names = param_names + feature_names

	# extract features
	print('Extracting features...')
	dataset_features = []
	for audio_filename in all_files:
	    
	    loudness_features = getFeatureMean(synth_name, audio_filename, 'loudness', num_loudness_features)
	    mfcc_features = getFeatureMean(synth_name, audio_filename, 'mfcc', num_mfcc_features)
	    chroma_features = getFeatureMean(synth_name, audio_filename, 'chroma', num_chroma_features)
	    specshape_features = getFeatureMean(synth_name, audio_filename, 'specshape', num_specshape_features)
	    sinefeaturefreqs_features = getFeatureMean(synth_name, audio_filename, 'sinefeaturefreqs', num_sinefeaturefreqs_features)
	    sinefeaturemags_features = getFeatureMean(synth_name, audio_filename, 'sinefeaturemags', num_sinefeaturemags_features)
	    
	    all_features = loudness_features + mfcc_features + chroma_features + specshape_features + sinefeaturefreqs_features + sinefeaturemags_features
	    dataset_features.append(all_features)

	dataset_features = np.array(dataset_features)

	# compute dataframe
	print('Saving dataset...')
	data = np.hstack((synth_params,dataset_features))
	synth_df = pd.DataFrame(data=data, columns=column_names)
	synth_df = synth_df.set_index(param_names)
	synth_df = synth_df.sort_index()
	print(synth_df)
	synth_df.to_csv(f'{synth_name}/features/lookup_table.csv')



if __name__ == '__main__':


	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--SYNTH_NAME', type=str, default='sin',
						help='name of the folder containing the synth')
	parser.add_argument('--N_PARAMS', type=int, default=2,
						help='number of parameters of the synth')
	parser.add_argument('--SUBDIV', type=int, default=10,
						help='granularity of the lookup table')
	parser.add_argument('--THREADS', type=int, default=5,
						help='number of parallel threads for computation')
	parser.add_argument('--WINDOW_SIZE', type=int, default=1024,
						help='size of FFT window for feature computation')
	parser.add_argument('--WINDOW_OVERLAP', type=int, default=512,
						help='overlap of FFT windows for feature computation')
	parser.add_argument('--UBUNTU', type=bool, default=False,
						help='True if the script runs on Ubuntu')
	args = parser.parse_args(sys.argv[1:])


	## ARGUMENTS OF THIS PROGRAM ARE:
	# synth name: name of the folder where to compute the lookup table
	# number of parameters: number of parameters of the synth in the folder
	# number of subdivisions: granularity of the lookup table
	# ubuntu or mac os: flag that tells the script where to find the pd executable
	synth_name = args.SYNTH_NAME
	N_PARAMS = args.N_PARAMS
	SUBDIV = args.SUBDIV
	THREADS = args.THREADS
	WINDOW_SIZE = args.WINDOW_SIZE
	WINDOW_OVERLAP = args.WINDOW_OVERLAP
	UBUNTU = args.UBUNTU

	## PD EXECUTABLE
	global pd_executable
	if not UBUNTU:
		# find the pd executable in your computer, the following works on mac
		pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' # on mac
	else:
		pd_executable = '/usr/bin/pd' # on linux


	#synth_name = 'sin'
	## CREATE FOLDERS FOR AUDIO AND FEATURES
	audio_path = f'./{synth_name}/audio'
	features_path = f'./{synth_name}/features'
	if os.path.exists(audio_path):
		shutil.rmtree(audio_path)
	if os.path.exists(features_path):
		shutil.rmtree(features_path)

	os.mkdir(audio_path)
	os.mkdir(features_path)
	os.mkdir(os.path.join(features_path, 'loudness'))
	os.mkdir(os.path.join(features_path, 'mfcc'))
	os.mkdir(os.path.join(features_path, 'chroma'))
	os.mkdir(os.path.join(features_path, 'specshape'))
	os.mkdir(os.path.join(features_path, 'sinefeaturefreqs'))
	os.mkdir(os.path.join(features_path, 'sinefeaturemags'))


	## GENERATE AUDIO
	# generate all possible combinations of parameters 
	#SUBDIV = 20 # subdvisions between 0 and 1
	#N_PARAMS = 2 # number of synth parameters
	elements = np.array(range(0, SUBDIV)) / SUBDIV
	#permutations = np.array(list(permutations(elements, N_PARAMS)))
	permutations = [p for p in itertools.product(elements, repeat=N_PARAMS)]
	print(f'num permutations: {len(permutations)}')

	command_strings = []
	audio_filenames = []
	for paramters in permutations:
		params_string = ''
		audio_filename = ''
		for i in range(len(paramters)):
			params_string += f'{paramters[i]} ' 
			if paramters[i] == 0:
				audio_filename += '0-'
			else:
				audio_filename += f'{paramters[i]}-'

		print(params_string)
		command_strings.append(params_string)
		audio_filenames.append(audio_filename[:-1])


	## RECORD MANY AUDIO FILES
	# multi-thread execution subdiv by subdiv
	pd_record_script_path = f'./{synth_name}/record.pd'
	subdiv = THREADS # num threads
	for i in range(int(len(command_strings) / subdiv)):
		processes = [Popen(pd_executable + f' -send "; synth_params {command_strings[i*subdiv + j]}; filename {audio_filenames[i*subdiv + j]}; " -nogui ' + pd_record_script_path, shell=True) for j in range(subdiv)]
		# collect statuses
		exitcodes = [p.wait() for p in processes]

	# remainder single-exectution
	remaining_indices = len(command_strings) - (i*subdiv+(subdiv-1))
	if remaining_indices > 0:
		for j in range(remaining_indices):
			command = pd_executable + f' -send "; synth_params {command_strings[(i*subdiv+(subdiv-1)) + j]}; filename {audio_filenames[(i*subdiv+(subdiv-1)) + j]}; " -nogui ' + pd_record_script_path
			os.system(command)


	## ANALYSE AUDIO FILES WITH FLUCOMA
	# multi-thread execution subdiv by subdiv
	pd_analysis_script_path = f'./analysis.pd'
	subdiv = THREADS # num threads
	for i in range(int(len(audio_filenames) / subdiv)):
		processes = [Popen(pd_executable + f' -send "; filename {audio_filenames[i*subdiv + j]}; synthname {synth_name}; fftsize {WINDOW_SIZE}; " -nogui ' + pd_analysis_script_path, shell=True) for j in range(subdiv)]
		# collect statuses
		exitcodes = [p.wait() for p in processes]

	# remainder single-exectution
	remaining_indices = len(audio_filenames) - (i*subdiv+(subdiv-1))
	if remaining_indices > 0:
		for j in range(remaining_indices):
			command = pd_executable + f' -send "; filename {audio_filenames[(i*subdiv+(subdiv-1)) + j]}; synthname {synth_name};  fftsize {WINDOW_SIZE}; " -nogui ' + pd_analysis_script_path
			os.system(command)


	## COMPUTE DATASET TABLE
	computeLookup(synth_name)
