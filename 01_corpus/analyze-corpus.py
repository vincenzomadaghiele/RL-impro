## THIS SCRIPT COMPUTES THE FEATURES FOR EACH AUDIO FILE SAVED IN '{corpus_name}/audio/...'
# the output of the script is a csv file for each audio file analyzed according to the features

import os
import shutil
import numpy as np
import pandas as pd
from subprocess import Popen # process on multiple threads


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


def getFeatureValues(corpus_name, audio_filename, feature_name, num_features):
	features = []
	min_shape = np.inf
	for i in range(num_features):
		feature_path = f'./{corpus_name}/features/{audio_filename}/{feature_name}/{audio_filename}-{feature_name}-{i}.txt'
		_, feature = readFeatureTxt(feature_path)
		if feature.shape[0] < min_shape:
			min_shape = feature.shape[0]
		features.append(feature)
	features = [feat[:min_shape] for feat in features]
	return np.array(features)



if __name__ == '__main__':


	corpus_name = 'GuitarSet'
	THREADS = 5
	UBUNTU = False

	## PD EXECUTABLE
	global pd_executable
	if not UBUNTU:
		# find the pd executable in your computer, the following works on mac
		pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' # on mac
	else:
		pd_executable = '/usr/bin/pd' # on linux

	# list all sound files in corpus    
	audio_path = f"./{corpus_name}/audio/"
	audio_files_list = os.listdir(audio_path)
	audio_filenames = [audio_file.split('.')[0] for audio_file in audio_files_list]

	## CREATE FOLDERS FOR AUDIO AND FEATURES
	features_path = f'./{corpus_name}/features'
	if not os.path.exists(audio_path):
		print(f'Place the audio files in the directory: "{audio_path}" ')
		os.mkdir(audio_path)
	if os.path.exists(features_path):
		shutil.rmtree(features_path)

	# make directories for analysis
	os.mkdir(features_path)
	for audio_file in audio_filenames:
		audio_features_path = os.path.join(features_path, audio_file)
		os.mkdir(audio_features_path)
		os.mkdir(os.path.join(audio_features_path, 'loudness'))
		os.mkdir(os.path.join(audio_features_path, 'mfcc'))
		os.mkdir(os.path.join(audio_features_path, 'chroma'))
		os.mkdir(os.path.join(audio_features_path, 'specshape'))
		os.mkdir(os.path.join(audio_features_path, 'sinefeaturefreqs'))
		os.mkdir(os.path.join(audio_features_path, 'sinefeaturemags'))


	## ANALYSE AUDIO FILES WITH FLUCOMA
	# multi-thread execution subdiv by subdiv
	pd_analysis_script_path = f'./corpus-analysis.pd'
	subdiv = THREADS # num threads
	for i in range(int(len(audio_filenames) / subdiv)):
		processes = [Popen(pd_executable + f' -send "; filename {audio_filenames[i*subdiv + j]}; corpusname {corpus_name}; " -nogui ' + pd_analysis_script_path, shell=True) for j in range(subdiv)]
		# collect statuses
		exitcodes = [p.wait() for p in processes]

	# remainder single-exectution
	remaining_indices = len(audio_filenames) - (i*subdiv+(subdiv-1))
	if remaining_indices > 0:
		for j in range(remaining_indices):
			command = pd_executable + f' -send "; filename {audio_filenames[(i*subdiv+(subdiv-1)) + j]}; corpusname {corpus_name}; " -nogui ' + pd_analysis_script_path
			os.system(command)


	## COMPUTE CSV FROM AUDIO FILES
	# one different csv file for each audio file in the corpus
	# with features ordered in time

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


	## OPEN CORPUS AUDIO FOLDER
	path = f"./{corpus_name}/audio"
	dir_list = os.listdir(path)
	all_files = [audio_file.split('.')[0] for audio_file in audio_files_list]
	print(f'Reading corpus with {len(all_files)} sound files')

	for audio_file in all_files:

		print(f'Analysing {audio_file}')

		loudness_features = getFeatureValues(corpus_name, audio_file, 'loudness', num_loudness_features)
		mfcc_features = getFeatureValues(corpus_name, audio_file, 'mfcc', num_mfcc_features)
		chroma_features = getFeatureValues(corpus_name, audio_file, 'chroma', num_chroma_features)
		specshape_features = getFeatureValues(corpus_name, audio_file, 'specshape', num_specshape_features)
		sinefeaturefreqs_features = getFeatureValues(corpus_name, audio_file, 'sinefeaturefreqs', num_sinefeaturefreqs_features)
		sinefeaturemags_features = getFeatureValues(corpus_name, audio_file, 'sinefeaturemags', num_sinefeaturemags_features)

		print(loudness_features.shape)
		print(mfcc_features.shape)
		print(chroma_features.shape)
		print(specshape_features.shape)

		# compute dataframe
		print('Saving dataset...')
		data = np.hstack((loudness_features.T,mfcc_features.T, 
							chroma_features.T, specshape_features.T, 
							sinefeaturefreqs_features.T, sinefeaturemags_features.T))
		audiofile_df = pd.DataFrame(data=data, columns=feature_names)
		audiofile_df = audiofile_df.sort_index()
		print(audiofile_df)
		audiofile_df.to_csv(f'{corpus_name}/features/{audio_file}/{audio_file}.csv')




