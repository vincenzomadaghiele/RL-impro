## define CONSTANTS (move to global constants.py script)

# number of features for each type
global num_loudness_features
global num_mfcc_features
global num_chroma_features
global num_specshape_features
global num_sinefeaturefreqs_features
global num_sinefeaturemags_features

num_loudness_features = 3
num_mfcc_features = 13
num_chroma_features = 12
num_specshape_features = 7
num_sinefeaturefreqs_features = 10
num_sinefeaturemags_features = 10

# define feature names
global loudness_feature_names
global mfcc_feature_names
global chroma_feature_names
global specshape_feature_names
global sinefeaturefreqs_feature_names
global sinefeaturemags_feature_names
global feature_names

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
