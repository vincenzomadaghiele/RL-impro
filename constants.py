## CONSTANTS


# PD executables
macos_pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' # on mac
ubuntu_pd_executable = '/usr/bin/pd' # on linux


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


def abbreviations2feats(features_list):
    features = []
    for feat in features_list:
        if feat == 'loudness':
            features += loudness_feature_names
        elif feat == 'mfcc':
            features += mfcc_feature_names
        elif feat == 'chroma':
            features += chroma_feature_names
        elif feat == 'specshape':
            features += specshape_feature_names
        elif feat == 'sinefeaturefreqs':
            features += sinefeaturefreqs_feature_names
        elif feat == 'sinefeaturemags':
            features += sinefeaturemags_feature_names
        elif feat == 'all':
            features += feature_names
        else:
            features.append(feat)
    
    features = [feat for feat in feature_names if feat in features]
    return features
