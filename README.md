# A reinforcement learning agent for live sound improvisation

This is the official repository of the RL improvisation project.
It contains all the code for the paper:

> Vincenzo Madaghiele, Stefano Fasciani.
> [**A listening agent for live control of synthesis parameters using reinforcement learning**]().
> In _Proceedings of AI and Music Creativity Conference (AIMC) 2024_, 9-11 September 2024, Oxford (UK).

## Instructions

The features are only computed using Flucoma to ensure compatibility between PD and python in a live scenario. 


### Generate lookup table for a custom synth
1. Create a PD synthesizer with a given number of synthesis parameters. The synthesizer is a pd abstraction called "synth.pd". The abstraction has only one input: a list of synthesis parameters as floating points betwwen 0 and 1. The synthesizer outputs sound according to the given list of parameters.
2. Save the synth in a folder called './00_synths/{synth name}/synth.pd'. In the same folder, copy the PD scripts "live.pd" and "record.pd".
3. Run the code "compute-lookup.py" to compute the lookup table of the synth. This code generates and records sound from the pd synth you chose by iterating through its parameters at equal intervals. The recorded sounds are then analysed according to a set of Flucoma descriptors in PD and saved as .txt files in the "features" folder. The .txt are then read and combined in the "lookup_table.csv" file
4. Run "visualize-lookup.py" to visualize the contents of the lookup table using TSNE, while playing the synth via the interactive map. This script allows to test which features are better at describing the parameter variations. In general, if similar parameter combinations are close to each other in the TSNE plot, the features are good at capturing the synthesizer. 


### Generate feature analysis for a custom corpus
1. Place your collection of audio tracks in the folder '01_corpus/{corpus_name}/audio'. 
2. Run the code '01_corpus/analyze-corpus.py'. This code outputs a csv file for each audio track in the corpus. The csv collects all the features extracted from the audio file. 


### Training the model
1. Set up the environment parameters by modifying the file 'environment_settings.json'. The field 'features_keep' allows to select which features to use for training among the ones extracted during the analysis phase; these features are used to descibe the state of the agent and the musician at a given point in time. The single name of a feature will select only that feature. Inserting the whole feature category will override single features and select the whole category (for example, selecting 'loudness-dB' will add it to the list of features, but adding 'loudness' will add all the features in that category).
2. The field 'features_reward' allows to select which features to use for the reward. These features will be used to calculate the similarity metric. The single name of a feature will select only that feature. Inserting the whole feature category will override single features and select the whole category.


#### Dependencies

The python dependencies for the project are collected in the "RLenv.yml" script. 
This project also the depends on Pure Data (PD), and the PD libraries Flucoma and zexy.