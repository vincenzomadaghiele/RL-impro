# A reinforcement learning agent for live sound improvisation

This project develops a musical agent for improvisation. The model employs reinforcement learning to adaptively control the parameters of a sound synthesizer in response to live audio from a musician. The agent is trained on a corpus of audio files that exemplify the musician’s instrument and stylistic repertoire. During training, the agent listens and learns to imitate the incoming sound according to a set of perceptual descriptors by continuously adjusting the parameters of the synthesizer it controls. To achieve this objective, the agent learns specific strategies that are characteristic of its autonomous behavior in a live interaction.

More information about the project is in the paper:
> Vincenzo Madaghiele, Stefano Fasciani.
> [**A listening agent for live control of synthesis parameters using reinforcement learning**]().
> In _Proceedings of AI and Music Creativity Conference (AIMC) 2024_, 9-11 September 2024, Oxford (UK).

## Running the code

### Install dependencies
This project employs python code for machine learning and Pure Data for sound synthesis. 

Pure Data (PD) is an open source computer music environment, it can be downloaded [here](https://puredata.info/downloads). The [Flucoma](https://www.flucoma.org/) library for Pure Data is used for computation of sound descriptor in its PD implementation. Installation instructions for Flucoma with PD can be found [here](https://learn.flucoma.org/installation/pd/). The `zexy` library for PD is used for OSC communication between python and PD, it can be installed by typing `zexy` in the deken externals manager (`Help -> find externals`) and clicking on `install`.

The python dependencies for the project can be installed in a custom conda environment by running the following code in the directory of this repository:
```
conda env create -f environment.yml
conda activate gymenv
```

### Playing with pre-trained agents
Pre-trained models are available for three basic synthesizers: sine wave, frequency modulation and granular. To use a pre-trained model controlling one of these synthesizers, run the following code:
```
python3 live-server.py --SYNTH_NAME <synthesizer name> --MODEL_NAME <model name> 
```
This script will activate the python server loading the RL model and load the PD patch `live.pd` corresponding to the chosen synthesizer.
The model name is a combination of a timestamp and the type of RL agent used, for example `1720616936-DQN`. The saved models can be found in the directory `00_synths/<synth-name>/gym_models/models`. For example, to use the granular synthesizer trained to match spectral shape, MFCCs and chroma descriptors, run this code:
```
python3 live-server.py --SYNTH_NAME granular --MODEL_NAME 1720616936-DQN
```



## Training agents with a custom synthesizer
It is possible to train the agent on any custom synthesizer coded in Pure Data, using any corpus of sond files representing the musician. 

### Making a custom synth in PD
Create a PD synthesizer with a given number of synthesis parameters. The synthesizer is a PD abstraction called `synth.pd`. The abstraction has only one input: a list of synthesis parameters as floating points betwwen 0 and 1. The synthesizer outputs sound according to the given list of parameters.
The file `synth.pd` should be saved in the directory `./00_synths/<synth-name>/synth.pd`. `<synth-name>` will be the name you assign to your custom synthesizer. In the same directory, copy the PD scripts `live.pd`, `live-analysis.pd` and `record.pd` from the other synthesizers in `./00_synths`.

### Generating the lookup table for a custom synth
The script `00_synths/compute-lookup.py` computes the lookup table of the synth. This code generates and records sound from the PD synth you chose by iterating through its parameters at equal intervals. The recorded sounds are then analysed using the Flucoma descriptors in PD and saved as .txt files in the directory `00_synths/<synth-name>/features`. The .txt are then combined in the `00_synths/<synth-name>/lookup_table.csv` file. An example of the command to run is:
```
cd 00_synths
python3 compute-lookup.py --SYNTH_NAME <synth-name> --N_params <number of synthesis parameters> --SUBDIV <granluarity of the lookup table> --WINDOW_SIZE <fft window size>
```

The script `./00_synths/visualize-lookup.py` generates an interactive 2D representation of the lookup table using PCA or TSNE, allowing to explore the lookup table and listen to how each parameter combination sounds like.
```
cd 00_synths
python3 visualize-lookup.py --SYNTH_NAME <synth-name> 
```

### Generating feature analysis for a custom corpus
Place your collection of audio tracks in the folder `01_corpus/<corpus name>/audio`. `<corpus name>` is the name you assign to your custom corpus.
The script `01_corpus/analyze-corpus.py` generates a .csv file for each audio track in the corpus, containing the descriptors resulting from the analysis. 
```
cd 01_corpus
python3 analyze-corpus.py --CORPUS_NAME <synth-name> --WINDOW_SIZE <fft window size>
```

### Training the agent
To train an agent from scratch using a custom synthesizer, et up the environment parameters by modifying the file `environment_settings.json`. 

The field `features_keep` allows to select which features to use for training among the ones extracted during the analysis phase; these features are used to descibe the state of the agent and the musician at a given point in time. The single name of a feature will select only that feature. Inserting the whole feature category will override single features and select the whole category (for example, selecting `loudness-dB` will add it to the list of features, but selecting `loudness` will add all the features in that category).
The field `features_reward` allows to select which features to use for the reward. These features will be used to calculate the similarity metric. The single name of a feature will select only that feature. Inserting the whole feature category will override single features and select the whole category.
```
python3 train.py
```

To follow the development of the training using tensorboard you can run the following code in a separate terminal, and then copy `` in your browser search filed. 
```
python3 -m tensorboard.main --logdir ./00_synths/<synth-name>/gym_models/logs
```

The trained agents are saved in the directory `./00_synths/<synth-name>/gym_models`.

### Playing with the agent
To play with a trained agent, run the following code, selecting a model from the folder `./00_synths/<synth-name>/gym_models/models`:
```
python3 live-server.py --SYNTH_NAME <synth-name> --MODEL_NAME <model-name>
```

## Cite
```
@inproceedings{madaghiele2024RLimpro,
  author    = {Madaghiele, Vincenzo and Fasciani, Stefano},
  title     = {{A listening agent for live control of synthesis parameters using reinforcement learning}},
  booktitle = {Proceedings of AI and Music Creativity Conference (AIMC)},
  year      = {2024},
  month     = {09},
  address   = {Oxford (UK)}
}
```


