# A listening agent for live control of synthesis parameters using reinforcement learning

This is the code repository for the paper submitted to AIMC 2024 conference. 

### Abstract:
This paper presents a novel approach for developing customized autonomous agents for musical improvisation. The model we propose employs reinforcement	learning to adaptively control the parameters of a digital synthesizer in response to live audio from a musician. The agent is trained on a corpus of audio files that exemplify the musician’s instrument and stylistic repertoire. During training, the agent learns to imitate the incoming sound according to a set of perceptual descriptors by continuously adjusting the parameters of the synthesizer it controls. To achieve this objective, the agent learns specific strategies that are characteristic of its autonomous behavior in a live interaction. In the paper we detail the design and implementation of the model, as well as discussing the agent’s application in three representative scenarios. 

## 0. Install dependencies
Using the model requires installing the python dependencies in a custom conda environment:
```
$ conda env create -f RLenv.yml
$ conda activate RLenv
```

## 1. Lookup table
The synthesizers used in the paper are in the folder 00_lookup_table. To compute the lookup table from the synthesizer open the Pure Data (PD) patch "00_lookup_table/synthesizer-name/synthesizer-name_collectFeatures.pd". Keeping the PD patch open, execute the following code from the main directory:
```
$ python3 00_lookup_table/rcv-live-lookup.py
```
Then in a separate terminal window, execute this other script to compute the features from the synthesizer:
```
$ python3 00_lookup_table/snd-live-lookup.py
```

## 2. Feature extraction from corpus
To extract the features from the training corpus, run the following code:
```
$ python3 01_target_corpus/corpus_guitar_train/analyse-corpus.py
$ python3 01_target_corpus/corpus_guitar_test/analyse-corpus.py
```
This python code iteratively calls the PD patches in the folder, which compute the descriptors for each file in the folders "01_target_corpus/corpus_guitar_train/corpus" and "01_target_corpus/corpus_guitar_test/corpus".

## 2. Train the model
To train the model, set the parameters in the files "environment_settings.json" and "model_settings.json", then run the following code:
```
$ python3 train.py
```

## 3. Live interaction
To use the agent in a live interaction, execute the python server. One of the pre-trained agents in the repository can be used:
```
$ python3 live-server.py
```
Once the live server is running, open the PD patch corresponding to the synthesizer the agent is controlling "00_lookup_table/synthesizer-name/synthesizer-name_interaction-connect.pd". From the patch, you can either play the same instrument as the agent, or play your own instrument through the PD live input adc~. 
