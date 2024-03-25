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

## 1. Feature extraction
The synthesizers used in the paper are in the folder 00_lookup_table. To compute the lookup table from the synthesizer open the Pure Data (PD) patch "00_lookup_table/<synthesizer-name>/<synthesizer-name>-collectFeatures.pd". Keeping the PD patch open, execute the following code from the main directory:
```
$ python3 00_lookup_table/rcv-live-lookup.py
```
Then in a separate terminal window, execute this other script:
```
$ python3 00_lookup_table/snd-live-lookup.py
```


## 2. Train the model


## 3. Live interaction

