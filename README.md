## Instructions

1. Create a PD synthesizer with a given number of synthesis parameters. The synthesizer is a pd abstraction called "synth.pd". The abstraction has only one input: a list of synthesis parameters as floating points betwwen 0 and 1. The synthesizer outputs sound according to the given list of parameters.
2. Save the synth in a folder called './00_synths/{synth name}/synth.pd'. In the same folder, copy the PD scripts "live.pd" and "record.pd".
3. Run the code "compute-lookup.py" to compute the lookup table of the synth. This code generates and records sound from the pd synth you chose by iterating through its parameters at equal intervals. The recorded sounds are then analysed according to a set of Flucoma descriptors in PD and saved as .txt files in the "features" folder. The .txt are then read and combined in the "lookup_table.csv" file
4. Run "visualize-lookup.py" to visualize the contents of the lookup table using TSNE, while playing the synth via the interactive map. This script allows to test which features are better at describing the parameter variations. In general, if similar parameter combinations are close to each other in the TSNE plot, the features are good at capturing the synthesizer. 


#### Dependencies

The python dependencies for the project are collected in the "RLenv.yml" script. 
This project also the depends on Pure Data (PD), and the PD libraries Flucoma and zexy.