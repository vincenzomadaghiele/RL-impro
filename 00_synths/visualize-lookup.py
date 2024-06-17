## VISUALIZE THE EXTRACTED LOOKUP TABLE USING TSNE AND PCA
# the code generates an interactive map that communicates with a live synth
# this can be used to check which features best capture the data
# the code only needs the lookup table in csv format

import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import IPython.display as Ipd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pythonosc import udp_client


## PD EXECUTABLE
global pd_executable
# find the pd executable in your computer, the following works on mac
pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' # on mac
#pd_executable = '/usr/bin/pd' # on linux


if __name__ == '__main__': 

	# options: 
	# synth name 
	# features to visualize: standard = all
	# method (TSNE or PCA): standard = TSNE

	synth_name = 'sin'

	# use live synth
	command = pd_executable + f' ./{synth_name}/live.pd'
	subprocess.Popen(command, shell=True)


	## LOAD LOOKUP TABLE
	lookup_table_path = f'./{synth_name}/features/lookup_table.csv'
	synth_df = pd.read_csv(lookup_table_path)
	param_names = [col for col in synth_df.columns if col.split('-')[0] == 'param']
	loudness_feature_names = [col for col in synth_df.columns if col.split('-')[0] == 'loudness']
	mfcc_feature_names = [col for col in synth_df.columns if col.split('-')[0] == 'mfcc']
	chroma_feature_names = [col for col in synth_df.columns if col.split('-')[0] == 'chroma']
	specshape_feature_names = [col for col in synth_df.columns if col.split('-')[0] == 'specshape']
	sinefeaturefreqs_feature_names = [col for col in synth_df.columns if col.split('-')[0] == 'sinefeaturefreqs']
	sinefeaturemags_feature_names = [col for col in synth_df.columns if col.split('-')[0] == 'sinefeaturemags']
	feature_names = loudness_feature_names + mfcc_feature_names + chroma_feature_names + specshape_feature_names + sinefeaturefreqs_feature_names + sinefeaturemags_feature_names
	synth_params = synth_df[param_names].values

	synth_df = synth_df.set_index(param_names)


	## SET FEATURES TO USE FOR VISUALIZATION
	features_keep = mfcc_feature_names
	X = synth_df[features_keep].values
	scaler = StandardScaler().fit(X)
	X_normalized = scaler.transform(X)

	pca = PCA(n_components=2).fit(X)
	X_PCA = pca.transform(X)

	tsne = TSNE(n_components=2, random_state=0)
	X_TSNE = tsne.fit_transform(X_normalized) 


	## SET OSC CLIENT FOR INTERACTIVE MAP
	port_snd = 6667 # send port to PD
	ip = "127.0.0.1" # localhost
	client = udp_client.SimpleUDPClient(ip, port_snd)

	def sendControlsToPD(paramsArray, client):
		paramsArray = [str(n) for n in paramsArray]
		msg = ' '.join(paramsArray)
		client.send_message("/params", msg)


	## LOAD INTERACTIVE FIGURE
	fig = plt.figure()
	ax = plt.axes()
	ax.scatter(X_TSNE[:,0], X_TSNE[:,1])
	ax.set_xlabel('1st t-SNE component')
	ax.set_ylabel('2nd t-SNE component')
	text = ax.text(0,0, "", va="bottom", ha="left")

	# find a way to get to the closest parameter based on position
	def onclick(event):
	    newpoint = np.array((event.xdata, event.ydata))
	    distances= np.linalg.norm(X_TSNE - newpoint, axis = 1)
	    min_index = np.argmin(distances)
	    tx = f'synth_params = {synth_params[min_index]}'
	    text.set_text(tx)
	    print(tx)
	    sendControlsToPD(synth_params[min_index], client)

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()


