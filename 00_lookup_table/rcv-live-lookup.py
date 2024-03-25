# this server gets mfcc from the live player as input
# processes them through the model
# and sends synth control parameters as output

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import numpy as np

import os
import pickle

# default servers

def liveFeaturesIn_handler(address, *args):
    #print(f"{address}: {args}")

	# check which feature is received
	feature = address.split('/')[-1]
	if feature == 'loudness':
		loudness = np.array(args).tolist()
		#print(f"{feature}: {loudness}")
		print(loudness[:5], loudness[7], loudness[21])
		features.append(loudness)
		with open(lookup_save_dir, 'wb') as f:
			pickle.dump(np.array(features), f)

		#print(np.array(features))
		#result = process(loudness)
		#sendControlsToPD(result, client)

def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")


if __name__ == '__main__': 

	# DEFINE
	ip = "127.0.0.1" # localhost
	port_rcv = 6666 # receive port from PD

	features = []
	#instrument_name = '05_rave_sol_ordinario'
	instrument_name = '01_FM3'
	feature_name = 'loudness_mfcc_spectral'
	save_dir = f'{instrument_name}/features'

	lookup_filename = f'{instrument_name}_{feature_name}_lookup_table.pkl'
	lookup_save_dir = os.path.join(save_dir, lookup_filename)

	# OSC SERVER
	# define dispatcher
	dispatcher = Dispatcher()
	dispatcher.map("/feats/*", liveFeaturesIn_handler)
	dispatcher.set_default_handler(default_handler)

	# define server
	server = BlockingOSCUDPServer((ip, port_rcv), dispatcher)
	server.serve_forever()  # Blocks forever



