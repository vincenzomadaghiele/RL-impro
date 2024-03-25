# this server gets mfcc from the live player as input
# processes them through the model
# and sends synth control parameters as output

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import numpy as np
from itertools import permutations
import itertools
import time
import os


if __name__ == '__main__': 

	# DEFINE
	ip = "127.0.0.1" # localhost
	port_snd = 6667 # send port to PD

	# define client
	client = udp_client.SimpleUDPClient(ip, port_snd)

	# generate all possible combinations of parameters 
	SUBDIV = 30 # subdvisions between 0 and 1
	N_PARAMS = 3 # number of synth parameters
	elements = np.array(range(0, SUBDIV)) / SUBDIV
	#permutations = np.array(list(permutations(elements, N_PARAMS)))
	permutations = [p for p in itertools.product(elements, repeat=N_PARAMS)]
	print(f'num permutations: {len(permutations)}')
	#np.savetxt('permutations.txt', permutations, delimiter=',') # save to txt

	for paramters in permutations:

		# execute pd and extract mfcc
		resultArray = [str(n) for n in paramters]
		synth_paramters = ' '.join(resultArray)
		print(synth_paramters)
		client.send_message("/result", synth_paramters)
		time.sleep(0.3)

