#! /usr/bin/env python
""" Utility to plot the first layer of convolutions learned by
the Deep q-network.

(Assumes dnn convolutions)

Usage:

plot_filters.py PICKLED_NN_FILE
"""

import sys
import matplotlib.pyplot as plt
import cPickle
import lasagne.layers
from run_nature import Parameters
from q_network import DeepQLearner
import q_learner
from lasagne.layers import dnn


net_file = open(sys.argv[1], 'r')
try:
    network = cPickle.load(net_file)
except EOFError:
    net_file.close()
    net_file = open(sys.argv[1], 'rb')
    network = cPickle.load(net_file)
print network
q_layers = lasagne.layers.get_all_layers(network.l_out)
for l in range(1,q_layers.__len__()):
    # q_layers[l].get_output_shape_for(img)
    if dnn.Conv2DDNNLayer.__instancecheck__(q_layers[l]):
        w = q_layers[1].W.get_value()
        count = 1
        plt.figure()
        for f in range(w.shape[0]):  # filters
            for c in range(w.shape[1]):  # channels/time-steps
                plt.subplot(w.shape[0], w.shape[1], count)
                img = w[f, c, :, :]
                plt.imshow(img, vmin=img.min(), vmax=img.max(),
                           interpolation='none', cmap='gray')
                plt.xticks(())
                plt.yticks(())
                count += 1
        # plt.show()
        plt.savefig("dnn{}.jpg".format(l))
        plt.cla()
        plt.clf()

        plt.close()
        print("Plot {}".format(l))
