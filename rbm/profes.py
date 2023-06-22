#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:18:27 2019

@author: mauri
"""


from rbm import *


r=RBM(num_visible=9,num_hidden=4)


training_data =np.array([[1,1,1,0,1,1,1,1,1],[1,1,1,0,1,1,0,0,1],[0,1,1,0,1,1,0,0,1],[0,1,1,0,1,1,0,1,1],[1,1,0,0,0,0,0,1,1],[1,1,0,0,1,1,0,0,1], [0,1,0,0,0,1,0,0,0],[0,1,1,0,0,1,1,1,0]])



r.train(training_data, max_epochs = 5000) # Don't run the training for more than 5000 epochs.

visible_data = np.array([[0,0,0,0,0,0,1,0,1],[0,0,0,1,0,0,0,0,1]]) 

r.run_visible(visible_data)





hidden_data = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
r.run_hidden(hidden_data)



hidden_data = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
r.run_hidden(hidden_data)














hola=r.daydream(100)
hola.shape







