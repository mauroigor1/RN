# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:47:18 2019

@author: Bruko
"""

from rbm import *


r=RBM(num_visible=6,num_hidden=2)


training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]) # A 6x6 matrix where each row is a training example and each column is a visible unit.
r.train(training_data, max_epochs = 5000) # Don't run the training for more than 5000 epochs.

visible_data = np.array([[0,0,0,1,1,0]]) 


r.run_visible(visible_data)

hidden_data = np.array([[1,0],[0,1]])
r.run_hidden(hidden_data)


hola=r.daydream(100)
hola.shape
