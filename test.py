__author__ = 'mg'

import numpy as np
import my_nn as my
X,Y = my.load_mat_data('ex3data1.mat',10)
nn = my.multilayer_nn([400,25,10])
X, Y, X_cv, Y_cv, X_test, Y_test = my.split_data(X, Y, 0.8, 0, 0.2)

nn.train(X,Y)
