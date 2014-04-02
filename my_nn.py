__author__ = 'mg'

import numpy as np

class multilayer_nn(object):

    def __init__(self, layer_sizes):

        # initialize layer_sizes, including input and output layers:
        self.layers = np.array(layer_sizes) # [400, 25, 10]

        # set depth of the net, including input and output layer
        self.depth = len(layer_sizes) # 3

        # indexes for splitting Theta
        self.Theta_ind = np.zeros(self.depth).astype(np.int) # [0, 10000, 10250]
        # indexes for splitting layers activations (no incoming Xdata)
        self.acts_ind = np.zeros(self.depth).astype(np.int) # [0, 25, 35]
        # filling indexes arrays
        if self.depth > 2:
            for i in xrange(1,self.depth):
                self.Theta_ind[i] = layer_sizes[i - 1]*layer_sizes[i] + self.Theta_ind[i - 1]
                self.acts_ind[i] = self.acts_ind[i - 1] + layer_sizes[i]
        else:
            return -1 # will create irrelevant error (because __init__ method cannot return),
            # should learn a better way to output errors

        # initialize Theta
        self.Theta = (1. - 2. * np.random.rand(self.Theta_ind[self.depth - 1]))
        # initiate derivatives array
        self.D = np.zeros_like(self.Theta)

        # set learning rate
        self.learning_rate = 2.
        self.min_iter = 20
        self.threshold = 0.0001

        # set regularization term
        self.Lambda = 1.

        # will initialize these in "train" method
        self.activations = None
        self.deltas = None

    def sigmoid(self,x):
        ''' Computes sigmoid function
        '''
        return 1.0 / (1.0 + np.exp(-x))

    def one_layer_forward(self, theta, acts):
        '''Compute activations in the next layer using current activations and 2D weights array.
        '''

        # sum up activations times the weights
        z = np.dot(theta,acts.T).T + 1.

        # return the activation function
        return self.sigmoid(z)

    def Forward(self, X):

        # forward propagation from input to the first hidden, or to the output, layer
        # get weights 2D array for the input layer
        theta = self.Theta[ 0:self.Theta_ind[1] ].reshape(self.layers[1], -1)
        # compute activations in the first hidden, or output, layer
        self.activations[ :, 0:self.acts_ind[1] ] = self.one_layer_forward(theta, X)

        # next layers
        # 0 hidden layer: range(2,2) = [] - no loop
        # 1 hidden: range(2,3) = [2]
        # ...
        for l in xrange(2, self.depth):

            # get 2D weights array for the current layer layer
            theta = self.Theta[ self.Theta_ind[l - 1]:self.Theta_ind[l] ].reshape(self.layers[l], -1)

            # starting and ending indexes of the activation compound array for the (l-1)-th hidden or output layer
            start = self.acts_ind[l - 1]
            stop = self.acts_ind[l]

            # starting and ending indexes of the activation compound array for the (l-2)-th hidden or input layer
            start_prev = self.acts_ind[l - 2]
            stop_prev = self.acts_ind[l - 1]

            # compute activations in the (l-1)-th layer
            self.activations[:,start:stop] = self.one_layer_forward(theta, self.activations[:,start_prev:stop_prev])

    def one_layer_back(self, delta_current, theta_current, acts_prev):
        '''Back-propagates one layer.
        Computes errors in the previous layer based on the errors of the current layer,
        activations of the previous layers, and the weights array that connect these two layers
        '''
        return np.dot(delta_current, theta_current) * (acts_prev - acts_prev**2)

    def Backward(self, Y):
        """Computes errors in output and hidden layers
        """

        # output layer:
        # starting and ending indexes in the activations compound array for the output layer
        start = self.acts_ind[self.depth - 2]
        stop = self.acts_ind[self.depth - 1]
        # compute the errors in the output layer
        self.deltas[:,start:stop] = self.activations[:,start:stop] - Y

        # hidden layers
        # no hidden layers, depth = 2: range(0,0,-1) = [] - no loop
        # 1 hidden layer, depth = 3: range(1,0,-1) = [1]
        # 2 hidden layers, depth = 4: range(2,0,-1) = [2,1]
        # ...
        for l in range(self.depth - 2,0,-1):

            # starting and ending indexes in the activations
            # compound array for the (l+1)-th hidden layer or output layer
            start = self.acts_ind[l]
            stop = self.acts_ind[l + 1]

            # starting and ending indexes in the activations
            # compound array for the l-th hidden layer
            start_prev = self.acts_ind[l - 1]
            stop_prev = self.acts_ind[l]

            # get 2D weights array for l-th layer
            theta = self.Theta[ self.Theta_ind[l]:self.Theta_ind[l + 1] ].reshape(self.layers[l + 1], -1)

            # calculate errors for the l-th layer
            self.deltas[:,start_prev:stop_prev] = self.one_layer_back(
                self.deltas[:,start:stop], theta, self.activations[:,start_prev:stop_prev])

    def derivatives(self, Xdata):
        """Computes partial derivatives of the cost function with respect to weights.
        Based on the basic backpropagation algorithm.
        """

        # number of examples
        m = Xdata.shape[0]

        # first hidden layer derivatives
        # D is local variable 2-dimensional array
        D = (1. / m) * np.dot(self.deltas[:,0:self.acts_ind[1]].T, Xdata)
        # write into global variable D, 1D array
        self.D[0:self.Theta_ind[1]] = D.ravel()

        # derivatives of other hidden layers
        # there will be no loop if only one hidden layer: range(1,1) = []
        for l in range(1,self.depth - 1):

            # reassign local D variable, 2-dimensional array
            D = (1. / m) * np.dot(self.deltas[ :,self.acts_ind[l]:self.acts_ind[l + 1] ].T,
                                  self.activations[ :,self.acts_ind[l - 1]:self.acts_ind[l] ] )

            # convert to 1D and copy results into the global D
            self.D[self.Theta_ind[l]:self.Theta_ind[l + 1]] = D.ravel()

        # add regularization term derivatives
        self.D = self.D + (self.Lambda / m) * self.Theta

    def cost(self, Y):
        ''' Cost function for a training labels set y and its predictions'''

        # (-1/m)
        factor = (-1. / Y.shape[0])

        # output activations
        output = self.activations[ :,self.acts_ind[-2]:self.acts_ind[-1] ]

        # regularization, lambda term
        lambda_term = (0.5 * self.Lambda / Y.shape[0]) * np.sum( self.Theta**2 )

        # compute and return the logistic cost function with regularization
        return ( factor * np.sum(Y * np.log(output) + (1. - Y) * np.log(1. - output)) ) + lambda_term

    def train(self, Xdata, Ydata):

        # initialize Theta
        self.Theta = (1. - 2. * np.random.rand(self.Theta_ind[self.depth - 1]))

        # number of examples
        m = Xdata.shape[0]

        # initialize compound array with activations of hidden and output layers
        self.activations = np.zeros((m, np.sum(self.acts_ind[self.depth - 1])))

        # initialize compound array with errors
        self.deltas = np.zeros_like(self.activations)

        iterate = True
        J = []
        count = 0
        while iterate:
            count+=1

            # compute self.activations
            self.Forward(Xdata)

            # add current cost value to the list
            J.append(self.cost(Ydata))

            # compute errors: self.deltas
            self.Backward(Ydata)

            # compute derivatives: self.D
            self.derivatives(Xdata)

            # update weights
            self.Theta -= self.learning_rate * self.D

            # after self.min_iter number of iterations
            # continue until J varies less than self.threshold
            if ((len(J) > self.min_iter) and (abs((J[-1] - J[-2])) < self.threshold)):
                print (J[-1] - J[-2])
                iterate = False

        #print J
        print count
        import matplotlib.pyplot as plt
        plt.plot(range(len(J)),J)
        plt.show()

    def test(self, Xtest, Ytest):
        # number of examples
        m = Xtest.shape[0]
        # reset compound array with activations of hidden and output layers
        self.activations = np.zeros((m,np.sum(self.acts_ind[self.depth - 1])))
        # compute self.activations
        self.Forward(Xtest)
        # compute cost
        J = self.cost(Ytest)
        # compute percent of correct answers
        predicted_ind = self.activations[ :, self.acts_ind[-2]:self.acts_ind[-1] ].argmax(axis=1)
        correct = 100. * np.sum(Ytest[range(m),predicted_ind]) / m
        print correct,"%"



def load_mat_data(mat_file, output_size):
    ''' load data from mat file
    returns X and Y - test examples and converted labels
    so that label 7 corresponds to [0,0,0,0,0,0,0,1,0,0]
    '''
    import scipy.io

    # get all training data
    mat = scipy.io.loadmat(mat_file)
    X = np.array(mat['X'])
    ydata = np.array(mat['y']) % 10 # convert 10s to 0s

    # total number of tests
    m = X.shape[0]

    # convert labels to output vectors
    Y = np.zeros((m, output_size))
    Y[np.arange(m), ydata.ravel()] = 1.

    return X, Y

def split_data(Xin, Yin, r_train, r_cv, r_test):
    if (r_train + r_cv + r_test > 1):
        print "total more than one"
    else:
        m = Xin.shape[0]
        m_train = int(r_train * m)
        m_cv = int(r_cv * m)
        m_test = m - m_train - m_cv

        ind = np.random.permutation(m)

        # train data:
        X = Xin[ ind[:m_train] ]
        Y = Yin[ ind[:m_train] ]
        # cross-validation data
        X_cv = Xin[ ind[ m_train:-m_test ] ]
        Y_cv = Yin[ ind[ m_train:-m_test ] ]
        # test data
        X_test = Xin[ ind[ -m_test: ] ]
        Y_test = Yin[ ind[ -m_test: ] ]

        return X, Y, X_cv, Y_cv, X_test, Y_test


"""
import numpy as np
import my_nn as my
X,Y = my.load_mat_data('ex3data1.mat',10)
nn = my.multilayer_nn([400,25,10])
X, Y, X_cv, Y_cv, X_test, Y_test = my.split_data(X, Y, 0.8, 0, 0.2)

nn.train(X,Y)
"""