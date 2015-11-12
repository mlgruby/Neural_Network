# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib.pyplot import plot
from sklearn.linear_model import LinearRegression as lm

# class for neural network
class Neural_Network:
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize):
        # init function for initializing hyperparameter
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        
        # initializing weight parameter (alpha and beta) uniformly between [-0.7, 0.7] (Weights for bias included)
        self.alpha = np.random.uniform(-0.7, 0.7, size=(self.inputLayerSize, self.hiddenLayerSize))
        self.beta = np.random.uniform(-0.7, 0.7, size=(self.hiddenLayerSize, self.outputLayerSize))
        self.alpha0 = np.random.uniform(-0.7, 0.7, size=(1, self.hiddenLayerSize)) # Bias weight for Hidden Layer
        self.beta0 = np.random.uniform(-0.7, 0.7, size=(1, outputLayerSize)) # Bias weight for output layer
    def forwardPass(self, X):
        # input forwading through network
        # returns yHat in one go
        self.z2 = np.dot(X, self.alpha) + self.alpha0 # bias term addition at hidden layer
        self.a2 = self.activation(self.z2)
        yHat = np.dot(self.a2, self.beta) + self.beta0 # bias term addidtion at output layer
        return yHat
        
    def activation(self, z):
        # sigmoid as a activation function
        return 1/(1+np.exp(-z))
    
    def activationPrime(self, z):
        # differentiation of sigmoid activation function
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def mseCostFunction(self, X, Y, c):
        # MSE claculation
        self.yHat = self.forwardPass(X)
        R = (1.0/X.shape[0])*sum((Y-self.yHat)**2)+c*(sum(np.dot(self.alpha, self.alpha.T)) + sum(np.dot(self.beta, self.beta.T)))# Regularization term
        return R
        
    def mseCostFunctionPrime(self, X, Y):
        # computing derivate of cost function w.r.t alpha and beta
        self.yHat = self.forwardPass(X)
        # for beta
        delta = -2*(Y-self.yHat)
        dJdbeta = (1.0/X.shape[0])*np.dot(self.a2.T, delta)
        # for alpha
        s = np.dot(delta, self.beta.T)*self.activationPrime(self.z2)
        dJdalpha = (1.0/X.shape[0])*np.dot(X.T, s)  
        
        return dJdalpha, dJdbeta
        
    def train(self, X, Y, eta, epoch,c ):
        # updates weights with given eta and epoch
        self.mses = np.zeros([epoch])
#        self.mses = 0
        for i in np.arange(epoch):
            self.mses[i] = self.mseCostFunction(X, Y, c)
#            self.mses = self.mseCostFunction(X, Y, c)
            print 'For epoch %5d, trainig MSE is %3.6f' %(i+1, self.mses[i])
            dalpha, dbeta = self.mseCostFunctionPrime(X, Y) # back-propagation
            self.alpha = self.alpha - eta * dalpha # updates alpha
            self.beta = self.beta - eta * dbeta # updates beta
#        return self.mese[i]
        
    def trainStop(self, X, Y, eta, c):
        # stops updating weights if last 10 MSE are not changing by 1%
        counter, i, mses = 0, 0, 0
        while counter < 10:
            diff = (mses - self.mseCostFunction(X, Y, c))/mses
            mses = self.mseCostFunction(X, Y, c)
            print 'For epoch %5d, trainig MSE is %3.6f' %(i+1, mses)
            dalpha, dbeta = self.mseCostFunctionPrime(X, Y) # back-propagation
            self.alpha = self.alpha - eta * dalpha # updates alpha
            self.beta = self.beta - eta * dbeta # updates beta
            i = i + 1 
            if diff < 0.01:
                counter = counter + 1
            else:
                counter = 0
        print 'stopped at epoch %5d, with training MSE %3.6f' %(i, mses)
#        return mses
            
    def predict(self, X):
        # predicts for test data
        return self.forwardPass(X)
        
    def testMSE(self, X_test, Y_test):
        # calculates MSE for test data
        yHat = self.forwardPass(X_test)
        tMSE = (1.0/X_test.shape[0])*sum((Y_test-yHat)**2)
        return tMSE
        
########################### Class Implementation ends here ####################################

# General Functions
def normalize(col):
    # return normalized column
    return ((col - np.mean(col))/np.std(col))
    
def testTrainSplit(DF, label, train_size):
    np.random.seed(0331)
    trainNumber = DF.shape[0]*train_size
    DF.apply(np.random.shuffle, axis=0)
    Y = (DF[label]).reshape(DF.shape[0], 1)
    X = (DF.drop(label,1)).values
    train_X = X[:trainNumber,:]
    test_X = X[trainNumber:,:]
    train_Y = Y[:trainNumber,:]
    test_Y = Y[trainNumber:,:]
    
    return train_X, train_Y, test_X, test_Y
    
# reading Auto data
auto = pd.read_csv('Auto.csv', usecols=(0,3,4,6,7))
auto.head()
# Linear Regrssion
train_X, train_Y, test_X, test_Y = testTrainSplit(auto, 'mpg', 0.5)
lmModel = lm(normalize=False)
lmModel.fit(train_X, train_Y)
yHat = lmModel.predict(test_X)
testMse = (1.0/test_X.shape[0])*np.sum(test_Y - yHat)
trainMse = (1.0/train_X.shape[0])*np.sum(train_Y - lmModel.predict(train_X))

# creating dummy variable for origin ---> origin1, origin2, origin3
for element in auto['origin'].unique():
    auto['origin'+str(element)] = (auto['origin'] == element).astype(int)

auto = auto.drop('origin', 1) # removing main origin column
auto.head()

#auto = auto.apply(normalize, 1)
#test and train split
train_X, train_Y, test_X, test_Y = testTrainSplit(auto, 'mpg', 0.5)

# normalizing auto train_X and test_X
train_X = ((pd.DataFrame(train_X)).apply(normalize, 1)).values
test_X = ((pd.DataFrame(test_X)).apply(normalize, 1)).values

NN = Neural_Network(6,3,1) # Neural Network object - Initializing weights
NN.train(train_X, train_Y, 0.1, 100, 0.03) # Training Neural Network
NN.trainStop(train_X, train_Y, 0.7, 0.03) # Automatically stops if last 10 MSE dosen't changes by 1%
NN.testMSE(test_X, test_Y) # test MSE
NN.predict(test_X) # Test Prediction

plot(NN.mses)

#############################################################################
# Below code is breaking somewhere ------> Need to check
# Cross validating function
#def cv(X, Y, folds, model_object):
#    subset_size = len(X)/folds
#    mses = np.zeros([folds])
#    for i in np.arange(folds):
#        testing_round_X = X[i*subset_size:][:subset_size]
#        training_round_X = X[:i*subset_size] + X[(i+1)*subset_size:]
#        testing_round_Y = Y[i*subset_size:][:subset_size]
#        training_round_Y = Y[:i*subset_size] + Y[(i+1)*subset_size:]
#        model_object.train(training_round_X, training_round_Y, 0.1, 100, 0.3)
#        mses[i] = model_object.testMSE(testing_round_X, testing_round_Y)
#    return np.mean(mses)
#    
#cvmse = np.zeros([10])
#for m in np.arange(10)+1:
#    NN = Neural_Network(6,m,1)
#    cvmse[m] = cv(train_X, train_Y, 4, NN)
#    
###########################################################################   

# 100 times model
mseTest = np.zeros([100])
epoch = 30
eta = 0.1
c = 0.3
NN = Neural_Network(6,3,1)
for i in range(100):
    NN.train(train_X, train_Y, eta, epoch, c)
    mseTest[i] = NN.testMSE(test_X, test_Y)

fig = plot.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.boxplot(mseTest)