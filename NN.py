# -*- coding: utf-8 -*-
import numpy as np


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
        

