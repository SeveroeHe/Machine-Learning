#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from scipy.stats import norm
import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    
    def sigmoid(self, Z):
        sigmoid = 1/(1+np.exp(-1*Z));

        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''
        
        return sigmoid
    
    
    def calculateGradient(self, weight, X, Y, regLambda):
        #= np.zeros((X.shape[1],1));     # d+1 by 1
        #Gradient = np.zeros((X.shape[1],1));
        hx = np.dot(X,weight);
        sigmad = self.sigmoid(hx)-Y;
        Xnew = np.transpose(X);
        Gradient = np.dot(Xnew,sigmad)+regLambda*weight;  
        ##for j in range(1, X.shape[1]):
        ##    Gradient[j,0] = float(np.dot((self.sigmoid(np.dot(X,weight))-Y)[:,0],X[:,j])) + regLambda * weight[j,0];
        #a = np.ones((X.shape[0],1));
        #temp = self.sigmoid(np.dot(X,weight))-Y;
        #Gradient[0,0] = np.dot(temp[:,0], a[:,0]);
        #        Gradient = np.dot(X.transpose(),(self.sigmoid(np.dot(X,weight))-Y))+regLambda*weight;   
        Gradient[0,0] -= regLambda * weight[0,0];    
         
        '''
        for j in range(0, X.shape[1]):
            total = 0;    #ith element in SUM
            for i in range(0, X.shape[0]):    # 0 to n-1
                total += (self.sigmoid(np.dot(weight[:,0], X[i,:]))-Y[i,0])*X[i,j];
            Gradient[j,0] = regLambda * weight[j,0] + total;   
            ##****************************refer to phase1
        '''
        ##****************************phase2
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        return Gradient    



    def update_weight(self,X,Y,weight):
        new_weight = weight - self.alpha * self.calculateGradient(weight,X,Y,self.regLambda);
               #****************************   phase3

        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        return new_weight
    
    def check_conv(self,weight,new_weight,epsilon):
        #****************************phase4
        sub = (new_weight - weight);
        sumsqr = np.dot(sub[:,0],sub[:,0]);
        norm2 = np.sqrt(sumsqr);
        #print("norm2 = " , norm2);
        if norm2 <=epsilon:
            return True;
        else:
            return False;


        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape    # n is the row, d is the column
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1));
        #weight = nweight = (self.weight).copy();

        for i in range(0,self.maxNumIters):
            self.new_weight = self.update_weight(X,Y,self.weight);
            if self.check_conv(self.weight,self.new_weight,self.epsilon)==True:
                self.weight = (self.new_weight).copy();
                break;
            self.weight = (self.new_weight).copy();
            i += 1;

            
        #compute origin weight
        '''
        py1 = float(sum(Y))/Y.shape[0];
        weight[0,0] = np.log((1-py1)/py1);
         #take sum prior and wi to wn
        
        for j in range(1, d+1):
            uj1 = uj0 = 0;
            count0 = 0;
            thetaj = 0;
            for i in range(0, n):
                if Y[i,0] == 0:
                    count0 += 1;
                    uj0 += X[i,j];
                else:
                    uj1 += X[i,j];
            uj0 = float(uj0)/count0;
            uj1 = float(uj1)/(n-count0);
            thetaj = np.var(X[:,j]);       #******************
            weight[0,0] += float(uj1**2-uj0**2)/(2*(thetaj**2));
            weight[j,0] = float(uj0 - uj1)/(thetaj**2);
        '''
               


        #train iteratio
        '''
        looptime = 1;
        new_weight = self.update_weight(X,Y,weight);
        while looptime<=self.maxNumIters and self.check_conv(weight,new_weight,self.epsilon)==False:
            weight = new_weight.copy();
            new_weight = self.update_weight(X,Y,weight);
            looptime += 1;
            #print("looptime = " , looptime);
        '''
        return self.weight

    def predict_label(self, X,weight):


        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]      
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]     # X is n by d+1

        result = np.zeros((n,1));
        for i in range(0, n):
            lgvalue = -1 * (np.dot(weight[:,0],X[i,:]));
            if lgvalue >= 0:
                result[i,0] = 0;
            else:
                result[i,0] = 1;

        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        #for each i (range 0 to n-1), predict and combine a vector
        Accuracy = 100 * (float)(sum(Y_predict == Y_test))/Y_test.shape[0];
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        
        return Accuracy
    
        