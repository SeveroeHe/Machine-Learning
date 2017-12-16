import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
    log_product = 0
    for i in x:
        log_product += i;

    return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
    
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
    x1y1 = 0;
    x1y0 = 0;
    D = np.zeros([2, XTrain.shape[1]]);
    y1 =sum(yTrain);
    y0 = yTrain.shape[0]-y1;
    for i in range(0, D.shape[1]):
        a = XTrain[:,i];
        x1y1 = 0;x1y0 = 0;
        for j in range(0, yTrain.shape[0]):
            if yTrain[j]==1:
                x1y1 += a[j];
            else:
                x1y0 += a[j];
        D[0,i] = (x1y0+beta_1-1)/(beta_0+beta_1-2+y0);
        D[1,i] = (x1y1+beta_1-1)/(beta_0+beta_1-2+y1);
        
	## Outputs ##
	# D - (2 by V) numpy ndarray

    return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n
    p = sum(yTrain)/yTrain.shape[0];
	## Outputs ##
	# p - float
    return 1-p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
    yHat = np.zeros(XTest.shape[0]);
    y0 = np.zeros(XTest.shape[0]);
    y1 = np.zeros(XTest.shape[0]);
    for i in range(0, XTest.shape[0]):   #   0~m case
        E = D.copy();
        for j in range(0, XTest.shape[1]):    #   0~v case
            if XTest[i, j]==0:
                E[0, j] = 1 - D[0,j];
                E[1, j] = 1 - D[1,j];
                
        y0[i] = logProd(np.log(E[0]))+np.log(p);
        y1[i] = logProd(np.log(E[1]))+np.log(1-p);
        
    for i in range(0, XTest.shape[0]):
        if y0[i]<y1[i]:
            yHat[i] = 1;
        else:
            yHat[i] = 0;
	## Outputs ##
	# yHat - 1D numpy ndarray of length m
    return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
    
	## Outputs ##
	# error - float
    error = float(sum(yHat!=yTruth))/yTruth.shape[0];
    return error
