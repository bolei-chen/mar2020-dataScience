## not finished yet
## to be continued....

import numpy as np

######### maybe it will work!!!!! ###########

def forwardPropogation(X, w, b):
    #### w --> 1 * n
    #### X --> n * m
    #### b vertor --> [b, b, b, b.....]
    z = np.dot(w, X) + b
    return z

def backwrdPropogation(X, y, z, n):
    z_deriv = (1 / n) * (z - y)
    w_deriv = np.dot(z_deriv, X.T)
    b_deriv = np.sum(z_deriv)
    return w_deriv, b_deriv

def lossFunction(z, y, n):
    loss = (1 / (2 * n)) * np.sum(np.square(z - y))
    return loss

def gradientDescent(w, b, w_deriv, b_deriv, step_size):
    w -= w_deriv * step_size
    b -= b_deriv * step_size
    return w, b

class MultipleLinearRegression:

    def __init__(self, step_size=0.01, epoch=20000):
        self.step_size = step_size
        self.epoch = epoch
        self.w = None
        self.b = None
        self.loss_history = []
        

    def fit(self, X, y):
        # initializing the parameters
        n_samples, n_features = X.shape
        self.w = np.zeros((1, n_features))
        self.b = 0
        n = n_samples
        
        ########### STOCHASTIC GRADIENT DESCENT ################
        for i in range(self.epoch):
            random_index = np.random.randint(0, n_samples)
            Xi = (X[random_index, :].reshape(1, X.shape[1])).T
            yi = np.array(y[random_index].reshape(1, 1))
            
            ### forward propogation
            zi = forwardPropogation(Xi, self.w, self.b)
            
            ### calculate and store loss
            loss = lossFunction(zi, yi, n)
            if i % 10 == 0:
                self.loss_history.append(loss)
            
            ### backward propogation
            w_deriv, b_deriv = backwrdPropogation(Xi, yi, zi, n)
            
            ### update weights and bais
            self.w, self.b = gradientDescent(self.w, self.b, w_deriv, b_deriv, self.step_size)
            


    def predict(self, X):
        X = X.T
        return np.dot(self.w, X) + self.b
        