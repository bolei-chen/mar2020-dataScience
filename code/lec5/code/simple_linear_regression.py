# video link for SGD: https://www.youtube.com/watch?v=k3AiUhwHQ28
# video link for GD: https://www.youtube.com/watch?v=AeRwohPuUHQ

import numpy as np

class SimpleLinearRegression:
    
    def __init__(self, step_size=0.01, epoch=1000):
        self.step_size = step_size
        self.epoch = epoch
        self.w = None
        self.b = None
        self.epoch = epoch
        
    def fit(self, X, y):
        # setting basic parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        ########### STOCHASTIC GRADIENT DESCENT ################
        
        #### i have not chek it yet
        #### hopefully it will work
        #### anyways i will come back to this later
        
        for i in range(self.epoch):
            random_index = np.random.randint(0, n_samples)
            Xi = X[random_index, :].reshape(1, X.shape[1])
            yi = y[random_index].reshape(1, 1)
            
            # calculate derivatives w.r.t w and b
            y_pred = np.dot(Xi, self.w) + self.b
            w_deriv = (1 / n_samples) * np.dot(Xi.T, (y_pred - yi))
            b_deriv = (1 / n_samples) * np.sum(y_pred - yi)
            
            # update weights and bias
            self.w -= self.step_size * w_deriv[0]
            self.b -= self.step_size * b_deriv
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b

