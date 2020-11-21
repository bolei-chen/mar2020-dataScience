import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

class Support_Vector_Machine:
    
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, features, labels):
        n_samples, n_features = features.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(features[i], features[j])
                
        # set parameters to solve the quadratic problem.
        P = cvxopt.matrix(np.outer(labels, labels) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(labels, (1, n_samples))
        b = cvxopt.matrix(0.0)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            
        # Solve quadratic problem.
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Deduce the lagrange multiplier alpha.
        alpha = np.ravel(solution['x'])
        
        # Support vectors have non zero lagrange multipliers.
        # Determine the support vectors
        support_vector_alpha = alpha > 1e-5
        ind = np.arange(len(alpha))[support_vector_alpha]
        self.alpha = alpha[support_vector_alpha]
        self.support_vector_features = features[support_vector_alpha]
        self.support_vector_labels = labels[support_vector_alpha]
        print(len(self.alpha), "support vectors out of", n_samples, "points.")
        
        # Deduce the b value.
        # b = y - X * w.
        self.b = 0
        for n in range(len(self.alpha)):
            # plus y
            self.b += self.support_vector_labels[n]
            # - X * w
            self.b -= np.sum(self.alpha * self.support_vector_labels * K[ind[n], support_vector_alpha])
        # calculate mean
        self.b /= len(self.alpha)
        
        # Deduce w if possible.
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.alpha)):
                # w = sigma(alphai * yi * xi).
                self.w += self.alpha[n] * self.support_vector_labels[n] * self.support_vector_features[n]
        else:
            self.w = None

    def predict(self, features):
        if self.w is not None:
            sign = np.sign(np.dot(features, self.w) + self.b)
            if sign == 1:
                return "spam"
            else:
                return "not_spam"
        else:
            wx = 0
            for i in range(len(features)):
                for alpha, sv_y, sv_x in zip(self.alpha, self.support_vector_labels, self.support_vector_features):
                    # s = w * x
                    # w = sigma(alphai * yi * xi).
                    wx += alpha * sv_y * np.sum(self.kernel(features[i], sv_x))
                wx /= len(features)
                y_predict = wx + self.b
                # class = y_predict + b.
                print(y_predict + self.b, self.b)
                sign = np.sign(y_predict)
                if sign == 1:
                    return "spam"
                else:
                    return "not_spam"



svm = Support_Vector_Machine(kernel=polynomial_kernel)
featrues = np.array([[0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1]])
labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
svm.fit(featrues, labels)
print(svm.predict([1, 0, 1, 0, 1, 0, 1]))