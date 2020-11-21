from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import re
import os

# Bag Of Words
def convertTxtToList(path, text):  
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                lineList = [line.rstrip('\n') for line in open(os.path.join(root, file), 'r', encoding="gb18030", errors="ignore")]
                i = 0
                while i < len(lineList):
                    if lineList[i] == "":
                        lineList.pop(i)
                    else:
                        i += 1
                email = []
                email += lineList
                text.append(email)
    return text

def wordExtraction(text):    
            ignore = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"] 
            words = re.sub("[^\w]", " ", text).split()    
            cleaned_text = [w.lower() for w in words if w not in ignore]    
            return cleaned_text

def tokenize(text):
    words = []
    for i in range(0, len(text)):
        w = wordExtraction(text[i])
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def generateBOW(text, vocab):
    words = []
    for sentence in text:
        words.append(wordExtraction(sentence))
        bag_vector = np.zeros(len(vocab))
    for w in words:
        for i, word in enumerate(vocab):
            if word == w: 
                bag_vector[i] += 1
    return bag_vector

# set up basic variables
spam_path = "lec2\code\email samples\spam"
nonspam_path = "lec2\code\email samples\\not spam"
spam_email = []
nonspam_email = []

spam_email_list = convertTxtToList(path=spam_path, text=spam_email)
nonspam_email_list = convertTxtToList(path=nonspam_path, text=nonspam_email)
email_list = spam_email_list + nonspam_email_list
feature_set = []
vocab_dict = []
print("shape for the email list is", np.shape(email_list))

# create dictionary
for i in range(0, len(email_list)):
    vocab = tokenize(email_list[i])
    vocab_dict += vocab
#print("Word List for Document \n{0} \n".format(vocab));
print("shape for feature set is", np.shape(vocab_dict))

# create feature set
for email in email_list:
    feature_set.append(generateBOW(email, vocab_dict))
print("shape for the dictionary is", np.shape(feature_set))

# create label set
label_set = []
value = 1.0
for i in range(0, 2):
    for j in range(0, 8):
        label_set.append(float(value))
    value -= 2
print("shape for the label set is", np.shape(label_set))

# Support Vector Machine

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
            return sign
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
                print(y_predict, self.b)
                sign = np.sign(y_predict)
                return sign

# split data
X_train, X_test, y_train, y_test = train_test_split(feature_set, label_set, test_size=0.3, random_state=4)
svm = Support_Vector_Machine(kernel=linear_kernel, C=1)
featrues = np.array(X_train)
labels = np.array(y_train)
# train the model
svm.fit(featrues, labels)
# check accuracy
y_predict = svm.predict(X_test)
print(classification_report(y_true=y_test, y_pred=y_predict))
print(accuracy_score(y_pred=y_predict, y_true=y_test))