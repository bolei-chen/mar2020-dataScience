import numpy as np
import re
import os

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
            ignore = ["a", "the", "is", "we", "he", "she"]  
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

# create dictionary
for i in range(0, len(email_list)):
    vocab = tokenize(email_list[i])
    vocab_dict += vocab
print("Word List for Document \n{0} \n".format(vocab));
print(np.shape(vocab_dict))

# create feature set
for email in email_list:
    feature_set.append(generateBOW(email, vocab_dict))
print(np.shape(feature_set))

# create label set
label_set = []
value = 1.0
for i in range(0, 2):
    for j in range(0, 8):
        label_set.append(float(value))
    value -= 2
print(np.shape(label_set))




