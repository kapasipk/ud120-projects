#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#########################################################
## First quiz
#########################################################
# t0 = time()
# clf = SVC(kernel='linear')
# clf.fit(features_train, labels_train)
# t1 = time()
# print "Training time =", (t1 - t0), "s"
# predicted_labels = clf.predict(features_test)
# t2 = time()
# print "Testing time =", (t2 - t1), "s"
# score = accuracy_score(predicted_labels, labels_test)
# print "Accuracy =", score
#########################################################
### Output
#########################################################
# Training time = 169.861778021 s
# Testing time = 17.8753809929 s
# Accuracy = 0.984072810011
#########################################################


#########################################################
## Next quiz
#########################################################
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
# t0 = time()
# clf = SVC(kernel='linear')
# clf.fit(features_train, labels_train)
# t1 = time()
# print "Training time =", (t1 - t0), "s"
# predicted_labels = clf.predict(features_test)
# t2 = time()
# print "Testing time =", (t2 - t1), "s"
# score = accuracy_score(predicted_labels, labels_test)
# print "Accuracy =", score
#########################################################
### Output
#########################################################
# Training time = 0.100518226624 s
# Testing time = 1.05672693253 s
# Accuracy = 0.884527872582
#########################################################


#########################################################
## Next quiz
#########################################################
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
# t0 = time()
# clf = SVC(kernel='rbf')
# clf.fit(features_train, labels_train)
# t1 = time()
# print "Training time =", (t1 - t0), "s"
# predicted_labels = clf.predict(features_test)
# t2 = time()
# print "Testing time =", (t2 - t1), "s"
# score = accuracy_score(predicted_labels, labels_test)
# print "Accuracy =", score
#########################################################
### Output
#########################################################
# Training time = 0.106473922729 s
# Testing time = 1.18873906136 s
# Accuracy = 0.616040955631
#########################################################


#########################################################
## Next quiz
#########################################################
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
# c_values = [10.0, 100.0, 1000.0, 10000.0]
# for c_value in c_values:
#   print "\nTraining with C =", c_value
#   t0 = time()
#   clf = SVC(kernel='rbf', C=c_value)
#   clf.fit(features_train, labels_train)
#   t1 = time()
#   print "Training time =", (t1 - t0), "s"
#   predicted_labels = clf.predict(features_test)
#   t2 = time()
#   print "Testing time =", (t2 - t1), "s"
#   score = accuracy_score(predicted_labels, labels_test)
#   print "Accuracy =", score
#########################################################
### Output
#########################################################
# Training with C = 10.0
# Training time = 0.107966184616 s
# Testing time = 1.14221382141 s
# Accuracy = 0.616040955631
# 
# Training with C = 100.0
# Training time = 0.103870868683 s
# Testing time = 1.1277410984 s
# Accuracy = 0.616040955631
# 
# Training with C = 1000.0
# Training time = 0.103864908218 s
# Testing time = 1.1568710804 s
# Accuracy = 0.821387940842
# 
# Training with C = 10000.0
# Training time = 0.105635881424 s
# Testing time = 0.924825906754 s
# Accuracy = 0.892491467577
#########################################################


#########################################################
### Next quiz
#########################################################
# t0 = time()
# clf = SVC(kernel='rbf', C=10000)
# clf.fit(features_train, labels_train)
# t1 = time()
# print "Training time =", (t1 - t0), "s"
# predicted_labels = clf.predict(features_test)
# t2 = time()
# print "Testing time =", (t2 - t1), "s"
# score = accuracy_score(predicted_labels, labels_test)
# print "Accuracy =", score
#########################################################
### Output
#########################################################
# Training time = 110.32753396 s
# Testing time = 10.983066082 s
# Accuracy = 0.990898748578
#########################################################


#########################################################
### Next quiz
#########################################################
# t0 = time()
# clf = SVC(kernel='rbf', C=10000)
# clf.fit(features_train, labels_train)
# t1 = time()
# print "Training time =", (t1 - t0), "s"
# predicted_labels = clf.predict(features_test)
# t2 = time()
# print "Testing time =", (t2 - t1), "s"
# score = accuracy_score(predicted_labels, labels_test)
# print "Accuracy =", score
# print "Label for 10:", predicted_labels[100]
# print "Label for 26:", predicted_labels[26]
# print "Label for 50:", predicted_labels[50]
#########################################################
### Output
#########################################################
# Training time = 107.910524845 s
# Testing time = 12.5942981243 s
# Accuracy = 0.990898748578
# Label for 10: 0
# Label for 26: 0
# Label for 50: 1
#########################################################


#########################################################
### Next quiz
#########################################################
# clf = SVC(kernel='rbf', C=10000)
# clf.fit(features_train, labels_train)
# predicted_labels = clf.predict(features_test)
# count = 0
# for label in predicted_labels:
#   if label == 1:
#     count += 1
# print count, " emails were predicted to be from Chris"
#########################################################
### Output
#########################################################
# 877  emails were predicted to be from Chris
#########################################################
