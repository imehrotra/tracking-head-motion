import numpy as np
import sys
from os import walk, path, makedirs
import ast
import argparse
import math
import re
import pickle
import window
import Metrics as met
from sklearn import neighbors, datasets, preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from itertools import izip_longest
from classify2 import create_dictionary


#reconfigure as necessary
dump_path = ""

def save(data, filename):
    '''
    Saves knn classification object, so it can be loaded
    '''
    if not path.exists(dump_path):
        makedirs(dump_path)
    filename = path.join(dump_path, filename)
    fileObject = open(filename, 'wb')
    pickle.dump(data, fileObject)
    fileObject.close()

def load(filename):
    '''
    load knn classification object
    '''
    filename = path.join(dump_path, filename)
    fileObject = open(filename, 'rb')
    return pickle.load(fileObject)



### Old functions no longer used for initial algorithm and training data (might come in handy) ###

def grouper(n, iterable, fillvalue=None):
    '''
    Collect data into fixed-length chunks or blocks
    '''
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def makeWindow(filename):
    '''
    Used to try splitting data originally
    '''
    with open(filename, "r") as f:
        if not ".txt" in filename:
            print(filename + " not a txt file")
        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            with open(filename+'_{0}'.format(i * n), 'w+') as fout:
                fout.writelines(g)


def toSingle(array,i):
    '''
    takes 2D array and selects for one column i
    '''
    dicts = []
    for each in array:
            dicts.append(each[i])

    #print dicts
    return dicts

def features_get_all_types(Y, Z, path):
    '''
    Given path to folder, will create dictionary and Y, Z arrays with all 65 features 
    '''
    data_dict = create_dictionary(path)
    
    # Classifies the nods as noisy data. If we want to recognize it, 
    # we can easily change the label
    # alphabetically: left down, left up, noisy  right down, right up
    dict_labels = {"bk": "noisy",  "ns": "noisy", "fd":"noisy", "ld":"left down", "lu":"left up", "rd":"right down", "ru":"right up"}
    label = ""
    lists_w_labels = []
    for date_key, d in data_dict.items():
        tmp = []

        for data_type, metric_list in sorted(d.iteritems()):
            if data_type == "label":
                label = metric_list #this one isn't actually a metric list 
                continue

            if not metric_list:
                print "no Metrics in data_type "+ data_type +" in date "+ date_key
                continue

            try:
                tmp.append(metric_list.getMax())
                tmp.append(metric_list.getMean())
                tmp.append(metric_list.getMed())
                tmp.append(metric_list.getMin())
                tmp.append(metric_list.getDev())
            except:
                for metric in metric_list:
                    tmp.append(metric.getMax())
                    tmp.append(metric.getMean())
                    tmp.append(metric.getMed())
                    tmp.append(metric.getMin())
                    tmp.append(metric.getDev())
            Y.append(tmp)
            Z.append(dict_labels[d["label"]])

 
    return Y, Z


def find_features(Y, Z):
    '''
    Input: Y and Z: input and output of classifier
    Finds most important features 
    '''
    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    scaler = preprocessing.StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    model = ExtraTreesClassifier()
    model = model.fit(Y_train, Z_train)

    print "feature importances"
    print(model.feature_importances_)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(Y[0])):
        print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])

    return model