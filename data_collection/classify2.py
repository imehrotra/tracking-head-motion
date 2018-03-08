import pandas as pd
import numpy as np
import sys
import os
from os import walk, path, makedirs
import ast
import argparse
import math
import re
import pickle
import Metrics as met
from sklearn import neighbors, datasets, preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier

epsilon = 0.000001

#global variables for storing training
Y = []
Z = []

from itertools import izip_longest

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

n = 40
def makeWindow(filename):
    with open(filename, "r") as f:
        if not ".txt" in filename:
            print(filename + " not a txt file")
        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            with open(filename+'_{0}'.format(i * n), 'w+') as fout:
                fout.writelines(g)


def toSingle(array,i):
        #takes 2D array and selects for one column i
        dicts = []
        for each in array:
                dicts.append(each[i])

        #print dicts
        return dicts
        
def rot_txt(filename):
  '''
  Input: A path to a accl and rotation textfile
  Return: a Metric 
  '''
  with open(filename, "r") as f:
    if not ".txt" in filename:
      print(filename + " not a txt file")
      return None
    dicts = []
    for line in f:
        structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
        list1 = []
        x = float(structure[2])
        y = float(structure[7])
        z = float(structure[12])
        #if (abs(0-x) < epsilon) and (abs(0-y) < epsilon) and (abs(0-z) < epsilon):
        if 0:
                continue
        else:
                list1.append(float(structure[2]))
                list1.append(float(structure[7]))
                list1.append(float(structure[12]))
                dicts.append(list1)
    dicts = np.array([np.array(xi) for xi in dicts])
    dicts_x = toSingle(dicts,0)
    dicts_y = toSingle(dicts,1)
    dicts_z = toSingle(dicts,2)
    dataX = met.Metrics()
    setattr(dataX,'mean',np.mean(dicts_x))
    dataX.max = np.max(dicts_x)
    dataX.min = np.min(dicts_x)
    dataX.dev = np.std(dicts_x)
    dataX.med = np.median(dicts_x)
    #print dataX#        dataY = met.Metrics()
    dataY = met.Metrics()
    dataY.max = np.max(dicts_y)
    dataY.min = np.min(dicts_y)
    dataY.dev = np.std(dicts_y)
    dataY.med = np.median(dicts_y)
    #print dataY
    dataZ = met.Metrics()
    setattr(dataZ,'mean',np.mean(dicts_z))
    dataZ.max = np.max(dicts_z)
    dataZ.min = np.min(dicts_z)
    dataZ.dev = np.std(dicts_z)
    dataZ.med = np.median(dicts_z)
        #print dataZ
    #return (dataX,dataZ)
    return (dataX,dataY,dataZ)
def xyz_accl(filename):
    '''
    Input: A path to x, y, or z accl, which only has one value
    Return: a metric
    '''
    with open(filename, "r") as f:
        if not ".txt" in filename:
            print(filename + " not a txt file")
            return None
        list_data = []
        for line in f:
            list_data.append(line)
    data = met.Metrics()
    setattr(data,'mean',np.mean(list_data))
    data.max = np.max(list_data)
    data.min = np.min(list_data)
    data.dev = np.std(list_data)
    data.med = np.median(list_data)
    #print dataW

    return data
def attitude_txt(filename):
    '''
    Input: A path to attitude
    Return: a metric
    '''
    with open(filename, "r") as f:
        if not ".txt" in filename:
            print(filename + " not a txt file")
            return None
        dicts = []
        for line in f:
            structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
            list1 = []
            x = float(structure[2])
            y = float(structure[7])
            z = float(structure[12])
            #if (abs(0-x) < epsilon) and (abs(0-y) < epsilon) and (abs(0-z) < epsilon):
        #                               continue
        #                       else:
            list1.append(float(structure[2]))
            list1.append(float(structure[7]))
            list1.append(float(structure[12]))
            list1.append(float(structure[17]))
            dicts.append(list1)
    dicts = np.array([np.array(xi) for xi in dicts])
    dicts_x = toSingle(dicts,0)
    dicts_y = toSingle(dicts,1)
    dicts_z = toSingle(dicts,2)
    dicts_w = toSingle(dicts,3)
    dataW = met.Metrics()
    setattr(dataW,'mean',np.mean(dicts_w))
    dataW.max = np.max(dicts_w)
    dataW.min = np.min(dicts_w)
    dataW.dev = np.std(dicts_w)
    dataW.med = np.median(dicts_w)
    #print dataW
    dataX = met.Metrics()
    setattr(dataX,'mean',np.mean(dicts_x))
    dataX.max = np.max(dicts_x)
    dataX.min = np.min(dicts_x)
    dataX.dev = np.std(dicts_x)
    dataX.med = np.median(dicts_x)
    #print dataX
    dataY = met.Metrics()
    setattr(dataY,'mean',np.mean(dicts_y))
    dataY.max = np.max(dicts_y)
    dataY.min = np.min(dicts_y)
    dataY.dev = np.std(dicts_y)
    dataY.med = np.median(dicts_y)
    #print dataY
    dataZ = met.Metrics()
    setattr(dataZ,'mean',np.mean(dicts_z))
    dataZ.max = np.max(dicts_z)
    dataZ.min = np.min(dicts_z)
    dataZ.dev = np.std(dicts_z)
    dataZ.med = np.median(dicts_z)
    #print dataZ
    return (dataX,dataY,dataZ,dataW)
def get_all_types(path):
    '''
    Given folder, will extract arrays for each type of data.
    bd = [[zAccl][yAccl][xAccl][][]p[]]
    '''
    data_dict = create_dictionary(path)
    action_type = ["bd", "bu", "ns", "fd", "fu", "ld", "lu", "rd", "ru"]

    # Right now, I'm classifying the nods as noisy data. If we want to recognize it, 
    # we can easily change the label
    dict_labels = {"bd": "noisy", "bu": "noisy", "ns": "noisy", "fd":"noisy", "fu":"noisy", "ld":"left down", "lu":"left up", "rd":"right down", "ru":"right up"}
    label = ""
    for date_key, d in data_dict.items():
        tmp = []
        for data_type, metric_list in d.items():
            if data_type == "label":
                label = metric_list #this one isn't actually a metric list 
                continue

            if not metric_list:
                print "no Metrics in data_type "+ data_type +" in date "+ date_key
                continue

            for metric in metric_list:
                tmp.append(metric.getMax())
                tmp.append(metric.getMean())
                tmp.append(metric.getMed())
                tmp.append(metric.getMin())
                tmp.append(metric.getDev())

        Y.append(tmp)
        Z.append(dict_labels[label])



def extract_data(d, filename, path):
    '''
    Given filename, extract that particular type of data by assigning a Metric
    to a dictionary.... at the end we'll have a dictionary of metrics. Each dict value
    is a tuple, with at least one Metric
    '''
    fullname = os.path.join(path,filename)
    data_type = filename[3]
    if (data_type == "x"):
        d["xaccl"] = xyz_accl(fullname)

    elif (data_type == "y"):
        d["yaccl"] = xyz_accl(fullname)

    elif (data_type == "z"):
        d["zaccl"] = xyz_accl(fullname) 

    elif (data_type == "r"):
        d["rot"] = rot_text(fullname)

    elif (data_type == "a"):            
        d["att"] = attitude_text(fullname)

    elif (data_type == "u"):
        d["uaccl"] = rot_text(fullname) #like rot, user Accl has 3 places...

def create_dictionary(path):
    '''
    {"DATE": {"label": ACTION_LABEL, "rot": [], "xaccl": [], "yaccl": [], "zAccl": [], "attitude": [], "uAccl: []"}}
    '''
    data_type = ["label", "rot", "xaccl", "yaccl", "zaccl", "att", "uaccl"]
    data_dict = {}
    for filename in os.listdir(path):
        date = filename[-14: -4]
        d = data_dict.get(date)
        if d is None:
            d = dict.fromkeys(data_type)
            d["label"] = filename[:2]
            data_dict[date] = d
        extract_data(d, filename, path) #Have to implement this
    return data_dict
#dataframe is a list of Metrics... 


 
def train():
    '''
    trains the data and prints the accuracy score
'''
    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    scaler =  preprocessing.StandardScaler().fit(Y_train)
    Y_test = scaler.transform(Y_test)
    for n in range(1,20):
        knn = neighbors.KNeighborsClassifier(n_neighbors=n)
        knn.fit(Y_train, Z_train)
        z_pred = knn.predict(Y_test)
        misclassified = Y_test[Z_test != z_pred]
        print "misclassified ones,", misclassified
        if accuracy_score(Z_test, z_pred) > maxAccurate:
            maxAccurate = accuracy_score(Z_test, z_pred)
            index = n
        print "accuracy score, ", accuracy_score(Z_test, z_pred)
        print "confusion_matrix, ", confusion_matrix(Z_test, z_pred)
        #print classification_report(Z_test, z_pred)
    print "best accuracy: ",maxAccurate," nearest neighbor:", n

def train2():
    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    scaler =  preprocessing.StandardScaler().fit(Y_train)
    Y_test = scaler.transform(Y_test)
    knn = tree.DecisionTreeClassifier()
    knn = knn.fit(Y_train, Z_train)
    z_pred = knn.predict(Y_test)
    misclassified = Y_test[Z_test != z_pred]
    print "misclassified ones,", misclassified
    if accuracy_score(Z_test, z_pred) > maxAccurate:
        maxAccurate = accuracy_score(Z_test, z_pred)
            
    print "accuracy score, ", accuracy_score(Z_test, z_pred)
    print "confusion_matrix, ", confusion_matrix(Z_test, z_pred)

    model = ExtraTreesClassifier()
    model = model.fit(Y_train, Z_train)
 #   print(Y_train)
    print "feature importances"
    print(model.feature_importances_)

    

    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',help = 'Specifies a particular input file to test')
    parser.add_argument('-a','--filea',help = 'Specifies a particular attitude file to test')
    parser.add_argument('-fd','--folder',help = 'Specifies a particular folder to test')
    parser.add_argument('-fa','--folderA',help = 'Specifies a particular attitude folder to test')
    parser.add_argument('-t', '--test', help = 'Trains folders')
    args = parser.parse_args()
    if args.file:
            txt_path = args.file
            dataFrame = rot_txt(txt_path)
    if args.filea:
            txt_path = args.filea
            dataFrame = attitude_txt(txt_path)
    if args.folder:
            txt_path = args.folder
            dataFrame = folder_accel(txt_path)
            capture = overall_mean(dataFrame,1)
            overall_dev(dataFrame,1)
            overall_max(dataFrame,1)
            overall_min(dataFrame,1)
    if args.folderA:
            txt_path = args.folderA
            dataFrame = folder_att(txt_path)
            overall_mean(dataFrame,0)
            overall_dev(dataFrame,0)
            overall_max(dataFrame,0)
            capture = overall_min(dataFrame,0)
    if args.test:
            get_all_types("all_data")

           # train()
            train2()

if __name__ == '__main__':
    main()