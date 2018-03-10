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

    dataX = met.Metrics(in_min=np.min(dicts_x), in_max=np.max(dicts_x), in_mean=np.mean(dicts_x), in_dev=np.std(dicts_x), in_med=np.median(dicts_x))
    dataY = met.Metrics(in_min=np.min(dicts_y), in_max=np.max(dicts_y), in_mean=np.mean(dicts_y), in_dev=np.std(dicts_y), in_med=np.median(dicts_y))
    dataZ = met.Metrics(in_min=np.min(dicts_z), in_max=np.max(dicts_z), in_mean=np.mean(dicts_z), in_dev=np.std(dicts_z), in_med=np.median(dicts_z))

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
            list_data.append(float(line))    
    data = met.Metrics(in_min=np.min(list_data), in_max=np.max(list_data), in_mean=np.mean(list_data), in_dev=np.std(list_data), in_med=np.median(list_data))
    return (data)
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
    dataX = met.Metrics(in_min=np.min(dicts_x), in_max=np.max(dicts_x), in_mean=np.mean(dicts_x), in_dev=np.std(dicts_x), in_med=np.median(dicts_x))
    dataY = met.Metrics(in_min=np.min(dicts_y), in_max=np.max(dicts_y), in_mean=np.mean(dicts_y), in_dev=np.std(dicts_y), in_med=np.median(dicts_y))
    dataZ = met.Metrics(in_min=np.min(dicts_z), in_max=np.max(dicts_z), in_mean=np.mean(dicts_z), in_dev=np.std(dicts_z), in_med=np.median(dicts_z))
    dataW = met.Metrics(in_min=np.min(dicts_w), in_max=np.max(dicts_w), in_mean=np.mean(dicts_w), in_dev=np.std(dicts_w), in_med=np.median(dicts_w))
    #print dataZ
    return (dataX,dataY,dataZ,dataW)
def get_all_types(path):
    '''
    Given folder, will extract arrays for each type of data.
    bd = [[zAccl][yAccl][xAccl][][]p[]]
    '''
    data_dict = create_dictionary(path)
    
    action_type = ["bk", "ns", "fd", "ld", "lu", "rd", "ru"]

    # Right now, I'm classifying the nods as noisy data. If we want to recognize it, 
    # we can easily change the label
    # alphabetically: left down, left up, noisy  right down, right up
    dict_labels = {"bk": "noisy",  "ns": "noisy", "fd":"noisy", "ld":"left down", "lu":"left up", "rd":"right down", "ru":"right up"}
    label = ""
    lists_w_labels = []
    for date_key, d in data_dict.items():
        tmp = []
        ### We can use a different factor chooser?
        ### So feature1 =  "att" (4 metrics), 2= "rot" (3 metrics) , 3="uaccl" (3 metrics), 4= "xaccl" (1 metric), 5= "yaccl" (1 metric), 6="zaccl" (1 metric)
        ### att_x [0-4 features], att_y [5:9], att_z [10:14], att_w [15_19]
        ### rot_x [20:24],rot_y [25:29], rot_z [30:34]
        ### uaccl_x [35:39],uaccl_y [40:44], uaccl_z [45:49]
        ### xAccL: [50:54] 

        ### top ranked features: 54 (xaccl: dev), 50 (uaccl_z dev), 51(xaccl: max), 
        ### 52 (xAccl med),32 (rot_z med),26 (rot_y mean), 31 (rot_z mean), 33 (rot_z min)
        ### 39 (uaccl_x dev)

        ### FOR GETTING TOP FEATURES
 
        if (d["xaccl"] is not None) and (d["uaccl"] is not None) and (d["rot"] is not None):
            # xaccl: dev
            tmp.append(d["xaccl"].getDev())

            # uaccl_z dev
            tmp.append(d["uaccl"][2].getDev())

            # xaccl: max
            tmp.append(d["xaccl"].getMax())

            # rot_z med
            tmp.append(d["rot"][2].getMed())

            #rot_y mean
            tmp.append(d["rot"][1].getMean())

            # rot_z mean
            tmp.append(d["rot"][2].getMean())

            # rot_z min
            tmp.append(d["rot"][2].getMin())
            Y.append(tmp)
            Z.append(dict_labels[d["label"]])
        else:
            print date_key


        #uaccl_x dev
        #tmp.append(d["uaccl"][0].getDev())



        #CODE FOR GETTING ALL FEATURES
        '''
        for data_type, metric_list in sorted(d.iteritems()):
            if data_type == "label":
                label = metric_list #this one isn't actually a metric list 
                continue

            if not metric_list:
                print "no Metrics in data_type "+ data_type +" in date "+ date_key
                continue

 #           print metric_list
 #           print data_dict
           # for metric in metric_list:

            
            try:
                tmp.append(metric_list.getMax())
                tmp.append(metric_list.getMean())
                tmp.append(metric_list.getMed())
                tmp.append(metric_list.getMin())
                tmp.append(metric_list.getDev())
            except:
                for metric in metric_list:
                   # print metric
                    tmp.append(metric.getMax())
                    tmp.append(metric.getMean())
                    tmp.append(metric.getMed())
                    tmp.append(metric.getMin())
                    tmp.append(metric.getDev())
                   # print "error"
        '''

        #lists_w_labels.append(tmp +[date_key])


'''
parse test trial
'''
def convert(file,path):
    '''
    Given folder, will extract arrays for each type of data.
    bd = [[zAccl][yAccl][xAccl][][]p[]]
    '''
    data_dict = to_dictionary(file,path)
    
    action_type = ["bk", "ns", "fd", "ld", "lu", "rd", "ru"]

    # Right now, I'm classifying the nods as noisy data. If we want to recognize it, 
    # we can easily change the label
    # alphabetically: left down, left up, noisy  right down, right up
    dict_labels = {"bk": "noisy",  "ns": "noisy", "fd":"noisy", "ld":"left down", "lu":"left up", "rd":"right down", "ru":"right up"}
    label = ""
    YVals = []
    ZVals = []
    
    for date_key, d in data_dict.items():
        tmp = []
        
        ### FOR GETTING TOP FEATURES
 
        # xaccl: dev
        tmp.append(d["xaccl"].getDev())

        # uaccl_z dev
        tmp.append(d["uaccl"][2].getDev())

        # xaccl: max
        tmp.append(d["xaccl"].getMax())

        # rot_z med
        tmp.append(d["rot"][2].getMed())

        #rot_y mean
        tmp.append(d["rot"][1].getMean())

        # rot_z mean
        tmp.append(d["rot"][2].getMean())

        # rot_z min
        tmp.append(d["rot"][2].getMin())

        #uaccl_x dev
        #tmp.append(d["uaccl"][0].getDev())

        YVals.append(tmp)
        ZVals.append(dict_labels[d["label"]])
        #lists_w_labels.append(tmp +[date_key])
        return(YVals,ZVals)


def extract_data(d, filename, path):
    '''
    Given filename, extract that particular type of data by assigning a Metric
    to a dictionary.... at the end we'll have a dictionary of metrics. Each dict value
    is a tuple, with at least one Metric
    '''
    fullname = os.path.join(path,filename)
    data_type = filename[3]
    # 5* (1 * 3 + 3*2 + 4) = 65 features 
    if (data_type == "x"):
        d["xaccl"] = xyz_accl(fullname) 

    elif (data_type == "y"):
        d["yaccl"] = xyz_accl(fullname)

    elif (data_type == "z"):
        d["zaccl"] = xyz_accl(fullname) 

    elif (data_type == "r"):
        d["rot"] = rot_txt(fullname)

    elif (data_type == "a"):            
        d["att"] = attitude_txt(fullname)

    elif (data_type == "u"):
        d["uaccl"] = rot_txt(fullname) #like rot, user Accl has 3 places...


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

def to_dictionary(filename,path):
   
    data_type = ["label", "rot", "xaccl", "yaccl", "zaccl", "att", "uaccl"]
    data_dict = {}

    date = filename[-14: -4]
    d = data_dict.get(date)
    if d is None:
        d = dict.fromkeys(data_type)
        d["label"] = filename[:2]
            
        data_dict[date] = d
    extract_data(d, filename, path) #Have to implement this
    return data_dict

 
def train():
    '''
    trains the data and prints the accuracy score
'''
    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    #scaler =  preprocessing.StandardScaler().fit(Y_train)
    #Y_test = scaler.transform(Y_test)
    scaler = preprocessing.StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    for n in range(1,20):
        knn = neighbors.KNeighborsClassifier(n_neighbors=n)
        knn.fit(Y_train, Z_train)
        z_pred = knn.predict(Y_test)
        #misclassified = Y_test[Z_test != z_pred]
        #print "misclassified ones,", misclassified
        if accuracy_score(Z_test, z_pred) > maxAccurate:
            maxAccurate = accuracy_score(Z_test, z_pred)
            index = n
        print "accuracy score, ", accuracy_score(Z_test, z_pred)
        print "confusion_matrix, "
        print confusion_matrix(Z_test, z_pred)

        #print classification_report(Z_test, z_pred)
    print "best accuracy: ",maxAccurate," nearest neighbor:", index

'''
def find_misclassified(arr, lists_w_labels):
    
    #Input: array of arrays that have
    #go through dictionary to find misclassified files
    

    for a in arr:
        print "finding: "
        print a
        for l in lists_w_labels:
            date = l[-1]
            metrics = l[:-1]
            found = True
            for index, el in enumerate(a):
                if round(el, 5) != round(metrics[index], 5):
                    found = False
            if found:
                print "misclassified: "+ date 

        ### for each misclassified, 
'''
def train2():
    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    scaler =  preprocessing.StandardScaler().fit(Y_train)
    Y_test = scaler.transform(Y_test)

    model = ExtraTreesClassifier()
    model = model.fit(Y_train, Z_train)
 #   print(Y_train)
    print "feature importances"
    print(model.feature_importances_)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(Y[0])):
        print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])
    model = tree.DecisionTreeClassifier()
    model.fit(Y_train, Z_train)
    z_pred = model.predict(Y_test)
    misclassified = Y_test[Z_test != z_pred]
    print "misclassified ones,", misclassified
            
    print "accuracy score, ", accuracy_score(Z_test, z_pred)
    print "confusion_matrix, ", confusion_matrix(Z_test, z_pred)
    return model


def classify(YVals,ZVals):
    '''
    trains the data and prints the accuracy score
'''
    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(YVals,ZVals,random_state = 0)
    #scaler =  preprocessing.StandardScaler().fit(Y_train)
    #Y_test = scaler.transform(Y_test)
    scaler = preprocessing.StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(Y_train, Z_train)
    z_pred = knn.predict(Y_test)
        #misclassified = Y_test[Z_test != z_pred]
        #print "misclassified ones,", misclassified
    print "accuracy score, ", accuracy_score(Z_test, z_pred)
    print "confusion_matrix, "
    print confusion_matrix(Z_test, z_pred)
    return knn
        #print classification_report(Z_test, z_pred)

        
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

#            train2()
            train()

if __name__ == '__main__':
    main()
