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

epsilon = 0.000001

#global variables for storing training
Y = []
Z = []

'''dump_path = path.join('..', 'output')
def save(data, filename):   
    if not path.exists(dump_path):
        makedirs(dump_path)
    filename = path.join(dump_path, filename)
    fileObject = open(filename, 'wb')
    pickle.dump(data, fileObject)
    fileObject.close()


def load(filename):
 #   filename = path.join(dump_path, filename)
    fileObject = open(filename, 'rb')
    return pickle.load(fileObject)
'''
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
        
def accl_txt(filename):
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
#                        if (abs(0-x) < epsilon) and (abs(0-y) < epsilon) and (abs(0-z) < epsilon):
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
    return (dataX,dataZ)
  #return (dataX,dataY,dataZ)

def gyro_txt(filename):
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
#    dataX = met.Metrics()
#    setattr(dataX,'mean',np.mean(dicts_x))
#    dataX.max = np.max(dicts_x)
#    dataX.min = np.min(dicts_x)
#    dataX.dev = np.std(dicts_x)
#    dataX.med = np.median(dicts_x)
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
def folder_accel(path):
    dataFrame = []
    for filename in os.listdir(path): 
        if not ".txt" in filename:
            print(filename + " not a txt file")
            continue
        dataFrame.append(accl_txt(os.path.join(path, filename)))
  #print dataFrame[1]
    return dataFrame
def folder_att(path):
    dataFrame = []
    for filename in os.listdir(path): 
        if not ".txt" in filename:
            print(filename + " not a txt file")
            continue
        dataFrame.append(gyro_txt(os.path.join(path, filename)))
  #print dataFrame[1]
    return dataFrame
def overall_mean(data,i):
    dictsX = []
    dictsY = []
    dictsZ = []
    dictsW = []
    for each in data:
        dictsX.append(each[0].getMean())
        dictsY.append(each[1].getMean())
        dictsZ.append(each[2].getMean())
        if i == 0:
            dictsW.append(each[3].getMean())
    print "x mean: ", np.mean(dictsX)
    print "y mean: ", np.mean(dictsY)
    print "z mean: ", np.mean(dictsZ)
    if i == 0:
        print "w mean: ", np.mean(dictsW)
        return(dictsX,dictsY,dictsZ,dictsW)
    return (dictsX,dictsY,dictsZ)
def overall_dev(data,i):
    dictsX = []
    dictsY = []
    dictsZ = []
    dictsW = []
    for each in data:
        dictsX.append(each[0].getDev())
        dictsY.append(each[1].getDev())
        dictsZ.append(each[2].getDev())
        if i == 0:
            dictsW.append(each[3].getMean())
    print "x Dev: ", np.mean(dictsX)
    print "y Dev: ", np.mean(dictsY)
    print "z Dev: ", np.mean(dictsZ)
    if i == 0:
        print "w Dev: ", np.mean(dictsW)
def overall_max(data,i):
    dictsX = []
    dictsY = []
    dictsZ = []
    dictsW = []
    for each in data:
        dictsX.append(each[0].getMax())
        dictsY.append(each[1].getMax())
        dictsZ.append(each[2].getMax())
        if i == 0:
            dictsW.append(each[3].getMean())
    print "x Max: ", np.mean(dictsX)
    print "y Max: ", np.mean(dictsY)
    print "z Max: ", np.mean(dictsZ)
    if i == 0:
        print "w Max: ", np.mean(dictsW)
def overall_min(data,i):
        dictsX = []
        dictsY = []
        dictsZ = []
        dictsW = []
        for each in data:
            dictsX.append(each[0].getMin())
            dictsY.append(each[1].getMin())
            dictsZ.append(each[2].getMin())
            if i == 0:
                dictsW.append(each[3].getMean())
        print "x Min: ", np.mean(dictsX)
        print "y Min: ", np.mean(dictsY)
        print "z Min: ", np.mean(dictsZ)
        if i == 0:
            print "w Max: ", np.mean(dictsW)
            return(dictsX,dictsY,dictsZ,dictsW)
        return (dictsX,dictsY,dictsZ)
def add_train(capture,i, s):
        # i is 0 for right tilt
        # adds the data into the training test set (global variables)
    for each in capture:
        for one in each:
            tmp = []
            tmp.append(one.getMax())
#            tmp.append(one.getMean())
#            tmp.append(one.getMed())
            tmp.append(one.getMin())
#            tmp.append(one.getDev())

            Y.append(tmp)
            if i == 0:
                if s == "u": 
                    Z.append("counterclockwise") #right tilt u
                else:
                    Z.append("clockwise") #right tild d
            elif i == 1:
                if s == "u":
                    Z.append("clockwise")
                else:
                    Z.append("counterclockwise")
            else:
                Z.append("sit")
 
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
            dataFrame = accl_txt(txt_path)
    if args.filea:
            txt_path = args.filea
            dataFrame = gyro_txt(txt_path)
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
            dataFrameRotateLU = folder_accel("rotate_lt_u")
            add_train(dataFrameRotateLU,1,"u")
            dataFrameRotateLD = folder_accel("rotate_lt_d")
            add_train(dataFrameRotateLD,1, "d")
            dataFrameRotateRU = folder_accel("rotate_rt_u")
            add_train(dataFrameRotateRU,0, "u")
            dataFrameRotateRD = folder_accel("rotate_rt_d")
            add_train(dataFrameRotateRD,0, "d")
        #    dataFrameRotateS = folder_accel("rotate_sit")
        #    add_train(dataFrameRotateS,2, "s")
            train()

if __name__ == '__main__':
    main()
