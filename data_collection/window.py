import pandas as pd
import numpy as np
import sys
import os
from os import walk, path, makedirs
import ast
import argparse
import math
import re

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
#global variables for storing training

from itertools import izip_longest


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


        #print dataZ
    #return (dataX,dataZ)
    return (dicts_x,dicts_y,dicts_z)
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
    return (list_data)
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
   #print dataZ
    return (dicts_x,dicts_y,dicts_x,dicts_w)


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


def test_window(knn, scaler, data_dict):
    count_l = 0
    count_r = 0
    count_n = 0    
    for date_key, d in data_dict.items():
        if d["rot"]==None or d["uaccl"] == None or d["xaccl"] == None:
            continue
        if d["label"] == "lu":
            #print "lu"
            l, r, n = window(knn, scaler, d)
            count_l += l
            count_r += r
            count_n += n
    print "lu"
    print ("left:", count_l )
    print ("right:", count_r )
    print ("noisy:", count_n )

    count_l = 0
    count_r = 0
    count_n = 0    
    for date_key, d in data_dict.items():
        if d["rot"]==None or d["uaccl"] == None or d["xaccl"] == None:
            continue
        if d["label"] == "ld":
            #print "lu"
            l, r, n = window(knn, scaler, d)
            count_l += l
            count_r += r
            count_n += n
    print "ld"
    print ("left:", count_l )
    print ("right:", count_r )
    print ("noisy:", count_n )

    count_l = 0
    count_r = 0
    count_n = 0    
    for date_key, d in data_dict.items():
        if d["rot"]==None or d["uaccl"] == None or d["xaccl"] == None:
            continue
        if d["label"] == "ru":
            #print "lu"
            l, r, n = window(knn, scaler, d)
            count_l += l
            count_r += r
            count_n += n
    print "ru"
    print ("left:", count_l )
    print ("right:", count_r )
    print ("noisy:", count_n )

    count_l = 0
    count_r = 0
    count_n = 0    
    for date_key, d in data_dict.items():
        if d["rot"]==None or d["uaccl"] == None or d["xaccl"] == None:
            continue
        if d["label"] == "rd":
            #print "lu"
            l, r, n = window(knn, scaler, d)
            count_l += l
            count_r += r
            count_n += n
    print "rd"
    print ("left:", count_l )
    print ("right:", count_r )
    print ("noisy:", count_n )


    count_l = 0
    count_r = 0
    count_n = 0    
    for date_key, d in data_dict.items():
        if d["label"] == "ns":
            #print "lu"
            l, r, n = window(knn, scaler, d)
            count_l += l
            count_r += r
            count_n += n
    print "ns"
    print ("left:", count_l )
    print ("right:", count_r )
    print ("noisy:", count_n )


def window(knn, scaler, d):
    roty = d["rot"][1]
    rotz = d["rot"][2]
    uaccelz = d["uaccl"][2]
    xaccl = d["xaccl"]
    cur_roty = []
    cur_rotz = []
    cur_uaccelz = []
    cur_xaccl = []

    ld = False
    rd = False
    count_l = 0
    count_r = 0
    count_n = 0

    while len(roty) > 50:

        cur_roty = roty[:75]
        cur_rotz = rotz[:75]
        cur_uaccelz = uaccelz[:]
        cur_xaccl = xaccl[:75]

        # deleted first 25
        del roty[:10]            
        del rotz[:10]    
        del uaccelz[:10]    
        del xaccl[:10] 

        # make into arrays
        n_roty = np.array(cur_roty)
        n_rotz = np.array(cur_rotz)
        n_uaccelz = np.array(cur_uaccelz)
        n_xaccl = np.array(cur_xaccl)
        tmp = []
        tmp.append(n_xaccl.std())
        tmp.append(n_uaccelz.std())
        tmp.append(n_xaccl.max())
        tmp.append(np.median(n_roty))
        tmp.append(n_roty.mean())
        tmp.append(n_rotz.mean())
        tmp.append(n_rotz.min())
        Features = []
        Features.append(tmp)

        #print "features", Features
        scaler.transform(Features)
        label = knn.predict(Features)
        #print("result:", label)

        if label == "right down":
            count_r +=1
            rd = True
        elif label == "right up":
            count_r += 1
            if rd:
        #        print "RIGHT TILT"
                rd = False
            #flush(rotz,roty,uaccelz,xaccl)

        if label == "left down":
            count_l +=1

            ld = True
        elif label == "left up":
            count_l+=1
            if ld: 
        #        print "LEFT TILT"
                ld = False
            #flush(rotz,roty,uaccelz,xaccl)

        else:
            pass
            #print "NOISY"
    if count_l + count_r == 0:
        count_n+=1
    #print "DONE"
    return count_l, count_r, count_n


        
def main():
#            train2()
    pass
    # data_dict = create_dictionary("all_data")

    # for n in range(1,20):
    #     print n
    #     knn, scaler = classify2.temp(n) #train()
    #     test_window(knn, scaler, data_dict)

if __name__ == '__main__':
    main()