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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from itertools import izip_longest


def toSingle(array,i):
    '''
    takes 2D array and selects for one column i
    '''
    dicts = []
    for each in array:
            dicts.append(each[i])

    return dicts

def count_print(knn, scaler, data_dict, label):
    '''
    Helper function for test_window; iterates over dictionary and runs on
    all files with given label
    '''
    count_lu = 0
    count_ld = 0
    count_rd = 0
    count_ru = 0
    count_n = 0
    for date_key, d in data_dict.items():
        if d["rot"]==None or d["uaccl"] == None or d["xaccl"] == None:
            continue
        if d["label"] == label:
            lu, ld, rd, ru, n = window(knn, scaler, d)
            count_lu += lu
            count_ld += ld
            count_rd += rd
            count_ru += ru
            count_n += n
    print label
    print ("left u:", count_lu )
    print ("left d:", count_ld )
    print ("right u:", count_ru )
    print ("right d:", count_rd )
    print ("noisy:", count_n )

def test_window(knn, scaler, data_dict):
    '''
    Runs the segmentation algorithm over data stored in dictionary to see if 
    uniform frames results in accurate classification
    '''
    count_print(knn, scaler, data_dict, "ru")
    count_print(knn, scaler, data_dict, "rd")
    count_print(knn, scaler, data_dict, "lu")
    count_print(knn, scaler, data_dict, "rd")
    count_print(knn, scaler, data_dict, "ns")


def window(knn, scaler, d):
    '''
    Segments given data lists into uniform, overlapping frames, and tries to 
    classify
    '''
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
    count_lu = 0
    count_ld = 0
    count_rd = 0
    count_ru = 0
    count_n = 0

    while len(roty) > 50:

        cur_roty = roty[:50]
        cur_rotz = rotz[:50]
        cur_uaccelz = uaccelz[:]
        cur_xaccl = xaccl[:50]

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
            count_rd +=1
            rd = True
        elif label == "right up":
            count_ru += 1
            if rd:
        #        print "RIGHT TILT"
                rd = False
            #flush(rotz,roty,uaccelz,xaccl)

        if label == "left down":
            count_ld +=1

            ld = True
        elif label == "left up":
            count_lu+=1
            if ld: 
        #        print "LEFT TILT"
                ld = False
            #flush(rotz,roty,uaccelz,xaccl)

        else:
            pass
            #print "NOISY"
    if count_lu + count_ld + count_ru+count_rd== 0:
        count_n+=1
    #print "DONE"
    return count_lu, count_ld, count_ru, count_rd, count_n


        
def main():
    pass


if __name__ == '__main__':
    main()
