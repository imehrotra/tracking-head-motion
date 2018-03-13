import numpy as np
import sys
import os 
import ast
import argparse
import math
import re
import pickle
import window
import Metrics as met
import utils
from sklearn import neighbors, datasets, preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from itertools import izip_longest


def rot_txt(filename):
  '''
  Input: A path to an userAccel and rotrate textfile. Parses for key values
  Return: a Metric 
  '''
  with open(filename, "r") as f:
    if not ".txt" in filename:
      print(filename + " not a txt file")
      return None
    dicts_x = []
    dicts_y = []
    dicts_z = []
    for line in f:
        structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
        list1 = []
        x = float(structure[2])
        y = float(structure[7])
        z = float(structure[12])
    
        dicts_x.append(x)
        dicts_y.append(y)
        dicts_z.append(z)

    dataX = met.Metrics(in_min=np.min(dicts_x), in_max=np.max(dicts_x), in_mean=np.mean(dicts_x), in_dev=np.std(dicts_x), in_med=np.median(dicts_x))
    dataY = met.Metrics(in_min=np.min(dicts_y), in_max=np.max(dicts_y), in_mean=np.mean(dicts_y), in_dev=np.std(dicts_y), in_med=np.median(dicts_y))
    dataZ = met.Metrics(in_min=np.min(dicts_z), in_max=np.max(dicts_z), in_mean=np.mean(dicts_z), in_dev=np.std(dicts_z), in_med=np.median(dicts_z))

    return (dataX,dataY,dataZ)


def window_rot_txt(filename):
  '''
  Input: A path to userAccel and rotrate textfile.
  Return: Lists of data for x, y, z
  *Modified version that is employed to train knn algorithm in overlapping frames manner
  '''
  with open(filename, "r") as f:
    if not ".txt" in filename:
      print(filename + " not a txt file")
      return None
    dicts_x = []
    dicts_y = []
    dicts_z = []
    for line in f:
        structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
        list1 = []
        x = float(structure[2])
        y = float(structure[7])
        z = float(structure[12])

        dicts_x.append(x)
        dicts_y.append(y)
        dicts_z.append(z)
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
<<<<<<< HEAD
    data = met.Metrics(in_min=np.min(list_data), in_max=np.max(list_data), in_mean=np.mean(list_data), in_dev=np.std(list_data), in_med=np.median(list_data))
    return (data)
=======
    return (list_data)
>>>>>>> 12f317eb14c35e51cc4e0356f26c475048bc1487

def window_xyz_accl(filename):
    '''
    Input: A path to x, y, or z accl, which only has one value
    Return: A list of data 
    *Modified version that is employed to train knn algorithm in overlapping frames manner
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
        dicts_x = []
        dicts_y = []
        dicts_z = []
        dicts_w = []

        for line in f:
            structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
            x = float(structure[2])
            y = float(structure[7])
            z = float(structure[12]) 
            w = float(structure[17])
            dicts_x.append(x)
            dicts_y.append(y)
            dicts_z.append(z)
            dicts_w.append(w)

    dataX = met.Metrics(in_min=np.min(dicts_x), in_max=np.max(dicts_x), in_mean=np.mean(dicts_x), in_dev=np.std(dicts_x), in_med=np.median(dicts_x))
    dataY = met.Metrics(in_min=np.min(dicts_y), in_max=np.max(dicts_y), in_mean=np.mean(dicts_y), in_dev=np.std(dicts_y), in_med=np.median(dicts_y))
    dataZ = met.Metrics(in_min=np.min(dicts_z), in_max=np.max(dicts_z), in_mean=np.mean(dicts_z), in_dev=np.std(dicts_z), in_med=np.median(dicts_z))
    dataW = met.Metrics(in_min=np.min(dicts_w), in_max=np.max(dicts_w), in_mean=np.mean(dicts_w), in_dev=np.std(dicts_w), in_med=np.median(dicts_w))

    return (dataX,dataY,dataZ,dataW)

def window_attitude_txt(filename):
    '''
    Input: A path to attitude
    Return: a metric
    *Modified version that is employed to train knn algorithm in overlapping frames manner
    '''
    with open(filename, "r") as f:
        if not ".txt" in filename:
            print(filename + " not a txt file")
            return None
        dicts_x = []
        dicts_y = []
        dicts_z = []
        dicts_w = []

        for line in f:
            structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
            x = float(structure[2])
            y = float(structure[7])
            z = float(structure[12]) 
            w = float(structure[17])
            dicts_x.append(x)
            dicts_y.append(y)
            dicts_z.append(z)
            dicts_w.append(w)
    return (dicts_x,dicts_y,dicts_z,dicts_w)

def window_get_all_types(Y,Z,path):
    '''
    Function for extracting data for window testing of algorithm, using uniform frames
    Inputs: Y,Z: arrays for features and dict_labels; path: folder to read from
    Return: Y, Z filled for the given folder
    '''
    data_dict = window_create_dictionary(path)
    
    dict_labels = {"bk": "noisy",  "ns": "noisy", "fd":"noisy", "ld":"left down", "lu":"left up", "rd":"right down", "ru":"right up"}
    label = ""
    lists_w_labels = []
    for date_key, d in data_dict.items():
        tmp = []
 
        if (d["xaccl"] is not None) and (d["uaccl"] is not None) and (d["rot"] is not None):
            
            if(len(d["xaccl"]) <= 80):
                tmp.append(np.std(d["xaccl"]))
                tmp.append(np.max(d["xaccl"]))
                tmp.append(np.std(d["uaccl"][2]))
                tmp.append(np.median(d["rot"][2]))
                tmp.append(np.mean(d["rot"][2]))
                tmp.append(np.min(d["rot"][2]))
                tmp.append(np.mean(d["rot"][1]))
                Y.append(tmp)
                Z.append(dict_labels[d["label"]])
            else:

                while(len(d["xaccl"]) > 50):
                      tmp = []
                      tempListA = d["xaccl"][:50]
                      del d["xaccl"][:10]
                      tmp.append(np.std(tempListA))
                      tmp.append(np.max(tempListA))

                      tempListB = d["uaccl"][2][:50]
                      del d["uaccl"][2][:10]                      
                      tmp.append(np.std(tempListB))

                      tempListC = d["rot"][2][:50]
                      del d["rot"][2][:10]
                      tmp.append(np.median(tempListC))
                      tmp.append(np.mean(tempListC))
                      tmp.append(np.min(tempListC))

                      tempListD = d["rot"][1][:50]
                      del d["rot"][1][:10]
                      tmp.append(np.mean(tempListD))

                      Y.append(tmp)
                      Z.append(dict_labels[d["label"]])
    return Y, Z


def get_all_types(Y, Z, path):
    '''
    Function for extracting data for  testing of algorithm, WITHOUT uniform frames
    Inputs: Y,Z: arrays for features and dict_labels; path: folder to read from
    Return: Y, Z filled for the given folder with top features
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
        # From the find_features training:
        # feature1 =  "att" (4 metrics), 2= "rot" (3 metrics) , 3="uaccl" (3 metrics), 4= "xaccl" (1 metric), 5= "yaccl" (1 metric), 6="zaccl" (1 metric)
        # att_x [0-4 features], att_y [5:9], att_z [10:14], att_w [15_19]
        # rot_x [20:24],rot_y [25:29], rot_z [30:34]
        # uaccl_x [35:39],uaccl_y [40:44], uaccl_z [45:49]
        # xAccL: [50:54] 
        #
        # top ranked features: 54 (xaccl: dev), 50 (uaccl_z dev), 51(xaccl: max), 
        # 52 (xAccl med),32 (rot_z med),26 (rot_y mean), 31 (rot_z mean), 33 (rot_z min)
        # 39 (uaccl_x dev)

        # Code for getting top features
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

    return Y, Z


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
        d["rot"] = rot_txt(fullname)

    elif (data_type == "a"):            
        d["att"] = attitude_txt(fullname)

    elif (data_type == "u"):
        d["uaccl"] = rot_txt(fullname) #like rot, user Accl has 3 places...


def window_extract_data(d, filename, path):
    '''
    Modified extract_data for uniform window method--necessary because previous 
    method already put them into Metrics, which can't be sliced like lists.

    Given filename, extract that particular type of data by assigning data list
    to a dictionary.... at the end we'll have a dictionary of data lists. 
    '''
    fullname = os.path.join(path,filename)
    data_type = filename[3]
    # 5* (1 * 3 + 3*2 + 4) = 65 features 
    if (data_type == "x"):
        d["xaccl"] = window_xyz_accl(fullname) 

    elif (data_type == "y"):
        d["yaccl"] = window_xyz_accl(fullname)

    elif (data_type == "z"):
        d["zaccl"] = window_xyz_accl(fullname) 

    elif (data_type == "r"):
        d["rot"] = window_rot_txt(fullname)

    elif (data_type == "a"):            
        d["att"] = window_attitude_txt(fullname)

    elif (data_type == "u"):
        d["uaccl"] = window_rot_txt(fullname) #like rot, user Accl has 3 places...

def create_dictionary(path):
    '''
    Given folder, will create dictionary with the date as key, lining up all the 
    data from different text files for easy access
    Returns dictionary of the following form:
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
        extract_data(d, filename, path) 

    return data_dict

def window_create_dictionary(path):
    '''
    Modified from create_dictionary to use the modified window_extract_data() function

    Given folder, will create dictionary with the date as key, lining up all the 
    data from different text files for easy access
    Returns dictionary of the following form:
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
        window_extract_data(d, filename, path) 

    return data_dict
 
def train(Y, Z):
    '''
    Input: Y, Z: arrays with input and output for classification
    Return: classifier knn, scaler fitted to training data

    Trains the data and prints the accuracy scores of a range of k values
    '''
    get_all_types(Y,Z,"all_data")

    maxAccurate = 0;
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z)
    scaler = preprocessing.StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    for n in range(1,20):
        knn = neighbors.KNeighborsClassifier(n_neighbors=n)
        knn.fit(Y_train, Z_train)
        z_pred = knn.predict(Y_test)
        if accuracy_score(Z_test, z_pred) > maxAccurate:
            maxAccurate = accuracy_score(Z_test, z_pred)
            index = n
        print "accuracy score, ", accuracy_score(Z_test, z_pred)
        print "confusion_matrix, "
        print confusion_matrix(Z_test, z_pred)

    print "best accuracy: ",maxAccurate," nearest neighbor:", index
    
    return knn, scaler

def window_train(Y, Z):
    '''
    Modified version of train to use uniform overlapping frames
    Input: Y, Z: arrays with input and output for classification
    Return: classifier knn, scaler fitted to training data

    Trains the data and prints the accuracy scores of a range of k values
    '''
    window_get_all_types(Y, Z, "all_data")
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    scaler = preprocessing.StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)
    for n in range(1,20):
        print n

        knn = neighbors.KNeighborsClassifier(n_neighbors=n)
        knn.fit(Y_train, Z_train)
        z_pred = knn.predict(Y_test)

        print "accuracy score, ", accuracy_score(Z_test, z_pred)
 
        data_dict = window_create_dictionary("all_data")
        window.test_window(knn, scaler, data_dict)
    return knn, scaler

def classify(Y, Z):
    '''
    Input: Y, Z: arrays with input and output for classification
    Return: classifier knn, scaler fitted to training data

    Trains the data and prints the accuracy score for one model
    '''
    get_all_types(Y, Z, "all_data")

    Y_train, Y_test, Z_train, Z_test = train_test_split(Y,Z,random_state = 0)
    scaler = preprocessing.StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    
    knn = neighbors.KNeighborsClassifier(n_neighbors=10) #Isha changed 1 to 10
    knn.fit(Y_train, Z_train)
    z_pred = knn.predict(Y_test)

    print "accuracy score, ", accuracy_score(Z_test, z_pred)
    print "confusion_matrix, "
    print confusion_matrix(Z_test, z_pred)

    return knn, scaler


        
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
            Y = []
            Z = []

            classify(Y, Z)


if __name__ == '__main__':
    main()
