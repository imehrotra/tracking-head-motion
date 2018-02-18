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

epsilon = 0.000001
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
                        if (abs(0-x) < epsilon) and (abs(0-y) < epsilon) and (abs(0-z) < epsilon):
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
        dataX.max = np.amax(dicts_x)
        dataX.min = np.amin(dicts_x)
        dataX.dev = np.std(dicts_x)
        #print dataX
        dataY = met.Metrics()
        setattr(dataY,'mean',np.mean(dicts_y))
        dataY.max = np.amax(dicts_y)
        dataY.min = np.amin(dicts_y)
        dataY.dev = np.std(dicts_y)
        #print dataY
        dataZ = met.Metrics()
        setattr(dataZ,'mean',np.mean(dicts_z))
        dataZ.max = np.amax(dicts_z)
        dataZ.min = np.amin(dicts_z)
        dataZ.dev = np.std(dicts_z)
        #print dataZ
	return (dataX,dataY,dataZ)

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
        dataW.max = np.amax(dicts_w)
        dataW.min = np.amin(dicts_w)
        dataW.dev = np.std(dicts_w)
        #print dataW
        dataX = met.Metrics()
        setattr(dataX,'mean',np.mean(dicts_x))
        dataX.max = np.amax(dicts_x)
        dataX.min = np.amin(dicts_x)
        dataX.dev = np.std(dicts_x)
        #print dataX
        dataY = met.Metrics()
        setattr(dataY,'mean',np.mean(dicts_y))
        dataY.max = np.amax(dicts_y)
        dataY.min = np.amin(dicts_y)
        dataY.dev = np.std(dicts_y)
        #print dataY
        dataZ = met.Metrics()
        setattr(dataZ,'mean',np.mean(dicts_z))
        dataZ.max = np.amax(dicts_z)
        dataZ.min = np.amin(dicts_z)
        dataZ.dev = np.std(dicts_z)
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
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-f','--file',help = 'Specifies a particular input file to test')
        parser.add_argument('-a','--filea',help = 'Specifies a particular attitude file to test')
        parser.add_argument('-fd','--folder',help = 'Specifies a particular folder to test')
        parser.add_argument('-fa','--folderA',help = 'Specifies a particular attitude folder to test')
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
                overall_mean(dataFrame,1)
                overall_dev(dataFrame,1)
                overall_max(dataFrame,1)
                overall_min(dataFrame,1)
        if args.folderA:
                txt_path = args.folderA
                dataFrame = folder_att(txt_path)
                overall_mean(dataFrame,0)
                overall_dev(dataFrame,0)
                overall_max(dataFrame,0)
                overall_min(dataFrame,0)

if __name__ == '__main__':
    main()
