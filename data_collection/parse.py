import pandas as pd
import numpy as np
import sys
import os
import ast
import argparse
import math
import re

class Metrics(object):
        def __init__(self):
                self.min = None
                self.max = None
                self.dev = None
                self.mean = None

        def __repr__(self):
                return self.__str__()
        def __str__(self):
                return str({
                        'max': self.id,
                        'max': self.user,
                        'dev': self.category,
                        'mean': self.timestamp
                })
        

def extract_txt(filename):
	'''
	Input: A path to a textfile
	Return: a pandas dataframe
	'''
	with open(filename, "r") as f:
		if not ".txt" in filename:
			print(filename + " not a txt file")
			return None
		dicts = []
		for line in f:
                    structure = re.split('([(]?)(.*?)([)]?)(,|$)',line)
                    list1 = []
                    list1.append(structure[2])
                    list1.append(structure[7])
                    list1.append(structure[12])
                    list1.append(structure[17])
                    dicts.append(list1)
                    
	df = pd.DataFrame(dicts)
	#print dicts
	return df
		

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-f','--file',help = 'Specifies a particular input file to test')
        #parser.add_argument('-fd','--folder',help = 'Specifies a particular folder name for testing. Default is "Data"')
        args = parser.parse_args()
        if args.file:
                txt_path = args.file
                dataFrame = extract_txt(txt_path)

if __name__ == '__main__':
    main()
