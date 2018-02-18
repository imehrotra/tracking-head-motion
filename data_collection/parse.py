import pandas as pd
import numpy as np
import sys
import os
import ast
import argparse
import math
import re


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
                    structure = re.split(', () ',line)
                    dicts.append(structure)
	df = pd.DataFrame(dicts)
	#print df
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
