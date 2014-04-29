#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy

def chip_data(num):
	f = open("Network1_expression_data.tsv")
        lines = f.readlines()
	arr = numpy.array([])
	for x in lines:
		tmp = x.split("\t")
		arr.append(tmp[num])
	print arr
	return arr

def chip_data_dif(a,b):
	threshold = 0.3

        f = open("Network1_expression_data.tsv")
        lines = f.readlines()
	f.close()

        arr = []
	i = 0
	res = numpy.zeros(len(lines))
        for x in lines:
		if i != 0:
                	tmp = x.split("\t")
			tmp_d = math.fabs(float(tmp[int(a)])-float(tmp[int(b)]))
			if float(threshold) > tmp_d:
                		#arr.append(1)
				res[i-1] = 1
			else:
				#arr.append(0)
				res[i-1] = 0
		i += 1
	#return arr
	return res

def convert():
	#tranform GXX to XX
	f = open("DREAM5_NetworkInference_GoldStandard_Network3_300.csv")
	tmp_f = f.read()
	lines = tmp_f.split("\r")
	f.close()

	f1 = open("Nw3_G_300.csv","w")
	for x in lines:
		tmp = x.split(",")
		f1.write(tmp[0].strip("G"))
		f1.write(",")
		f1.write(tmp[1].strip("G"))
		f1.write(",")
		f1.write(tmp[2].strip("\n"))
		f1.write("\r")
	f1.close()
		

if __name__ == "__main__":
	import itertools
	
	convert()

	"""
	f = open("Network1_expression_data.tsv")
	lines = f.readlines()
	f.close()
	tm = lines[0].split("\t")
	cols = len(tm)
	print cols

	#You cannot change this part because of something excel error
	f1 = open("DREAM5_NetworkInference_GoldStandard_Network1.tsv")
	lins = f1.readlines()
	f1.close()


	combi = list(itertools.combinations(range(cols), 2))
	train = numpy.zeros(len(lines))
	res_train = numpy.zeros(2)
	
	i = 0
	for y in lins:
		print y
		tmp =  y.split("\t")
		#print chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)
		numpy.column_stack((train,chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)))
		if int(tmp[2].strip()) == 0:
			numpy.vstack((res_train,[1,0]))
		else:
			numpy.vstack((res_train,[0,1]))
		i += 1
	"""
