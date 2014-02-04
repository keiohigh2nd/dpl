#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

def chip_data(num):
	f = open("Network1_expression_data.tsv")
        lines = f.readlines()
	arr = []
	for x in lines:
		tmp = x.split("\t")
		arr.append(tmp[num])
	print arr
	return arr

def chip_data_dif(a,b):
        f = open("Network1_expression_data.tsv")
        lines = f.readlines()
	f.close()

        arr = []
	i = 0
        for x in lines:
		if i != 0:
                	tmp = x.split("\t")
                	arr.append(math.fabs(float(tmp[int(a)])-float(tmp[int(b)])))
		i += 1
        return arr

if __name__ == "__main__":
	import itertools

	f = open("Network1_expression_data.tsv")
	lines = f.readlines()
	f.close()
	tm = lines[0].split("\t")
	cols = len(tm)
	print cols

	for x in lines:
		tmp = x.split("\t")
		#print tmp[0]

	chip_data_dif(3,4)

	combi = list(itertools.combinations(range(cols), 2))
	test = []
	for x in combi:
		if x[0] < 195:
			test.append(chip_data_dif(x[0],x[1]))
		

		
