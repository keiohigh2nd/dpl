import dbn
import sys, os, getopt
sys.path.append("/home/appl/opencv-2.4.6.1/lib/python2.6/site-packages")
import numpy
import cv2
from common import anorm, getsize

def convert_img_to_nparray(file):
	img_tmp = cv2.imread(file, 0)	
	img = cv2.resize(img_tmp, (0,0), fx=0.5, fy=0.5) 
	
	x = len(img)
	y = len(img[0])
	
	train = numpy.array([0])

	for i in range(x):
		for j in range(y):
			train = numpy.vstack((train,img[i,j]/float(255)))

	train = numpy.delete(train,0,0)
	print len(train)
	return train			

def cut_mitosis(file,res):
	dir = "/home/keiohigh2nd/icpr2014/A04/frames/x40/"
	name = file + ".tiff"
	src = cv2.imread(dir+name, 1)
	i = 0
	for x in res:
		dst = src[int(x[1]):int(x[1])+100, int(x[0]):int(x[0])+100]
		name = str(i) + name
		cv2.imwrite(name,dst)
		i += 1

def read_csv(file):
	print file
        f = open(file)
	data = f.read()
	if not data:
		print "no mitosis"
		return 1
        lines = data.split("\n")
	res = []
        for x in lines:
		if x:
                	tmp = x.split(",")
			print tmp
                	res_tmp = []
			res_tmp.append(tmp[0])
			res_tmp.append(tmp[1])
			res.append(res_tmp)
	return res
		

def read_foldr(foldr):
	files = os.listdir(foldr)
	current = os.getcwd()

	for file in files:
		if int(file.find("not")) == int(-1) and int(file.find("jpg")) == int(-1):
			tmp = foldr + "/"
			res = read_csv(tmp+file)
			if res != 1:
				cut_mitosis(file[0:8],res)
		print file[0:8]


if __name__ == "__main__":
	read_foldr("/home/keiohigh2nd/icpr2014/A04/mitosis")
	
