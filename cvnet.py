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

def read_foldr(foldr):
	files = os.listdir(foldr)
	current = os.getcwd()

	size = 20*20
	train = numpy.array(numpy.zeros(size))

	print current
	for file in files:
		train = convert_img_to_nparray(current+file,train)

	train = numpy.delete(train,0,0)
	return train

if __name__ == "__main__":
        #dbn.test_dbn()

	read_foldr("/home/keiohigh2nd/dpl")

