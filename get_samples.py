import random

#This script makes random dataset(rand_**Network.csv) from **_Network.csv

def get_sample(num, max, file):
  f = open(file)
  lines = f.readlines()
  f.close()

  tmp_file = "rand_" + file
  f1 = open(tmp_file, "w")

  for x in xrange(num):
    tmp = random.randint(0,max)
    t = lines[tmp].split(",")
    if int(t[0].strip("G")) < max and  int(t[1].strip("G")) < max:
    	f1.write(lines[tmp])

  f1.close()

def make_train_and_test(ratio, file):
  tmp_file = "rand_" + file
  f = open(tmp_file)
  lines = f.readlines()
  f.close()

  test_file = "test_" + tmp_file
  test = open(test_file, "w")

  train_file = "train_" + tmp_file
  train = open(train_file, "w")

  i = 0
  max = len(lines)*ratio
  print max
  for i in xrange(len(lines)):
	if i < max:
		train.write(lines[i])
	else:
		test.write(lines[i])

  test.close()
  train.close()      

if __name__ == "__main__":
  #get_sample(100,400, "1_DREAM5_NetworkInference_GoldStandard_Network3.csv")
  #get_sample(100,400, "0_DREAM5_NetworkInference_GoldStandard_Network3.csv")
 
  make_train_and_test(0.8, "1_DREAM5_NetworkInference_GoldStandard_Network3.csv") 
