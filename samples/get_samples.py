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

  test_file = "test_Nw3.csv" 
  test = open(test_file, "a+")

  train_file = "train_Nw3.csv"
  train = open(train_file, "a+")

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
  Num_Paths = 200
  Num_Genes = 500

  get_sample(Num_Paths, Num_Genes, "1_DREAM5_NetworkInference_GoldStandard_Network3.csv")
  get_sample(Num_Paths, Num_Genes, "0_DREAM5_NetworkInference_GoldStandard_Network3.csv")

  ##File is always overwritten. If you change the parameter, please remove test and train_Nw3.csv
  train_ratio = 0.8 
  make_train_and_test(train_ratio, "1_DREAM5_NetworkInference_GoldStandard_Network3.csv") 
  make_train_and_test(train_ratio, "0_DREAM5_NetworkInference_GoldStandard_Network3.csv") 
