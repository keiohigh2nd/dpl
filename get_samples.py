import random

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
    
if __name__ == "__main__":
  get_sample(100,400,"1_DREAM5_NetworkInference_GoldStandard_Network3.csv")
  
