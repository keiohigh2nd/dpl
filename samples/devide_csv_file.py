
#This script devides the fiele into 1 class and 0  class
def extract_file(num):
  file = "DREAM5_NetworkInference_GoldStandard_Network3.csv"

  f = open(file)
  data = f.read()
  lines = data.split("\r")
  f.close

  tmp = str(num) + "_" + file
  f1 = open(tmp, "w")
  for line in lines:
    tmp = line.split(",")
    if int(tmp[2].strip("\r")) == int(num):
	f1.write(line)
	f1.write("\n")
    
  f1.close()

if __name__ == "__main__":
  extract_file(1)
  extract_file(0)

