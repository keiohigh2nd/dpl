
for i in `seq 1 100`
  do
    python2.7 get_samples.py
    python2.7 sda_dropout.py

    rm test_Nw3.csv
    rm train_Nw3.csv
  done

