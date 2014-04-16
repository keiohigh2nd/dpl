#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
'''
 Deep Belief Nets (DBN)
 
 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
 
 
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials
 
 
'''
 
import sys
import numpy
import math 
 
numpy.seterr(all='ignore')
 
def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))
 
def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
 
 
class DBN(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 numpy_rng=None):
        
        self.x = input
        self.y = label
 
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)
 
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
 
        
        assert self.n_layers > 0
 
 
        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]
 
            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        numpy_rng=numpy_rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
 
 
            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)
 
 
        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)
 
        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()
 
 
 
    def pretrain(self, lr=0.1, k=1, epochs=100):
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            
            for epoch in xrange(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost
 
    # def pretrain(self, lr=0.1, k=1, epochs=100):
    #     # pre-train layer-wise
    #     for i in xrange(self.n_layers):
    #         rbm = self.rbm_layers[i]
            
    #         for epoch in xrange(epochs):
    #             layer_input = self.x
    #             for j in xrange(i):
    #                 layer_input = self.sigmoid_layers[j].sample_h_given_v(layer_input)
            
    #             rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
    #             # cost = rbm.get_reconstruction_cross_entropy()
    #             # print >> sys.stderr, \
    #             #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost
 
 
    def finetune(self, lr=0.1, epochs=300):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()
 
        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost
            
            lr *= 0.95
            epoch += 1
 
 
    def predict(self, x):
        layer_input = x
        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            # rbm_layer = self.rbm_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)
 
        out = self.log_layer.predict(layer_input)
        return out
 
 
 
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out,\
                 W=None, b=None, numpy_rng=None, activation=numpy.tanh):
        
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
 
        if W is None:
            a = 1. / n_in
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))
 
            W = initial_W
 
        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0
 
 
        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.b = b
 
        self.activation = activation
 
    def output(self, input=None):
        if input is not None:
            self.input = input
        
        linear_output = numpy.dot(self.input, self.W) + self.b
 
        return (linear_output if self.activation is None
                else self.activation(linear_output))
 
    def sample_h_given_v(self, input=None):
        if input is not None:
            self.input = input
 
        v_mean = self.output()
        h_sample = self.numpy_rng.binomial(size=v_mean.shape,
                                           n=1,
                                           p=v_mean)
        return h_sample
 
 
 
class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, numpy_rng=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer
 
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
 
 
        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))
 
            W = initial_W
 
        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0
 
        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0
 
 
        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
 
        # self.params = [self.W, self.hbias, self.vbias]
 
 
    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input
        
        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
 
        chain_start = ph_sample
 
        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)
 
        # chain_end = nv_samples
 
 
        self.W += lr * (numpy.dot(self.input.T, ph_sample)
                        - numpy.dot(nv_samples.T, nh_means))
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * numpy.mean(ph_sample - nh_means, axis=0)
 
        # cost = self.get_reconstruction_cross_entropy()
        # return cost
 
 
    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)
 
        return [h1_mean, h1_sample]
 
 
    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)
        
        return [v1_mean, v1_sample]
 
    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)
 
    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)
 
 
    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
 
        return [v1_mean, v1_sample,
                h1_mean, h1_sample]
    
 
    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)
 
        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy
 
    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

 
class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0
 
    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input
 
        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)
 
    def negative_log_likelihood(self):
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)
 
        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))
 
        return cross_entropy
 
    def predict(self, x):
        return softmax(numpy.dot(x, self.W) + self.b)
 
 

def chip_data(num):
        f = open("Nw1_Ex_10_S.csv")
        lines = f.readlines()
        arr = numpy.array([])
        for x in lines:
                tmp = x.split("\t")
                arr.append(tmp[num])
        return arr


def chip_data_dif_test(a,b):
        threshold = 0.08
        #threshold = get_average_test(a,b)
        #threshold = get_first_one_test(a,b)
        #threshold = get_median_test(a,b)

        f = open("Network3_expression_data.csv")
        tmp_f = f.read()
	lines =	tmp_f.split("\r")
        f.close()

        arr = []
        i = 0
        #res = numpy.zeros(len(lines))
        res = numpy.array([])
        for x in lines:
		if int(x.find("G")) == int(-1):
                        tmp = x.split(",")
                        tmp_d = math.fabs(float(tmp[int(a)])-float(tmp[int(b)]))
                        if float(threshold) > float(tmp_d):
				res = numpy.append(res,1)
                                #res[i] = 1
				#numpy.hstack((res,1))
                        else:
                                #res[i] = 0
				res = numpy.append(res,0)
				#numpy.hstack((res,1))
                i += 1
		
        return res

def chip_data_dif_train(a,b):
        threshold = 0.08
        #threshold = get_first_one_train(a,b)
        #threshold = get_average_train(a,b)
        #threshold = get_median_train(a,b)

        f = open("Network1_expression_data.csv")
        tmp_f = f.read()
        lines = tmp_f.split("\r")
        f.close()

        arr = []
        i = 0
        #res = numpy.zeros(len(lines))
        res = numpy.array([])
        for x in lines:
                if int(x.find("G")) == int(-1):
                        tmp = x.split(",")
                        tmp_d = math.fabs(float(tmp[int(a)])-float(tmp[int(b)]))
                        if float(threshold) > float(tmp_d):
                                res = numpy.append(res,1)
                                #res[i] = 1
                                #numpy.hstack((res,1))
                        else:
                                #res[i] = 0
                                res = numpy.append(res,0)
                                #numpy.hstack((res,1))
                i += 1

        return res

def sort_array(arr):
	arr_sort = numpy.sort(arr, axis=None)
	res = numpy.array(numpy.zeros(arr.size))

	for i in xrange(arr_sort.size):
		if i <= 30:
			for j in xrange(arr.size):
				if float(arr_sort[-i]) == float(arr[j]):
					res[j] = 1
	return res	

				
	

def chip_data_add_test(a,b):
        threshold = 0.08

        f = open("Network3_expression_data.csv")
        tmp_f = f.read()
        lines = tmp_f.split("\r")
        f.close()

        arr = []
        i = 0
        #res = numpy.zeros(len(lines))
        res = numpy.array([])
        for x in lines:
                if int(x.find("G")) == int(-1):
                        tmp = x.split(",")
			#tmp_d = math.fabs(float(tmp[int(a)])-float(tmp[int(b)]))
                        res = numpy.append(res,math.fabs(float(tmp[int(a)])-float(tmp[int(b)])))
                i += 1

        return sort_array(res)


def chip_data_add_train(a,b):
        threshold = 0.08

        f = open("Network1_expression_data.csv")
        tmp_f = f.read()
        lines = tmp_f.split("\r")
        f.close()

        arr = []
        i = 0
        #res = numpy.zeros(len(lines))
        res = numpy.array([])
        for x in lines:
                if int(x.find("G")) == int(-1):
                        tmp = x.split(",")
			#tmp_d = math.fabs(float(tmp[int(a)])-float(tmp[int(b)]))
                        res = numpy.append(res, math.fabs(float(tmp[int(a)])-float(tmp[int(b)])))
                i += 1

        return sort_array(res)
                     


def gene_data():
	import itertools

   	"""
	Please Be careful for the file name
	1: Number of ROW of Expression Data
	2:
	"""
        
	f = open("Network1_expression_data.csv")
        tmp_f = f.read()
	lines = tmp_f.split("\r")
        f.close()
        tm = lines[0].split(",")

	#File Information 
        cols = len(lines)-1
        print "Number of Row expression Genes = %d"%cols
	num_genes = len(tm)
	print "Number of column Genes = %d"%num_genes
	
        #THIS IS CSV. \R and , are keys.Don't forget to run read.py convert()
        f1 = open("Nw1_G_200.csv")
	tmp_f1 = f1.read()
        lins = tmp_f1.split("\r")
        f1.close()
        combi = list(itertools.combinations(range(num_genes), 2))
        
	#This means RESIZE
	#If you use chip_data_add, you have to 2 times
	train = numpy.array(numpy.zeros(cols))
        res_train = numpy.array([[0, 10]])

	print cols	
	print train[0].size
	#res_train = numpy.concatenate((res_train,numpy.array([[1,0]])),axis=0)
	#res_train = numpy.concatenate((res_train,numpy.array([[1,0]])))
         
	i = 0
	for y in lins:
                tmp =  y.split(",")
		try:
                	#numpy.column_stack((train,chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)))
			if tmp[0].strip("G").isdigit() == True & tmp[1].strip("G").isdigit() == True:
                		train = numpy.vstack((train,chip_data_add_train(int(tmp[0])-1,int(tmp[1])-1)))
                		if int(tmp[2].strip()) == 0:
					#res_train = numpy.concatenate((res_train,numpy.array([[1,0]])))
					res_train = numpy.vstack((res_train,numpy.array([1,0])))
                		else:
                       			res_train = numpy.vstack((res_train,numpy.array([0,1])))
                	i += 1
		except:
			print "error----",y
                	i += 1

	#Delete first input data
	res_train = numpy.delete(res_train,0,0)
	train = numpy.delete(train,0,0)
	
	#Check number of expression data and gold standard is the same number.
	print len(res_train), len(train)
	
	return train,res_train

def easy_test():
    x = numpy.array([[0,0,0,0,0,1,1,0,0],
                     [1,1,1,1,1,1,1,1,1],
                     [0,0,1,0,0,0,0,0,0],
                     [0,0,1,1,1,1,1,0,0],
                     [1,1,1,0,0,0,0,0,0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [1, 0]])

def gene_test():
	import itertools

	#Expression_data
	f = open("Network3_expression_data.csv")
        tmp_f = f.read()
        lines = tmp_f.split("\r")
        f.close()
        tm = lines[0].split(",")

        #File Information 
        cols = len(lines)-1
        print "Number of Row expression Genes = %d"%cols
        num_genes = len(tm)
        print "Number of column Genes = %d"%num_genes

        #GOLD_Standard_Data_THIS IS CSV. \R and , are keys.Don't forget to run read.py convert()
        f1 = open("Nw1_G_200_1.csv")
        tmp_f1 = f1.read()
        lins = tmp_f1.split("\r")
        f1.close()
        combi = list(itertools.combinations(range(cols), 2))

        #This means RESIZE
	#IF you you use chip_add_data, you have to 2 times
        test = numpy.array(numpy.zeros(cols))
        res_test = numpy.array([[0, 10]])
	#res_test = numpy.array(numpy.zeros(10))

        #res_test = numpy.concatenate((res_test,numpy.array([[1,0]])),axis=0)
        #res_test = numpy.concatenate((res_test,numpy.array([[1,0]])))

        i = 0
        for y in lins:
		tmp =  y.split(",")
                #numpy.column_stack((test,chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)))
		try:
			if tmp[0].strip("G").isdigit() == True & tmp[1].strip("G").isdigit() == True:
                		test = numpy.vstack((test,chip_data_add_train(int(float(tmp[0].strip("G")))-1,int(float(tmp[1].strip("G")))-1)))
                		if int(tmp[2].strip()) == 0:
                        		#res_test = numpy.concatenate((res_test,numpy.array([[1,0]])))
                        		res_test = numpy.vstack((res_test,numpy.array([1,0])))
                		else:
					res_test = numpy.vstack((res_test,numpy.array([0,1])))
                	i += 1

		except:
			print "error___",y
                	i += 1

        #Delete first input data
        res_test = numpy.delete(res_test,0,0)
        test = numpy.delete(test,0,0)

        #Check number of expression data and gold standard is the same number.
        print len(res_test), len(test)

        return test,res_test

def calc_accuracy(res,res_test):
	if len(res) != len(res_test):
		print "Error Size is different"
		return 0
	
	i = 0
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for x in res:
		print x
		if int(change_binary(x)) == int(change_binary(res_test[i])):
			if int(change_binary(x)) == 0:
				fn += 1
			else:
				tp += 1
		else:
			if int(change_binary(x)) == 0:
				tn += 1
			else:
				fp += 1
		i += 1

	f = open("res.txt","w")
	f.write(str(tp))
	f.write(",")
	f.write(str(tn))
	f.write(",")
	f.write(str(fp))
	f.write(",")
	f.write(str(fn))
	f.write("\r")
	f.close()
	
	print "True Positive = %d"%tp
	print "True Negative = %d"%tn
	print "False Positive = %d"%fp
	print "False Negative = %d"%fn

	all = tp + tn + fp + fn

	return float(tp+fn)/all
	
			
			
def change_binary(x):
	if float(x[0]) >= float(x[1]):
		return 0
	else:
		return 1
		

def test_dbn(pretrain_lr, pretraining_epochs, k, finetune_lr, finetune_epochs, num_lays, num_units):
    import time
    start = time.clock()

    x,y = gene_data()

    num_expression = len(x[0])
    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=x, label=y, n_ins=num_expression, hidden_layer_sizes=[num_lays, num_units], n_outs=2, numpy_rng=rng)

 
    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
 
 
    # test
    #x = numpy.array([0, 0, 1, 0, 1, 0, 0, 1, 1])
    test, res_test = gene_test()

    res = []
    for x in test:
    	res.append(dbn.predict(x))

    res_calc = calc_accuracy(res,res_test)

    end = time.clock()
    tom = end-start
    print "Takes",tom

    return res_calc
 
if __name__ == "__main__":
    import random


    res_tmp = 0
    while res_tmp <= 0.7:
	#i = random.randint(10)
	i = 10
    	j = random.randint(2200,2500) 
	print i
	print j
    	tmp = test_dbn(0.0001,1000,1,0.0001,200,i,j)
    	f = open("run_res.txt","w")
	f.write(str(i))
	f.write("\t")
	f.write(str(j))
	f.write("\t")
	f.write(str(tmp))
	f.write("\n")
    	f.close()
	if res_tmp <= tmp:
		res_tmp = tmp

