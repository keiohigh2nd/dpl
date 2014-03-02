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
 
 
    def finetune(self, lr=0.1, epochs=100):
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

def limited_for_train(a,b):
	if a < 30 and b < 30:
		return 1
	else:
		return 0

def gene_data():
	import itertools

   	#convert()
        f = open("Nw1_Ex_10.csv")
        tmp_f = f.read()
	lines = tmp_f.split("\r")
        f.close()
        tm = lines[0].split(",")
        cols = len(tm)
        print "Number of Genes = %d"%cols

        #You cannot change this part because of something excel error
        f1 = open("Nw1_G_10.csv")
	tmp_f1 = f1.read()
        lins = tmp_f1.split("\r")
        f1.close()
        combi = list(itertools.combinations(range(cols), 2))
        
	
	train = numpy.zeros(len(lines))
        res_train = numpy.array([[0, 0]])
	
	#res_train = numpy.concatenate((res_train,numpy.array([[1,0]])),axis=0)
	print res_train
	#res_train = numpy.concatenate((res_train,numpy.array([[1,0]])))
         
	i = 0
	for y in lins:
                tmp =  y.split(",")
                #print chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)
                #numpy.column_stack((train,chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)))
                train = numpy.concatenate((train,chip_data_dif(int(tmp[0].strip("G"))-1,int(tmp[1].strip("G"))-1)),axis=0)
                if int(tmp[2].strip()) == 0:
			res_train = numpy.concatenate((res_train,numpy.array([[1,0]])))
                else:
                        res_train = numpy.concatenate((res_train,numpy.array([[0,1]])))
                i += 1
	print res_train	
	return train,res_train

def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):

    #x,y = gene_data()

    
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
 
    

    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=x, label=y, n_ins=9, hidden_layer_sizes=[6, 6], n_outs=2, numpy_rng=rng)

 
    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
 
 
    # test
    # 1,0,1
    #x = numpy.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    x = numpy.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    #x = numpy.array([0, 0, 1, 0, 1, 0, 0, 1, 1])
    
    print dbn.predict(x)
 
 
if __name__ == "__main__":
    test_dbn()
