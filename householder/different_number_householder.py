import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import scipy
import pickle
import sys
from time import process_time

import inspect

# Parameters
CTX = mx.gpu(0)

NUM_LAYER = 100
NUM_UNITS = 500
LEARNING_RATE = 1e-4
INPUT_STD = 1
OUTPUT_STD = 1
ACT = 'tanh'
NUM_LAYER_LIST = [500]
LEARNING_RATE_LIST = [LEARNING_RATE]
NUM_HOUSEHOLDER_V = 20

n_epoch = 100
trial_times = 100
plot_batch = 1

def dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def Rsq(J):
    K = J@J.T
    sum_a = np.diag(K@K).sum()
    sum_b = np.diag(K).sum()
    return sum_a/sum_b**2, sum_a, sum_b

def eye(F, N):
    return F.one_hot(F.arange(N, ctx=CTX), N)

class Householder(mx.gluon.HybridBlock):
    def __init__(self, N, use_bias=True, dtype='float32', bias_initializer='zeros', **kwargs):
        super(Householder, self).__init__(**kwargs)
        with self.name_scope():
            self.N = N
            self.num_v = NUM_HOUSEHOLDER_V
            # Number of Householder vectors in a layer
            self.v_concat = self.params.get('v_concat',
                shape=(N, self.num_v), init=mx.initializer.Normal(), dtype=dtype)
            if use_bias:
                self.bias = self.params.get('bias', shape=(N,), init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None

    def hybrid_forward(self, F, x, v_concat, bias=None):
        I = eye(F, self.N)
        for iV in range(self.num_v):
            v = F.slice(v_concat, begin=(None, iV), end=(None, iV+1))
            vv = F.broadcast_div(F.dot(v, F.transpose(v)), F.dot(F.transpose(v), v))
            H = I - 2*vv
            x = F.FullyConnected(x, H, None, no_bias=True, num_hidden=self.N, flatten=True, name='fwd%d'%iV)
        x = x + bias
        return x

    def __repr__(self):
        s = '{name}({N}, linear)'
        return s.format(name=self.__class__.__name__,
                        N=str(self.N))

class MLP(gluon.Block):
    def __init__(self, num_layer=10, num_units=500, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            data = mx.sym.var('input')
            self.dense_list = [mx.gluon.nn.Dense(num_units, use_bias=True)]
            for iL in range(num_layer-1):
                self.dense_list.append(
                    Householder(num_units, use_bias=True)
                )
            self.dense_list.append(mx.gluon.nn.Dense(10, use_bias=True))
        for dense in self.dense_list:
            self.register_child(dense)
        self.input = None
        self.hidden_list = []

    def forward(self, x):
        self.input = x
        self.input.attach_grad()
        self.hidden_list = []
        for dense in self.dense_list[0:-1]:
            # x = dense(x)
            x = dense(x)
            # x.attach_grad()
            self.hidden_list.append(x)
            if ACT == 'relu':
                x = nd.relu(x)
            elif ACT == 'tanh':
                x = nd.tanh(x)
            elif ACT == 'hard-tanh':
                x = 2*nd.hard_sigmoid(x, alpha=0.5, beta=0.5)-1
        x = self.dense_list[-1](x)
        # x.attach_grad()
        self.hidden_list.append(x)
        return x


# Data
def data_xform(data):
    return nd.moveaxis(data, 2, 0).astype('float32') / 255
mnist = mx.test_utils.get_mnist()
train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=100)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=100)
all_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=50000)

metric = mx.metric.create([mx.metric.Accuracy(), mx.metric.CrossEntropy()])
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()

# Record for different activation
ln_net = MLP(num_layer=NUM_LAYER, num_units=NUM_UNITS)
n_seed = 1
n_lr = 1
lr_range = [LEARNING_RATE]
batch_size = 100
rsq_period = 100

rsq_array = np.zeros((n_seed, n_lr, 600*n_epoch//rsq_period))
BRA = {}
start = process_time()

for seed in range(0, n_seed):
    print('Seed %d'%seed)
    # weight_container = np.zeros((n_lr, n_epoch*600//rsq_period, NUM_LAYER-2, NUM_UNITS, NUM_UNITS))
    for iA, lr in enumerate(lr_range):
        print('Learning rate: %2E'%lr)
        mx.random.seed(seed)
        np.random.seed(seed)
        ln_net.collect_params().initialize(
            mx.init.Xavier(rnd_type='gaussian', magnitude=2 if ACT == 'relu' else 1),ctx=CTX, force_reinit=True)
        opt = mx.optimizer.SGD(learning_rate=lr)
        trainer = gluon.Trainer(ln_net.collect_params(), opt)

        for e in range(n_epoch):
            print('Epoch %d' %e)
            sys.stdout.flush()
            b = 0
            for inputs, labels in train_loader:
                inputs = inputs.as_in_context(CTX)
                labels = labels.as_in_context(CTX)

                with autograd.record():
                    outputs = ln_net(inputs)
                    loss = loss_function(outputs, labels)

                loss.backward()
                trainer.step(batch_size=inputs.shape[0])

                if b % rsq_period == 0:
                    metric.reset()
                    val_output_list = []
                    for inputs, labels in val_loader:
                        inputs = inputs.as_in_context(CTX)
                        labels = labels.as_in_context(CTX)
                        outputs = ln_net(inputs)
                        metric.update(labels, mx.nd.softmax(outputs))
                        val_output_list.append(outputs.asnumpy())
                    te_metric = metric.get()[1]
                    Y = np.concatenate(val_output_list, axis=0)
                    Y = (Y-np.mean(Y, axis=0)).T
                    rsq_y = Rsq(Y)[0]
                    rsq_array[seed, iA, (e*600+b)//rsq_period] = rsq_y
                    print('Batch: %d, Rsq: %.6f, Acc: %.3f' %(b, rsq_y, te_metric[0]))
                    BRA = {'Batch':e*600+b,'Rsq':rsq_y,'Acc':te_metric[0]}
                    file = str(str(seed) + '_' + str(e) + '_' + str(b) + '.pickle')
                    dump(BRA, file)
                    sys.stdout.flush()

                    for layer_index in range(1, NUM_LAYER-1):
                        # weight_container[seed, iA, (e*600+b)//rsq_period, layer_index-1, :, :] = ln_net.dense_list[layer_index].weight.data().asnumpy()
                        pass
                b += 1
            # pickle.dump(weight_container, open('weight_seed%d.pickle'%seed, 'wb'))

    print('\nNetwork width = %d\nNetwork depth = %d\nNonlinear type = %s\nLearning rate = %.2E\nBatch size = %d'
         %(NUM_UNITS, NUM_LAYER, "Tanh(SGD)", LEARNING_RATE, batch_size))

end = process_time()
time = abs(start - end)
print("time:", time)