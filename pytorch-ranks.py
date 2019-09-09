import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import powerlaw
import sklearn
from sklearn.decomposition import TruncatedSVD


import pandas as pd 
import matplotlib.pyplot as plt






# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.1
depth = 10

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)



def analyze_model(model, plot=True):
    min_svs = []
    rank_loss = []
    alphas = []
    Qs = []
    matrix_ranks = []
    
    pretrained_model = model()
    
    model_name = model.__name__
    model_name = model_name.replace("_",' ')
    
    for im, m in enumerate(pretrained_model.modules()):
        if isinstance(m, nn.Linear):
                
            W = np.array(m.weight.data.clone().cpu())
            M, N = np.min(W.shape), np.max(W.shape)
            Q=N/M 
            
            Qs.append(Q)

            sv, _ = RMT_Util.singular_spectrum(W)
            smallest_sv = np.min(sv)
            min_svs.append(smallest_sv)
            
            rank = np.linalg.matrix_rank(W)
            rank_loss.append(100*(M-rank)/M)
            
            if ((plot) and (Q > 1.2)):
                print(m)
                plt.hist(sv*sv, bins=100)
                plt.title(" N {} M {} Rank {}".format(N,M,rank))
                plt.show()
                
    return min_svs, rank_loss, Qs


# Xavier Initialization on every layer
def xavier_initialization(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth):
        super(NeuralNet, self).__init__()
        self.input = nn.Linear(input_size, hidden_size) 
        self.hidden = nn.ModuleList()
        for k in range(depth-1):
            self.hidden.append(nn.Linear(hidden_size,hidden_size))
        self.output = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.input(x)
        for layer in self.hidden:
            out = torch.tanh(layer(out))
            # out = torch.relu(layer(out))
            # out = layer(out)
        out = self.output(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes, depth).to(device)
model.apply(xavier_initialization)
print (model)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()            
        optimizer.step()

    grads_gradient = {}
    grads_list = []
    weight_list = []
    weight_dict = []
#         for param in model.named_parameters():
#             print (param[0])
#         for name, para in (model.named_parameters()):
#             grads_gradient[name] = para.grad
    for name,param in model.named_parameters():
        # print ("Gradient in ",epoch, " epoch = ",param.grad)
        # # print ("Name in ",epoch, " epoch = ",param.name)
        # print ("Data in ",epoch, " epoch = ",param.data)
        # print ("Mean of weight = ",param.data.mean())
        # print ("Mean of grad = ",param.grad.mean())

        grads_list.append(param.grad)
        weight_list.append(param.data)

        # weight_sum = torch.sum(model.hidden.weight.data)
        # print ("Weight sum = ",weight_sum)
grads_list = grads_list[1::2]
weight_list = weight_list[1::2]
print ("Len of grads_list = ", len(grads_list))
print ("Len of weight list = ", len(weight_list))



# for index in model.modules():
#     print ("Index weight data = ",index.weight.data)

all_min_svs = []
all_rank_loss = []
all_alphas = []
all_Qs = []

#print(model)
min_svs, rank_loss, Qs  = analyze_model(model, plot=True)
all_min_svs.extend(min_svs)
all_rank_loss.extend(rank_loss)
#all_alphas.extend(alphas)
all_Qs.extend(Qs)




# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


def mean_abs(matrix):
    return torch.abs(matrix.double()).mean()
def variance(matrix):
    return torch.var(matrix)


mean_dic = {}
var_dic = {}

mean_list = []
var_list = []

for index in grads_gradient:
    mean_dic[index] = mean_abs(grads_gradient[index])
    var_dic[index] = variance(grads_gradient[index])

for index in grads_list:
    mean_list.append(mean_abs(index))
    var_list.append(variance(index))
# mean_list[0], mean_list[-1] = mean_list[-1], mean_list[0]
# var_list[0], var_list[-1] = var_list[-1], var_list[0]
# print ("Mean list = ",mean_list)
# print ("Variance list = ", var_list)

# For the output layer is exponetially greater than the rest
mean_no_out = mean_list[:-1]
var_no_out = var_list[:-1]

print ("Len of mean_list = ",len(mean_list) ,"\nLen of mean_list_without_output = " ,len(mean_no_out))
# print ("Mean of weights = ",mean_dic)
# print ("Variance of weights = ",var_dic)
# for index in grads_gradient:
#     print (grads_gradient[index])



x_axis = list(range(depth+1))
plt.plot(x_axis,mean_list,label='mean')
# plt.yscale('log')
# plt.savefig('100_depth_weight_mean_without_xavier_linear_activation')
plt.plot(x_axis,var_list,label='variance')
plt.legend(loc='upper left')
plt.yscale('log')
plt.savefig('./graph/MNIST_%d_depth_weight_mean_variance_with_xavier_linear_activation' % depth)





