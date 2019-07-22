# From https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

'''
STEP 1: LOADING DATASET
'''

train_dataset = dsets.CIFAR10(root='./data', 
							train=True, 
							transform=transforms.ToTensor(),
							download=True)

test_dataset = dsets.CIFAR10(root='./data', 
						   train=False, 
						   transform=transforms.ToTensor())


'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 3000
hidden_layer_num = 10
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
										   batch_size=batch_size, 
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
										  batch_size=batch_size, 
										  shuffle=False)


'''
STEP 3: CREATE MODEL CLASS
'''
class FeedforwardNeuralNetModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(FeedforwardNeuralNetModel, self).__init__()
		# Linear function 1: 32x32x3 --> 100
		self.input = nn.Linear(input_dim, hidden_dim) 
		
		self.hidden = nn.ModuleList()
		for k in range((hidden_layer_num-1)):
			self.hidden.append(nn.Linear(hidden_dim,hidden_dim))
		self.output = nn.Linear(hidden_dim, output_dim)


	def forward(self, x):
		# Feed Forward
		x = self.input(x)
		for layer in self.hidden:
			x = torch.nn.functional.relu(layer(x))
		output = torch.nn.functional.softmax(self.output(x), dim = 1)
		return output



'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 32*32*3
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print (torch.cuda.is_available())

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()


'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.0001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 32*32*3
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print (torch.cuda.is_available())

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()


'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.0001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        #######################
        #  USE GPU FOR MODEL  #
        #######################
        images = images.view(-1, 32*32*3).requires_grad_().to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                images = images.view(-1, 32*32*3).requires_grad_().to(device)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                #######################
                #  USE GPU FOR MODEL  #
                #######################
                # Total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))



