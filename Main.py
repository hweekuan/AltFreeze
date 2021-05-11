#!/usr/bin/env python
# coding: utf-8

#import key packages
import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from Blocks import EnsembleModel
from Blocks import CNNBlock1
from Blocks import CNNBlock2
from Blocks import CNNBlock3
from Blocks import MLPBlock
#set seed for reproducibility, could do extra for cuda but would slow performance
random.seed(12345)
torch.manual_seed(12345)
np.random.seed(12345)
device = torch.device("cuda:0")

#set some parameters here

learnrate = 0.001
OPTIM = 'ADAM default'
activation = 'ReLU'
nepochs = 100

#downloading the data
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# In[3]:

train_len = len(trainset)
test_len = len(testset)
index = list(range(train_len)) 
print(train_len, test_len)


# shuffle data for "randomness" 
np.random.shuffle(index)


#Generate training sets, validations sets with no overlap together with test sets
split1 = int(0.45*train_len)
split2 = int(0.9*train_len)
train_index1 = index[:split1]
train_index2 = index[split1:2*split1]
val_index = index[split2:]
index2 = list(range(test_len))
np.random.shuffle(index2)
split3 = int(0.1 * test_len)
test_index = index2[:split3]
train_loader1 = torch.utils.data.DataLoader(trainset, sampler = train_index1, batch_size = 50, num_workers = 8)  #batch size 10 because when it was 100 had memory issues
train_loader2 = torch.utils.data.DataLoader(trainset, sampler = train_index2, batch_size = 50, num_workers = 8)  #batch size 10 because when it was 100 had memory issues
val_loader = torch.utils.data.DataLoader(trainset, sampler = val_index, batch_size = 50, num_workers = 8)
test_loader = torch.utils.data.DataLoader(testset, sampler = test_index)  #test set for running every epoch needs to be small
test_loader_big = torch.utils.data.DataLoader(testset)



#size of the sets
trainset1_size = len(train_index1)
trainset2_size = len(train_index1)
val_size = len(val_index)


#Load the Model
ensemblemodel = EnsembleModel()
ensemblemodel.to(device)

optimizer = optim.Adam(ensemblemodel.parameters(), lr = learnrate)
print('optimizer', optimizer)


# In[15]:


criterion = nn.CrossEntropyLoss()


# In[10]:

#training
trainingloss = []
validationloss = []
testaccuracy = []
#print("Time = " time.perf_counter())
for epoch in range(nepochs):
	ensemblemodel.train()
	running_loss = 0.0
	
	if epoch%2 == 0 & epoch > 75:
		for param in cnnblock1.parameters():
			param.requires_grad_(False)
        
		for param in cnnblock2.parameters():
        		param.requires_grad_(False)
        
		for param in cnnblock3.parameters():
			param.requires_grad_(True)
        
		for param in mlpblock.parameters():
			param.requires_grad_(True)
        

		for i, data in enumerate(train_loader1,0):
			inputs, set2labels = data[0].to(device), data[1].to(device)
	            
			optimizer.zero_grad()
	            
			outputs = ensemblemodel(inputs)
			loss = criterion(outputs, set2labels)
			loss.backward()
			optimizer.step()
	            #print stats
			running_loss += loss.item()
			#Training loss once at the end of each epoch
			if i%450 == 449:
				trainingloss.append(running_loss/450)
				print(running_loss/450)
				running_loss = 0.0
			#if i%4500 == 4499:
				#trainingloss.append(running_loss/4500)
				#print(running_loss/4500)
				#running_loss = 0.0
	
		#Validation loss once at end of epoch
	                
        
	if epoch%2 == 1 & epoch > 75:
		for param in cnnblock1.parameters():
			param.requires_grad_(True)
        
		for param in cnnblock2.parameters():
        		param.requires_grad_(True)
        
		for param in cnnblock3.parameters():
			param.requires_grad_(False)
        
		for param in mlpblock.parameters():
			param.requires_grad_(False)

		
		for i, data in enumerate(train_loader2,0):
			inputs, set2labels = data[0].to(device), data[1].to(device)
	            
			optimizer.zero_grad()
	            
			outputs = ensemblemodel(inputs)
			loss = criterion(outputs, set2labels)
			loss.backward()
			optimizer.step()
	            #print stats
			running_loss += loss.item()
			#Training loss once at the end of each epoch
			if i%450 == 449:
				trainingloss.append(running_loss/450)
				print(running_loss/450)
				running_loss = 0.0
			#if i%4500 == 4499:
				#trainingloss.append(running_loss/4500)
				#print(running_loss/4500)
				#running_loss = 0.0
	
		#Validation loss once at end of epoch

	if epoch <= 75:
#		for param in cnnblock1.parameters():
#			param.requires_grad_(True)
        
#		for param in cnnblock2.parameters():
#        		param.requires_grad_(True)
        
#		for param in cnnblock3.parameters():
#			param.requires_grad_(True)
        
#		for param in mlpblock.parameters():
#			param.requires_grad_(True)

		
		for i, data in enumerate(train_loader1,0):
			inputs, set2labels = data[0].to(device), data[1].to(device)
	            
			optimizer.zero_grad()
	            
			outputs = ensemblemodel(inputs)
			loss = criterion(outputs, set2labels)
			loss.backward()
			optimizer.step()
	            #print stats
			running_loss += loss.item()
			#Training loss once at the end of each epoch
			if i%450 == 449:
				trainingloss.append(running_loss/450)
				print(running_loss/450)
				running_loss = 0.0
			#if i%4500 == 4499:
				#trainingloss.append(running_loss/4500)
				#print(running_loss/4500)
				#running_loss = 0.0
	
	ensemblemodel.eval()
	running_loss2 = 0.0
	for i,data in enumerate(val_loader): 
		inputs,vallabels = data[0].to(device),data[1].to(device)
		outputs = ensemblemodel(inputs)
		lloss = criterion(outputs, vallabels)   	
                
		running_loss2 += lloss.item()
		if i%100 == 99:
			validationloss.append(running_loss2/100)
			print(running_loss2/100)
			running_loss2 = 0.0
                        
        #Provides test accuracy at each epoch, 10% of test set     
        # set to testing                           
	correct_count,all_count = 0,0
	for i, data in enumerate(test_loader,0):
		inp,labels = data[0].to(device), data[1].to(device)
		with torch.no_grad():
			logps = ensemblemodel(inp)
		
		ps = torch.exp(logps)
		ps = ps.cpu()
		probab = list(ps.numpy()[0])
		pred_label = probab.index(max(probab))
		true_label = labels.cpu()
		if (true_label == pred_label):
			correct_count +=1
		all_count +=1
		
	print("\nModel Accuracy =", (correct_count/all_count))
	testaccuracy.append(correct_count/all_count)
	print(epoch)
	#print("Time = " time.perf_counter())

print("finished training")

torch.save(ensemblemodel.state_dict(), '/home/brian_chen/mytitandir/ensemblemodel.pth')

fig1, ax1 =plt.subplots()
ax1.plot(trainingloss)
ax1.plot(validationloss)
ax1.legend(["Training Loss", "Validation Loss"])
ax1.set_xlabel("epochs", fontsize = 20)
ax1.set_ylabel("loss", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} val:{} \n lr:{} optim:{} activ:{}'.format(trainset1_size,trainset2_size,val_size,learnrate,OPTIM,activation),loc='upper left', prop = dict(fontweight = "normal", size= 10))
ax1.add_artist(anchored_text)
plt.savefig('cifar10_alt_loss2')

plt.clf()

fig2, ax2 =plt.subplots()
ax2.plot(testaccuracy)
ax2.legend(["Test Accuracy"])
ax2.set_xlabel("epoch", fontsize = 20)
ax2.set_ylabel("Test Accuracy", fontsize = 20)
anchored_text = AnchoredText('set1:{} set2:{} val:{} \n lr:{} optim:{} activ:{}'.format(trainset1_size,trainset2_size,val_size,learnrate,OPTIM,activation),loc='lower right', prop = dict(fontweight = "normal", size= 10))
ax2.add_artist(anchored_text)
ax2.add_artist(anchored_text)
plt.savefig('cifar10_alt_test2')



plt.clf()
