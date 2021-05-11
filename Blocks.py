#Blocks for the Neural Net

#import key packages
import torch
import numpy as np
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn





class CNNBlock1(nn.Module):
	def __init__(self):
		super(CNNBlock1, self).__init__()
		self.conv1 = nn.Conv2d(3, 48, kernel_size = 3, padding = 1)
		torch.nn.init.xavier_normal_(self.conv1.weight)  #use xavier normal initialisation for all
		self.batchnorm1 = nn.BatchNorm2d(48)		
		self.conv2 = nn.Conv2d(48, 96, kernel_size = 3, padding = 1)
		torch.nn.init.xavier_normal_(self.conv2.weight)
		self.batchnorm2 = nn.BatchNorm2d(96)		
		self.dropout = nn.Dropout(0.4)
		 
    
	def forward(self, x):
		x = self.conv1(x)
		x = self.batchnorm1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.batchnorm2(x)
		x = F.relu(F.max_pool2d(x,2))
		#x = self.dropout(x)
		return x


class CNNBlock2(nn.Module):
	def __init__(self):
		super(CNNBlock2, self).__init__()
		self.conv3 = nn.Conv2d(96, 192, kernel_size = 4, padding = 1)
		torch.nn.init.xavier_normal_(self.conv3.weight)
		self.batchnorm1 = nn.BatchNorm2d(192)        
		self.conv4 = nn.Conv2d(192, 384, kernel_size = 4, padding = 1)
		torch.nn.init.xavier_normal_(self.conv4.weight)
		self.batchnorm2 = nn.BatchNorm2d(384)
		self.dropout = nn.Dropout(0.4)
            
	def forward(self, x):
		x = self.conv3(x)
		x = self.batchnorm1(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = self.batchnorm2(x)
		x = F.relu(F.max_pool2d(x,2))
		#x = self.dropout(x)
		return x


class CNNBlock3(nn.Module):
    def __init__(self):
        super(CNNBlock3, self).__init__()
        self.conv5 = nn.Conv2d(384, 384, kernel_size = 4, padding = 1)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        self.batchnorm1 = nn.BatchNorm2d(384)
        self.conv6 = nn.Conv2d(384, 384, kernel_size = 4, padding = 1)
        torch.nn.init.xavier_normal_(self.conv6.weight)
        self.batchnorm2 = nn.BatchNorm2d(384)
        #self.dropout = nn.Dropout(0.7)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
    	#This block has no maxpooling
        x = self.conv5(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
cnnblock3 = CNNBlock3()

#MLP block for the branch
class MLPBlock(nn.Module):
	def __init__(self):
		super(MLPBlock, self).__init__()
		self.fc1 = nn.Linear(384, 10)
		torch.nn.init.xavier_normal_(self.fc1.weight) 
		self.dropout1 = nn.Dropout(0.2)
    
	def forward(self, x):
		x = F.avg_pool2d(x,5)
		x = x.view(-1, 1*1*384)
		x = self.fc1(x)
		x = self.dropout1(x)		
		x = F.relu(x)
		return x	
mlpblock = MLPBlock()


cnnblock1 = CNNBlock1()
cnnblock2 = CNNBlock2()
cnnblock3 = CNNBlock3()
mlpblock = MLPBlock()

#Combine these blocks

class EnsembleModel(nn.Module):
	def __init__(self):
		super(EnsembleModel, self).__init__()
		self.cnnblock1 = cnnblock1
		self.cnnblock2 = cnnblock2
		self.cnnblock3 = cnnblock3
		self.mlpblock = mlpblock
    
	def forward(self, x):
		x1 = self.cnnblock1(x)
		x2 = self.cnnblock2(x1)
		x3 = self.cnnblock3(x2)
		x4 = self.mlpblock(x3)
		return F.log_softmax(x4, dim=1)


