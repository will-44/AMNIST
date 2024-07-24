import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

	def __init__(self):

		# Call the parent method
		super(Model, self).__init__()


		# First batch normalization layer
		self.bn1 = nn.BatchNorm2d(num_features=1)

		# First convolutional layer
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), bias=False)

		# Second batch normalization layer
		self.bn2 = nn.BatchNorm2d(num_features=32)
		# Second convolutional layer
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), bias=False)

		# Third batch normalization layer
		self.bn3 = nn.BatchNorm2d(num_features=32)
		# Third convolutional layer
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), bias=False)

		# Fourth batch normalization layer
		self.bn4 = nn.BatchNorm2d(num_features=32)
		# Fourth convolitional layer
		self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), bias=True)

		# Fifth batch normalization layer
		self.bn5 = nn.BatchNorm2d(num_features=32)
		# Fifth convolitional layer
		self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), bias=True)

		# Fully connected layer (expressed as a conv layer)
		self.fc1 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), bias=True)

		self.fc2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1,1), bias=True)

		# Softmax (we use LogSoftMax for stability purpose)
		self.lsf = nn.LogSoftmax(dim=1)


    # Forward pass
	def forward(self, x):

		# Apply forward pass, and return digit (c) and box coordinates (r)
		# Compute 1st layer (batchnorm + conv + relu + maxpool)
		# (N,1,100,100) > (N,32,49,49)
		x = F.max_pool2d(F.relu(self.conv1(self.bn1(x))), (2,2))

		# Compute 2nd layer (batchnorm + conv + relu + maxpool)
		# (N,32,49,49) > (N,32,23,23)
		x = F.max_pool2d(F.relu(self.conv2(self.bn2(x))), (2,2))

		# Compute 3rd layer (batchnorm + conv + relu + maxpool)
		# (N,32,23,23) > (N,32,10,10)
		x = F.max_pool2d(F.relu(self.conv3(self.bn3(x))), (2,2))

		# new layer to down to 5
		# (N,32,10,10) > (N,32,4,4)
		x = F.max_pool2d(F.relu(self.conv4(self.bn4(x))), (2,2))

		#new layer to down to 1
		# (N,32,5,5) > (N,32,1,1)
		x = F.max_pool2d(F.relu(self.conv5(self.bn5(x))), (2,2))

		# Classification layer (fully connectd + logsoftmax)
		# (N,32,1,1) > (N,10)
		c = self.lsf(torch.squeeze(self.fc1(x)))

		# x = self.conv4(x)
		r = F.relu(torch.squeeze(self.fc2(x)))
		return c, r























