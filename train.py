from dataset import AMNIST
from model import Model

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import torch
import torch.nn.functional as F
import torch.optim as optim

# Choose the device to use (either GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset (train and test)
dset_train = AMNIST(train=True)
dset_test = AMNIST(train=False)

# Create data loaders
loader_train = torch.utils.data.DataLoader(dataset=dset_train, batch_size=16, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset=dset_test, batch_size=16, shuffle=False)

# Load the untrained model
model = Model().to(device)

# Choose the Adam optimizer
optimizer = optim.Adam(model.parameters())

# Let's train over 20 epochs
for epoch in range(0, 20):

	# Model goes in training mode
	model.train()

	# This will accumulate the total loss for all samples
	total = 0.0

	for image, label, box in progressbar.progressbar(loader_train):

		# Push tensors to CPU/GPU
		image = image.to(device)
		image = image.unsqueeze(1)
		target_label = label.to(device)
		target_box = box.to(device)

		# Reset optimizer
		optimizer.zero_grad()

		# Make a prediction with the model
		predicted_label, predicted_box = model(image)
		
		# Compute the loss
		# COMPUTE THE LOSS HERE
		loss = F.nll_loss(predicted_label, target_label) + F.mse_loss(predicted_box, target_box)
		# loss = torch.nn.CrossEntropyLoss(predicted_label, target_label) + torch.nn.CrossEntropyLoss(predicted_box, target_box)
		# Propagate gradient
		loss.backward()

		# Update weights
		optimizer.step()

		# Keep track of total loss
		total += loss.item()

	# Print loss for this epoch
	total /= len(loader_train)
	print("Train loss = %f" % total)

	# Model goes in eval mode
	model.eval()
	
	# This will accumulate the total loss for all samples
	total = 0.0

	for image, label, box in progressbar.progressbar(loader_test):

		# Push tensors to CPU/GPU
		image = image.to(device)
		image = image.unsqueeze(1)
		target_label = label.to(device)
		target_box = box.to(device)

		# Make a prediction with the model
		predicted_label, predicted_box = model(image)

		# Compute the loss
		# SAME LOSS HERE AS IN TRAINING
		loss = F.nll_loss(predicted_label, target_label) + F.mse_loss(predicted_box, target_box)
		# Keep track of total loss
		total += loss.item()

	# Print loss for this epoch
	total /= len(loader_test)
	print("Test loss = %f" % total)

# Take the last 16 samples and plot them
for i in range(0, 16):

	image_np = image[i,:,:].detach().cpu().numpy()
	predicted_label_np = np.argmax(predicted_label[i].detach().cpu().numpy())
	predicted_box_np = predicted_box[i,:].detach().cpu().numpy()

	dset_test.plot(image_np, predicted_label_np, predicted_box_np)

