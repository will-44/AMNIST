from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision

class AMNIST(Dataset):

	def __init__(self, train=True, number=10000):

		# Load MNIST dataset
		self.train_mnist = torchvision.datasets.MNIST(root='data', train=train, download=True)

		# Image size, and pixels per digit
		self.image_x = 100
		self.image_y = 100
		self.digit_x = 28
		self.digit_y = 28

		# We will create only 100 noise instances, to limit memory usage
		self.noise_instances = 100
		
		# Number of samples
		self.number = number

		# Random seed to ensure repeatability
		np.random.seed(0)

		# Choose each digit for each image
		self.digits = np.random.randint(0, len(self.train_mnist), number)

		# Choose the digit position
		self.xs = np.random.randint(0, self.image_x - self.digit_x, number)
		self.ys = np.random.randint(0, self.image_y - self.digit_y, number)

		# Generate noise and associate noise index for each observation
		self.noise = np.random.uniform(0.0, 1.0, (self.noise_instances, self.image_x, self.image_y))
		self.noise_index = np.random.randint(0, self.noise_instances, number)

	def __len__(self):

		return self.number

	def __getitem__(self, idx):

		# Create empty image
		image = np.zeros((self.image_x, self.image_y), dtype=np.float32)

		# Get digit position and label
		x = self.xs[idx]
		y = self.ys[idx]
		digit = self.digits[idx]

		# Add digit in the image
		image[x:(x+self.digit_x), y:(y+self.digit_y)] = np.array(self.train_mnist[digit][0]) / 255.0
		# Add noise
		image += self.noise[self.noise_index[idx]]
		# Make sure all pixels lie in the range [0,1]
		image = np.clip(image, 0.0, 1.0)

		# Load digit label and box position
		label = self.train_mnist[digit][1]
		box = np.asarray([x,y], dtype=np.float32)

		return image, label, box

	def plot(self, image, label, box):

		# Display the image, with the digit label in title and box in red
		fig, ax = plt.subplots(1)
		# img = 1.0 - image
		ax.imshow(image[0], cmap='gray', vmin=0.0, vmax=1.0)
		ax.add_patch(patches.Rectangle((box[1],box[0]),28,28,linewidth=1,edgecolor='r',facecolor='none'))
		plt.title(label)
		plt.show()
