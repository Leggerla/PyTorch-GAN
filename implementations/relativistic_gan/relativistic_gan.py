import argparse
import os
import numpy as np
import math
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from scipy.signal import correlate
# from pytorch_forecasting.utils import autocorrelation

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("charts", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./", help="folder with all the data")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--vector_size", type=int, default=100, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def correlation(gen):
	coor = []
	for i in range(gen.shape[0]):
		coor.append(torch.from_numpy(correlate(gen[i, :].cpu(), gen[i, :].cpu(), 'same')))  # /gen.shape[-1]))
	return torch.stack(coor)


class StockDataset(torch.utils.data.Dataset):
	'Characterizes a dataset for PyTorch'

	def __init__(self, data_path):
		'Initialization'
		window, roll = opt.vector_size, opt.vector_size
		self.base = self.rolling_periods(pd.read_csv(data_path + 'base.csv', usecols=[1]), window, roll)
		self.associate = self.rolling_periods(pd.read_csv(data_path + 'associate.csv', usecols=[1]), window, roll)

	def __len__(self):
		'Denotes the total number of samples'
		return self.base.shape[0]

	def __getitem__(self, index):
		'Generates one sample of data'
		# Get vector and label
		X = self.base[index, :]
		y = self.associate[index, :]

		return X, y

	def rolling_periods(self, df, window, roll):
		res = []
		array = torch.tensor(df.values)
		max = torch.max(array)
		min = torch.min(array)
		enum = array.shape[0]
		for i in torch.arange(0, enum, step=window):
			if enum < i + roll:
				break
			res.append(array[i:i + roll][:, 0])
		return 2 * (torch.stack(res) - min + 1e-8) / (max - min + 1e-8) - 1


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		
		def generator_block(in_filters, out_filters, kernel_size=(2, 5), padding=(2, 2, 1, 0)):
			block = [nn.ZeroPad2d(padding), nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size),
				 nn.BatchNorm2d(out_filters),nn.LeakyReLU(0.2, inplace=True)]
			return block

		self.model = nn.Sequential(
			*generator_block(1, 512),
			*generator_block(512, 256),
			*generator_block(256, 128),
			*generator_block(128, 64),
			*generator_block(64, 32),
			*generator_block(32, 16),
			*generator_block(16, 8),
			*generator_block(8, 4, kernel_size=(3, 3), padding=(1, 1, 1, 1)),
			*generator_block(4, 2, kernel_size=(3, 3), padding=(1, 1, 1, 1)),
			*generator_block(2, 1, kernel_size=(2, 1), padding=(0, 0, 0, 0)))

	def forward(self, base, z):
		out = torch.cat([base[:, None, :], z[:, None, :]], dim=1)
		out = out[:, None]	
		out = self.model(out)
		out = torch.squeeze(out)
		return out


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		def discriminator_block(in_filters, out_filters):
			block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
			return block

		self.model = nn.Sequential(
			*discriminator_block(opt.channels, 64),
			*discriminator_block(64, 64),
			*discriminator_block(64, 64),
			*discriminator_block(64, 64),
			*discriminator_block(64, 32),
			*discriminator_block(32, 32),
			*discriminator_block(32, 32),
			*discriminator_block(32, 32)
		)

		# The height and width of downsampled image
		ds_size = 2*opt.vector_size
		self.adv_layer = nn.Sequential(nn.Linear(32, 100), nn.LeakyReLU(0.2, inplace=True), 
					       nn.Linear(100, 50), nn.LeakyReLU(0.2, inplace=True),
					       nn.Linear(50, 1), nn.LeakyReLU(0.2, inplace=True))

	def forward(self, base, associate):
		out = torch.cat([base, associate], dim=1)[:, None, :]
		out = self.model(out)
		out = out.view(out.shape[0], -1)
		validity = self.adv_layer(out)
		return validity


# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Configure data loader
dataset = StockDataset(opt.data_path)
dataloader = torch.utils.data.DataLoader(
	dataset,
	batch_size=opt.batch_size,
	shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ----------
#  Training
# ----------
d_real_losses = torch.zeros(opt.n_epochs)
d_fake_losses = torch.zeros(opt.n_epochs)
g_losses = torch.zeros(opt.n_epochs)
best_autocorrelation = -float('inf')
best_similarity = -float('inf')
best_corr_dist = float('inf')
best_list = []
best_loss = float('inf')
for epoch in range(opt.n_epochs):
	sum_d_real_loss = []
	sum_d_fake_loss = []
	sum_g_loss = []
	for i, (base, associate) in enumerate(dataloader):

		base, associate = base.to(device), associate.to(device)
		# Adversarial ground truths
		valid = Variable(Tensor(associate.shape[0], 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(associate.shape[0], 1).fill_(0.0), requires_grad=False)

		# Configure input
		real_base = Variable(base.type(Tensor))
		real_associate = Variable(associate.type(Tensor))
		
		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G.zero_grad()

		# Sample noise as generator input
		z = Variable(Tensor(np.random.normal(0, 1, (base.shape[0], opt.latent_dim))))

		# Generate a batch of images
		gen_associate = generator(real_base, z)

		real_pred = discriminator(real_base, real_associate).detach()
		fake_pred = discriminator(real_base, gen_associate)

		if opt.rel_avg_gan:
		    g_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), valid)
		else:
		    g_loss = adversarial_loss(fake_pred - real_pred, valid)

		# Loss measures generator's ability to fool the discriminator
		#??? g_loss = adversarial_loss(discriminator(gen_associate), valid)
		
		if g_loss.item() < best_loss:
			best_loss = g_loss.item()
			torch.save(real_associate.data, "charts/best_loss_real.pt")
			torch.save(gen_associate.data, "charts/best_loss_gen.pt")
			torch.save(generator.state_dict(), "best_loss_generator.pt")
			torch.save(discriminator.state_dict(), "best_loss_discriminator.pt")
			
		g_loss.backward()
		optimizer_G.step()

		for n in range(1):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			optimizer_D.zero_grad()

			# Predict validity
			real_pred = discriminator(real_base, real_associate)
			fake_pred = discriminator(real_base, gen_associate.detach())

			if opt.rel_avg_gan:
			    real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
			    fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
			else:
			    real_loss = adversarial_loss(real_pred - fake_pred, valid)
			    fake_loss = adversarial_loss(fake_pred - real_pred, fake)

			d_loss = (real_loss + fake_loss) / 2

			d_loss.backward()
			optimizer_D.step()
		
		sum_d_real_loss.append(real_loss.item())
		sum_d_fake_loss.append(fake_loss.item())
		sum_g_loss.append(g_loss.item())

		sp_vix_real_corr = correlate(real_base.data.cpu(), real_associate.data.cpu(), 'same')
		sp_vix_gen_corr = correlate(real_base.data.cpu(), gen_associate.data.cpu(), 'same')

		corr_dist = np.linalg.norm(sp_vix_real_corr - sp_vix_gen_corr)
		if corr_dist <= best_corr_dist:
			if corr_dist < best_corr_dist:
				best_corr_dist = corr_dist.item()
				print('Correlation distance', epoch, best_corr_dist)
				batch_num = epoch * opt.batch_size + i
				best_list.append(torch.tensor([batch_num, best_corr_dist]))

			torch.save(real_associate.data, "charts/real.pt")
			torch.save(gen_associate.data, "charts/gen.pt")
			torch.save(sp_vix_real_corr, "charts/real_corr.pt")
			torch.save(sp_vix_gen_corr, "charts/gen_corr.pt")
			torch.save(generator.state_dict(), "generator.pt")
			torch.save(discriminator.state_dict(), "discriminator.pt")
			torch.save(torch.stack(best_list), "correlation distance.pt")

		'''matrix_autocorr = correlation(gen_associate.data[:, 0, :])
		#mask = torch.logical_and(torch.mean(gen_associate.data[:, 0, :], dim=0) > 0.3, torch.mean(gen_associate.data[:, 0, :], dim=0) < 0.5)
		#mask = torch.logical_and(torch.std(gen_associate.data[:, 0, :], dim=0) > 0.02, mask)
		mask = torch.logical_and(matrix_autocorr >= -0.01, matrix_autocorr <= 0.1)
		#autocorr = torch.sum(mask).float()
		autocorr = torch.sum(mask).float()
		if autocorr >= best_autocorrelation:
			if autocorr > best_autocorrelation:
				best_autocorrelation = autocorr.item()
				print('Autocorrelation', epoch, best_autocorrelation)
			torch.save(real_associate.data, "charts/real.pt")
			torch.save(gen_associate.data, "charts/gen.pt")
			torch.save(matrix_autocorr, "charts/autocorr.pt")

			torch.save(generator.state_dict(), "generator.pt")
			torch.save(discriminator.state_dict(), "discriminator.pt")'''

		similarity = torch.sum(torch.mul(real_associate.data, gen_associate.data))
		if similarity > best_similarity:
			best_similarity = similarity
			torch.save(real_associate.data, "charts/similarity_real.pt")
			torch.save(gen_associate.data, "charts/similarity_gen.pt")

	d_real_losses[epoch] = torch.mean(torch.tensor(sum_d_real_loss))
	d_fake_losses[epoch] = torch.mean(torch.tensor(sum_d_fake_loss))
	g_losses[epoch] = torch.mean(torch.tensor(sum_g_loss))

	if epoch % 10 == 0:
		print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
		  % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
		torch.save(d_real_losses, 'd_real_losses.pt')
		torch.save(d_fake_losses, 'd_fake_losses.pt')
		torch.save(g_losses, 'g_losses.pt')

torch.save(real_associate.data, "charts/final_real.pt")
torch.save(gen_associate.data, "charts/final_gen.pt")
torch.save(sp_vix_real_corr, "charts/final_real_corr.pt")
torch.save(sp_vix_gen_corr, "charts/final_gen_corr.pt")
torch.save(generator.state_dict(), "final_generator.pt")
torch.save(discriminator.state_dict(), "final_discriminator.pt")

torch.save(d_real_losses, 'd_real_losses.pt')
torch.save(d_fake_losses, 'd_fake_losses.pt')
torch.save(g_losses, 'g_losses.pt')
