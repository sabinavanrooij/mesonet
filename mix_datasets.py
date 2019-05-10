import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle       
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse
import matplotlib.pyplot as plt
from random import shuffle
from PIL import ImageFile, Image
from itertools import islice


def get_mixed_batches(batch_size, get_train = True):

	if get_train: 
		real_imgs_ffhq = torchvision.datasets.ImageFolder(root='../forensic-transfer/FFHQ/train',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		real_imgs_celeba = torchvision.datasets.ImageFolder(root='../forensic-transfer/celebA/train/',
		transform=torchvision.transforms.Compose([torchvision.transforms.CenterCrop(178), torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		real_imgs_faceforensics = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensics_images/FaceForensics_source_to_target_images/train/faceforensics_real/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
		
		real_imgs_faceforensicspp = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensicspp/original_sequences/real_cropped/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_stargan = torchvision.datasets.ImageFolder(root='../forensic-transfer/stargan/train',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_pggan = torchvision.datasets.ImageFolder(root='../forensic-transfer/pggan_fake/train/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256),
		torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_faceswap = torchvision.datasets.ImageFolder(root='../forensic-transfer/FaceSwap/FaceSwap/train',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_faceforensics = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensics_images/FaceForensics_source_to_target_images/train/faceforensics_fake',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_deepfake = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensicspp/manipulated_sequences/Deepfakes/fake_cropped',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_stylegan = torchvision.datasets.ImageFolder(root='../forensic-transfer/stylegan-master/train/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
		
	else: 
		real_imgs_ffhq = torchvision.datasets.ImageFolder(root='../forensic-transfer/FFHQ/test',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		real_imgs_celeba = torchvision.datasets.ImageFolder(root='../forensic-transfer/celebA/test/',
		transform=torchvision.transforms.Compose([torchvision.transforms.CenterCrop(178), torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		real_imgs_faceforensics = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensics_images/FaceForensics_source_to_target_images/test/faceforensics_real/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
		
		real_imgs_faceforensicspp = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensicspp/original_sequences/real_cropped/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_stargan = torchvision.datasets.ImageFolder(root='../forensic-transfer/stargan/test',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_pggan = torchvision.datasets.ImageFolder(root='../forensic-transfer/pggan_fake/test/',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256),
		torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_faceswap = torchvision.datasets.ImageFolder(root='../forensic-transfer/FaceSwap/FaceSwap/test',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_faceforensics = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensics_images/FaceForensics_source_to_target_images/test/faceforensics_fake',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_deepfake = torchvision.datasets.ImageFolder(root='../forensic-transfer/faceforensicspp/deepfake_test/cropped_squared',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))

		fake_imgs_stylegan = torchvision.datasets.ImageFolder(root='../forensic-transfer/stylegan-master/test',
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
	

	num_real1 = batch_size/2/4
	num_real2 = batch_size/2/4
	num_real3 = batch_size/2/4
	num_real4 = batch_size/2/4
	num_fake1 = round(batch_size/2/6)
	num_fake2 = round(batch_size/2/6)
	num_fake3 = round(batch_size/2/6)
	num_fake4 = round(batch_size/2/6)
	num_fake5 = round(batch_size/2/6)+1
	num_fake6 = round(batch_size/2/6)+1

	num_list = [num_real1,num_real2,num_real3, num_real4, num_fake1,num_fake2,num_fake3,num_fake4,num_fake5, num_fake6]
	
	if get_train: 
		indic = list(range(len(fake_imgs_deepfake)))
	else: 
		indic = list(range(len(real_imgs_ffhq)))

	real1_dataloader = torch.utils.data.DataLoader(real_imgs_ffhq,
	    batch_size=int(num_real1), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	real2_dataloader = torch.utils.data.DataLoader(real_imgs_celeba,
	    batch_size=int(num_real2), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	real3_dataloader = torch.utils.data.DataLoader(real_imgs_faceforensics,
	    batch_size=int(num_real3), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	real4_dataloader = torch.utils.data.DataLoader(real_imgs_faceforensicspp,
	    batch_size=int(num_real4), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	fake1_dataloader = torch.utils.data.DataLoader(fake_imgs_stargan,
	    batch_size=int(num_fake1), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	fake2_dataloader = torch.utils.data.DataLoader(fake_imgs_pggan,
	    batch_size=int(num_fake2), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	fake3_dataloader = torch.utils.data.DataLoader(fake_imgs_faceswap,
	    batch_size=int(num_fake3), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))

	fake4_dataloader = torch.utils.data.DataLoader(fake_imgs_faceforensics,
	    batch_size=int(num_fake4), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))
	
	fake5_dataloader = torch.utils.data.DataLoader(fake_imgs_deepfake,
		    batch_size=int(num_fake5), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))
	
	fake6_dataloader = torch.utils.data.DataLoader(fake_imgs_stylegan,
		    batch_size=int(num_fake6), shuffle=False, drop_last=True, sampler=torch.utils.data.dataloader.RandomSampler(indic))


	batches_list = zip(real1_dataloader, real2_dataloader, real3_dataloader, real4_dataloader, fake1_dataloader, fake2_dataloader, fake3_dataloader, fake4_dataloader, fake5_dataloader, fake6_dataloader)


	return batches_list