from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64
imageSize = 64 # Size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 4)

# Takes a neural network and initilize its weights
def weights_init(nn):
    classname = nn.__class__.__name__
    if classname.find('Conv') != -1:
        nn.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.weight.data.normal_(1.0, 0.02)
        nn.bias.data.fill_(0)
        
# Defining the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                # inverse convolution
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), 
                # normalize the feature maps
                nn.BatchNorm2d(512),
                # Rectify linear units
                nn.ReLU(True),
                
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                nn.Tanh()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output
    
# Creating the generator
genNet = Generator()
genNet.apply(weights_init)

# Defining the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias= False),
                nn.LeakyReLU(0.2, inplace = True),
                
                nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace = True),
                
                nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace = True),
                
                nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace = True),
                
                nn.Conv2d(512, 1, 4, 1, 0, bias = False),
                nn.Sigmoid()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output.view(-1) # flatten the result of the convolutions
    
# Creating the discriminator
disNet = Discriminator()
disNet.apply(weights_init)

# Training the GANs
criterion = nn.BCELoss()
optimizerDis = optim.Adam(disNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerGen = optim.Adam(genNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range (25):
    for i, data in enumerate(dataloader, 0):
        
        # Updating the weights of the discriminator's neural network 
        disNet.zero_grad()
        # Training the discriminator with a real image of the dataset
        real, _ = data
        input = Variable(real)
        # Populate the tensor with ones because the images are real
        target = Variable(torch.ones(input.size()[0]))
        # Puts the input images through the discriminator
        output = disNet(input)
        # Compute error of the discriminator 
        disErrReal = criterion(output, target)
        
        # Training the discriminator with a fake image from the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # 100 feature maps of size 1x1
        fake = genNet(noise)
        # Populate the tensor with zeros because the images are fake
        target = Variable(torch.zeros(input.size()[0]))
        # Puts the fake images through the discriminator
        output = disNet(fake.detach())
        # Compute error of the discriminator 
        disErrFake = criterion(output, target)
        
        # Backpropagation
        disErr = disErrReal + disErrFake
        disErr.backward()
        optimizerDis.step()
        
        
        # Updating the weights of the generator's neural network
        genNet.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = disNet(fake)
        genErr = criterion(output, target)
        genErr.backward()
        optimizerGen.step()
        
        # Print the losses and save the real and generated images every 100 steps
        print('[%d/%d][%d/%d] Loss_Dis: %.4f Loss_Gen: %.4f' % (epoch, 25, i, len(dataloader), disErr.data[0], genErr.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = genNet(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch) , normalize = True)
        
        
        
        
        
    
    
    
    
    
    
    
    
    