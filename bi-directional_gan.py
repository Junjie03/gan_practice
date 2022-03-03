import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt
#%matplotlib inline
import torch.autograd as autograd
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
    
NLAT =100

generator = nn.Sequential(
    # in: latent_size x 1 x 1
    # in: latent_size x 13 x 13

    nn.ConvTranspose2d(NLAT, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4
    # out: 512 x 2 x 2
    
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 512 x 6 x 6

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 256 x 8 x 8
    # out: 256 x 26 x 26

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 128 x 16 x 16
    # out: 128 x 52 x 52

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
    # in: 3 x 416 x 416
)
g = to_device(generator, device)



Encoder = nn.Sequential(
    # in: 3 x 64 x 64
    # in: 3 x 416 x 416
    
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32
    # out: 64 x 208 x 208

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16
    # out: 128 x 104 x 104

    nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8
    # out: 256 x 52 x 52

    nn.Conv2d(256,512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4
    # out: 512 x 26 x 26
    
    nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    #out: 512 x 6 x 6
    
    nn.Conv2d(512, NLAT, kernel_size=1, stride=1, padding=0, bias=False))
    #out: 1 x 1 x 1
    
e = to_device(Encoder, device)

class JointCritic(nn.Module):
    def __init__(self, x_mapping, z_mapping, joint_mapping):
    
        super().__init__()

        self.x_net = x_mapping
        self.z_net = z_mapping
        self.joint_net = joint_mapping

    def forward(self, x, z):
        assert x.size(0) == z.size(0)
        x_out = self.x_net(x)
        z_out = self.z_net(z)
        joint_input = torch.cat((x_out, z_out), dim=1)
        output = self.joint_net(joint_input)
        return output

def create_critic():
    x_mapping = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        #nn.InstanceNorm2d(64, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x 32 x 32
        # out: 64 x 208 x 208

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x 16 x 16
        # out: 128 x 104 x 104

        nn.Conv2d(128,128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        #nn.InstanceNorm2d(128, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 8 x 8
        # out: 256 x 52 x 52

        nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        #nn.InstanceNorm2d(256, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 4 x 4
        # out: 512 x 26 x 26

        nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        #nn.InstanceNorm2d(512, affine=True),
        nn.LeakyReLU(0.2, inplace=True))
        #out: 512 x 13 x 13


    z_mapping = nn.Sequential(
        nn.Conv2d(NLAT, 512, 1, 1, 0), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 1, 1, 0), nn.LeakyReLU(0.2))

    joint_mapping = nn.Sequential(
        nn.Conv2d(512 + 512, 1024, 1, 1, 0), nn.LeakyReLU(0.2),
        nn.Conv2d(1024, 1024, 1, 1, 0), nn.LeakyReLU(0.2),
        nn.Conv2d(1024, 1, 1, 1, 0))

    return JointCritic(x_mapping, z_mapping, joint_mapping)


#stats = (0.5), (0.5)

train_dir ='./data/project/only_test_kit'
image_size = 64
NLAT = 100


batch_size = 16
latent = torch.randn(batch_size, NLAT, 1, 1, device=device)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
#stats = (0.5), (0.5)
CRITIC_ITERATIONS = 5
#WEIGHT_CLIP = 0.01
LAMBDA_GP = 10
#file_dir ='../input/vertical-train-set'

def build_path(input_dir):
    dataset = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for x in filenames:
            if x.endswith(".jpg"):
                dataset.append(os.path.join(dirpath, x))
    return dataset

def load_image_binary(img,input_size = image_size):
    image =Image.open(img).convert('RGB')
    image = transforms.Resize((image_size, image_size))(image)
    #print('Loaded image...')
    #print('Image Size: {}'.format(image.size))
    return image

def prepare_dataset(dir_path):
    arr=[]
    dataset=build_path(dir_path)
    for i in dataset:
        arr.append(load_image_binary(i,image_size))
    return arr

class testkitDataset(Dataset):
    def __init__(self, X):
        'Initialization'
        self.X = X
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X
    transform = transforms.Compose([
        #T.ToPILImage(),
        #T.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)])
    

train_dataset=prepare_dataset(train_dir)
train_transformed_dataset=testkitDataset(train_dataset)
train_dl = DataLoader(train_transformed_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)


def criticize( x, z_hat, x_tilde, z):
    input_x = torch.cat((x, x_tilde), dim=0)
    input_z = torch.cat((z_hat, z), dim=0)
    output = c(input_x, input_z)
    data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
    return data_preds, sample_preds

def calculate_grad_penalty( x, z_hat, x_tilde, z):
    bsize = x.size(0)
    eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
    intp_x = eps * x + (1 - eps) * x_tilde
    intp_z = eps * z_hat + (1 - eps) * z
    intp_x.requires_grad = True
    intp_z.requires_grad = True
    C_intp_loss = c(intp_x, intp_z).sum()
    grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
    grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
    grads = torch.cat((grads_x, grads_z), dim=1)
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

c=create_critic()
c = to_device(c, device)



BATCH_SIZE = 16
ITER = 80000
#IMAGE_SIZE = 32
#NUM_CHANNELS = 3
DIM = 128
NLAT = 100
LEAK = 0.2

C_ITERS = 5       # critic iterations
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.9

optimizerEG = torch.optim.Adam(list(e.parameters()) + list(g.parameters()), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizerC = torch.optim.Adam(c.parameters(), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2))

EG_losses,C_losses=[],[]
curr_iter = C_iter = EG_iter =0
C_update,EG_update = True,False
print('Training starts')

while curr_iter < ITER:
    #for batch_idx,x in enumerate (train_dl,1):
    for x in tqdm(train_dl):
        x=x.to(device)
        if curr_iter ==0:
            int_x=x
            curr_iter+=1
        z = torch.randn(x.size(0), NLAT, 1, 1, device=device)
        z_hat,x_tilde = e(x),g(z)
        data_preds,sample_preds = criticize(x,z_hat,x_tilde,z)
        EG_loss = torch.mean(data_preds-sample_preds)
        C_loss = -EG_loss + LAMBDA * calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
        
        
        if C_update:
            optimizerC.zero_grad()
            C_loss.backward()
            C_losses.append(C_loss.item())
            optimizerC.step()
            C_iter += 1
        if C_iter == C_ITERS:
            C_iter = 0
            C_update, EG_update = False, True
            continue
                   
                   
        if EG_update:
            optimizerEG.zero_grad()
            EG_loss.backward()
            EG_losses.append(EG_loss.item())
            optimizerEG.step()
            EG_iter += 1

        if EG_iter == EG_ITERS:
            EG_iter = 0
            C_update, EG_update = True, False
            curr_iter += 1
        else:
            continue
        
        if curr_iter % 100 == 0:
            print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
          % (curr_iter, ITER, EG_loss.item(), C_loss.item()))

            

torch.save(g.state_dict(), r'./data/project/g_bigan_onlysmushtestkit_size64_bs16_lr5e5_iter80000.pt')
torch.save(e.state_dict(), r'./data/project/e_bigan_onlysmushtestkit_size64_bs16_lr5e5_iter80000.pt')
torch.save(c.state_dict(), r'./data/project/c_bigan_onlysmushtestkit_size64_bs16_lr5e5_iter80000.pt')