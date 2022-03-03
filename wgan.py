from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from torchvision import transforms
import torch
import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torchvision.utils import save_image
from tqdm.notebook import tqdm
import torch.nn.functional as F


image_size = 416
batch_size = 16
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
CRITIC_ITERATIONS = 5
#WEIGHT_CLIP = 0.01
LAMBDA_GP = 10
#stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
#stats = (0.5), (0.5)
train_dir ='./data/project/only_test_kit'

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


#%matplotlib inline

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break
        
show_batch(train_dl)


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
    
    
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)




discriminator = nn.Sequential(
    # in: 3 x 64 x 64
    # in: 3 x 416 x 416
    
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    #nn.BatchNorm2d(64),
    nn.InstanceNorm2d(64, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32
    # out: 64 x 208 x 208

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    #nn.BatchNorm2d(128),
    nn.InstanceNorm2d(128, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16
    # out: 128 x 104 x 104

    nn.Conv2d(128,128, kernel_size=4, stride=2, padding=1, bias=False),
    #nn.BatchNorm2d(128),
    nn.InstanceNorm2d(128, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8
    # out: 256 x 52 x 52

    nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1, bias=False),
    #nn.BatchNorm2d(256),
    nn.InstanceNorm2d(256, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4
    # out: 512 x 26 x 26
    
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    #nn.BatchNorm2d(512),
    nn.InstanceNorm2d(512, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    #out: 512 x 13 x 13
    
    nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0, bias=False),
    #nn.BatchNorm2d(512),
    nn.InstanceNorm2d(512, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    #out: 512 x 6 x 6

    nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=0, bias=False),
    #nn.BatchNorm2d(512),
    nn.InstanceNorm2d(512, affine=True),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 1 x 1 x 1
    # out: 1 x 2 x 2
    
    nn.Conv2d(512, 1, kernel_size=2, stride=1, padding=0, bias=False),
    #out: 1 x 1 x 1
    

    nn.Flatten(),
    nn.Sigmoid())


discriminator = to_device(discriminator, device)

latent_size = 128

generator = nn.Sequential(
    # in: latent_size x 1 x 1
    # in: latent_size x 13 x 13

    nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    # out: 512 x 4 x 4
    # out: 512 x 2 x 2
    
    nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 6 x 6
    
    nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 13 x 13

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8
    # out: 256 x 26 x 26

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16
    # out: 128 x 52 x 52
    
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32
    # out: 64 x 104 x 104

    nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32
    # out: 64 x 208 x 208
        
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
    # in: 3 x 416 x 416
)


xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
print(xb.shape)
#xb = torch.randn(batch_size, latent_size, 13, 13) # random latent tensors
fake_images = generator(xb)
print(fake_images.shape)
show_images(fake_images)

generator = to_device(generator, device)

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()
    cur_batch_size = real_images.shape[0]
    # Pass real images through discriminator
    real_preds = discriminator(real_images).reshape(-1)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    #real_targets = torch.ones(real_images.size(0), 169, device=device)
    #real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(cur_batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    #fake_targets = torch.zeros(fake_images.size(0), 169, device=device)
    fake_preds = discriminator(fake_images).reshape(-1)
    #fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()
    gp = gradient_penalty(discriminator, real_images, fake_images, device=device)
    loss_critic = (-(torch.mean(real_preds) - torch.mean(fake_preds)) + LAMBDA_GP * gp)
    # Update discriminator weights
    #loss = -(torch.mean(real_preds)-torch.mean(fake_preds))
    loss_critic.backward(retain_graph = True)
    opt_d.step()
    
    # clip critic weights between -0.01, 0.01
    #for p in discriminator.parameters():
        #p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        
        
    return loss_critic.item(), real_score, fake_score


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    #targets = torch.ones(batch_size, 169, device=device)
    #loss = F.binary_cross_entropy(preds, targets)
    loss_gen = -torch.mean(preds)
    
    # Update generator weights
    loss_gen.backward()
    opt_g.step()
    
    return loss_gen.item()



sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)
def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
#fixed_latent = torch.randn(64, latent_size, 13, 13, device=device)
save_samples(0, fixed_latent)
print(fixed_latent.shape)


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    
    for epoch in range(epochs):
        for real_images in tqdm(train_dl):
            real = real_images.to(device)
            
            for _ in range(CRITIC_ITERATIONS):
                #print(real_images.size)
                # Train discriminator
                loss_d, real_score, fake_score = train_discriminator(real, opt_d)
                # Train generator
                loss_g = train_generator(opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores



lr = 0.0001
epochs = 600
history = fit(epochs, lr)

torch.save(generator.state_dict(), r'./data/project/wgan_g_instancenorm_bs16_lr1e-4_600epoc.pt')
torch.save(discriminator.state_dict(), r'./data/project/wgan_d_instancenorm_bs16_lr1e-4_600epoc.pt')