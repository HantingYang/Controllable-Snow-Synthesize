import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1600, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
parser.add_argument("--crop_size", type=int, default=512, help="crop size of image patch")
parser.add_argument("--name", type=str, default="munit", help="name of the experiment")
parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0,1,2,3], help="List of GPU IDs to use")
parser.add_argument("--start_style_regr", type=int, default=300, help="epoch from which to start style regression")

opt = parser.parse_args()
print(opt)
gpu_ids = opt.gpu_ids
# Set the GPUs you want to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Set the device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Create sample and checkpoint directories
#os.makedirs("images/%s" % opt.name, exist_ok=True)
os.makedirs("images/%s_sample" % opt.name, exist_ok=True)
os.makedirs("images/%s_styleregresample" % opt.name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.name, exist_ok=True)

criterion_recon = torch.nn.L1Loss()


# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(gpu_ids[0])
Enc1 = nn.DataParallel(Enc1,gpu_ids)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(gpu_ids[0])
Dec1 = nn.DataParallel(Dec1,gpu_ids)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(gpu_ids[0])
Enc2 = nn.DataParallel(Enc2,gpu_ids)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(gpu_ids[0])
Dec2 = nn.DataParallel(Dec2,gpu_ids)
EncSh = StyleEncoderShared(dim=opt.dim, style_dim=opt.style_dim).to(gpu_ids[0])
EncSh = nn.DataParallel(EncSh,gpu_ids)
D1 = MultiDiscriminator().to(device)
D1 = nn.DataParallel(D1,gpu_ids)
D2 = MultiDiscriminator().to(device)
D2 = nn.DataParallel(D2,gpu_ids)
Dc = ContentDiscriminator(gpu_ids).to(device)
Dc = nn.DataParallel(Dc,gpu_ids)
    
    

if opt.epoch != 0:
    # Load pretrained models
    Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.name, opt.epoch)))
    Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.name, opt.epoch)))
    Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.name, opt.epoch)))
    Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.name, opt.epoch)))
    EncSh.load_state_dict(torch.load("saved_models/%s/EncSh_%d.pth" % (opt.name, opt.epoch)))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.name, opt.epoch)))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.name, opt.epoch)))
    Dc.load_state_dict(torch.load("saved_models/%s/Dc_%d.pth" % (opt.name, opt.epoch)))
else:
    # Initialize weights
    Enc1.apply(weights_init_normal)
    Dec1.apply(weights_init_normal)
    Enc2.apply(weights_init_normal)
    Dec2.apply(weights_init_normal)
    EncSh.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)
    Dc.apply(weights_init_normal)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 1
lambda_style_regr = 1

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters(), EncSh.parameters()), 
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
lr_dc = opt.lr / 2.5
optimizer_Dc = torch.optim.Adam(Dc.parameters(), lr=lr_dc, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_Dc = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Dc, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Configure dataloaders
transforms_train = [
    # transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.RandomCrop(opt.crop_size),
    transforms.ColorJitter(brightness=0.1, contrast=0.1,         # Adjust brightness and contrast
                           saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms_val = [
    # transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_train),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_val, mode="test"),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done, epoch):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    img_samples = None
    alphas = [0.1, 0.3, 0.6, 0.9]  # The alpha values
    for img1, img2, img1_path in zip(imgs["A"], imgs["B"], imgs["A_paths"]):
        # Create copies of image
        X1 = img1.unsqueeze(0).repeat(opt.style_dim, 1, 1, 1)
        # X1 [8, 3, 512, 1024]
        X1 = Variable(X1.type(Tensor))
        X2 = img2.unsqueeze(0).repeat(opt.style_dim, 1, 1, 1)
        X2 = Variable(X2.type(Tensor))
        # Get random style codes
        s_code = np.random.uniform(-1, 1, (opt.style_dim, opt.style_dim))
        s_code = Variable(Tensor(s_code))

        # Generate samples
        c_code_1, s_code_1 = Enc1(X1)
        s_code_1 = EncSh(s_code_1)
        _, s_code_2 = Enc2(X2)
        s_code_2 = EncSh(s_code_2)
        
        
        
        X12 = Dec2(c_code_1, s_code)
        # X12 [8, 3, 512, 1024]
        # Concatenate samples horisontally
        X12 = torch.cat([x for x in X12.data.cpu()], -1)
        img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
        
        styleregre_images = [img1.cpu().unsqueeze(0)]
        for alpha in alphas:
            # Get interpolated style code
            s_code_interpolated = interpolate_style(s_code_1, s_code_2, alpha)
            
            X12_alpha = Dec2(c_code_1[0].unsqueeze(0), s_code_interpolated[0].unsqueeze(0))
            
            styleregre_images.append(X12_alpha.data.cpu())

        styleregre_images = torch.cat(styleregre_images, -1).squeeze(0)
        
        # print(img1_path)
        head, tail = os.path.split(img1_path)
        # Concatenate with previous samples vertically
        # img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
        save_image(img_sample, "images/" + opt.name + "_sample" + "/epoch" + str(epoch) + "_"  + tail, nrow=5, normalize=True)
        save_image(styleregre_images, "images/" + opt.name + "_styleregresample" + "/epoch" + str(epoch) + "_" + tail, nrow=5, normalize=True)
        


# ----------
#  Training
# ----------

# Adversarial ground truths
valid = 1
fake = 0

def interpolate_style(style_a, style_b, alpha):
    return alpha * style_a + (1 - alpha) * style_b

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))

        # Sampled style codes
        style_1 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))
        style_2 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        s_code_1 = EncSh(s_code_1)
        c_code_2, s_code_2 = Enc2(X2)
        s_code_2 = EncSh(s_code_2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        s_code_21 = EncSh(s_code_21)
        c_code_12, s_code_12 = Enc2(X12)
        s_code_12 = EncSh(s_code_12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0
        


        # Losses
        loss_GAN_1 = lambda_gan * D1.module.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.module.compute_loss(X12, valid)
        loss_G_GAN_Acontent =  lambda_gan * Dc.module.compute_GAN_content_loss(c_code_1)
        loss_G_GAN_Bcontent =  lambda_gan * Dc.module.compute_GAN_content_loss(c_code_2)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 =  lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 =  lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0
        
        
        # print("epoch: ",epoch, "batch: ", i, "loss_GAN_1: ", loss_GAN_1, "loss_GAN_2: ", loss_GAN_2, "loss_ID_1: ", loss_ID_1, "loss_ID_2: ", loss_ID_2, "loss_s_1: ", loss_s_1, "loss_s_2: ", loss_s_2, "loss_c_1: ", loss_c_1, "loss_c_2: ", loss_c_2, "loss_cyc_1: ", loss_cyc_1, "loss_cyc_2: ", loss_cyc_2)

        # Total loss
        if epoch <= opt.start_style_regr:
            loss_G = (
                loss_GAN_1
                + loss_GAN_2
                + loss_ID_1
                + loss_ID_2
                + loss_s_1
                + loss_s_2
                + loss_c_1
                + loss_c_2
                + loss_cyc_1
                + loss_cyc_2
                + loss_G_GAN_Acontent
                + loss_G_GAN_Bcontent
            )
        else:
            # Style interpolation
            ### Sample alpha randomly
            alpha = torch.rand(1).item()
            ### Interpolate the style codes
            s_code_interpolated = interpolate_style(s_code_1, s_code_2, alpha)
            ### Translate images with the interpolated style code
            X_alpha = Dec2(c_code_1, s_code_interpolated)
            ### Extract the style from the synthesized image
            s_code_alpha_extracted = Enc2.module.style_encoder(X_alpha)
            # Style regression loss 
            loss_style_regr = lambda_style_regr * criterion_recon(s_code_alpha_extracted, s_code_interpolated)
            loss_G = (
                loss_GAN_1
                + loss_GAN_2
                + loss_ID_1
                + loss_ID_2
                + loss_s_1
                + loss_s_2
                + loss_c_1
                + loss_c_2
                + loss_cyc_1
                + loss_cyc_2
                + loss_G_GAN_Acontent
                + loss_G_GAN_Bcontent
                + loss_style_regr
            )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = D1.module.compute_loss(X1, valid) + D1.module.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = D2.module.compute_loss(X2, valid) + D2.module.compute_loss(X12.detach(), fake)

        loss_D2.backward()
        optimizer_D2.step()
        
        
        # content discriminator loss
        optimizer_Dc.zero_grad()
        
        loss_GAN_content = Dc.module.compute_loss(c_code_1.detach(), c_code_2.detach())
        
        loss_GAN_content.backward()
        optimizer_Dc.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0 and epoch >= 50:
            print("")
            print("epoch:",epoch," batches_done:",batches_done)
            sample_images(batches_done, epoch)
            

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch >= 150:
        # Save model checkpoints
        torch.save(Enc1.state_dict(), "saved_models/%s/Enc1_%d.pth" % (opt.name, epoch))
        torch.save(Dec1.state_dict(), "saved_models/%s/Dec1_%d.pth" % (opt.name, epoch))
        torch.save(Enc2.state_dict(), "saved_models/%s/Enc2_%d.pth" % (opt.name, epoch))
        torch.save(Dec2.state_dict(), "saved_models/%s/Dec2_%d.pth" % (opt.name, epoch))
        torch.save(D1.state_dict(), "saved_models/%s/D1_%d.pth" % (opt.name, epoch))
        torch.save(D2.state_dict(), "saved_models/%s/D2_%d.pth" % (opt.name, epoch))
