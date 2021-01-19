import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from RayTracingModels import UNet
from RayTracingUtils import Sampler, Renderer, Display

imageio.plugins.freeimage.download()
# Use cuda
torch.backends.cudnn.benchmark = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
## TODO
# Check kernel size depending on the projector disparity
# Add ssim to loss # doesn't really work
# Precalculate direc and ori? would make rendering quicker but also increase memory requirements
###

#-----------PARAMETERS--------------------
# Define I/O image folers
GTFolder = "./data/GroundTrueImages/"
inFolder = "./data/ProjectorImages_0000/"
imgSaveFolder = "./rendered/"
weightSaveFolder = "./weights/"
Path(imgSaveFolder).mkdir(parents=True, exist_ok=True)
Path(weightSaveFolder).mkdir(parents=True, exist_ok=True)

# Define image and view parameters
imgWidth = 800
imgHeight = 600
colorChannels = 3
numProjectors = 41
numViews = 101  # 101
views = list(range(numViews))
useLAB = True # Use LAB color space. False: BGR
primaryResolution = 3


# Training parameters
train = 1
test = 1
maxIters = 500
printFreq = 5
saveFreq = 20
startIter = 0
# Multipliers for loss functions, SSIM is currently commented out
alphaMSE = 1
alphaSSIM = 100
viewsPerIter = 15
LRAnnulmentRate = 200 # reduce the learning rate after how many iterations

# Model parameters
kernelSize = [3, 7]  # For 2 last axes
numLayers = 3
useDepthwise = True  # Lighter convolution type
# axisOrder = [0, 1, 2, 3]  # Originally 0 is batches, 1 is height, 2 is width and 3 is channels/projectors
# axisOrder = [0, 3, 1, 2]  # XY, or rather YX
# axisOrder = [0, 2, 3, 1]  # YZ
axisOrder = [0, 1, 3, 2]  # XZ

# Define projector and viewer positions
observerPositions = np.linspace([-500, 0, 0], [-500 + 10 * 100, 0, 0], 101)
projectorPositions = np.linspace([-1000, 0, 800], [-1000 + 50 * 40, 0, 800], 41)
observerPositions = torch.from_numpy(observerPositions).to(device)
projectorPositions = torch.from_numpy(projectorPositions).to(device)

# Reading images
inputSize = (imgHeight, imgWidth, colorChannels * numProjectors)
outputSize = (imgHeight, imgWidth, colorChannels * numViews)
inputs = np.zeros(inputSize)
outputs = np.zeros(outputSize)
#------------------END OF PARAMETERS-----------

for ii in range(0, numProjectors):
    img = cv2.imread(inFolder + "{0:0=4d}".format(ii) + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if useLAB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    inputs[:, :, ii * colorChannels:ii * colorChannels + colorChannels] = img

for ii in range(0, numViews):
    view = views[ii]
    img = cv2.imread(GTFolder + "{0:0=4d}".format(view) + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if useLAB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    outputs[:, :, ii * colorChannels:ii * colorChannels + colorChannels] = img

inputs = inputs[np.newaxis, ...]

# Define screen model
sampler = Sampler(height=imgHeight, width=imgWidth, primaryResolution=primaryResolution)
display = Display(observerPositions, height=imgHeight, width=imgWidth, diffusionPower=[40, 0], halfPhysSize=[200, 150],
                  viewerDistance=400,
                  projectorResolution=[imgWidth, imgHeight], primaryResolution=primaryResolution)
renderer = Renderer(m_DisplayModel=display, sampler=sampler, projectorPositions=projectorPositions,
                    projectorImages=inputs, height=imgHeight, width=imgWidth,
                    primaryResolution=3)
torch.set_default_dtype(torch.float32)

# Transform data from numpy to pytorch
# if memory is tight, one can move x to device in the training loop
axisOrder = [0, 1, 2, 3]  # Originally 0 is batches, 1 is height, 2 is width and 3 is channels/projectors
# axisOrder = [0, 3, 1, 2]  # XY, or rather YX
axisOrder = [0, 2, 3, 1]  # YZ
# axisOrder = [0, 1, 3, 2]  # XZ
# rendered uses [0, 3,1,2]. Below is required method for ensuring it gets the data in right format
# 2D convolution kernel is slid in the 2 last axes, and a 1x1 convolution in the second axis
axisOrderForRendering = [axisOrder.index(0), axisOrder.index(3), axisOrder.index(1), axisOrder.index(2)]
x = torch.from_numpy(inputs).permute(*axisOrder).float()  # .to(device)
y = torch.from_numpy(outputs).float()  # .permute(0, 3, 1, 2)
lossf = nn.L1Loss(reduction='sum')

# Define model
nChns = x.size(1)
layers = [nChns * 2 ** x for x in (range(numLayers))]
enc_chs = (nChns, *layers)
dec_chs = (*reversed(layers),)
model = UNet(inputSize=[x.size(-2), x.size(-1)], enc_chs=enc_chs, dec_chs=dec_chs, kernel=kernelSize,
             useDepthwise=useDepthwise).to(device)
# model = fullyConnectedModel().to(device)



scaler = GradScaler()
opt = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(opt, LRAnnulmentRate, gamma=0.8)
scheduler.last_epoch = startIter - 1

# with profiler.profile(use_cuda=True,profile_memory=True) as prof:
#     projectorImageshat = model(x).squeeze(0)
#     renderer.updateProjectorImages(projectorImageshat)
#     yhat = renderer.render(5)
# print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))

if train:
    """Main training loop.
        Gives projector views as optimized by the neural network.
        Renders a random view and an another one 50 views from that one
        Compares rendered views to GT and backpropagates"""
    print("Starting to train!")
    timer = time.time()
    if startIter > 0:
        model.load_state_dict(torch.load(weightSaveFolder + "UNET_iter_" + str(startIter) + ".weights"))
        opt.load_state_dict(torch.load(weightSaveFolder + "UNET_iter_" + str(startIter) + ".optimizer"))
    losses = 0
    for iter in range(startIter, maxIters + 1):
        opt.zero_grad()
        loss = torch.zeros(1).to(device)

        with autocast():
            projectorImageshat = model(x.detach().to(device)).permute(*axisOrderForRendering).squeeze(0)
            renderer.updateProjectorImages(projectorImageshat)
            views2 = np.arange(0, numViews)
            np.random.shuffle(views2)
            for ii in range(viewsPerIter):
                torch.cuda.empty_cache()

                view = views2[ii]
                # view = np.random.randint(0, numViews)
                yhat = renderer.render(view)
                yiter = y[:, :, view * 3:view * 3 + 3].detach().to(device)
                # ssimVal = ssim(yhat, yiter)
                loss = alphaMSE * lossf(yhat, yiter)  # + alphaSSIM*(1-ssimVal)
                torch.cuda.empty_cache()

                scaler.scale(loss).backward(retain_graph=True if ii != (viewsPerIter - 1) else False)
                losses += loss.detach().item()

        # this shouldnt be required, can make training ~15% slower
        # torch.cuda.empty_cache()

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if iter % printFreq == 0 and iter > startIter:
            print("iter: ", iter)
            print("Avg. loss per view: " + "{0:.2E}".format(losses / (printFreq * viewsPerIter)))
            print("Average time per iteration: ", (time.time() - timer) / printFreq, " seconds")
            timer = time.time()
            losses = 0
        if iter % saveFreq == 0 and iter > startIter:
            torch.save(model.state_dict(), weightSaveFolder + "UNET_iter_" + str(iter) + ".weights")
            torch.save(opt.state_dict(), weightSaveFolder + "UNET_iter_" + str(iter) + ".optimizer")
            print("LR: " + str(scheduler.get_last_lr()[0]))

if test:
    """Test loop.
    Saves both optimized projector images and the respective rendered views"""

    # Render views from unoptimized projector images for testing rendering
    sanityCheck = False

    if not sanityCheck:
        model = UNet(inputSize=[x.size(-2), x.size(-1)], enc_chs=enc_chs, dec_chs=dec_chs, kernel=kernelSize,
                     useDepthwise=useDepthwise).to(device).eval()
        # model = fullyConnectedModel().to(device).eval()
        model.load_state_dict(torch.load(weightSaveFolder + "UNET_iter_" + str(maxIters) + ".weights"))
    else:
        print("USING PROJECTOR IMAGES, NOT DISTORTED PROJECTOR IMAGES")

    with torch.no_grad():
        projectorImageshat = model(x.detach().to(device)) if not sanityCheck else x.to(device)
        projectorImageshat = projectorImageshat.permute(*axisOrderForRendering).squeeze(0)

        for iproj in range(numProjectors):
            outputImg2 = projectorImageshat[iproj * 3:iproj * 3 + 3, :, :].cpu().permute(1, 2, 0).numpy().astype(
                np.float32)
            outputImg2 = cv2.cvtColor(outputImg2, cv2.COLOR_BGR2RGB if not useLAB else cv2.COLOR_LAB2RGB)
            imageio.imwrite(imgSaveFolder + "ProjNr" + str(iproj) + ".exr", outputImg2)
            print("Saved projector ", iproj)

        renderer.updateProjectorImages(projectorImageshat.squeeze(0))
        for iview in range(numViews):
            outputImg2 = renderer.render(iview)
            outputImg2 = outputImg2.cpu().numpy().astype(np.float32)
            outputImg2 = cv2.cvtColor(outputImg2, cv2.COLOR_BGR2RGB if not useLAB else cv2.COLOR_LAB2RGB)

            imageio.imwrite(imgSaveFolder + "ViewNr" + str(iview) + ".exr", outputImg2)
            print("Saved view ", iview)
