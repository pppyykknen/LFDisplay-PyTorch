import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim



inFolder = "./renderedUNET/ViewNr"
# inFolder = "./renderedImgsTensPrimRes/ViewNr"
# inFolder2 = "./renderedImgsTensPrimRes/ViewNr"
inFolderGT = "D:/lightfields/LFDisplay/GroundTrueImages/"
# inFolderGT = "D:/lightfields/LFDisplay/bunnyfur/GroundTrueImages/"
# inFolderGT = "D:/lightfields/LFDisplay/dragon/GroundTrueImages/"

# inFolderUnoptimized = "D:/lightfields/LFDisplay/PerceivedImages_0000/"
# inFolderOptimized50iter = "D:/lightfields/LFDisplay/PerceivedImages_0050/"
# inFolderxy = "./renderedUNETxy/ViewNr"
# inFolderyz = "./renderedUNETyz/ViewNr"
# inFolderxz = "./renderedUNETxz/ViewNr"

inFolderxy = "./renderedUNETxy/ProjNr"
inFolderyz = "./renderedUNETyz/ProjNr"
inFolderxz = "./renderedUNETxz/ProjNr"

nViews = 41
downscale = 1.5
width = int(800 // downscale)
height = int(600 // downscale)
imgsUnopt = np.zeros((nViews, width, height, 3))
imgsPyT = np.zeros((nViews, width, height, 3))
imgsGT = np.zeros((nViews, width, height, 3))
imgsOpt50 = np.zeros((nViews, width, height, 3))
imgsxy = np.zeros((nViews, width, height, 3))
imgsyz = np.zeros((nViews, width, height, 3))
imgsxz = np.zeros((nViews, width, height, 3))

psnrs = np.zeros((3, nViews))
ssims = np.zeros((3, nViews))

# Load images, calculate SSIM/PSNR and crop for visualization. Save in numpy arrays
for ii in range(0, nViews, 1):
    # img = cv2.imread(inFolder + str(ii) + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #
    # imgsPyT[ii, :, :, :] = cv2.resize(img, (height, width))
    imgGT = cv2.imread(inFolderGT + "{0:04d}".format(ii) + ".exr",
                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    imgGT = np.clip(imgGT,0,1)
    imgsGT[ii, :, :, :] = cv2.resize(imgGT, (height, width))

    # img = cv2.imread(inFolderUnoptimized + "{0:04d}".format(ii) + ".exr",
    #                  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # imgsUnopt[ii, :, :, :] = cv2.resize(img, (height, width))
    # img = cv2.imread(inFolderOptimized50iter + "{0:04d}".format(ii) + ".exr",
    #                  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # imgsOpt50[ii, :, :, :] = cv2.resize(img, (height, width))
    img = cv2.imread(inFolderyz + str(ii) + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    imgsyz[ii, :, :, :] = cv2.resize(img, (height, width))

    psnrs[0, ii] = psnr(imgGT, img)
    ssims[0, ii] = ssim(imgGT, img, multichannel=True)

    img = cv2.imread(inFolderxy + str(ii) + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    imgsxy[ii, :, :, :] = cv2.resize(img, (height, width))

    psnrs[1, ii] = psnr(imgGT, img)
    ssims[1, ii] = ssim(imgGT, img, multichannel=True)

    img = cv2.imread(inFolderxz + str(ii) + ".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    imgsxz[ii, :, :, :] = cv2.resize(img, (height, width))

    psnrs[2, ii] = psnr(imgGT, img)
    ssims[2, ii] = ssim(imgGT, img, multichannel=True)

# out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (810, 1136), True)


def cvSubplot(imgs,  # 2d np array of imgs (each img an np arrays of depth 1 or 3).
              pad=10,  # number of pixels to use for padding between images. must be even
              titles=None,  # (optional) np array of subplot titles
              win_name='CV Subplot'  # name of cv2 window
              ):
    '''
    Makes cv2 based subplots. Useful to plot image in actual pixel size
    '''
    # print(np.shape(imgs))
    rows, cols = np.shape(imgs)[0:2]
    subplot_shapes = np.array([list(map(np.shape, x)) for x in imgs])
    sp_height, sp_width, depth = np.max(np.max(subplot_shapes, axis=0), axis=0)

    title_pad = 30
    if titles is not None:
        pad_top = pad + title_pad
    else:
        pad_top = pad

    frame = np.zeros((rows * (sp_height + pad_top), cols * (sp_width + pad), depth))

    for r in range(rows):
        for c in range(cols):
            img = imgs[r, c]
            h, w, _ = img.shape
            y0 = r * (sp_height + pad_top) + pad_top // 2
            x0 = c * (sp_width + pad) + pad // 2
            frame[y0:y0 + h, x0:x0 + w, :] = img

            if titles is not None:
                frame = cv2.putText(frame, titles[r, c], (x0, y0 - title_pad // 4), cv2.FONT_HERSHEY_COMPLEX, .5,
                                    (255, 255, 255))
    # print(frame.depth())
    # print(np.shape(frame))
    # array = frame
    # normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))  # this set the range from 0 till 1
    # img_array = (normalized_array * 255).astype(np.uint8)
    # out.write(frame.astype(np.uint8))
    cv2.imshow(win_name, frame)

cv2.imshow('CV Subplot', np.zeros((1,1)))
cv2.waitKey(0)
def vis(ii):
    # Main visualization function using cvSubplot function to display one frame of images

    # imgPyT = imgsPyT[ii]
    # imgUn = imgsUnopt[ii]
    # imgGT = imgsGT[ii]
    # imgOpt = imgsOpt50[ii]


    imgxy = imgsxy[ii]
    imgxz = imgsxz[ii]
    imgGT = imgsGT[ii]
    imgyz = imgsyz[ii]

    titles = np.array([
        ["GT: View " + str(ii),
         "Unoptimized: " + " {:.2f}/{:.2f} ".format(psnrs[0, ii], ssims[0, ii]),
        "50 Iter optimized: " + " {:.2f}/{:.2f} ".format(psnrs[1, ii], ssims[1, ii]),
         "UNET: " + " {:.2f}/{:.2f} ".format(psnrs[2, ii], ssims[2, ii])]])
    # img = np.array([[imgGT, imgUn], [imgOpt, imgPyT]])

    titles = np.array([
        ["GT: View " + str(ii),
         "YZ: " + " {:.2f}/{:.2f} ".format(psnrs[0, ii], ssims[0, ii]),
        "XY: " + " {:.2f}/{:.2f} ".format(psnrs[1, ii], ssims[1, ii]),
         "XZ: " + " {:.2f}/{:.2f} ".format(psnrs[2, ii], ssims[2, ii])]])
    img = np.array([[imgGT, imgyz, imgxy, imgxz]])
    cvSubplot(img, pad=5, titles=titles)
    cv2.waitKey(30)


flag = 1
while (flag):
    for ii in range(0, nViews, 1):  # [0,25,50,75,100]:
        vis(ii)

    for ii in reversed(range(0, nViews, 1)):  # [0,25,50,75,100]:
        vis(ii)
    # flag = 0
# out.release()
