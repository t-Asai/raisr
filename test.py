import cv2
import numpy as np
import os
import pickle
import sys
from gaussian2d import gaussian2d
from gettestargs import gettestargs
from hashkey import hashkey
from math import floor
# from matplotlib import pyplot as plt
import multiprocessing
from scipy import interpolate
from utils import getImageList, argwrapper


args = gettestargs()

# Define parameters
R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize / 2)
patchmargin = floor(patchsize / 2)
gradientmargin = floor(gradientsize / 2)

# Read filter from file
filtername = 'filter.p'
if args.filter:
    filtername = args.filter
with open(filtername, "rb") as fp:
    h = pickle.load(fp)

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

imagelist = getImageList()


def calculateValues(numOfRows, upscaledLR, row, col):
    if round(row * 100 / numOfRows) \
            != round((row + 1) * 100 / numOfRows):
        print('|{sharps}{spaces}|  {percent}%'.format(**{
            'sharps': '#' * round((row + 1) *
                                  100 / numOfRows / 2),
            'spaces': ' ' * (50 - round((row + 1) *
                                        100 / numOfRows / 2)),
            'percent': round((row + 1) * 100 / numOfRows)
        }), end='\r')
    # Get patch
    patch = upscaledLR[row - patchmargin: row + patchmargin + 1,
                       col - patchmargin: col + patchmargin + 1]
    patch = patch.ravel()
    # Get gradient block
    gradientblock = upscaledLR[row - gradientmargin: row + gradientmargin + 1,
                               col - gradientmargin: col + gradientmargin + 1]
    # Calculate hashkey
    angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
    # Get pixel type
    pixeltype = ((row - margin) % R) * R + ((col - margin) % R)
    return {
        "row": row - margin,
        "col": col - margin,
        "value": patch.dot(h[angle, strength, coherence, pixeltype])
    }


imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rUpscaling image {imagecount} of {len_list} ({image})'.format(**{
        'imagecount': imagecount,
        'len_list': len(imagelist),
        'image': image
    }))
    origin = cv2.imread(image)
    # Extract only the luminance in YCbCr
    ycrcvorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    grayorigin = ycrcvorigin[:, :, 0]
    # Normalized to [0,1]
    grayorigin = cv2.normalize(
        grayorigin.astype('float'),
        None,
        grayorigin.min() / 255,
        grayorigin.max() / 255,
        cv2.NORM_MINMAX
    )
    # Upscale (bilinear interpolation)
    heightLR, widthLR = grayorigin.shape
    heightgridLR = np.linspace(0, heightLR - 1, heightLR)
    widthgridLR = np.linspace(0, widthLR - 1, widthLR)
    bilinearinterp = interpolate.interp2d(
        widthgridLR, heightgridLR, grayorigin, kind='linear')
    heightgridHR = np.linspace(0, heightLR - 0.5, heightLR * 2)
    widthgridHR = np.linspace(0, widthLR - 0.5, widthLR * 2)
    upscaledLR = bilinearinterp(widthgridHR, heightgridHR)
    # Calculate predictHR pixels
    heightHR, widthHR = upscaledLR.shape
    predictHR = np.zeros((heightHR - 2 * margin, widthHR - 2 * margin))

    poo = multiprocessing.Pool()
    funcArgs = [(calculateValues, heightHR - margin,
                 upscaledLR, row, col)
                for row in range(margin, heightHR - margin)
                for col in range(margin, widthHR - margin)]
    results = poo.map(argwrapper, funcArgs)
    poo.close()
    poo.join()
    for result in results:
        predictHR[result["row"], result["col"]] = result["value"]

    # Scale back to [0,255]
    predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)
    # Bilinear interpolation on CbCr field
    result = np.zeros((heightHR, widthHR, 3))
    y = ycrcvorigin[:, :, 0]
    bilinearinterp = interpolate.interp2d(
        widthgridLR, heightgridLR, y, kind='linear')
    result[:, :, 0] = bilinearinterp(widthgridHR, heightgridHR)
    cr = ycrcvorigin[:, :, 1]
    bilinearinterp = interpolate.interp2d(
        widthgridLR, heightgridLR, cr, kind='linear')
    result[:, :, 1] = bilinearinterp(widthgridHR, heightgridHR)
    cv = ycrcvorigin[:, :, 2]
    bilinearinterp = interpolate.interp2d(
        widthgridLR, heightgridLR, cv, kind='linear')
    result[:, :, 2] = bilinearinterp(widthgridHR, heightgridHR)
    result[margin:heightHR - margin, margin:widthHR - margin, 0] = predictHR
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    cv2.imwrite('results/' + os.path.splitext(os.path.basename(image))
                [0] + '_result.bmp', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    imagecount += 1
    # Visualizing the process of RAISR image upscaling
    # if args.plot:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 4, 1)
    #     ax.imshow(grayorigin, cmap='gray', interpolation='none')
    #     ax = fig.add_subplot(1, 4, 2)
    #     ax.imshow(upscaledLR, cmap='gray', interpolation='none')
    #     ax = fig.add_subplot(1, 4, 3)
    #     ax.imshow(predictHR, cmap='gray', interpolation='none')
    #     ax = fig.add_subplot(1, 4, 4)
    #     ax.imshow(result, interpolation='none')
    #     plt.show()

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
