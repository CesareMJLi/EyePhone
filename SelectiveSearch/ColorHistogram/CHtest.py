import skimage.io
import skimage.segmentation
import skimage.util
import numpy as np
import os

def _calc_colour_hist(img):
    BINS = 25
    hist = np.array([])

    for colour_channel in (0, 1, 2):
        c = img[:,:, colour_channel]
        # print(c)
        # print("THE LENGTH OF ONE CHANNEL: %i" %len(c))
        """now c is a channel, it has length of img length, each element is a array with length of img width"""
        # calculate histogram for each colour and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
        # print(hist.shape)

    # L1 normalize
    print("The length of img %i" %len(img))
    hist = hist / (len(img)*len(img[0]))
    return hist


# original color histograms
# def _calc_colour_hist(img):

#     BINS = 25
#     hist = np.array([])

#     for colour_channel in (0, 1, 2):

#         # extracting one colour channel
#         c = img[:, colour_channel]
#         print(len(c))
#         print(len(c[0]))
#         # calculate histogram for each colour and join to the result
#         hist = np.concatenate(
#             [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])

#     # L1 normalize
#     hist = hist / len(img)

#     return hist


img = skimage.io.imread("/Users/cesare/python/EyePhone/SelectiveSearch/ColorHistogram/test.jpeg")

print("IMAGE SHAPE")
print(img.shape)

hist = _calc_colour_hist(img)

print("HISTOGRAM SHAPE")
print(hist.shape)

print(hist)

print(np.sum(hist))
# the final sum is 3, the sum of each channel is 1