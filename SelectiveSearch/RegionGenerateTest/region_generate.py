

import skimage.io
import skimage.segmentation
import skimage.util
import numpy
import os


def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """
    
    """
    # print("IMAGE SHAPE")
    print("IMAGE SHAPE")
    print(im_orig.shape) # the shape is a tuple
    # the result is (205, 246, 3) corresponding to the width*length*depth depth is a 3-dimension vector

    print("IMAGE AS FLOAT")
    # print(skimage.util.img_as_float(im_orig))
    x = skimage.util.img_as_float(im_orig)

    print("THE LENGTH OF IM_ORIG IS %i" %len(x))
    # we got 205

    print("THE WIDTH OF IM_ORIG IS %i" %len(x[0]))
    # we got 46

    print("THE SIZE OF EACH ELEMENT IN IM_ORIG IS %i" %len(x[0][0]))
    # we got 3

    print("each element is")
    print(x[0][0])
    # open the Image
    """

    # lista = [1,2,3,4,5]
    # print(lista[:2])
    # the result is [1,2], which means the lista[0], lista[1] are selected

    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
        #  this method offered each pixels a labels marking its segmentation in the picture

    # print(im_orig[0,0,:])
    # the result is [137 143 133]

    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
        # start from the shape[2], which means here it get the length and width of the original picture
    # print(im_orig[0,0,:])
    # the result is [137. 143. 133.   0.] the original vector is [137,143,133] and the fourth layer is added here
    # im_orig[:, :, 3] = im_mask
    # print(im_orig[:,:,3])
    # skimage.io.imsave('test_1.png', im_orig[:,:,3]/256)
    # skimage.io.imsave('test_2.png', im_orig[:,:,3]/1000)

    return im_orig

def _extract_regions(img):
    # img is the result of generate regions in the beginning width * length * 4-dimension vector

    R = {}

    # get hsv image the 0,1,2 element in 4-dimension vector
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    # print(hsv)
    print("The length of hsv is %i" %len(hsv))
    print("The width of hsv is %i" %len(hsv[0]))
    print("The element of hsv is %i" %len(hsv[0][0]))
    # the shape does not change

    # pass 1: count pixel positions
    for y, i in enumerate(img):
        # >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

        for x, (r, g, b, l) in enumerate(i):

            #  l is the label
            # here x,y is the location of the pixel

            # initialize a new region

            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

            # each item in R marks a l in the result of Fel-method,  it mark the min/max x/y

    # pass 2: calculate texture gradient
    # tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region
    count = 0
    for k, v in list(R.items()):

        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        count += len(masked_pixels)
        # print("The length of MP is %i" %len(masked_pixels))
        # print("The width of MP is %i" %len(masked_pixels[0]))
        # the masked_pixels is a list

        # find the pixels in the original img with the label k
        # if k<=1:
        #     print("MASKED PIXELS")
        #     print(masked_pixels)
        
        R[k]["size"] = len(masked_pixels / 4)
        # R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        # R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    # print(count) 
    # the result is the length*width

    return R

# filename = os.path.join(skimage.data_dir, 'test.jpeg')
img = skimage.io.imread("/Users/cesare/python/EyePhone/SelectiveSearch/RegionGenerateTest/test.jpeg")
img = _generate_segments(img,scale = 1.0, sigma = 0.8, min_size = 50)

_extract_regions(img)