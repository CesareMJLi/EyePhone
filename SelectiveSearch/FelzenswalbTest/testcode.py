
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

    # lista = [1,2,3,4,5]
    # print(lista[:2])
    # the result is [1,2], which means the lista[0], lista[1] are selected

    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
        #  this method offered each pixels a labels marking its segmentation in the picture

    print(im_orig[0,0,:])
    # the result is [137 143 133]

    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
        # start from the shape[2], which means here it get the length and width of the original picture
    print(im_orig[0,0,:])
    # the result is [137. 143. 133.   0.] the original vector is [137,143,133] and the fourth layer is added here
    im_orig[:, :, 3] = im_mask
    # print(im_orig[:,:,3])
    skimage.io.imsave('test_1.png', im_orig[:,:,3]/256)
    # skimage.io.imsave('test_2.png', im_orig[:,:,3]/1000)

    return im_orig

# filename = os.path.join(skimage.data_dir, 'test.jpeg')
img = skimage.io.imread("/Users/cesare/python/EyePhone/SelectiveSearch/FelzenswalbTest/test.jpeg")
_generate_segments(img,scale = 1.0, sigma = 0.8, min_size = 50)