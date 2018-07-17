# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch

path = "/Users/cesare/python/EyePhone/SelectiveSearch/"
filenames = ["test1.JPG","test2.JPG","test3.JPG"]

def main(filename):

    # loading astronaut image
    # img = skimage.data.astronaut()
    img  = skimage.io.imread(path+filename)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # in this part we deal with the rectangles set from SS, while we must select them, another idea is too add the parameter
        # of [CANDIDATE SIZE/LENGTH-WIDTH RATIO] into the selective search program.

        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000/1000 pixels
        # if r['size'] < 2000:
        if r['size'] < 1000:
            continue
        # distorted rects， we only want the 'squared' rectangles, nerther too long or too short
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    for n in filenames:
        main(n)
