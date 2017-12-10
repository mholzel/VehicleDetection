import datetime

import cv2
import itertools
import math
import matplotlib.pyplot as plt
import movify
import numpy as np
import utils
from scipy.ndimage.measurements import label
from collections import deque

import defaults


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, draw_img=None, heatmap=None, color=(255, 0, 0), scale=1, svc=defaults.svc(),
              scaler=defaults.scaler()):
    img = utils.rgbTo(img, defaults.colorspace())

    # Select only the part of the image that we want to search at this scale
    height = img.shape[0]
    ystart = height // 2
    ystop = utils.ystop(height, scale)
    img = img[ystart:ystop, :, :]

    # Scale the image
    if scale != 1:
        img = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)))

    # Define dimensions that are used in calculating the size of the hog features.
    window = 64
    pix_per_cell = defaults.pix_per_cell()
    cell_per_block = defaults.cell_per_block()
    nxblocks = (img.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img.shape[0] // pix_per_cell) - cell_per_block + 1
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    orient = defaults.orient()
    block_norm = defaults.block_norm()
    hog1 = utils.get_hog_features(img[:, :, 0], orient, pix_per_cell, cell_per_block, block_norm, False, False)
    hog2 = utils.get_hog_features(img[:, :, 1], orient, pix_per_cell, cell_per_block, block_norm, False, False)
    hog3 = utils.get_hog_features(img[:, :, 2], orient, pix_per_cell, cell_per_block, block_norm, False, False)
    for xb in range(nxsteps):
        for yb in range(nysteps):

            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            if ypos + nblocks_per_window >= hog1.shape[0] or xpos + nblocks_per_window >= hog1.shape[1]:
                continue

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1, -1)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Scale features and make a prediction
            try:
                test_features = scaler.transform(hog_features)
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    y = ytop_draw + ystart
                    x = xbox_left
                    if heatmap is not None:
                        heatmap[y:y + win_draw, x:x + win_draw] += 1
                    if draw_img is not None:
                        cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                      (xbox_left + win_draw, ytop_draw + win_draw + ystart), color, 6)
            except Exception as e:
                print(e)


def multiscale_finder(imgs, threshold=2, scales=defaults.scales(), display=False, svc=defaults.svc(),
                      scaler=defaults.scaler()):
    # If imgs is a numpy object, then assume that imgs is already a loaded image
    if 'numpy' in str(type(imgs)):
        imgs = [imgs]

    # Create space for the subplots if displaying the output
    if display:
        cols = 4
        rows = math.ceil(len(imgs) / cols)
        fig, ax = plt.subplots(rows, cols)
        ax = np.array(ax).ravel()
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0, left=0, top=1, right=1)

    # Now, for each image, find the cars
    for imgIndex, img in enumerate(imgs):

        heatmap = np.zeros(img.shape[:2], dtype=np.uint8)
        drawing_img = img.copy()
        colors = itertools.cycle([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        for scale in scales:
            find_cars(img, drawing_img, heatmap, next(colors), scale, svc, scaler)

        # Now remove points form the heatmap with a value less than the threshold
        if threshold >= 1:
            heatmap[heatmap < threshold] = 0

        if display:
            # Now let's scale the heatmap so that the largest value is 255
            heatmapImage = np.uint8(255 * (heatmap / np.max(heatmap)))

            h, w, c = img.shape
            combined = np.zeros((2 * h, w, c), dtype=np.uint8)
            for i in range(c):
                combined[:h, :, i] = drawing_img[:, :, i]
                combined[h:, :, i] = heatmapImage
            ax[imgIndex].imshow(combined)
            ax[imgIndex].axis('off')

    if display:
        plt.show()
    return drawing_img, heatmap, imgs


class Summer():
    def __init__(self, n, threshold):
        self.n = n
        self.frames = deque()
        self.summed = None
        self.threshold = threshold

    def addFrame(self, frame):
        ''' Add a frame to the deque of frames. '''
        if self.summed is None:
            self.summed = np.copy(frame)
            return np.copy(self.summed)
        elif len(self.frames) == self.n:
            # If the deque already holds the max number of elements...
            removed = self.frames.popleft()
            self.summed -= removed
        self.frames.append(np.copy(frame))
        self.summed += frame
        return np.copy(self.summed)


def label_heatmap(heatmap):

    width = heatmap.shape[1]
    labels, n = label(heatmap)
    tmp = np.zeros_like(heatmap)
    boxes = []
    for i in range(1, n + 1):
        rows, cols = np.where(labels == i)
        mincol = min(cols)
        maxcol = max(cols)
        minrow = min(rows)
        maxrow = max(rows)
        if mincol + maxcol > width:
            boxes.append([(mincol, minrow), (maxcol, maxrow)])
            tmp[minrow:maxrow+1,mincol:maxcol+1] = 1

    oldN = n
    labels, n = label(tmp)

    while oldN != n:
        tmp = np.zeros_like(tmp)
        boxes = []
        for i in range(1, n + 1):
            rows, cols = np.where(labels == i)
            mincol = min(cols)
            maxcol = max(cols)
            minrow = min(rows)
            maxrow = max(rows)
            if mincol + maxcol > width:
                boxes.append([(mincol, minrow), (maxcol, maxrow)])
                tmp[minrow:maxrow + 1, mincol:maxcol + 1] = 1

        oldN = n
        labels, n = label(tmp)
    return boxes


def draw_with_labels(imgs, summer=None, scales=defaults.scales(), threshold=2, display=False, svc=defaults.svc(),
                     scaler=defaults.scaler()):
    if summer is not None:
        thresh = threshold
        threshold = summer.threshold
    drawing_img, heatmap, imgs = multiscale_finder(imgs, scales=scales, threshold=threshold, display=False, svc=svc,
                                                   scaler=scaler)
    if summer is not None:
        heatmap = summer.addFrame(heatmap)
        heatmap[heatmap < thresh] = 0
    boxes = label_heatmap(heatmap)

    # Now let's show all of the bounding boxes
    img = imgs[0].copy()
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], (255, 0, 0), 3)
    if display:
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0, left=0, top=1, right=1)
        ax.imshow(img)
        ax.axis('off')
        plt.pause(1)
    return img


if __name__ == "__main__":
    if False:
        imgs = [utils.load('project_video/frame' + str(1000 + 2 * i) + '.jpg') for i in range(10)]
        if True:
            for img in imgs:
                scales = [1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
                draw_with_labels(img, display=True)
                plt.pause(.01)
            plt.pause(100)
        else:
            multiscale_finder(imgs, display=True)
    else:
        inputVideo = 'project_video'
        svc = defaults.svc()
        scaler = defaults.scaler()
        scales = [1.75, 2, 2.25, 2.5, 2.75, 3, 3.25 ]
        # scales = [1.5,2,2.5,3.5]

        # scales = defaults.scales()
        scaleString = '_'.join((str(scale) for scale in scales))
        s = 30
        thresh1 = 4
        summer = Summer(s, thresh1)
        threshold = 150
        stamp = str(s) + "_" + str(thresh1) + "_" + str(threshold) + "_xxx_" + scaleString
        movify.convert(inputVideo + '.mp4', inputVideo + '_' + stamp + '.mp4',
                       lambda x: draw_with_labels(x, scales=scales, threshold=threshold, svc=svc, scaler=scaler,
                                                  summer=summer))
