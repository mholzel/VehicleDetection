import os, pickle


def block_norm():
    return 'L1'


def colorspace():
    return 'ycrcb'


def channel():
    return 'all'


def orient():
    return 8


def pix_per_cell():
    return 16


def cell_per_block():
    return 3


def cv2Colors():
    return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]


def scales():
    minScale = 2
    maxScale = 4
    step = .5
    return [minScale + step * i for i in range(int(maxScale - minScale / step + 1))]


def picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    return colorspace + "_" + str(channel) + "_" + str(orient) + "_" + str(pix_per_cell) + "_" + str(
        cell_per_block) + "_" + block_norm + ".p"


def svc(colorspace=colorspace(), channel=channel(), orient=orient(), pix_per_cell=pix_per_cell(),
        cell_per_block=cell_per_block(), block_norm=block_norm()):
    filename = picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    path = os.path.join('pickled_svms', filename)
    return pickle.load(open(path, "rb"))


def scaler(colorspace=colorspace(), channel=channel(), orient=orient(), pix_per_cell=pix_per_cell(),
           cell_per_block=cell_per_block(), block_norm=block_norm()):
    filename = picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    path = os.path.join('pickled_scalers', filename)
    return pickle.load(open(path, "rb"))
