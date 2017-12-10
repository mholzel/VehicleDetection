import cv2, defaults, math, time
from skimage.feature import hog


def load(imgPath):
    # If the imgPath is not a string, then we assume that it is already an image
    if isinstance(imgPath, str):
        return cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    else:
        return imgPath


def get_hog_features(img, orient, pix_per_cell, cell_per_block, block_norm, vis, feature_vec):
    '''
    Get HOG features for square images
    '''
    return hog(img,
               orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               block_norm=block_norm,
               visualise=vis, feature_vector=feature_vec)


def features(image,
             colorspace=defaults.colorspace(),
             channel=defaults.channel(),
             orient=defaults.orient(),
             pix_per_cell=defaults.pix_per_cell(),
             cell_per_block=defaults.cell_per_block(),
             block_norm=defaults.block_norm(),
             feature_vec=True):
    '''
    Get the feature vector for the specified parameters, deferring to the defaults specified in defaults.py
    if an argument is not provided.
    '''
    image = rgbTo(image, colorspace=colorspace)
    if len(image.shape) < 3:
        return get_hog_features(image, orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, vis=False, feature_vec=feature_vec, block_norm=block_norm)
    elif channel == 'all':
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(
                get_hog_features(image[:, :, channel], orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, vis=False, feature_vec=feature_vec, block_norm=block_norm))
        return hog_features
    else:
        return get_hog_features(image[:, :, channel], orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, vis=False, feature_vec=feature_vec, block_norm=block_norm)


def rgbTo(image, colorspace='rgb'):
    colorspace = colorspace.lower()
    if colorspace == 'rgb':
        return image
    elif colorspace == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif colorspace == 'hls':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif colorspace == 'gray':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif colorspace == 'ycrcb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif colorspace == 'luv':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)


class Timer():
    def __init__(self):
        self.t0 = time.time()

    def print(self, message):
        ''' Print the elapsed time with a message '''
        t = time.time()
        print(round(t - self.t0, 5), message)
        self.t0 = t

    def reset(self):
        self.t0 = time.time()
        
def ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def ystop(height, scale):
    if scale <= 1:
        return math.floor(height * .66)
    elif scale >= 4:
        return height
    else:
        return math.floor(height * (.66 + (scale - 1) / 3 * (1 - .66)))

