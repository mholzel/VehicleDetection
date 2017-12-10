from sklearn.preprocessing import StandardScaler
import data, os, pickle, utils
import numpy as np


def formatFeatures(features):
    features = np.array(features)
    return features.reshape(features.shape[0], -1)


def picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    return colorspace + "_" + str(channel) + "_" + str(orient) + "_" + str(pix_per_cell) + "_" + str(
        cell_per_block) + "_" + block_norm + ".p"


def saveFeatures(filename, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    # Define the feature_extractor
    feature_extractor = lambda img: utils.features(img, colorspace, channel, orient, pix_per_cell, cell_per_block,
                                                   block_norm)
    allFeatures = lambda dataset: [feature_extractor(utils.load(file)) for file in dataset]

    # Load the data
    labels = ['carTrain', 'carValid', 'notCarTrain', 'notCarValid']

    # Now for each dataset, compute the features
    features = {}
    for dataset, label in zip(data.get(), labels):
        features[label] = allFeatures(dataset)

    # Now pickle the features
    pickle.dump(features, open(filename, "wb"))
    return features


def loadFeatures(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm, reload=True):
    dir = 'C:/Users/matth/Desktop/pickled_features'
    filename = picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    path = os.path.join(dir, filename)
    if not os.path.isfile(path):
        # We still need to compute the features
        # Put a placeholder file so that other threads don't start this
        open(path, 'a').close()
        return saveFeatures(path, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    else:
        if reload:
            return pickle.load(open(path, "rb"))
        else:
            return None


def saveScaler(filename, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    # First, load the features
    features = loadFeatures(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    carTrainFeatures = formatFeatures(features['carTrain'])
    notCarTrainFeatures = formatFeatures(features['notCarTrain'])

    # Next, scale the training features
    X = np.vstack((carTrainFeatures, notCarTrainFeatures)).astype(np.float64)
    scaler = StandardScaler().fit(X)

    # Now pickle the scaler and return it
    pickle.dump(scaler, open(filename, "wb"))
    return scaler


def loadScaler(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm, reload=True):
    dir = 'pickled_scalers'
    filename = picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    path = os.path.join(dir, filename)
    if not os.path.isfile(path):
        # We still need to compute the scaler
        # Put a placeholder file so that other threads don't start this
        open(path, 'a').close()
        return saveScaler(path, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    else:
        if reload:
            return pickle.load(open(path, "rb"))
        else:
            return None


for colorspace in ['rgb', 'hsv', 'luv', 'ycrcb', 'hls']:
    for channel in [0, 'all']:
        for orient in [4, 8, 16]:
            for pix_per_cell in [8, 16]:
                for cell_per_block in [1, 2, 3]:
                    for block_norm in ['L1']:
                        loadScaler(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm, False)
