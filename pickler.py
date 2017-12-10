import data, os, pickle, utils
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


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


def formatFeatures(features):
    features = np.array(features)
    return features.reshape(features.shape[0], -1)


def train(clf, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    # First, load the features
    features = loadFeatures(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    carTrainFeatures = formatFeatures(features['carTrain'])
    notCarTrainFeatures = formatFeatures(features['notCarTrain'])

    # Next, scale the training features
    X = np.vstack((carTrainFeatures, notCarTrainFeatures)).astype(np.float64)
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)

    # Define the training labels
    y = np.hstack((np.ones(len(carTrainFeatures)), np.zeros(len(notCarTrainFeatures))))

    # and fit the classifier to the data
    clf.fit(scaled_X, y)


def validate(clf, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    # First, load the features
    features = loadFeatures(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    carTrainFeatures = formatFeatures(features['carTrain'])
    carValidFeatures = formatFeatures(features['carValid'])
    notCarTrainFeatures = formatFeatures(features['notCarTrain'])
    notCarValidFeatures = formatFeatures(features['notCarValid'])

    # Next, generate a scaler based on the training features
    X = np.vstack((carTrainFeatures, notCarTrainFeatures)).astype(np.float64)
    scaler = StandardScaler().fit(X)

    # Finally, compute the score
    X = np.vstack((carValidFeatures, notCarValidFeatures)).astype(np.float64)
    scaled_X = scaler.transform(X)
    y = np.hstack((np.ones(len(carValidFeatures)), np.zeros(len(notCarValidFeatures))))
    return clf.score(scaled_X, y)


def saveSVM(filename, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
    # Train and validate the SVM
    clf = LinearSVC()
    train(clf, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)

    # Then pickle it and return it
    pickle.dump(clf, open(filename, "wb"))
    return clf


def loadSVM(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm, reload=True):
    dir = 'pickled_svms'
    filename = picklename(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
    path = os.path.join(dir, filename)
    if not os.path.isfile(path):
        # We still need to train the svm
        # Put a placeholder file so that other threads don't start this
        open(path, 'a').close()
        return saveSVM(path, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
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


class Params():
    def __init__(self, colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm):
        self.colorspace = colorspace
        self.channel = channel
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.block_norm = block_norm

    def fields(self):
        return (self.colorspace, self.channel, self.orient, self.pix_per_cell, self.cell_per_block, self.block_norm)

    def __eq__(self, rhs):
        return self.fields() == rhs.fields()

    def __hash__(self):
        return hash(self.fields())

    def __str__(self):
        s = ''
        s += 'colorspace: ' + self.colorspace + ', '
        s += 'channel: ' + str(self.channel) + ', '
        s += 'orient: ' + str(self.orient) + ', '
        s += 'pix_per_cell: ' + str(self.pix_per_cell) + ', '
        s += 'cell_per_block: ' + str(self.cell_per_block) + ', '
        s += 'block_norm: ' + self.block_norm
        return s


def loadSVMScores():
    # First, load our scores from file
    filename = 'pickled_svms/scores.p'
    if os.path.isfile(filename):
        scores = pickle.load(open(filename, "rb"))
    else:
        scores = {}

    # Now compute all of the scores that aren't already in the dictionary
    for colorspace in ['rgb', 'hsv', 'luv', 'ycrcb', 'hls']:
        for channel in [0, 'all']:
            for orient in [4, 8, 16]:
                for pix_per_cell in [8, 16]:
                    for cell_per_block in [1, 2, 3]:
                        for block_norm in ['L1']:
                            params = Params(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
                            if params not in scores:
                                timer = utils.Timer()
                                svm = loadSVM(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm)
                                scores[params] = validate(svm, colorspace, channel, orient, pix_per_cell,
                                                          cell_per_block, block_norm)
                                timer.print("seconds to validate")
                                pickle.dump(scores, open(filename, "wb"))
    return scores

if __name__ == "__main__":
    for colorspace in ['rgb', 'hsv', 'luv', 'ycrcb', 'hls']:
        for channel in [0, 'all']:
            for orient in [4, 8, 16]:
                for pix_per_cell in [8, 16]:
                    for cell_per_block in [1, 2, 3]:
                        for block_norm in ['L1']:
                            loadScaler(colorspace, channel, orient, pix_per_cell, cell_per_block, block_norm, False)
