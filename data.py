import glob
import random

def get():
    # Define the directories to recursively search for car and non-car images
    carDir = "training_images/car_images"
    notCarDir = "training_images/not_car_images"

    # Load all of the images
    isImage = lambda f: f.endswith(".png") or f.endswith(".jpg")
    images = lambda dir: [f for f in glob.iglob(dir + '/**/*', recursive=True) if isImage(f)]
    carImagePaths = images(carDir)
    notCarImagePaths = images(notCarDir)

    # Next, split into training and test data
    train = 0.8
    random.seed(0)
    random.shuffle(carImagePaths)
    random.shuffle(notCarImagePaths)
    carTrain = int(train * len(carImagePaths))
    carTrain, carValid = carImagePaths[:carTrain], carImagePaths[carTrain:]
    notCarTrain = int(train * len(notCarImagePaths))
    notCarTrain, notCarValid = notCarImagePaths[:notCarTrain], notCarImagePaths[notCarTrain:]
    return carTrain, carValid, notCarTrain, notCarValid
