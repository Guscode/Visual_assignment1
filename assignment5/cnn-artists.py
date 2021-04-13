# data tools
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import pandas as pd

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

#Define plot_history function
def plot_history(H, epochs):
    # visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
#Define main function
def main():

    #Add the terminal argument
    ap = argparse.ArgumentParser()
    
    #Let users define # of epochs
    ap.add_argument("-e","--epochs", default = 10,type = int,
                    help="Specify amount of epochs, default: 10" )
    
    ap.add_argument("-t","--training_path", required=True,type = str,
                    help="Specify training path" )
    ap.add_argument("-v","--validation_path", required=True,type = str,
                    help="Specify validation path" )

    #parse arguments
    args = vars(ap.parse_args())
    
    #Path to training data
    train_path = args["training_path"]
    
    #Listing artists
    artists = os.listdir(train_path)
    artists.pop(4)
    
    #Define empty lists to store image paths and Y values for training data
    all_img=[]
    trainY = []
    
    #Loop through artists and extract filenames
    for i in artists:
        imgs = os.listdir(os.path.join(train_path, i))
        
        #Add artist names and image paths to lists
        trainY.extend(np.repeat(i, len(os.listdir(os.path.join(train_path, i)))  ))
        all_img = all_img +["/".join([train_path, i, n]) for n in imgs]

    
    #same process for testing data
    test_path = args["validation_path"]
    
    #Empty lists for storing testing image paths and Y values
    test_img=[]
    testY = []
    
    #looping through each artist in the validation folder
    for i in artists:
        imgs = os.listdir(os.path.join(test_path, i))
        testY.extend(np.repeat(i, len(os.listdir(os.path.join(test_path, i)))  ))
        test_img = test_img +["/".join([test_path, i, n]) for n in imgs]
        
    #Defining dimensions to resize the images    
    dim = (200,200)

    #Loading each image, resizing them and saving in list
    trainX = [cv2.resize(cv2.imread(filepath), dim, interpolation = cv2.INTER_AREA) for filepath in all_img]
    testX = [cv2.resize(cv2.imread(filepath), dim, interpolation = cv2.INTER_AREA) for filepath in test_img]
    
    #Normalize images
    trainX = np.array(trainX).astype('float')/255.
    testX = np.array(testX).astype('float')/255.
    
    # converting artist names to one-hot vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # initialize label names
    labelNames = artists
    
    # define model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), 
                     padding="same", 
                     input_shape=(200, 200, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # FC => RELU
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
    
    # train model
    H = model.fit(trainX, trainY, 
              validation_data=(testX, testY), 
              batch_size=32,
              epochs=args["epochs"],
              verbose=1)
    
    #Predicting the test set
    predictions = model.predict(testX, batch_size=32)
    
    #Saving classification report 
    results_df = pd.DataFrame(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames,
                                output_dict=True)).transpose()
    results_df.to_csv("classification_report.csv")
    
    #Saving history plot
    plt.savefig("history_plot.png",plot_history(H,args["epochs"]))
    
if __name__=="__main__":
    main()
