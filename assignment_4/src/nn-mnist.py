#!/usr/bin/python

#import packages
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import argparse
import pandas as pd
import joblib

import numpy as np

from utils.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

#Create class 
class nn_mnist:

    #Create init function loading data and creating self.args with arguments from argparse
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args["mnist"])

    #Get data from dataframe format into np.array and perform min/max normalization
    def data_wrangle(self):
        self.y = np.array(self.data.y)
        self.data = self.data.drop("y", axis=1)
        self.X = np.array(self.data)
        self.X = (self.X - self.X.min())/(self.X.max() - self.X.min())

    #Make train and test split with optional test_split argument from self.args
    #Perform the labelBinarizer    
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                    self.y, 
                                                    random_state=9,
                            test_size=self.args["test_split"])
        self.y_train = LabelBinarizer().fit_transform(self.y_train)
        self.y_test = LabelBinarizer().fit_transform(self.y_test)

    #Define the neural network with specified layers from self.args
    #Train the model with specified amount of epochs
    #Save the model if save_model_path is provided
    def nn_run(self):
        
        if sum(self.args["layers"]) < int(self.X_train.shape[1]) +10:
            layers = [self.X_train.shape[1]]+ self.args["layers"]+[10]
        else:
            print(f"Number of hidden layers should be below {self.X_train.shape[1]+10}. Using default hidden layers: [32,16]")
            layers = [self.X_train.shape[1], 32,16,10]
        print("[INFO] training network...")
        
        self.nn = NeuralNetwork([self.X_train.shape[1], 32, 16, 10])
        print("[INFO] {}".format(self.nn))
        self.nn.fit(self.X_train, self.y_train, epochs=self.args["epochs"])
        if self.args["save_model_path"] != "":
            out_path = os.path.join(self.args["save_model_path"], "nn_model.pkl")
            joblib.dump(self.nn, out_path)

    #Print results to terminal
    #Save results to .csv file at desire output
    def results(self):
        predictions = self.nn.predict(self.X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(self.y_test.argmax(axis=1), predictions))
        results_df = pd.DataFrame(classification_report(self.y_test.argmax(axis=1),
                                                        predictions,
                                                      output_dict=True)).transpose()
        output_path = os.path.join(self.args["output"], "results_df_nn.csv")
        results_df.to_csv(output_path)


    #Load test_image if provided
    #Wrangle the data into the right format
    #Predict value using the neural network and print result
    def pred_new_number(self):
        test_image = cv2.imread(self.args["test_image"])
        gray = cv2.bitwise_not(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        flatten = [item for sublist in compressed for item in sublist]
        flatten_scaled=(np.array(flatten) - np.array(flatten).min())/(np.array(flatten).max() - np.array(flatten).min())
        flatten_reshaped = flatten_scaled.reshape(1, -1)
        prediction = self.nn.predict(flatten_reshaped).argmax(axis=1)
        print(f"The test image is predicted to show a {str(prediction)}")

    #Run all the functions
    def run(self):
        self.data_wrangle()
        self.split()
        self.nn_run()
        self.results()
        if self.args["test_image"] != "":
            self.pred_new_number()


def main():
    
    #Add all the terminal arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-m","--mnist", required = True,
                    help="Path to the Mnist dataset in .csv format" )
    ap.add_argument("-o","--output", required = False,
                    default = "",
                    help="Add output path to store results in a different folder")
    ap.add_argument("-ts","--test_split", required = False,
                    default = 0.2, type = float,
                    help="Add size of test data. default = 0.2")
    ap.add_argument("-l","--layers",required = False, nargs="+", default=[32,16],
                    type=int,
                    help="Add hidden layer dimensions. default=[32,16]")
    ap.add_argument("-t","--test_image", required = False,
                    default = "",
                    help="Add picture file to predict number")
    ap.add_argument("-e","--epochs", required = False,
                    default = 100, type=int,
                    help="Amount of epochs. Default=100")
    ap.add_argument("-s","--save_model_path", required = False,
                    default = "", type=str,
                    help="Define output path for saving trained neural network model")
    

    #parse arguments
    args = vars(ap.parse_args())

    #Run everything
    nn = nn_mnist(args)
    nn.run()
    
if __name__=="__main__":
    main()
