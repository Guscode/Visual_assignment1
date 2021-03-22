#!/usr/bin/python

#import packages
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import argparse
import pandas as pd

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#Create class 
class lr_mnist:

    #Create init function loading data and creating self.args with arguments from argparse
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args["mnist"])

    #Get data from dataframe format into np.array  and define classes
    def data_wrangle(self):
        self.y = np.array(self.data.y)
        self.data = self.data.drop("y", axis=1)
        self.X = np.array(self.data)
        self.classes = sorted(set(self.y))
        self.nclasses = len(self.classes)

    #Make train and test split with optional test_split argument from self.args
    def split(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                    self.y, 
                                                    random_state=9,
                                                    test_size=self.args["test_split"])
        self.X_train_scaled = X_train/255.0
        self.X_test_scaled = X_test/255.0

    #Run logistic regression with specified penalty and solver
    def log_reg(self):
        self.clf = LogisticRegression(penalty=self.args["penalty"], 
                         tol=0.1, 
                         solver=self.args["solver"],
                         multi_class='multinomial').fit(self.X_train_scaled,self.y_train)
    

    #Print results to terminal
    #Save results to .csv file at desire output
    def results(self):
        y_pred = self.clf.predict(self.X_test_scaled)
        cm = metrics.classification_report(self.y_test, y_pred)
        print(cm)
        results_df = pd.DataFrame(metrics.classification_report(self.y_test, y_pred, output_dict=True)).transpose()
        output_path = os.path.join(self.args["output"], "results_df.csv")
        results_df.to_csv(output_path)


    #Load test_image if provided
    #Wrangle the data into the right format
    #Predict value using the neural network and print result
    def pred_new_number(self):
        test_image = cv2.imread(self.args["test_image"])
        gray = cv2.bitwise_not(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        flatten = [item for sublist in compressed for item in sublist]
        flatten_scaled=np.array(flatten)/255.0
        flatten_reshaped = flatten_scaled.reshape(1, -1)
        prediction = self.clf.predict(flatten_reshaped)
        print(f"The test image is predicted to show a {str(prediction)}")

    #Run all the functions
    def run(self):
        self.data_wrangle()
        self.split()
        self.log_reg()
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
    ap.add_argument("-s","--solver", type = str, required = False,
                    default = "saga",
                    help="Add solver algorithm - can be 'newton-cg', ‘sag’, ‘saga’ and ‘lbfgs’. default='saga'.")
    ap.add_argument("-ts","--test_split", required = False,
                    default = 0.2, type = float,
                    help="Add size of test data. default = 0.2")
    ap.add_argument("-p","--penalty",  type = str,required = False,
                    default = "none",
                    help="Add norm used in penalization. Can be 'l2' or None. default=None.")
    ap.add_argument("-t","--test_image", required = False,
                    default = "",
                    help="Add picture file to predict number")

    #parse arguments
    args = vars(ap.parse_args())

    #Run everything
    lr = lr_mnist(args)
    lr.run()
    
if __name__=="__main__":
    main()
