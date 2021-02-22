#Import packages
import os
import sys
import re
import matplotlib as mpl
import pandas as pd

sys.path.append(os.path.join(".."))

import cv2
import numpy as np

import matplotlib.pyplot as plt
import glob

#Create function
def get_img_histsim(image_path, image_folder): #Taking in path for comparison image, and folder with images
    path = os.path.join(image_folder) #create path
    images = glob.glob(path+"/*.jpg") #Get all .jpg files from path
    comp_img = cv2.imread(image_path) #read the comparison image
    comp_img_hist = cv2.calcHist([comp_img], [0,1,2], None, histSize=[8,8,8], ranges=[0,256,0,256,0,256]) #make the comparison image histogram
    comp_img_hist_nm = cv2.normalize(comp_img_hist, comp_img_hist, 0,255,cv2.NORM_MINMAX) #MinMax nomralize the histogram
    
    if image_path in images:
        images.remove(image_path) #Remove the comparison image from the folder, if is is in there
    
    df = pd.DataFrame([], columns=["file", "chisq"]) #create dataframe for storing filenames and chisquared values
    for image_file in images: #loop through each file
        image = cv2.imread(os.path.join(image_file)) #load image

        hist = cv2.calcHist([image], [0,1,2], None, histSize=[8,8,8], ranges=[0,256,0,256,0,256]) #create histogram
    
        hist_nm = cv2.normalize(hist, hist, 0,255,cv2.NORM_MINMAX) #normalize histogram

        chisq_score = round(cv2.compareHist(comp_img_hist_nm, hist_nm, method = cv2.HISTCMP_CHISQR),2) #Get the chisquared score and rounf to 2 decimal places
        df2 = pd.DataFrame([[image_file,chisq_score]], columns=["file", "chisq"]) #Make small dataframe from the results
        df = df.append(df2, ignore_index=True) #append small dataframe to the total
    return df #return dataframe

def main(image_path = "jpg/image_0001.jpg",image_folder= "jpg"): #I specified the same function for main with minor adjustments
    path = os.path.join(image_folder)
    images = glob.glob(path+"/*.jpg")

    comp_img = cv2.imread(image_path)
    comp_img_hist = cv2.calcHist([comp_img], [0,1,2], None, histSize=[8,8,8], ranges=[0,256,0,256,0,256])
    comp_img_hist_nm = cv2.normalize(comp_img_hist, comp_img_hist, 0,255,cv2.NORM_MINMAX)
    
    if image_path in images:
        images.remove(image_path)
    
    df = pd.DataFrame([], columns=["file", "chisq"])
    for image_file in images:
        image = cv2.imread(os.path.join(image_file))

        hist = cv2.calcHist([image], [0,1,2], None, histSize=[8,8,8], ranges=[0,256,0,256,0,256])
    
        hist_nm = cv2.normalize(hist, hist, 0,255,cv2.NORM_MINMAX)

        chisq_score = round(cv2.compareHist(comp_img_hist_nm, hist_nm, method = cv2.HISTCMP_CHISQR),2)
        df2 = pd.DataFrame([[image_file,chisq_score]], columns=["file", "chisq"])
        df = df.append(df2, ignore_index=True)
    df.to_csv("image_similarity.csv") #Here i write the df to a csv file in order to save the results
    
if __name__=="__main__":
    main()
