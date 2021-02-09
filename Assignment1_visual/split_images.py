###Importing packages
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(".."))
import cv2 #openCv

#Making sure there is an output folder
if not os.path.exists('quadrant_split'):
    os.makedirs('quadrant_split')

    
def quadrant_split(path_to_image_folder):
    
    #Locating meme_folder
    image_folder_path = os.path.join(path_to_image_folder)

    #Getting filenames
    filenames = [name for name in os.listdir(image_folder_path)]

    #Creating empty lists for storing info
    widths =[]
    heights = []
    new_filenames = []

    #Looping through files, saving dimensions and filenames, and the split images.
    for file in filenames:
        path_to_image = os.path.join(image_folder_path, file) #Make path to file
        image= cv2.imread(path_to_image) #Load image
    
        height = int(image.shape[0]) #Get height of image
        width = int(image.shape[1]) #Get width of image

        topleft = image[ 0:int(height/2),0:int(width/2)] #split image into 4 quadrants using height and width elements
        topright = image[0:int(height/2),int(width/2):width]
        bottomleft = image[int(height/2):height, 0:int(width/2)]
        bottomright = image[int(height/2):height, int(width/2):width]
    
        topleft_path = os.path.join("quadrant_split", os.path.splitext(file)[0]+"_topleft.jpg") #Save split images
        cv2.imwrite(topleft_path, topleft)
    
        topright_path = os.path.join("quadrant_split", os.path.splitext(file)[0]+"_topright.jpg")
        cv2.imwrite(topright_path, topright)
        
        bottomleft_path = os.path.join("quadrant_split", os.path.splitext(file)[0]+"_bottomleft.jpg")
        cv2.imwrite(bottomleft_path, bottomleft)
        
        bottomright_path = os.path.join("quadrant_split", os.path.splitext(file)[0]+"_bottomright.jpg")
        cv2.imwrite(bottomright_path, bottomright)
        
        widths.append(width) #Add data to lists
        heights.append(height)
        new_filenames.append(os.path.splitext(file)[0]+"_topleft.jpg")
        new_filenames.append(os.path.splitext(file)[0]+"_topright.jpg")
        new_filenames.append(os.path.splitext(file)[0]+"_bottomleft.jpg")
        new_filenames.append(os.path.splitext(file)[0]+"_bottomright.jpg")
        
    


    #Saving lists in pandas data frame
    df = pd.DataFrame(list(zip(widths,heights, new_filenames)), columns=['width', 'height', 'new_filename']) 

    #Making output path
    output_path = os.path.join("quadrant_split","new_image_data.csv")

    #Creating .csv file in the folder containing the new images
    df.to_csv(output_path)


#When executed in terminal, the script should print this guide:
def main():
    print("Creating folder with split images and .csv-file including width, height and filename of new images")
    return quadrant_split()

# Declare namespace - we'll go over this more
if __name__=="__main__":
    main()