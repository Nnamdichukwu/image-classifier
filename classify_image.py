import os
from pathlib import Path
from PIL import Image
from resizeimage import resizeimage
import sys
import numpy as np
import shutil
import json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras.models import load_model

default_model = 'default_model.h5'
model_extension = ".h5"
json_file = 'classification_results.json'  # the file name
image_extensions = ('jpeg', 'png', 'jpg', 'JPG')  # add others

truth_values = ['yes', 'y','true','1', 'on']
false_values = ['no','n', 'false','0', 'off']



def import_model(model_name):
    model= default_model
 
    classifier = load_model(model)

    return classifier

def predictor(input_type, folder_or_image, model, directory_folder=None):
    """
    Accepts either a folder or an image, and a model argument that's the ML model
    to use for the function. 

    """

    # Load the model
   
    model = default_model
    classifier = import_model(model)

    
    folder_ = folder_or_image
    prediction = []  # list of file names that are the prediction
    not_prediction = []  # list of file names that are not the prediction
    #First create prediction folder inside provided folder
   
        
    prediction_folder = os.path.join(
            directory_folder, 'hotel_image')
    not_prediction_folder = os.path.join(
            directory_folder, 'not_hotel_image')
        
    # Create the folders using their paths
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)
        
    if not os.path.exists(not_prediction_folder):
         os.mkdir(not_prediction_folder)
        
    for image in prediction_folder:
        img_exists= os.path.isfile(image)
        if img_exists:
            continue
    for images in not_prediction_folder:
        img_exist= os.path.isfile(images)
        if img_exist:
            continue
    #resize width if larger than 1024
    with os.scandir(prediction_folder) as imgs:
        for img in imgs:
            im= Image.open(open(img,'rb'))
            try:
               if im.size[0] > 1024:
                  new_img_width= resizeimage.resize_width(im,1024)
            except:
                pass 
             
    #We need to exclude the prediction folder as we walk the provided directory
    exclude = set(['predictions'])
    for root, dirs, files in os.walk(folder_):
        # Below code modifies dirs in place using the default topdown=True, check os.walk help doc
        dirs[:]= [d for d in dirs if d not in exclude]
        
        for file in tqdm(files):

            # Didn't remove the file-type validation here as some files in the supplied
            # directory may not be images, unlike up where only an image is supplied.
            if file.lower().endswith(image_extensions):
                outcome = test(classifier, os.path.join(root, file))

                # Add to JSON list and then copy it to its respective folder
                
                 
                if outcome == True:
                    not_prediction.append(file)
                    shutil.copy(os.path.join(root, file),
                                not_prediction_folder)
                

                else:
                    prediction.append(file)
                    shutil.copy(os.path.join(root, file),
                                prediction_folder)

        
        prediction.clear() # clear the list containing the prediction names for use in the next iterated folder
        not_prediction.clear()  # Do the same for the not_prediction list

    return

def test(classifier, test_img):

    test_image = prepImage(test_img)
    result = classifier.predict(test_image)
    return printResult(result)

def prepImage(testImage):

    test_image = image.load_img(testImage, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    return test_image


def printResult(result):
    if result[0][0] == 1:
        prediction = True
    else:
        prediction = False

    return prediction
def create_folder_and_classify_image(path):
    list_folder=[]
    for files in os.listdir(path):
        list_folder.append(files)
    for folder in list_folder:
       
             
        if len(os.path.basename(folder)) ==1:
            for new_directory in os.listdir(path+'/'+folder):
                
               
                for new_folder in os.listdir(path+'/'+folder+'/'+new_directory):
                    # print(folder)
                    default_model = 'default_model.h5'
                    hotel_website_folder= path+'/'+folder+'/'+new_directory+"/"+new_folder
                    if os.path.basename(hotel_website_folder) == 'hotel_website':
                
                        predictor('folder', hotel_website_folder, default_model, path+'/'+folder+'/'+new_directory)
                    # predict('predict -path ' + fie)
        
create_folder_and_classify_image('/home/hotelsng/test/')