#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import shutil
# from app import main as predict
from predict_model import predictor


# In[3]:




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


# In[ ]:




