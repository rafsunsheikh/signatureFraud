from django.shortcuts import render
import numpy as np # linear algebra
from numpy import asarray
from numpy import savetxt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import random
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import os
from os import listdir
from tensorflow.keras.preprocessing.image import load_img,img_to_array
print("Load Successful")

################## ADD SIGNATURE #####################
def add_signature(request):
    
    ########### Load person number #############
    file = open(r'static/number.txt', 'r')
    number = int(file.read())
    file.close()
    print("number Load Successful")

    ############ Load Dataset data from file ############
    # file4 = open(r'static/dataset.txt', 'r')
    # dataset = str(file4.read())
    # file4.close()
    # dataset = pd.read_csv(r"static/dataset.csv")

    #####Create Dataset#######
    dataset_dir = r"static/preprocessed_data/"
    image_size=224
    labels = []
    dataset = []
    def create_dataset(image_category,label):
        for img in tqdm(image_category):
            image_path = os.path.join(dataset_dir,img)
            try:
                image = cv2.imread(image_path,cv2.IMREAD_COLOR)
                image = cv2.resize(image,(image_size,image_size))
            except:
                continue
            
            dataset.append([np.array(image),np.array(label)])
        random.shuffle(dataset)
        print("create_dataset function run Successful")
        return dataset

    #person_01 = os.listdir (r"static\Real\person_01")
    name = 'person_0{}'.format(number)
    image_list = os.listdir(r'static/Real/{}'.format(name))
    dataset = create_dataset(image_list,number)
    print("dataset create Successful")
    print(dataset)

    ########### Write new person number ############
    file2 = open(r'static/number.txt', 'w')
    number += 1
    file2.write(str(number))
    file2.close()
    
    print("number increment Successful")

    ############# Write new dataset data ####################
    # file3 = open(r'static/dataset.txt', 'w')
    # file3.write(str(dataset))
    # file3.close()
    savetxt(r'static/dataset.csv', dataset, fmt='%s', delimiter=',')
    print("dataset write Successful")
    return dataset

################## CHECK SIGNATURE #####################
def check_signature(request):
    a = 10
