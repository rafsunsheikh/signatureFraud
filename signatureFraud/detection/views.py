from cv2 import data
from django.shortcuts import render, redirect
from django.contrib import messages
import numpy as np # linear algebra
from numpy import asarray
from numpy import savetxt
from numpy import testing
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Sequential
from keras import layers
from tensorflow.keras.layers import Flatten,Dense
import os
import csv
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
    # dataset = pd.read_csv(r"static/dataset.csv", error_bad_lines=False)
    

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
    print(image_list)
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
    with open(r"static/dataset.npy", 'wb') as f:
        np.save(f, dataset, allow_pickle=True, fix_imports=True)
    # textfile = open(r"static/dataset.txt", "w")
    # for row in dataset:
    #     np.savetxt(textfile, row)
    # textfile.close()
    # file3.write(str(dataset))
    # file3.close()
    # savetxt(r'static/dataset.csv', dataset, delimiter=',', fmt='%s',)
    # with open(r'static/dataset.csv', 'a') as fd:
    #     # fd.write(dataset)
    #     writer = csv.writer(fd)
    #     writer.writerow(dataset)
    print("dataset write Successful")
    
    file = open(r'static/number.txt', 'r')
    number = int(file.read())
    file.close()
    
    
    with open(r"static/dataset.npy", 'rb') as f:
        dataset = np.load(f, allow_pickle=True)
    image_size=224
    x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
    y = np.array([i[1] for i in dataset])
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y,test_size=0.2)


    print((x_train1.shape,y_train1.shape))
    print((x_test1.shape,y_test1.shape))


    y_train1 = to_categorical(y_train1)
    y_test1 = to_categorical(y_test1)


    print((x_train1.shape,y_train1.shape))
    print((x_test1.shape,y_test1.shape))


    vgg16_weight_path = r'static/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg = VGG16(
        weights=vgg16_weight_path,
        include_top=False, 
        input_shape=(224,224,3)
    )


    for layer in vgg.layers:
        layer.trainable = False

    
    model = Sequential()
    model.add(vgg)
    model.add(Dense(256, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(Dense(128, activation='sigmoid'))
    model.add(layers.Dropout(rate=0.2))
    model.add(Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(number, activation="sigmoid"))


    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    # history = model.fit(x_train,y_train,batch_size=32,epochs=80,validation_data=(x_test,y_test))
    history = model.fit(x_train1,y_train1,batch_size=32,epochs=80,validation_data=(x_test1,y_test1))
    
    model.save(r'static/my_model')
    
    
    messages.success(request, f' Signature added successfully!')
    return redirect('index')

################## CHECK SIGNATURE #####################
def check_signature(request):
    # dataset = pd.read_csv(r"static/dataset.csv", error_bad_lines=False)
    # dataset = np.loadtxt(r"static/dataset.txt")
    # dataset1 = np.array(dataset)
    

    

    # #Dimension of the dataset
    # print((x_train.shape,y_train.shape))
    # # print((x_val.shape,y_val.shape))
    # print((x_test.shape,y_test.shape))
    #Dimension of the dataset
    

    #y_train=to_categorical(y_train)
    # y_test=to_categorical(y_test)
    

    # #Dimension of the dataset
    # print((x_train.shape,y_train.shape))
    # # print((x_val.shape,y_val.shape))
    # print((x_test.shape,y_test.shape))
    #Dimension of the dataset
    

    #Image Data Augmentation
    # train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

    # test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True, zoom_range=.1)

    #Fitting the augmentation defined above to the data
    # train_generator.fit(x_train1)
    # test_generator.fit(x_test1)
    a = 0


def remove_signature(request):
    a = 0

def show_signature_list(request):
    a = 0



    
