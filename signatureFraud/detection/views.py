from cv2 import data
from django.shortcuts import render, redirect
from django.contrib import messages
import numpy as np # linear algebra
# from npy_append_array import NpyAppendArray
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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from . import chequeCrop
print("Load Successful")


#####Create Dataset#######
dataset_dir = r"static/preprocessed_data/"

######################## Create Dataset function ############################
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


############################## Train model ################################
def model_train():
    file = open(r'detection/number.txt', 'r')
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
    print("Y:\n")
    print(y)


    y_train1 = to_categorical(y_train1)
    y_test1 = to_categorical(y_test1)


    print((x_train1.shape,y_train1.shape))
    print((x_test1.shape,y_test1.shape))

    print("Catagorical values:")
    print(y_train1)
    print("\n")
    print(y_test1)


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


################## ADD SIGNATURE #####################
def add_signature(request):
    
    # ########### Load person number #############
    file = open(r'detection/number.txt', 'r')
    number = int(file.read())
    file.close()
    print("number Load Successful")

    if(number < 10):
        name = 'person_0{}'.format(number)
    else:
        name = 'person_{}'.format(number)
    
    image_list = os.listdir(r'static/Real/{}'.format(name))
    print(image_list)
    dataset = create_dataset(image_list,number)
    print("New dataset create Successful")
    print(dataset)

    ############# Load and append with the previous dataset #############
    with open(r"static/dataset.npy", 'rb') as f:
        dataset_old = np.load(f, allow_pickle=True)

    dataset_1 = np.array(dataset_old)
    dataset_2 = np.array(dataset)

    print(dataset_1.shape)
    print(dataset_2.shape)
    
    

    dataset_new = np.append(dataset_old, dataset, axis =0)
    dataset.clear()

    ########### Write new person number ############
    file2 = open(r'detection/number.txt', 'w')
    number += 1
    file2.write(str(number))
    file2.close()
    
    print("number increment Successful")

    with open(r"static/dataset.npy", 'wb') as f:
        np.save(f, dataset_new, allow_pickle=True, fix_imports=True)
    print("dataset write Successful")
    
    model_train()
        
    messages.success(request, f' Signature added successfully!')
    return redirect('index')



    ############## When Dataset null ################
    ########## Load person number #############
    # file = open(r'/home/rafsunsheikh/Desktop/signatureFraud/signatureFraud/detection/number.txt', 'r')
    # number = int(file.read())
    # file.close()
    # print("number Load Successful")

    # if(number < 10):
    #     name = 'person_0{}'.format(number)
    # else:
    #     name = 'person_{}'.format(number)
    
    # image_list = os.listdir(r'static/Real/{}'.format(name))
    # print(image_list)
    # dataset = create_dataset(image_list,number)
    # print("New dataset create Successful")
    # print(dataset)

    # ############# Load and append with the previous dataset ############
    # dataset_2 = np.array(dataset)

    # print(dataset_2.shape)

    # ########### Write new person number ############
    # file2 = open(r'/home/rafsunsheikh/Desktop/signatureFraud/signatureFraud/detection/number.txt', 'w')
    # number += 1
    # file2.write(str(number))
    # file2.close()
    
    # print("number increment Successful")

    
    # with open(r"static/dataset.npy", 'wb') as f:
    #     np.save(f, dataset, allow_pickle=True, fix_imports=True)
    
    # print("dataset write Successful")
    
    # model_train()
    
    # messages.success(request, f' Signature added successfully!')
    # return redirect('index')


################## CHECK SIGNATURE #####################
def check_signature(request):
   
#################### Load raw cheque image ######################
    file = open(r'detection/number.txt', 'r')
    number = int(file.read())
    number = number - 1 
    file.close()

    cheque_image_path = r'static/cheque_image/myImage0.jpg'

    cheque_image = cv2.imread(cheque_image_path, cv2.IMREAD_COLOR)
    # cheque_image = cv2.flip(cheque_image, 1)
    # cheque_image = cv2.rotate(cheque_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cheque_image = cv2.resize(cheque_image, (740, 346))

    # crop_img = cheque_image[218:282, 493:728]
    # set_save_directory = r'static/'
    # os.chdir(set_save_directory)

    # filename = 'savedImage.jpg'
    # cv2.imwrite(filename, crop_img)
    # cv2.imwrite(filename, cheque_image)

######################  Start checking #############################
    # image_path = r'static/savedImage.jpg'
    # image_path = crop_img
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv2.resize(crop_img, (image_size, image_size))
    image = cv2.resize(cheque_image, (image_size, image_size))

    image_array = np.array(image)
    x_image = image_array.reshape(-1, image_size, image_size, 3)


    new_model = tf.keras.models.load_model(r'static/my_model')
    y_image_pred = new_model.predict(x_image)
    print("Y image pred:", y_image_pred)
    print("Y image pred 6 test:", y_image_pred[0][6])
    y_image_pred_index = np.argmax(y_image_pred, axis = 1)
    print("Y image pred final:", y_image_pred_index)
    row = y_image_pred_index[0]
    print("Y image pred 6 test with row:", y_image_pred[0][row])
    print("Row:", row)
    # print(y_image_pred)
    if y_image_pred_index == number:
        messages.success(request, ' Signature does not matched!' )
        return redirect('index') 
    elif y_image_pred[0][row] < 0.95:
        messages.success(request, ' Signature does not matched!' )
        return redirect('index') 
    else:
        messages.success(request, ' Signature matched successfully with Person {}'.format(y_image_pred_index) )
        return redirect('user')


def scan_image(request):
    chequeCrop.document()

def remove_signature(request):
    a = 0

def show_signature_list(request):
    a = 0



    
