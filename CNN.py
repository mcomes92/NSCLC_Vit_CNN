#!/usr/bin/env python


import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from matplotlib import gridspec
import scipy
import h5py
import pydicom
from natsort import natsorted, ns
import timeit
from matplotlib.ticker import MaxNLocator
import random
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
#from keras.utils.np_utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import DenseNet121,DenseNet201
from tensorflow.keras.applications import Xception, InceptionV3
from tensorflow.keras.applications import EfficientNetB2,EfficientNetB4,EfficientNetB5, EfficientNetB7 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Flatten, Input, BatchNormalization, Activation, Concatenate

from tensorflow.keras import backend as K
#K.set_session(tf.Session())

from vit_keras import vit, utils
#from swintransformer import SwinTransformer
import torch
import timm
import tfimm

from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import RobustScaler,StandardScaler   
from sklearn.decomposition import PCA
from sklearn.svm import SVC # "Support vector classifier"  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
from sklearn import metrics
from sklearn.metrics import fbeta_score
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import class_weight


import gc

import sklearn
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

hf = h5py.File('data.h5', 'r')
arr = hf.get('data')
y2 = hf.get('label')
arr=np.array(arr)
y2=np.array(y2)

auc_vec=[]
acc_vec=[]
sens_vec=[]
pre_vec=[]
spe_vec=[]

auc_vec2=[]
acc_vec2=[]
sens_vec2=[]
pre_vec2=[]
spe_vec2=[]

N=len(y2)

#N=len(y_train_hold_out)
#N1=len(y_test_hold_out)

patients=np.zeros(N)
for l in range (0,N):
    patients[l]=l

#patients1=np.zeros(N1)
#for l in range (0,N1):
    #patients1[l]=l

dz1 = pd.DataFrame({'Id_paziente': patients})
dz2 = pd.DataFrame({'Id_paziente': patients})

n_rounds2=20
n_rounds1=10
kf = StratifiedKFold(n_splits=n_rounds1)
kf.get_n_splits(arr,y2)
y_pred=np.zeros((len(y2),1))
#y_pred2=np.zeros(((n_rounds1,len(y_test_hold_out),1)))

vec_acc=[]
vec_val_acc=[]
vec_loss=[]
vec_val_loss=[]
vec_auc=[]
vec_val_auc=[]

image_size=224
from tensorflow.keras.applications.efficientnet import preprocess_input
vit_model = ResNet50(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3)) 
        #vit_model.trainable=True

data_augmentation = tf.keras.Sequential([
layers.RandomFlip("horizontal_and_vertical"),
layers.RandomRotation(0.2),
layers.RandomContrast(0.2)])

inputs = tf.keras.Input(shape=(image_size, image_size, 3))
x = data_augmentation(inputs)
x = vit_model(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(1, 'sigmoid')(x)
model2 = tf.keras.Model(inputs, outputs)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model2.compile(optimizer=optimizer, loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

k=0
for count in range(0,n_rounds2):
    k1=0
    for train_index, test_index in kf.split(arr,y2):
        
        x_train, x_test = np.array(arr)[train_index], np.array(arr)[test_index]
        y_train, y_test = y2[train_index], y2[test_index]
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        class_weights = class_weight.compute_class_weight("balanced",
                                                 classes=np.unique(y_train),
                                                 y=y_train)
        
        class_weights = {l:c for l,c in zip(np.unique(y_train), class_weights)}

        #es_callback = keras.callbacks.EarlyStopping(monitor='val_auc', patience=1)

        history=model2.fit(x_train, y_train,epochs=30,batch_size=10,validation_data=(x_test, y_test))

        y_pred[test_index]=model2.predict(x_test)
        
        vec_acc.append(history.history['binary_accuracy'])
        vec_val_acc.append(history.history['val_binary_accuracy'])
        vec_loss.append(history.history['loss'])
        vec_val_loss.append(history.history['val_loss'])
        vec_auc.append(history.history['auc'])
        vec_val_auc.append(history.history['val_auc'])

        #del vit_model, model2, history
        
        k1 +=1

        #if(k1>4):
            #break

    y_pred_final=y_pred[...,0]
    y_pred_final_binary=np.where(y_pred_final > 0.5, 1, 0)
    
    dz1.insert(loc=len(dz1.columns), column='round_'+str(k)+'', value=y_pred_final)
    dz2.insert(loc=len(dz2.columns), column='round_'+str(k)+'', value=y_pred_final_binary)

    auc_vec.append(roc_auc_score(y2,y_pred_final))
    acc_vec.append(metrics.accuracy_score(y2, y_pred_final_binary))
    sens_vec.append(metrics.recall_score(y2, y_pred_final_binary))
    spe_vec.append(metrics.recall_score(y2, y_pred_final_binary,pos_label=0))
    pre_vec.append(metrics.precision_score(y2, y_pred_final_binary))
    
    k+=1

g=open('Performances_training.dat',"w")
g.write("%s %.2f %.2f\n" % ('AUC:',np.average(auc_vec),np.std(auc_vec)))
g.write("%s %.2f %.2f\n" % ('Accuracy:',np.average(acc_vec), np.std(acc_vec)))
g.write("%s %.2f %.2f\n" % ('Sensitivity:', np.average(sens_vec),np.std(sens_vec)))  
g.write("%s %.2f %.2f\n" % ('Specificity:', np.average(spe_vec),np.std(spe_vec)))        
g.write("%s %.2f %.2f\n" % ('Precision:', np.average(pre_vec),np.std(pre_vec)))
g.close()

hf = h5py.File('acc_val_data.h5', 'w')
hf.create_dataset('auc', data=vec_auc)
hf.create_dataset('val_auc', data=vec_val_auc)
hf.create_dataset('acc', data=vec_acc)
hf.create_dataset('val_acc', data=vec_val_acc)
hf.create_dataset('loss', data=vec_loss)
hf.create_dataset('val_loss', data=vec_val_loss)
hf.close()

gf = h5py.File('metrics.h5', 'w')
gf.create_dataset('auc', data=auc_vec)
gf.create_dataset('acc', data=acc_vec)
gf.create_dataset('sens', data=sens_vec)
gf.create_dataset('spe', data=spe_vec)
gf.create_dataset('pre', data=pre_vec)
gf.close()

dz1.to_csv('y_pred_final.csv',decimal='.',float_format='%.3f')
dz2.to_csv('y_pred_final_binary.csv',decimal='.',float_format='%.1f')

sys.exit()
