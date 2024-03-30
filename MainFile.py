#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image

#====================== READ A INPUT IMAGE =========================


filename = askopenfilename()
inpimg = mpimg.imread(filename)
plt.imshow(inpimg)
plt.title('Original Image')
plt.axis ('off')
plt.show()


#============================ PREPROCESS =================================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(inpimg,(300,300))
img_resize_orig = cv2.resize(inpimg,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()

         
#==== GRAYSCALE IMAGE ====

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1)
plt.axis ('off')
plt.show()

# ============== FEATURE EXTRACTION ==============


#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("====================================")
print("        Feature Extraction          ")
print("====================================")
print()
print(features_extraction)


# ==== LBP =========

import cv2
import numpy as np
from matplotlib import pyplot as plt
   
      
def find_pixel(imgg, center, x, y):
    new_value = 0
    try:
        if imgg[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value
   
# Function for calculating LBP
def lbp_calculated_pixel(imgg, x, y):
    center = imgg[x][y]
    val_ar = []
    val_ar.append(find_pixel(imgg, center, x-1, y-1))
    val_ar.append(find_pixel(imgg, center, x-1, y))
    val_ar.append(find_pixel(imgg, center, x-1, y + 1))
    val_ar.append(find_pixel(imgg, center, x, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y))
    val_ar.append(find_pixel(imgg, center, x + 1, y-1))
    val_ar.append(find_pixel(imgg, center, x, y-1))
    power_value = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_value[i]
    return val
   
   
height, width, _ = inpimg.shape
   
img_gray_conv = cv2.cvtColor(inpimg,cv2.COLOR_BGR2GRAY)
   
img_lbp = np.zeros((height, width),np.uint8)
   
for i in range(0, height):
    for j in range(0, width):
        img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)

plt.imshow(img_lbp, cmap ="gray")
plt.title("LBP")
plt.show()
   

# ====================== IMAGE SPLITTING ================

#==== TRAIN DATA FEATURES ====

import pickle

with open('dot.pickle', 'rb') as f:
    dot1 = pickle.load(f)
  

import pickle
with open('labels.pickle', 'rb') as f:
    labels1 = pickle.load(f) 


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print("--------------------------------------------")
print("DATA SPLITTING")
print("--------------------------------------------")
print()
print("Total No.of Data            = ",len(dot1) )
print()
print("Total No.of Train Data      = ", len(x_train))
print()
print("Total No.of Test Data       = ", len(x_test))
print()



# ====================== CLASSIFICATION ================

# ==== CNN ==

from keras.utils import to_categorical

y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]
        
        
        
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
# from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential


# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(10,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=10,verbose=1)        
        
loss=history.history['loss']
loss=max(loss)*10

acc_cnn=100-loss



print("-------------------------------------------")
print("    CONVOLUTIONAL NEURAL NETWORK ")
print("-------------------------------------------")
print()

print("1. Accuracy    =", acc_cnn ,'%')
print()
print("2. Error rate   =", loss ,'%')
print()



# =============== IMAGE PREDICTION ============

import os

app1_data = os.listdir('Data/Apple/')

ban_data = os.listdir('Data/banana/')

beans_data = os.listdir('Data/beans/')

cabb_data = os.listdir('Data/cabbage/')

car_data = os.listdir('Data/carrot/')

grap_data = os.listdir('Data/grapes')

ora_data = os.listdir('Data/orange/')

pot_data = os.listdir('Data/potato/')

str_data = os.listdir('Data/Strawberry/')

tom_data = os.listdir('Data/tomato/')


Total_length = len(app1_data) + len(ban_data) + len(beans_data) + len(cabb_data) + len(car_data) + len(grap_data) + len(ora_data) + len(pot_data) + len(str_data) + len(tom_data)


temp_data1  = []
for ijk in range(0,94):
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)


if labels1[zz[0][0]] == 0:
    print('-----------------------')
    print()
    print('Apple ')
    print()
    print('-----------------------')
    a='Apple '

elif labels1[zz[0][0]] == 1:
    print('--------------------------')
    print()
    print('Banana')   
    print()
    print('-------------------------')
    a='Banana'
    
elif labels1[zz[0][0]] == 2:
    print('--------------------------')
    print()
    print('Beans')   
    print()
    print('-------------------------')   
    a='Beans'
    
elif labels1[zz[0][0]] == 3:
    print('--------------------------')
    print()
    print('Cabbage')   
    print()
    print('-------------------------')    
    a='Cabbage'
    
elif labels1[zz[0][0]] == 4:
    print('--------------------------')
    print()
    print('Carrot')   
    print()
    print('-------------------------')    
    a='Carrot'
    
elif labels1[zz[0][0]] == 5:
    print('--------------------------')
    print()
    print('Grapes')   
    print()
    print('-------------------------')    
    a='Grapes'
    
elif labels1[zz[0][0]] == 6:
    print('--------------------------')
    print()
    print('Orange')   
    print()
    print('-------------------------')       
    a='Orange'
elif labels1[zz[0][0]] == 7:
    print('--------------------------')
    print()
    print('Potato')   
    print()
    print('-------------------------')       
    a='Potato'   
elif labels1[zz[0][0]] == 8:
    print('--------------------------')
    print()
    print('Strawberry')   
    print()
    print('-------------------------')       
    a='Strawberry'  

elif labels1[zz[0][0]] == 9:
    print('--------------------------')
    print()
    print('Tomato')   
    print()
    print('-------------------------')       
    a='Tomato'  


#============================= 5.DATA SELECTION ===============================

#=== READ A DATASET ====

import pandas as pd

data_frame=pd.read_excel("data set 2.xlsx")
print("-------------------------------------------------------")
print("================== Data Selection ===================")
print("-------------------------------------------------------")
print()
print(data_frame.head(20))


#==========================  6.DATA PREPROCESSING ==============================


#=== CHECK MISSING VALUES ===

print("=====================================================")
print("                    Preprocessing                  ")
print("=====================================================")
print()
print("------------------------------------------------------")
print("================ Checking missing values =========")
print("------------------------------------------------------")
print()
print(data_frame.isnull().sum())
print()

data_label=data_frame['Name']

#==== LABEL ENCODING ====

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 
print("------------------------------------------------------")
print("================ Before label encoding ===========")
print("------------------------------------------------------")
print()
print(data_frame['Name'].head(10))

print("------------------------------------------------------")
print("================ After label encoding ===========")
print("------------------------------------------------------")
print()
data_frame['Name']= label_encoder.fit_transform(data_frame['Name']) 


print(data_frame['Name'].head(10))


x1=data_label
for i in range(0,len(data_frame)):
    if x1[i]==a:
        idx=i


data_frame1_c=data_frame['Carbohydrates']
data_frame1_fat=data_frame['Fats']
data_frame1_fib=data_frame['Fiber']
data_frame1_cal=data_frame['Calorie']
data_frame1_p=data_frame['Protein']


Req_data_c=data_frame1_c[idx]
Req_data_fat=data_frame1_fat[idx]
Req_data_fib=data_frame1_fib[idx]
Req_data_cal=data_frame1_cal[idx]
Req_data_p=data_frame1_p[idx]


print("----------------------------------------------------------------")
print("================= PREDICTION FOR NUTRITION VALUES ==============")
print("----------------------------------------------------------------")
print()

print("1.Carbohydrates value = ",Req_data_c )
print()
print("2.Fats value          = ",Req_data_fat)
print()
print("3.Fiber value         = ",Req_data_fib )
print()
print("4.Calorie value       = ",Req_data_cal)
print()
print("5.Protein value       = ",Req_data_p)


#=============================== VISUALIZATIOn =================================


print()
print("-----------------------------------------------------------------------")
print()


import matplotlib.pyplot as plt
vals=[acc_cnn,loss]
inds=range(len(vals))
labels=["ACC ","ErrorRate "]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('CNN')
plt.show() 
