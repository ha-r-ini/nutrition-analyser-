import os 
import cv2
import numpy as np
import matplotlib.image as mpimg

# === Carrot 

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



dot1= []
labels1 = []
for img in app1_data:
        # print(img)
        img_1 = mpimg.imread('Data/Apple/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

        
for img in ban_data:
    try:
        img_2 = mpimg.imread('Data/banana/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

for img in beans_data:
    try:
        img_2 = mpimg.imread('Data/beans'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None
        
        
for img in cabb_data:
    try:
        img_2 = mpimg.imread('Data/cabbage/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(3)
    except:
        None


        
for img in grap_data:
    try:
        img_2 = mpimg.imread('Data/carrot/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(4)
    except:
        None

############        
for img in grap_data:
    try:
        img_2 = mpimg.imread('Data/grapes/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(5)
    except:
        None
        
for img in ora_data:
    try:
        img_2 = mpimg.imread('Data/orange/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(6)
    except:
        None

for img in pot_data:
    try:
        img_2 = mpimg.imread('Data/potato/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(7)
    except:
        None

for img in str_data:
    try:
        img_2 = mpimg.imread('Data/Strawberry/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(8)
    except:
        None

        
for img in tom_data:
    try:
        img_2 = mpimg.imread('Data/tomato/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(9)
    except:
        None




import pickle
with open('dot.pickle', 'wb') as f:
    pickle.dump(dot1, f)
    
with open('labels.pickle', 'wb') as f:
    pickle.dump(labels1, f)        
        
