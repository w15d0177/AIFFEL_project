#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback
from sklearn.preprocessing import LabelEncoder
from PIL import Image 
import glob
import os


# In[2]:


import random # random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # accesso ai file
import datetime # dati temporali (date, ore)
import tensorflow as tf # Per creare reti neurali
import pandas as pd # Tabelle
import matplotlib # Grafici
import matplotlib.pyplot as plt #Grafici
import cv2 # Immagini
import keras # Semplifica la creazione di reti neurali
# from keras.optimizers import Adam # allenatore
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow import keras # idem
from sklearn.model_selection import train_test_split # Divide i dati in training set e test set
from sklearn.preprocessing import LabelEncoder # Converte le parole in numeri ("Gatto" = 0, "Cane" = 1, ...)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Per gestire il dataset di immagini
from keras.models import Sequential # Per definire una rete neurale semplice, sequenziale
from tensorflow.keras import layers # I livelli di una rete neurale
from keras.layers import Dense, Dropout, Activation, Flatten # Componenti di una rete neurale
from keras.layers import Conv2D, MaxPooling2D # Convoluzioni e pooling
import seaborn as sns # Altri grafici
from sklearn.metrics import confusion_matrix # Per capire quanto è bravo il modello
from sklearn.utils import class_weight # Per bilanciare il dataset


# In[3]:


def dataset(direct):
    images = []
    labels = []
    classes = []
    label_encoder = LabelEncoder()
    
    train_dir = os.listdir(direct)
    for i in train_dir:
        if i == "Train_Test_Valid":
            continue
        class_path = os.path.join(direct, i)
        
        # 디렉터리인지 확인
        if os.path.isdir(class_path):
            classes.append(i)
            for j in os.listdir(class_path):
                file_path = os.path.join(class_path, j)
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    img = cv2.resize(img, (224, 224))  # Resize image
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(i)
                
    images = np.array(images)
    labels = label_encoder.fit_transform(labels)

    return images, labels, classes


# In[4]:


images, labels, classes = dataset('./jellyfish')

random_indexes = random.sample(range(len(images)), 10)

# Plot a few images at random indexes
fig, ax = plt.subplots(2, 5, figsize=(15, 6))
for idx, ax in zip(random_indexes, ax.flatten()):
    ax.imshow(images[idx])
    ax.set_title(f"Label: {classes[labels[idx]]}")
    ax.axis('off')
plt.suptitle('Random Sample Images from Dataset')
plt.show()


# In[5]:


# training set (80%)  test set (20%)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
# training set (75%)  "validation set" (25%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

classes = np.unique(classes)


# In[6]:


# 데이터 증강 설정 

train_datagen = ImageDataGenerator(
    rotation_range = 10, # 이미지를 최대 10도까지 랜덤하게 회전 
    zoom_range = 0.1, # 이미지를 10%까지 랜덤하게 확대 또는 축소 
    width_shift_range = 0.2, # 이미지를 가로로 20%
    height_shift_range = 0.2, # 세롤로 20%까지 랜덤하게 이동 
    horizontal_flip = True, # 이미지를 좌우로 뒤집음
    vertical_flip = True, # 이미지를 위아래로 뒤집음 
)

test_val_datagen = ImageDataGenerator() # 검증 및 테스트 데이터 생성 설정 
# 검증과 테스트 데이터에는 증강을 적용하지 않음 
# 따라서 단순히 ImageDataGenerator()만 사용하여 원본 이미지를 그대로 사용

# 학습 데이터를 batch_size=20으로 나누어 train_datagen에 설정된 증강을 적용해 학습용 배치를 생성
train_generator = train_datagen.flow(X_train, y_train, batch_size=20)

# 검증 데이터를 batch_size=20으로 나누어 증강 없이 생성
val_generator = test_val_datagen.flow(X_val, y_val, batch_size=20)

# 테스트 데이터를 batch_size=20으로 나누어 생성
# shuffle=False로 설정하여 데이터를 랜덤하게 섞지 않음 
# 테스트에서는 예측 결과의 순서가 일정해야 하기 때문에 보통 shuffle=False로 설정
test_generator = test_val_datagen.flow(X_test, y_test, batch_size=20, shuffle=False)


# In[7]:


wandb.login(key = "c4e33984a0f1d0c7e209f455add7b4da4718e070")

#import os
#import wandb

#wandb.login(key=os.getenv("WANDB_API_KEY"))


# In[8]:


sweep_config = {
    "name": "sweep_test_core",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "method": "random",
    "parameters": {
        "learning_rate" : {
            "min" : 0.001,
            "max" : 0.1
            },
        "epoch" : {
            "distribution" : "int_uniform",
            "min" : 5,
            "max" : 10
            }
                    
        }
    }


# In[12]:


def train():
    default_config = {
        "input" : (28,28,1),
        "filter" : 16,
        "kernel" : (3,3),
        "activation" : "relu",
        "learning_rate" : 0.005,
        "optimizer" : "adam",
        "loss" : "sparse_categorical_crossentropy",
        "metrics" : ["accuracy"],
        "epoch" : 5,
        "batch_size" : 32
    }

    wandb.init(config = default_config)
    config = wandb.config

    # Model
    
    # 모델 초기화
    model = Sequential()

    # 첫 번째 convolutional layer
    model.add(Conv2D(16, 3, input_shape=(224, 224, 3), activation='relu'))

    # pooling으로 downsampling 및 convolutional layer 추가 
    model.add(MaxPooling2D(2))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2))

    # 2D metrics를 1D vector로 flattening하여 완전 연결 층 준비 
    model.add(Flatten())
          
    # 완전 연결 층 (flatten된 특징의 비선형 조합을 학습하기 위한 relu)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
          
    # 출력 층 (다중 클래스 분류 (6개의 클래스)에 적합)
    model.add(Dense(6, activation='softmax'))
          
    # 모델 컴파일 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    
    history = model.fit(
    train_generator, 
    epochs=100, 
    validation_data=val_generator
    )
    
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    # print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    
    
    
    
    # wandb.log 함수 안에 기록하고 싶은 정보를 담습니다.
    
    wandb.log({"Test Accuracy Rate: " : round(test_accuracy * 100, 2),
               "Test Error Rate: " : round((1 - test_accuracy) * 100, 2)})


# In[13]:


# entity와 project에 본인의 아이디와 프로젝트명을 입력하세요

sweep_id = wandb.sweep(sweep_config,
                       entity = "wisdom-jihyekim-aiffel",
                       project = "WandB_with_first_CNN_used_by_Jindeok")

# run the sweep
wandb.agent(sweep_id,
            function=train,
            count=10)


# In[ ]:




