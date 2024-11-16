#!/usr/bin/env python
# coding: utf-8

# In[119]:


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


# In[120]:


import os
import cv2
import numpy as np
import pandas as pd

# #Change this value based on your preferences
train_dir = "./jellyfish/Train_Test_Valid/Train"
train_dataframe = pd.DataFrame(columns=["path", "class"])


# In[121]:


for class_name in os.listdir(train_dir):
  class_dir = os.path.join(train_dir, class_name)
  for image_name in os.listdir(class_dir):
    image_path = os.path.join(class_dir, image_name)
    train_dataframe.loc[len(train_dataframe.index)] = [image_path, class_name]


# In[122]:


train_dataframe


# In[123]:


xdim = 224
ydim = 224


# In[124]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

main_datagen=ImageDataGenerator(rescale=1./255., # 정규화 
                               horizontal_flip = True, # 수평으로 뒤집기
                                vertical_flip = True, # 수직으로 뒤집기 
                               rotation_range = 5) # +5도 또는 -5도 범위 내에서 회전 


# In[125]:


# 데이터프레임의 "path"라는 열의 데이터를 NumPy 배열로 변환하여 X에 담기 
X = np.array(train_dataframe["path"])


# In[126]:


wandb.login(key = "c4e33984a0f1d0c7e209f455add7b4da4718e070")

#import os
#import wandb

#wandb.login(key=os.getenv("WANDB_API_KEY"))


# In[127]:


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


# In[128]:


from sklearn.model_selection import KFold
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def train_with_kfold():
    
    default_config = {
            #"input" : (28,28,1),
            #"filter" : 16,
            #"kernel" : (3,3),
            #"activation" : "relu",
            "learning_rate" : 0.005,
            #"optimizer" : "adam",
            #"loss" : "sparse_categorical_crossentropy",
            #"metrics" : ["accuracy"],
            "epoch" : 5,
            "batch_size" : 32
        }
    
    wandb.init(config = default_config)
    config = wandb.config
    
    
    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    cvScores = []
    i = 1

    labels = list(train_dataframe["class"].unique())  # 고정된 클래스 레이블 사용

    for train_index, test_index in kf.split(X):
        print(f"Fold: {i} ==================================================================")
    
        # 폴드 내 훈련 및 테스트 데이터 설정
        train_data = X[train_index]
        test_data = X[test_index]
        train_dataframe_inside = train_dataframe.loc[train_dataframe["path"].isin(list(train_data))]
        validation_dataframe = train_dataframe.loc[train_dataframe["path"].isin(list(test_data))]
    
        # labels = list(train_dataframe_inside["class"].unique())
    
        if train_dataframe_inside.empty or validation_dataframe.empty:
            print(f"Empty data detected in fold {i}. Skipping.")
            continue
    
    
        # 데이터 제너레이터 설정
        train_generator = main_datagen.flow_from_dataframe(
            dataframe=train_dataframe_inside,
            #directory=None,
            x_col="path",
            y_col="class",
            class_mode="categorical",
            classes=labels,
            target_size=(xdim, ydim),
            color_mode="rgb",
            batch_size=32
        )
    
        validation_generator = main_datagen.flow_from_dataframe(
            dataframe=validation_dataframe,
            #directory=None,
            x_col="path",
            y_col="class",
            class_mode="categorical",
            classes=labels,
            target_size=(xdim, ydim),
            color_mode="rgb",
            batch_size=32
        )
    
        # 모델 정의
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(xdim, ydim, 3)),  # Input 레이어 추가
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPooling2D((4,4), padding='same'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPooling2D((4,4), padding='same'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPooling2D((4,4), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            #tf.keras.layers.Dense(6, activation='softmax')
            tf.keras.layers.Dense(len(labels), activation='softmax')
        ])
    
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
        # 조기 종료 콜백 설정
        early_stopping = EarlyStopping(
            monitor='val_accuracy',         # 평가할 지표 (val_loss를 사용할 수도 있음)
            patience=10,                    # 10 에포크 동안 개선이 없으면 학습 중단
            restore_best_weights=True       # 가장 좋은 가중치를 복원
        )
    
    
    
         # W&B 설정
        wandb.init(
            project="WandB_with_kfold_used_by_Jindeok",
            name=f"fold-{i}",
            reinit=True  # 폴드별로 새 W&B 실행을 생성
        )
       
    
        # wandb.config 설정
        #wandb.config = {
            #"epoch": 10,  # 에포크 수
            #"batch_size": 32,  # 배치 크기
            #"learning_rate": 0.001  # 학습률
        #}
    
    
        # 모델 학습
        model_history = model.fit(
            train_generator,
            # epochs=config.epoch,
            epochs=wandb.config.epoch,
            validation_data=validation_generator,
            callbacks=[early_stopping, WandbCallback()]  # 조기 종료 콜백 추가, W&B 콜백 추가
        )
    
        # 학습 결과 시각화
        #plt.figure(figsize=(10,5))
        #plt.plot(model_history.history['accuracy'], label='Train Accuracy')
        #plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
        #plt.xlabel('Epoch')
        #plt.ylabel('Accuracy')
        #plt.ylim([0.5, 1])
        #plt.legend(loc='lower right')
        #plt.title(f'Fold {i} - Training and Validation Accuracy')
        #plt.show()

        # 폴드 성능 평가
        scores = model.evaluate(validation_generator)
        cvScores.append(scores[1] * 100)
        print(f"Validation Accuracy for fold {i}: {scores[1] * 100:.2f}%")
    
        # i += 1
    
    
        # W&B 실행 종료
        wandb.finish()

        i += 1
    


# In[129]:


# entity와 project에 본인의 아이디와 프로젝트명을 입력하세요

sweep_id = wandb.sweep(sweep_config,
                       entity = "wisdom-jihyekim-aiffel",
                       project = "WandB_with_kfold_used_by_Jindeok")

# run the sweep
wandb.agent(sweep_id,
            function=train_with_kfold,
            count=10)


# In[130]:


print(f"Mean CV Accuracy: {np.mean(cvScores):.2f}% ± {np.std(cvScores):.2f}%")


# In[131]:


avgScores = np.mean(cvScores)
stdScores = np.std(cvScores)
print(f"Average KFold Cross Validation Score: {avgScores}")
print(f"Standard Deviation KFold Cross Validation Score: {stdScores}")

