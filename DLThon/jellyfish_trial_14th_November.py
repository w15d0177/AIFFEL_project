#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 도구 불러오기 

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


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape( -1, 28, 28, 1)
X_test = X_test.reshape( -1, 28, 28, 1)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5)

# 시각화를 위해 sample 데이터를 준비합니다.
X_sample = X_test[:50]
y_sample = y_test[:50]

CLASS_NAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# In[ ]:


# wandb.login(key = "c4e33984a0f1d0c7e209f455add7b4da4718e070")


# In[5]:


import os
import wandb

wandb.login(key=os.getenv("WANDB_API_KEY"))


# In[6]:


# Sweep의 config 설정하기 

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


# In[7]:


# train 함수의 완결성 있는 run 구조 

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

    model=keras.models.Sequential()
    model.add(keras.layers.Conv2D(config.filter, config.kernel, activation=config.activation, input_shape=config.input))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # 머신 러닝 학습때 여러가지 optimzier를 사용할 경우나 learning rate를 조절할 경우에는 아래와 같은 형태의 코드를 응용합니다.

    if config.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate = config.learning_rate)
    
    model.compile(optimizer = optimizer,
                  loss = config.loss,
                  metrics = config.metrics)

    # WandbCallback 함수는 후술합니다.
    
    model.fit(X_train, y_train,
              epochs = config.epoch,
              batch_size = config.batch_size,
              validation_data = (X_val, y_val),
              callbacks = [WandbCallback(validation_data = (X_sample, y_sample),
                                        lables = CLASS_NAMES,
                                        predictions = 10,
                                        input_type = "images")])
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    
    # wandb.log 함수 안에 기록하고 싶은 정보를 담습니다.
    
    wandb.log({"Test Accuracy Rate: " : round(test_accuracy * 100, 2),
               "Test Error Rate: " : round((1 - test_accuracy) * 100, 2)})


# In[11]:


# entity와 project에 본인의 아이디와 프로젝트명을 입력하세요

sweep_id = wandb.sweep(sweep_config,
                       entity = "wisdom-jihyekim-aiffel",
                       project = "jellyfish-trial")

# run the sweep
wandb.agent(sweep_id,
            function=train,
            count=10)

