#!/usr/bin/env python
# coding: utf-8

# In[14]:


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


# In[15]:


import os
import cv2
import numpy as np
import pandas as pd

# #Change this value based on your preferences
train_dir = "./jellyfish/Train_Test_Valid/Train"
train_dataframe = pd.DataFrame(columns=["path", "class"])


# In[16]:


for class_name in os.listdir(train_dir):
  class_dir = os.path.join(train_dir, class_name)
  for image_name in os.listdir(class_dir):
    image_path = os.path.join(class_dir, image_name)
    train_dataframe.loc[len(train_dataframe.index)] = [image_path, class_name]


# In[17]:


train_dataframe


# In[18]:


xdim = 224
ydim = 224


# In[19]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

main_datagen=ImageDataGenerator(rescale=1./255., # 정규화 
                               horizontal_flip = True, # 수평으로 뒤집기
                                vertical_flip = True, # 수직으로 뒤집기 
                               rotation_range = 5) # +5도 또는 -5도 범위 내에서 회전 


# In[20]:


# 데이터프레임의 "path"라는 열의 데이터를 NumPy 배열로 변환하여 X에 담기 
X = np.array(train_dataframe["path"])


# In[21]:


wandb.login(key = "c4e33984a0f1d0c7e209f455add7b4da4718e070")

#import os
#import wandb

#wandb.login(key=os.getenv("WANDB_API_KEY"))


# In[22]:


sweep_config = {
    "name": "sweep_test_core",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "method": "random",
    #"method": "grid",  # Grid search
    "batch_size": {"values": [16, 32]},
    "parameters": {
        "learning_rate" : {
            "min" : 0.001,
            "max" : 0.01
            },
        "epoch" : {
            "distribution" : "int_uniform",
            "min" : 5,
            "max" : 10
            }
                    
        }
    }


# In[23]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def train_with_kfold():
    default_config = {
        "learning_rate": 0.005,
        "epoch": 5,
        "batch_size": 32
    }

    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    
    cvScores = []
    labels = list(train_dataframe["class"].unique())

    all_fold_histories = []  # 각 폴드별 기록 저장
    train_val_diffs = []  # 훈련-검증 정확도 차이를 저장
    
    
    # W&B 실행 초기화 (Run 이름에서 폴드 정보 제외)
    wandb.init(
        config=default_config,
        project="WandB_with_kfold_used_by_Jindeok_revised_again",
        name=f"CrossValidation_{wandb.util.generate_id()}",
        reinit=True
    )
    config = wandb.config
    
    
    fold = 1
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"Fold: {fold} ==================================================================")

        train_data = X[train_index]
        test_data = X[test_index]
        train_dataframe_inside = train_dataframe.loc[train_dataframe["path"].isin(list(train_data))]
        validation_dataframe = train_dataframe.loc[train_dataframe["path"].isin(list(test_data))]

        if train_dataframe_inside.empty or validation_dataframe.empty:
            print(f"Empty data detected in fold {fold}. Skipping.")
            continue

        train_generator = main_datagen.flow_from_dataframe(
            dataframe=train_dataframe_inside,
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
            x_col="path",
            y_col="class",
            class_mode="categorical",
            classes=labels,
            target_size=(xdim, ydim),
            color_mode="rgb",
            batch_size=32
        )

       

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(xdim, ydim, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.MaxPooling2D((4, 4), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(labels), activation='softmax')
        ])

    
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )

        model_history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config.epoch,
            verbose=1,
            callbacks=[early_stopping]
        )

        # 각 폴드의 기록 저장
        all_fold_histories.append(model_history.history)
        
        
        # 훈련-검증 성능 차이 저장
        final_train_accuracy = model_history.history["accuracy"][-1]
        final_val_accuracy = model_history.history["val_accuracy"][-1]
        train_val_diffs.append(final_train_accuracy - final_val_accuracy)
        
        
        
        
        
        # 폴드별 성능 로그
        scores = model.evaluate(validation_generator)
        cvScores.append(scores[1] * 100)
        print(f"Validation Accuracy for Fold {fold}: {scores[1] * 100:.2f}%")

        
        
        wandb.log({
            "fold": fold,
            "final_validation_accuracy": scores[1] * 100,
            "fold_train_accuracy": final_train_accuracy,
            "fold_val_accuracy": final_val_accuracy,
            "accuracy_diff": final_train_accuracy - final_val_accuracy
        })
        
        
        
        #wandb.log({
            #"fold": fold,
            #"final_validation_accuracy": scores[1] * 100,
            #"fold_train_accuracy": model_history.history["accuracy"][-1],
            #"fold_val_accuracy": model_history.history["val_accuracy"][-1]
        #})
        
        fold += 1 
        
        
    # 모든 폴드의 정확도 시각화
    plt.figure(figsize=(12, 8))
    for fold, history in enumerate(all_fold_histories, start=1):
        plt.plot(history['val_accuracy'], label=f'Fold {fold} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Across Folds')
    plt.legend()
    plt.show()
    
    
    
    # 훈련-검증 성능 차이 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(train_val_diffs) + 1), train_val_diffs, color="orange", edgecolor="black")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy Difference (Train - Validation)")
    plt.title("Train vs. Validation Accuracy Difference per Fold")
    plt.xticks(range(1, len(train_val_diffs) + 1), labels=[f"Fold {i}" for i in range(1, len(train_val_diffs) + 1)])
    plt.show()
    
    
    
    
    # 평균 및 표준 편차 계산 및 출력
    mean_accuracy = np.mean(cvScores)
    std_accuracy = np.std(cvScores)
    print(f"Cross-validation scores: {cvScores}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"Accuracy Standard Deviation: {std_accuracy:.2f}%")

    # 전체 결과 W&B 로그 추가
    wandb.log({
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy
    })
    
    wandb.finish()
    


# In[24]:


# entity와 project에 본인의 아이디와 프로젝트명을 입력하세요

sweep_id = wandb.sweep(sweep_config,
                       entity = "wisdom-jihyekim-aiffel",
                       project = "WandB_with_kfold_used_by_Jindeok_revised_again")


# run the sweep
wandb.agent(sweep_id,
            function=train_with_kfold,
            count=10)


# In[25]:


def compare_wandb_runs(project_name):
    # W&B API를 사용해 프로젝트의 모든 Runs 정보 가져오기
    api = wandb.Api()
    runs = api.runs(project_name)

    # 데이터를 저장할 리스트
    data = []
    
    for run in runs:
        # 1. Run 상태 필터링: 완료된 Run만 포함
        if run.state != "finished":
            continue

        # 2. 필요한 메트릭 추출 (mean_accuracy, std_accuracy 등)
        mean_accuracy = run.summary.get("mean_accuracy", None)
        std_accuracy = run.summary.get("std_accuracy", None)
        final_val_accuracy = run.summary.get("final_validation_accuracy", None)
        run_id = run.id
        run_name = run.name

        # 3. 메트릭 값이 None이거나 문자열인 경우, NaN으로 처리
        try:
            mean_accuracy = float(mean_accuracy) if mean_accuracy is not None else np.nan
            std_accuracy = float(std_accuracy) if std_accuracy is not None else np.nan
            final_val_accuracy = float(final_val_accuracy) if final_val_accuracy is not None else np.nan
        except ValueError:
            mean_accuracy = np.nan
            std_accuracy = np.nan
            final_val_accuracy = np.nan

        # 4. 리스트에 추가
        data.append({
            "Run ID": run_id,
            "Run Name": run_name,
            "Mean Accuracy": mean_accuracy,
            "Std Accuracy": std_accuracy,
            "Final Validation Accuracy": final_val_accuracy
        })

    # DataFrame으로 변환
    df = pd.DataFrame(data)

    # NaN 값 제거
    df = df.dropna(subset=["Mean Accuracy"])

    # 데이터프레임 정렬 (기본: Mean Accuracy 기준 내림차순)
    df = df.sort_values(by="Mean Accuracy", ascending=False)

    print("\nW&B Run Performance Comparison:")
    print(df)

    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.barh(df["Run Name"], df["Mean Accuracy"], color="skyblue", edgecolor="black")
    plt.xlabel("Mean Accuracy (%)")
    plt.ylabel("Run Name")
    plt.title(f"Comparison of W&B Runs in Project: {project_name}")
    plt.gca().invert_yaxis()  # 가장 높은 성능이 상단에 오도록 설정
    plt.show()

    return df


# In[27]:


project_name = "WandB_with_kfold_used_by_Jindeok_revised_again"  # 비교하려는 W&B 프로젝트 이름
comparison_df = compare_wandb_runs(project_name)

