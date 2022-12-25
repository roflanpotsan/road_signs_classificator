import numpy as np
import pandas as pd
import cv2
import keras
import tensorflow as tf
import os

# обработка входных данных
labels = pd.read_csv("./sources/train_extra.csv")["class_number"].to_numpy()

def get_img_data(path):
    img_files = sorted(os.listdir(path))
    print(img_files)
    data = None
    k = 0
    for i in range(len(img_files)):
        if (int(img_files[i][:-4]) > 888889):
            img_files[i] = f"99999{k}.png"
        file_name = img_files[i]
        file_path = os.path.join(path, file_name)
        img = cv2.imread(file_path)
        print(file_path)
        if data is None:
            data = np.zeros(shape=(len(img_files), img.shape[0], img.shape[1], img.shape[2]))
        data[i] = img
        if (int(img_files[i][:-4]) > 888889):
            k+=1
    return data / 255.0

img_data = get_img_data('./sources/train/train')

# определение слоев модели
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(48, (3, 3), activation=tf.nn.relu, input_shape=(48, 48, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(67, activation=tf.nn.softmax)
])

# определение параметров обучения
model.compile(optimizer=tf.keras.optimizers.Adam(), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

# обучение модели
history = model.fit(img_data, labels, epochs=12)

# сохранение данных об обучении
with open("history.csv", mode='w') as file:
   pd.DataFrame(history.history).to_csv(file)

# сохранение обученной модели
model.save("my_model")
