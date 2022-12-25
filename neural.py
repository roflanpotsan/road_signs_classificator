import numpy as np
import pandas as pd
import cv2
import keras
import tensorflow as tf
import os

# обученная модель
class NeuralPredictor:
	def __init__(self):
		self.labels = pd.read_csv('./sources/classes.csv', encoding = 'cp1251')['class_identifier'].to_numpy()
		self.model = model = keras.models.load_model("./my_model")
		self.class_id = -1
	def get_classification(self, data):
		self.class_id = np.argmax(self.model.predict(data), axis=1)
		return self.labels[self.class_id][0]
	def get_class_id(self):
		return self.class_id
# обработка входного изображения
def get_data(img_path):
	img = cv2.resize(cv2.imread(img_path), (48,48))
	data = np.zeros(shape=(1, img.shape[0], img.shape[1], img.shape[2]))
	data[0] = img
	return data
