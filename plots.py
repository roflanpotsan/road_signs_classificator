import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# обработка данных обучения
plot_data = pd.read_csv("history.csv")
labels_c = pd.read_csv("./sources/train_extra.csv")["class_number"].to_numpy()
labels_f = pd.read_csv("./sources/train_extra.csv")["filename"].to_numpy()

# построение графика зависимости точности модели от текущей эпохи
plt.plot(plot_data["accuracy"])
plt.title("Точность модели (Accuracy)")
plt.ylabel("Точность")
plt.xlabel("Эпоха")
plt.legend(["train"], loc="upper left")
plt.show()

# построение графика зависимости значений функции утраты модели от текущей эпохи
plt.plot(plot_data["loss"])
plt.title("Значение функции утраты (Loss)")
plt.ylabel("Утрата")
plt.xlabel("Эпоха")
plt.legend(["train"], loc="upper left")
plt.show()

# построение графика распределения данных обучения
a = np.zeros(shape=(67))

for i in range(67):
	index = np.where(labels_c == i)
	count = len(index[0])
	a[i] = count
b = np.arange(0,67,1)
plt.plot(b,a)
plt.title("Дистрибуция обучающей выборки по классам")
plt.ylabel("Количество изображений")
plt.xlabel("Класс")
plt.legend(["train_data"], loc="upper left")
plt.show()