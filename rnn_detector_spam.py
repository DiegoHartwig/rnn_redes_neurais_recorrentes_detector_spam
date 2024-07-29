# -*- coding: utf-8 -*-
"""RNN_Detector_Spam.ipynb

## Detector de Spam com Rede Neural Recorrente
Diego Hartwig - 2024

Importação das bibliotecas
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

"""Carregando a base de dados"""

!wget http://www.razer.net.br/datasets/spam.csv

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()

df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['labels', 'data']
df["b_labels"] = df["labels"].map({"ham": 0, "spam": 1})
y = df["b_labels"].values
df.head()

"""# Separação em Treino e Teste"""

x_train, x_test, y_train, y_test = train_test_split(df["data"], y, test_size=0.33)

"""# Tokenização"""

num_words = 20000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)
word2index = tokenizer.word_index
V = len(word2index)
print("%s Tokens" % V)

"""# Tamanho das sequencias - padding"""

data_train = pad_sequences(sequences_train)
T = data_train.shape[1]
print("Tamanho das sequencias: %s" % T)

data_test = pad_sequences(sequences_test, maxlen=T)
print("data_train.shape = ", data_train.shape)
print("data_test.shape = ", data_test.shape)

"""# Definindo o modelo"""

D = 20
M = 5

i = Input(shape=(T, ))
x = Embedding(V + 1, D)(i)
x = LSTM(M)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(i, x)

"""# Exibindo o modelo"""

model.summary()

"""# Compilando e Treinando o modelo"""

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

epochs = 5

r = model.fit(
    data_train, y_train,
    validation_data=(data_test, y_test),
    epochs=epochs
)

"""# Plotando função de perda e acurácia"""

plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0, epochs, step=1), labels=range(1, epochs+1))
plt.legend()
plt.show()

plt.plot(r.history["accuracy"], label="accuracy")
plt.plot(r.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0, epochs, step=1), labels=range(1, epochs+1))
plt.legend()
plt.show()

"""# Predição de um novo texto"""

texto = "UNBELIEVABLE OFFER!!! Hello! You have been selected to receive an EXCLUSIVE 70% discount on all our electronic products. Don't miss this unique opportunity! Click the link below to take advantage before it's gone![SuperDiscountElectronics.com]Offer valid for a limited time only!!! "

seq_texto = tokenizer.texts_to_sequences([texto])
data_texto = pad_sequences(seq_texto, maxlen=T)

pred = model.predict(data_texto)
print(pred)
print("SPAM" if pred >= 0.5 else "OK")