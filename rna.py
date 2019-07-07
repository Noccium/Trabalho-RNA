import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#visualização dos dados
data = pd.read_csv('voice.csv')
#print(data.head())

#verificar existência de campos nulos
#print(np.where(pd.isnull(data)))

#exibe a quantidade de dados referentes a cada gênero
#print("Número de homens: {}".format(data[data.label == 'male'].shape[0]))
#print("Número de mulheres: {}".format(data[data.label == 'female'].shape[0]))

#verifica a correlação entre os dados
#colormap = plt.cm.viridis
#plt.figure(figsize=(12,12))
#plt.title('Correlação de Pearson das variáveis', y=1.05, size=15)
#xd = sns.heatmap(data.iloc[:,:-1].astype(float).corr(),linewidths=0.1,vmax=1.0,
#                    square=True, cmap=colormap, linecolor='white', annot=False)
#plt.show()

#print(data.dtypes)
#converte os tipos object em um valor numérico discreto
#male = 1, female = 0
data['label'] = pd.Categorical(data['label'])
data['label'] = data.label.cat.codes

x_train = tf.constant(data.iloc[:,:-1])
y_train = tf.constant(data['label'])
print(x_train.dtype)
print(y_train.dtype)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(200, input_dim=20,activation='relu'))
model.add(tf.keras.layers.Dense(150, activation='tanh'))
model.add(tf.keras.layers.Dense(100, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mae', 'acc'])

model.fit(x_train, y_train, batch_size=3168, epochs=10000, steps_per_epoch=1)

predictions = model.predict(x_train, steps=1)
print(predictions)

score = model.evaluate(x_train, y_train, verbose=0, steps=1)
print(model.metrics_names)
print(score)
#model.save('my_model.h5')
