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
print("Número de homens: {}".format(data[data.label == 'male'].shape[0]))
print("Número de mulheres: {}".format(data[data.label == 'female'].shape[0]))

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

epochs = 15000
batch_size = 3168
test_size = 1000
x_train = tf.constant(data.iloc[:,:-1])
y_train = tf.constant(data['label'])

x_test1 = tf.constant(data.iloc[:int(test_size/2),:-1])
x_test2 = tf.constant(data.iloc[1584:int(1584+test_size/2),:-1])
x_test = tf.concat(values=[x_test1,x_test2], axis=0)

y_test1 = tf.constant(data.iloc[:int(test_size/2),20])
y_test2 = tf.constant(data.iloc[1584:int(1584+test_size/2),20])
y_test = tf.concat(values=[y_test1,y_test2], axis=0)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, input_dim=20, activation='elu'))
model.add(tf.keras.layers.Dense(15, activation='tanh'))
model.add(tf.keras.layers.Dense(10, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.01)
#Adagrad = tf.keras.optimizers.Adagrad(lr=0.01)
model.compile(loss='binary_crossentropy',
                optimizer='Adagrad',
                metrics=['mae', 'acc'])

#Treinamento do modelo
H = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                steps_per_epoch=1,
                validation_steps=1
                )

#Avaliação do modelo
predictions = model.predict(x_test, steps=1)
print("------- Predictions -------")
print(predictions)

score = model.evaluate(x_train, y_train, verbose=1, steps=1)
#print(model.metrics_names)
#print(score)

#Salva o modelo com base na acurácia
old_model = tf.keras.models.load_model('my_model.h5')
old_score = old_model.evaluate(x_train, y_train, verbose=1, steps=1)

if score[2] > old_score[2]:
    print("Novo modelo salvo!")
    model.save('my_model.h5')

print("Acurácia modelo = ", old_score[2])
print("Nova acurácia = ", score[2])
#Plota o gráfico
plt.figure()
#plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,epochs), H.history["val_acc"], label="val_acc")
plt.title("Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia/Loss")
plt.legend()
plt.show()
