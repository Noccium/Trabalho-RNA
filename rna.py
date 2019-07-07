import numpy as np
import tensorflow as tf

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, input_dim=2,activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['mae', 'acc'])

model.fit(X, y, batch_size=4, epochs=1000)

predictions = model.predict(X)
print(predictions)

score = model.evaluate(X, y, verbose=0)
print(model.metrics_names)
print(score)
