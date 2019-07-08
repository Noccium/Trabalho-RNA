
import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('voice.csv')

data['label'] = pd.Categorical(data['label'])
data['label'] = data.label.cat.codes

x_train = tf.constant(data.iloc[:,:-1])
y_train = tf.constant(data['label'])

x_test1 = tf.constant(data.iloc[:50,:-1])
x_test2 = tf.constant(data.iloc[1584:1634,:-1])
x_test = tf.concat(values=[x_test1,x_test2], axis=0)

y_test1 = tf.constant(data.iloc[:50,20])
y_test2 = tf.constant(data.iloc[1584:1634,20])
y_test = tf.concat(values=[y_test1,y_test2], axis=0)

'''
sess = tf.Session()
print(x_test.eval(session=sess))
print(y_test.eval(session=sess))
print(x_test)
print(y_test)
'''

model = tf.keras.models.load_model('my_model.h5')
model.summary()

score = model.evaluate(x_train, y_train, verbose=0, steps=1)
print(model.metrics_names)
print('loss = ', score[0])
print('mae = ', score[1])
print('acc =', score[2])
