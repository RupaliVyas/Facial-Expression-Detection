import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")   

#================ DATA PREPROCESSING ==========================

data = pd.read_csv('./fer2013.csv')

width, height = 48, 48

datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).as_matrix()


# # print(X)
# # print(X.shape) =>(35887, 48, 48, 1)
# # print(y)
# # print(y.shape) =>(35887, 7)


X-= np.mean(X, axis=0)
X/= np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(48,48, 1)))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(48,48, 1)))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape=(48,48, 1)))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape=(48,48, 1)))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(256, (3,3), input_shape=(48,48, 1)))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(256, (3,3), input_shape=(48,48, 1)))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, validation_split=0.1, epochs=20,batch_size=128,verbose=1)
#model.evaluate(X_test, y_test,batch_size=128,verbose=0)
#model.save('cnn1.model')

cvscores = []

reconstructed_model = tf.keras.models.load_model("cnn1.model")
scores = reconstructed_model.evaluate(X_test, y_test,batch_size=128,verbose=0)
print("%s: %.2f%%" % (reconstructed_model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))













