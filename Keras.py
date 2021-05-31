import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('drive/My Drive/Colab Notebooks/LinearRegression/trainingdata3.csv')

dataset = df.values
dataset
X = dataset[:,0:12]
Y = dataset[:,12]


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.8)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.2)


def my_metric_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`



model = Sequential([Dense(32, activation='relu', input_shape=(12,)),    
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(8, activation='relu'),
                    Dense(4, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(1, activation='linear'),])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[my_metric_fn])
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.losses.MeanAbsoluteError()])
# tuner = kt.Hyperband(model_builder,
#                      objective = 'val_accuracy', 
#                      max_epochs = 10,
#                      factor = 3,
#                      project_name = 'intro_to_kt')


hist = model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_data=(X_val, Y_val))
