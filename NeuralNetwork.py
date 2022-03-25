import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import numpy as np
from keras.utils.vis_utils import plot_model

# read the dataset into csv
dataset = pd.read_csv(r'C:\Users\Joshua\PycharmProjects\COVID Prediction\trainingdata.csv').values

X = dataset[:, 0:12]
y = dataset[:,12]


# perform preprocessing in the form of min max scaling
min_max = preprocessing.MinMaxScaler()
X_scaled = min_max.fit_transform(X)

# split into training set, test set, and validation set
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_scaled, y, test_size=0.8, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.2, random_state=1)

# define errors
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# create neural network
model = Sequential([Dense(32, activation='relu', input_shape=(12,), name='dense_1'),
                    Dense(32, activation='relu', name='dense_2'),
                    Dense(16, activation='relu', name='dense_3'),
                    Dense(8, activation='relu', name='dense_4'),
                    Dense(4, activation='relu', name='dense_5'),
                    Dense(32, activation='relu', name='dense_6'),
                    Dense(1, activation='linear', name='dense_7'),])

model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=root_mean_squared_error)


h = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val))

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    print(f'Predicted: {y_pred[i]}, Actual: {[y_test[i]]}')

print('\n')

print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error: ", metrics.mean_squared_error(y_test, y_pred))
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))