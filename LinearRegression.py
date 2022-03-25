import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn import metrics

# read the dataset into csv
dataset = pd.read_csv(r'C:\Users\Joshua\PycharmProjects\COVID Prediction\trainingdata.csv')


# perform preprocessing

X = pd.DataFrame(preprocessing.scale(dataset.iloc[:,:-1]))
y = pd.DataFrame(dataset.iloc[:,-1])



y_pred_results = []
y_test_results = []

MAE = []
MSE = []
RMSE = []
model = LinearRegression()
# create a loop to average the results
for i in range(50, 0, -1):
    # separate the data into training sets and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # create the Linear Regression model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_results = model.predict(X_test)
    y_test_results = y_test.values

    MAE.append(metrics.mean_absolute_error(y_test, y_pred))
    MSE.append(metrics.mean_squared_error(y_test, y_pred))
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# results
for i in range(len(y_pred_results)):
    print(f'Predicted: {y_pred_results[i]}, Actual: {y_test_results[i]}')

print('\n')

# print error metrics
print("Average mean absolute error: ", sum(MAE)/len(MAE))
print("Average mean squared error: ", sum(MSE)/len(MSE))
print("Average root mean squared error: ", sum(RMSE)/len(RMSE))

pre = []
t = []
for i in range(20):
    pre.append(y_pred_results[i][0])
    t.append(y_test_results[i][0])

experiments = list(range(1, 21))


X_axis = np.arange(len(experiments))

plt.bar(X_axis - 0.2, pre, 0.4, label="Predicted")
plt.bar(X_axis + 0.2, t, 0.4, label="Actual")

plt.xticks(X_axis, experiments)
plt.xlabel("Experiments")
plt.ylabel("Transit App Volume Percentage")
plt.title("Predicted vs Actual Transit App Volume Percentage")
plt.legend()
plt.grid()

plt.show()






# bar graph for weights of each attribute
l = ["gas price", "unemployment\nrate", "death\nincrease", "hospitalized\nincrease", 'hospitalized\ncurrently',
     'negative\nincrease', 'positive\nincrease', 'day of\nthe week', 'month', 'weekend', 'holiday', 'normal']

plt.bar(l, model.coef_[0])
plt.title("Attribute Weights")
plt.xlabel("Attributes")
plt.ylabel("Weights")

for x, y in zip(l, model.coef_[0]):
    label = "{:.2f}".format(y)

    plt.annotate(label,
                 (x, y),
                 textcoords="offset points",
                 xytext=(0, 3),
                 ha='center')
plt.grid()

plt.show()

