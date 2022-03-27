import autogluon as ag
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
from sklearn.model_selection import train_test_split

#load dataset into dataframe object
dataset = TabularDataset('/home/joshua/trainingdata.csv')

sample_size = 100
train_data = dataset.sample(n=sample_size, random_state=0)

label_column = 'actual'

#dir = '/home/joshua/TransitProject/modelStorage'

#predictor = TabularPredictor.load(dir)

#predictor = TabularPredictor(label=label_column, path=dir).fit(train_data, num_bag_folds=5, num_bag_sets=1,
#        num_stack_levels=3)


test_data = dataset
y_test = test_data[label_column]
test_data_nolab = test_data.drop(columns=[label_column])




# Tune the hyperparameters of the neural network
hp_tune = True

nn_options = {
        'num_epochs': 100,
        }

gbm_options = {
        'num_boost_round': 100,
        }

hyperparameters = {
        'GBM': gbm_options,
        'NN': nn_options,
        }

time_limits = 2*60 
num_trials = 5
search_strategy = 'skopt'
dir = '/home/joshua/TransitProject/stackStorage1'

predictor = TabularPredictor.load(dir)

#predictor = TabularPredictor(label=label_column, path=dir).fit(train_data, num_bag_folds=5, num_bag_sets=1,
 #       num_stack_levels=3)

y_pred = predictor.predict(test_data_nolab)


perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



#results = predictor.fit_summary(show_plot=True)

for i in range(100):
    print(f'Predicted: {y_pred[i]}, Actual: {y_test[i]}')

print('\n')
print(perf)

