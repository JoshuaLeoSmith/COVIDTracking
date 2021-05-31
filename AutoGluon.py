import autogluon as ag
from autogluon import TabularPrediction as task 
import numpy as np

# load dataset into dataframe object
train_data = task.Dataset(file_path='drive/My Drive/Colab Notebooks/LinearRegression/trainingdata3.csv')

# establish training subsample
subsample_size = 528
train_data = train_data.sample(n=subsample_size, random_state=0)
# print(train_data.head)

# identify and describe label column
label_column = 'actual'
print("Summary of actual column: \n", train_data['actual'].describe())

#identify model output directory
dir = 'drive/My Drive/Colab Notebooks/LinearRegression/AgModelDir'

# create testing data that is different from training
new_data = task.Dataset(file_path='drive/My Drive/Colab Notebooks/LinearRegression/trainingdata3.csv')
test_data = new_data.sample(n=subsample_size, random_state=1)

# label column test data
y_test = test_data[label_column]

# delete label column
test_data_nolabel = test_data.drop(labels=[label_column], axis=1)

val_data = new_data.sample(n=subsample_size, random_state=2)

# metric = 'accuracy'

hp_tune = True  # whether or not to do hyperparameter optimization

nn_options = {  # specifies non-default hyperparameter values for neural network models
    'num_epochs': 100,  # number of training epochs (controls training time of NN models)
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
    'layers': ag.space.Categorical([100],[1000],[200,100],[300,200,100]),  # each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
}


y_pred = predictor.predict(test_data_nolabel)
print("Predictions:  ", list(y_pred)[:5])
print("Actual: ", list(y_test)[:5])
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=False)


results = predictor.fit_summary()



gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {  # hyperparameters of each model type
                   'GBM': gbm_options,
                   'NN': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                  }  # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limits = 2*60  # train various models for ~2 min
num_trials = 5  # try at most 3 different hyperparameter configurations for each type of model
search_strategy = 'skopt'  # to tune hyperparameters using SKopt Bayesian optimization routine
output_dir = 'drive/My Drive/Colab Notebooks/LinearRegression/AgStack'
# predictor = task.fit(train_data=train_data, tuning_data=val_data, label=label_column,
#                     num_trials=num_trials, time_limits=time_limits,
#                      hyperparameter_tune=hp_tune, hyperparameters=hyperparameters,
#                      search_strategy=search_strategy)

predictor = task.fit(train_data=train_data, label=label_column, output_directory=output_dir,
                     num_bagging_folds=5, num_bagging_sets=1, stack_ensemble_levels=3, tuning_data=val_data, eval_metric='mean_absolute_error')
#predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir, eval_metric='mean_absolute_error')
