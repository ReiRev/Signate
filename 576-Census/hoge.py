import os
import json
from utils import load_datasets, load_target
import models
from models.tuning import beyesian_optimization
from models.evaluation import cross_validation_score
import json

n_trials = 10

config = json.load(open('./config/default.json'))
# X_train, X_test = load_datasets(["Age", "AgeSplit", "EducationNum"])
X_train, X_test = load_datasets(config['features'])
y_train = load_target('Y')

nn = models.NN({})
optimized_params = beyesian_optimization(nn, X_train, y_train, {
    'layers': 12,
    'dropout': [0.00001, 0.9],
    'units': [10, 1000],
    'nb_epoch': [100, 100000],
}, n_trials)
