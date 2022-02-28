import os
import json
from utils import load_datasets, load_target, save_submission
import models
from models.tuning import beyesian_optimization
from models.evaluation import cross_validation_score
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from scikeras.wrappers import KerasClassifier
import warnings
# warnings.filterwarnings('ignore')

config = json.load(open('./config/default.json'))
# X_train, X_test = load_datasets(["Age", "AgeSplit", "EducationNum"])
X_train, X_test = load_datasets(config['features'])
y_train = load_target('Y')

n_jobs = 1


def nn_model(layers, meta):
    """
    This function compiles and returns a Keras model.
    Should be passed to KerasClassifier in the Keras scikit-learn API.
    """
    X_shape_ = meta["X_shape_"]

    dropout = 0.1
    units = 1000
    model = Sequential()
    model.add(Dense(units, input_shape=(X_shape_[1], )))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    for l in range(layers - 1):
        model.add(Dense(units))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta', metrics=['accuracy'])

    return model


estimators = [
    # ('lgbm-shallow', LGBMClassifier(max_depth=5, random_state=0)),
    # ('lgbm-middle', LGBMClassifier(max_depth=8, random_state=0)),
    # ('lgbm-deep', LGBMClassifier(max_depth=-1, random_state=0)),
    # ('rf', RandomForestClassifier(random_state=0, n_jobs=n_jobs)),
    # ('ert', ExtraTreesClassifier(random_state=0, n_jobs=n_jobs)),
    # ('ridge', RidgeClassifier(random_state=0)),
    ('nn-shallow',  make_pipeline(StandardScaler(),
                                  KerasClassifier(model=nn_model, loss='binary_crossentropy',
                                  batch_size=32, epochs=100, layers=3))),
    ('nn-deep',  make_pipeline(StandardScaler(),
                               KerasClassifier(model=nn_model, loss='binary_crossentropy',
                                               batch_size=32, epochs=100, layers=10)))

]
final_estimator = VotingClassifier(
    estimators=[
        ('lgbm-shallow', LGBMClassifier(max_depth=5, random_state=0)),
        ('lgbm-middle', LGBMClassifier(max_depth=8, random_state=0)),
        ('lgbm-deep', LGBMClassifier(max_depth=-1, random_state=0)),
        ('rf', RandomForestClassifier(random_state=0, n_jobs=n_jobs)),
        ('ert', ExtraTreesClassifier(random_state=0, n_jobs=n_jobs)),
        ('ridge', RidgeClassifier(random_state=0)),
        ('nn-shallow',  make_pipeline(StandardScaler(),
                                      KerasClassifier(model=nn_model, batch_size=128, epochs=1000, verbose=False, random_state=0))),
        ('nn-deep',  make_pipeline(StandardScaler(),
                                   KerasClassifier(model=nn_model, batch_size=128, epochs=1000, verbose=False, random_state=0)))
    ],
    voting='hard',
    n_jobs=n_jobs
)

for model in estimators:
    model = model[1]
    print(model)
    cv_score = cross_val_score(model, X_train, y_train, n_jobs=n_jobs, verbose=0,
                               cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=True))
    print(cv_score)
    print(np.mean(cv_score))
