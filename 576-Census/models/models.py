from .base import Model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt


class Lgbm(Model):
    def fit(self, X_train, y_train):
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X_train, y_train, test_size=0.3,
                             random_state=0, stratify=y_train)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)
        self.model = lgb.train(self.params, lgb_train,
                               valid_sets=[lgb_train, lgb_eval], verbose_eval=False)

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        return y_pred


class RandomForest(Model):
    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(self.scaler.transform(X_train), y_train)

    def predict(self, X_pred):
        y_pred = self.model.predict(self.scaler.transform(X_pred))
        return y_pred


class ERT(Model):
    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.model = ExtraTreesClassifier(**self.params)
        self.model.fit(self.scaler.transform(X_train), y_train)

    def predict(self, X_pred):
        y_pred = self.model.predict(self.scaler.transform(X_pred))
        return y_pred


class NN(Model):
    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

        patience = self.params['patience']
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        batch_size = self.params['batch_size']

        model = Sequential()
        model.add(Dense(units, input_shape=(X_train.shape[1],)))
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
        early_stopping = EarlyStopping(
            monitor='loss', patience=patience, verbose=0)
        model.compile(loss='binary_crossentropy',
                      optimizer='adagrad',  metrics=['accuracy'])

        self.history = model.fit(self.scaler.transform(X_train), y_train,
                                 epochs=nb_epoch, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
        self.model = model

    def predict(self, X_pred):
        y_pred = self.model.predict(self.scaler.transform(X_pred)).flatten()
        return y_pred

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def plot(self):
        plt.plot(self.history.epoch,
                 self.history.history["accuracy"], label="acc")
        plt.plot(self.history.epoch,
                 self.history.history["loss"], label="loss")
        plt.xlabel("epoch")
        plt.legend()
