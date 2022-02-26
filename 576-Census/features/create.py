import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


# class Pclass(Feature):
#     def create_features(self):
#         self.train['Pclass'] = train['Pclass']
#         self.test['Pclass'] = test['Pclass']


class Age(Feature):
    def create_features(self):
        self.train['age'] = train['age']
        self.test['age'] = test['age']


class AgeSplit(Feature):
    def create_features(self):
        bins = [-1, 17, 21, 22, 23, 28, 33, 37, 100]
        self.train['age-split'] = pd.cut(train['age'],
                                         bins=bins, labels=[i for i in range(1, len(bins))])
        self.test['age-split'] = pd.cut(test['age'],
                                        bins=bins, labels=[i for i in range(1, len(bins))])


class Fnlwgt(Feature):
    def create_features(self):
        self.train['fnlwgt'] = train['fnlwgt']
        self.test['fnlwgt'] = test['fnlwgt']


class EducationNum(Feature):
    def create_features(self):
        self.train['education-num'] = train['education-num']
        self.test['education-num'] = test['education-num']


class Sex(Feature):
    def create_features(self):
        self.train['sex'] = train['sex'].map({'Male': 1, 'Female': 0})
        self.test['sex'] = test['sex'].map({'Male': 1, 'Female': 0})


class WorkClass(Feature):
    def create_features(self):
        self.train = pd.get_dummies(
            train['workclass'], prefix="workclass", prefix_sep="-")
        self.test = pd.get_dummies(
            test['workclass'], prefix="workclass", prefix_sep="-")


class Education(Feature):
    def create_features(self):
        column = 'education'
        prefix = column
        self.train = pd.get_dummies(
            train[column], prefix=prefix, prefix_sep="-")
        self.test = pd.get_dummies(
            test[column], prefix=prefix, prefix_sep="-")


class MaritalStatus(Feature):
    def create_features(self):
        column = 'marital-status'
        prefix = column
        self.train = pd.get_dummies(
            train[column], prefix=prefix, prefix_sep="-")
        self.test = pd.get_dummies(
            test[column], prefix=prefix, prefix_sep="-")


class Occupation(Feature):
    def create_features(self):
        column = 'occupation'
        prefix = column
        self.train = pd.get_dummies(
            train[column], prefix=prefix, prefix_sep="-")
        self.test = pd.get_dummies(
            test[column], prefix=prefix, prefix_sep="-")


class Relationship(Feature):
    def create_features(self):
        column = 'relationship'
        prefix = column
        self.train = pd.get_dummies(
            train[column], prefix=prefix, prefix_sep="-")
        self.test = pd.get_dummies(
            test[column], prefix=prefix, prefix_sep="-")


class Race(Feature):
    def create_features(self):
        column = 'race'
        prefix = column
        self.train = pd.get_dummies(
            train[column], prefix=prefix, prefix_sep="-")
        self.test = pd.get_dummies(
            test[column], prefix=prefix, prefix_sep="-")


class NativeCountry(Feature):
    def create_features(self):
        column = 'native-country'
        prefix = column
        self.train = pd.get_dummies(
            train[column], prefix=prefix, prefix_sep="-")
        self.test = pd.get_dummies(
            test[column], prefix=prefix, prefix_sep="-")


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
