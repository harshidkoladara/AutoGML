from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# classifiers and regressors
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor, ElasticNet, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge


from math import sqrt
import pandas as pd
import numpy as np
import os

# Plot
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
import pickle


class ML_Model():
    def __init__(self, path, raw):
        self.path = path

        self.data_file = [x for x in os.listdir(self.path) if x[0:3] != 'raw']
        self.raw = raw

        data = pd.read_csv(path + self.data_file[0])
        for x in raw.columns:
            if((list(data[data['Columns'] == x]['Target'])[0] == True) and (list(data[data['Columns'] == x]['Dtype'])[0] == 'bool' or list(
                    data[data['Columns'] == x]['Dtype'])[0] == 'category')):
                self.type = "supervised"
                self.learning = "classification"
            elif(list(data[data['Columns'] == x]['Target'])[0] == True):
                self.type = "supervised"
                self.learning = "regression"

        self.target_column = data[data['Target'] == True]['Columns'].values
        self.x = raw.drop(self.target_column, axis=1).values
        self.y = raw[self.target_column[0]].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42)

    def classification_model(self):
        log_clf = LogisticRegression(
            penalty='l2', solver="liblinear", random_state=42, C=100.0)

        rnd_clf = RandomForestClassifier(
            n_estimators=10, random_state=42)

        svm_clf = SVC(gamma="auto", probability=True,
                      random_state=42, kernel="rbf")

        sgd_clf = SGDClassifier(loss="log", penalty="l1", random_state=42)

        gaussian_clf = GaussianNB()

        mlp_clf = MLPClassifier(random_state=42, solver='adam')

        decision_clf = DecisionTreeClassifier(random_state=42)

        extra_tree = ExtraTreesClassifier(random_state=42)

        estimators = [('log_clf', log_clf), ('rnd_clf', rnd_clf),
                      ('svm_clf', svm_clf), ('sgd_clf', sgd_clf),
                      ('gaussian_clf', gaussian_clf), ('decision_clf', decision_clf),
                      ('extra_tree', extra_tree), ('mlp_clf', mlp_clf)]

        voting_clf = VotingClassifier(estimators, voting='soft')

        voting_clf.fit(self.X_train, self.y_train.ravel())
        estimator_accuracy_score = [accuracy_score(self.y_test.ravel(), estimator.predict(self.X_test))
                                    for estimator in voting_clf.estimators_]

        print(estimator_accuracy_score)
        train_accuracy = voting_clf.score(self.X_train, self.y_train.ravel())
        valid_accuracy = voting_clf.score(self.X_test, self.y_test.ravel())

        filename = f'{self.path}/model.h5'
        pickle.dump([voting_clf, "Classifier", [round(train_accuracy*100, 2), round(valid_accuracy*100, 2)]],
                    open(filename, 'wb'))

    def regression_model(self):
        linear = LinearRegression()
        lgbm = LGBMRegressor(random_state=42)
        xgb = XGBRegressor(random_state=42)
        kernelRidge = KernelRidge()
        elastic = ElasticNet(random_state=42)
        bayesian = BayesianRidge()
        gbr = GradientBoostingRegressor(random_state=42)
        # svr = SVR()
        random = RandomForestRegressor(n_estimators=10, random_state=1)

        estimators = [
            ('linear', linear),
            ('lgbm', lgbm),
            ('xgb', xgb),
            ('kernelRidge', kernelRidge),
            ('elastic', elastic),
            ('bayesian', bayesian),
            ('gbr', gbr),
            # ('svr', svr),
            ('random', random)
        ]
        voting_regressor = VotingRegressor(estimators)
        voting_regressor.fit(self.X_train, self.y_train.ravel())
        linear.fit(self.X_train, self.y_train.ravel())

        # for x in estimators:
        #     x[1].fit(self.X_train, self.y_train.ravel())
        #     print(x[0], ": ", sqrt(mean_squared_error(
        #         self.y_test.ravel(), x[1].predict(self.X_test))), r2_score(self.y_test.ravel(), x[1].predict(self.X_test)))

        pred_on_train = voting_regressor.predict(self.X_train)
        pred_on_test = voting_regressor.predict(self.X_test)

        r2_score_train = r2_score(self.y_train.ravel(), pred_on_train)
        r2_score_test = r2_score(self.y_test.ravel(), pred_on_test)

        rmse_train = sqrt(mean_squared_error(
            self.y_train.ravel(), pred_on_train))
        rmse_test = sqrt(mean_squared_error(self.y_test.ravel(), pred_on_test))

        # for i in range(self.X_test.shape[0]):
        #     print(self.X_test[i], self.y_test[i],
        #           voting_regressor.predict(self.X_test[i].reshape(1, -1)), linear.predict(self.X_test[i].reshape(1, -1)))

        # print(r2_score_train, r2_score_test)
        # print(rmse_train, rmse_test)

        filename = f'{self.path}/model.h5'
        pickle.dump([voting_regressor, 'Regressor', [round(r2_score_train, 4), round(r2_score_test, 4), round(rmse_train, 4), round(rmse_test, 4)]],
                    open(filename, 'wb'))
