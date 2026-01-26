'''
In this file we have Heart-Disease-Prediction-Without-AUC-ROC
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings("ignore")
import logging
from log_code import setup_logging
logger =  setup_logging('main')
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

class HEART:
    def __init__(self , path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info("The Data loaded Successfully")
            logger.info(f"We have: {self.df.shape[0]} Rows and {self.df.shape[1]} Columns")
            logger.info(f"Missing Values:\n{self.df.isnull().sum()}")
            self.x = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]

            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            logger.info(f'training sample{len(self.x_train)},{len(self.x_test)},{len(self.y_train)},{len(self.y_test)}')
        except Exception as e:
                error_type,error_msg,error_line = sys.exc_info()
                logger.info("Error in Line no : {error_line.tb_lineno} : due to {error_msg}")
    def knn_algo(self):
        try:
            self.reg_knn = KNeighborsClassifier(n_neighbors = 2)
            self.reg_knn.fit(self.x_train,self.y_train)
            logger.info(f'=============KNN Algorithm============')
            logger.info(f'Train Accuracy : {acuracy_score(self.y_train,self.reg_knn.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_knn.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")
    def logistic_regression(self):
        try:
            self.reg_lr = LogisticRegression()
            self.reg_lr.fit(self.x_train,self.y_train)
            logger.info(f"=============Logistic Regression Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_lr.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_lr.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")
    def naive_bayes(self):
        try:
            self.reg_nb = GaussianNB()
            self.reg_nb.fit(self.x_train,self.y_train)
            logger.info(f"=============Naive Bayes Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_nb.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_nb.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")
    def Decision_tree(self):
        try:
            self.reg_dt = DecisionTreeClassifier(criterion = 'entropy')
            logger.info(f'=============Decision Tree Algorithm============')
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_dt.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_dt.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")
    def Random_forest(self):
        try:
            self.reg_rf = RandomForestClassifier(n_estimators = 7,criterion = 'entropy')
            self.reg_rf.fit(self.x_train,self.y_train)
            logger.info(f"=============Random Forest Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_rf.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_rf.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")

    def AdaBoost(self):
        try:
            self.reg_ada = LogisticRegression()
            self.reg_ada = AdaBoostClassifier(n_estimators = 7,learning_rate = 1.0)
            self.reg_ada.fit(self.x_train,self.y_train)
            logger.info(f"=============AdaBoost Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_ada.predict(self.x_train)) * 100}')
            logger.info(f'test Accuracy : {accuracy_score(self.y_test,self.reg_ada.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")
    def Gradient_boosting(self):
        try:
            self.reg_gb = GradientBoostingClassifier(n_estimators = 7,criterion = 'friedman_mse')
            self.reg_gb.fit(self.x_train,self.y_train)
            logger.info(f"=============Gradient Boosting Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_gb.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_gb.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
    def xgboost(self):
        try:
            self.reg_xgb = XGBClassifier()
            self.reg_xgb.fit(self.x_train,self.y_train)
            logger.info(f"=============XGBoost Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_xgb.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_xgb.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
    def SVC(self):
        try:
            self.reg_svc = SVC()
            self.reg_svc.fit(self.x_train,self.y_train)
            logger.info(f"=============SVC Algorithm============")
            logger.info(f'Train Accuracy : {accuracy_score(self.y_train,self.reg_svc.predict(self.x_train)) * 100}')
            logger.info(f'Test Accuracy : {accuracy_score(self.y_test,self.reg_svc.predict(self.x_test)) * 100}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


if __name__ == '__main__':
    try:
        data  = 'heart.csv'
        obj = HEART(data)
        obj.knn_algorithm()
        obj.logistic_regression()
        obj.navie_bayes()
        obj.Decision_tree()
        obj.Random_forest()
        obj.AdaBoost()
        obj.Gradient_boosting()
        obj.XGBoost()
        obj.SVC()
    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')