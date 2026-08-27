import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from dataset import load_data


def split_data():
    y = df['Y']
    X = df.drop('Y',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=42, stratify= y )
    logging.info("Dataset split successfully")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data()

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    logging.info("Trained successfully")
    return model


# KNN
knn = KNeighborsClassifier(n_neighbors=20)

trained_knn = train_model(knn, X_train, y_train)
'''
# Logistic Regression
with open("../models/project1/lr.pkl","wb") as file:
    pickle.dump(logreg, file)
# SVM 
with open("../models/project1/svm.pkl","wb") as file:
    pickle.dump(svm, file)
# Decision Tree
with open("../models/project1/normal_tree.pkl","wb") as file:
    pickle.dump(normal_tree, file)
# Xgboost
with open("../models/project1/xgboost.pkl","wb") as file:
    pickle.dump(xgboost, file)
# Random Forest
with open("../models/project1/rf.pkl","wb") as file:
    pickle.dump(rf, file)
# Random Forest Tuned
with open("../models/project1/rf_tuned.pkl","wb") as file:
    pickle.dump(rf_tuned, file)

'''
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)