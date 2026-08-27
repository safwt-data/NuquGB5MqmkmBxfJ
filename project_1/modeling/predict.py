import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# KNN
with open("../models/project1/knn.pkl", "wb") as file:
    pickle.load(knn, file)
# Make predictions
knn_predictions = knn.predict(X_test)
# logistic Regression
with open("../models/project1/lr.pkl","wb") as file:
    pickle.load(logreg, file)
# Make predictions
lr_predictions = logreg.predict(X_test)
# SVM 
with open("../models/project1/svm.pkl","wb") as file:
    pickle.load(svm, file)
# Make predictions
svm_predictions = svm.predict(X_test)
# Decision Tree
with open("../models/project1/normal_tree.pkl","wb") as file:
    pickle.load(normal_tree, file)
# Make predictions
normal_tree_predictions = normal_tree.predict(X_test)
# Xgboost
with open("../models/project1/xgboost.pkl","wb") as file:
    pickle.load(xgboost, file)
# Make predictions
xgboost_predictions = xgboost.predict(X_test)
# Random Forest
with open("../models/project1/rf.pkl","wb") as file:
    pickle.load(rf, file)
# Make predictions
rf_predictions = rf.predict(X_test)
# Random Forest Tuned
with open("../models/project1/rf_tuned.pkl","wb") as file:
    pickle.load(rf_tuned, file)
# Make predictions
predictions = rf_tuned.predict(X_test)