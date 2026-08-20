import pickle
# KNN
with open("../models/project1/knn.pkl", "wb") as file:
    pickle.dump(knn, file)
# logistic Regression
with open("../models/project1/lr.pkl","wb") as file:
    pickle.dump(logreg, file)
# SVM 
with open("../models/project1/svm.pkl","wb") as file:
    pickle.dump(logreg, file)
# Decision Tree
with open("../models/project1/normal_tree.pkl","wb") as file:
    pickle.dump(logreg, file)
# Xgboost
with open("../models/project1/xgboost.pkl","wb") as file:
    pickle.dump(logreg, file)
# Random Forest
with open("../models/project1/rf.pkl","wb") as file:
    pickle.dump(logreg, file)
# Random Forest Tuned
with open("../models/project1/rf_tuned.pkl","wb") as file:
    pickle.dump(logreg, file)