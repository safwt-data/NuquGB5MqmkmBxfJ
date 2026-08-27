import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from project_1.dataset import load_data
from project_1.modeling.train import split_data
from project_1.modeling.train import train_model

def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    logging.info("Predictions successfully")
    return predictions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    knn = KNeighborsClassifier(n_neighbors=20)
    trained_knn = train_model(knn, X_train, y_train)
    knn_predictions = make_predictions(trained_knn, X_test)

    logreg = LogisticRegression(random_state =42)
    trained_logreg = train_model(logreg, X_train, y_train)
    logreg_predictions = make_predictions(trained_logreg, X_test)

    svm = SVC(random_state =42)
    trained_svm = train_model(svm, X_train, y_train)
    svm_predictions = make_predictions(trained_svm, X_test)

    normal_tree = DecisionTreeClassifier(random_state=42,
                                     class_weight='balanced')
    trained_normal_tree = train_model(normal_tree, X_train, y_train)
    tree_predictions = make_predictions(trained_normal_tree,X_test)

    xgboost = XGBClassifier(random_state =42)
    trained_xgboost = train_model(xgboost,X_train,y_train)
    xgboost_predications = make_predictions(trained_xgboost,X_test)

    rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced')
    trained_rf = train_model(rf,X_train,y_train)
    rf_predictions = make_predictions(trained_rf, X_test)

    tuned_rf = RandomForestClassifier(random_state=42,  
                            class_weight='balanced',
                            max_depth = 17,
                            min_samples_leaf = 5,
                            min_samples_split = 6,
                            n_estimators = 675
                           )
    trained_tuned_rf = train_model(tuned_rf, X_train, y_train)
    tuned_rf_predictions = make_predictions(trained_tuned_rf, X_test)

# python -m project_1.modeling.predict



